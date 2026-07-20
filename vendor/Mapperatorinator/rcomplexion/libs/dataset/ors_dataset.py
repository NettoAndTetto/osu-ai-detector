from __future__ import annotations

import json
import os
import random
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from omegaconf import DictConfig
from slider import Beatmap
from torch.utils.data import IterableDataset

from .data_utils import create_sequences, tokenize_events
from .osu_parser import OsuParser
from ..tokenizer import Tokenizer


class OrsDataset(IterableDataset):
    __slots__ = (
        "path",
        "start",
        "end",
        "args",
        "parser",
        "tokenizer",
        "beatmap_files",
        "test",
        "shared",
        "sample_weights",
    )

    def __init__(
            self,
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            beatmap_files: Optional[list[Path]] = None,
            test: bool = False,
            shared: Namespace = None,
    ):
        """Manage and process ORS dataset.

        Attributes:
            args: Data loading arguments.
            parser: Instance of OsuParser class.
            tokenizer: Instance of Tokenizer class.
            beatmap_files: List of beatmap files to process. Overrides track index range.
            test: Whether to load the test dataset.
        """
        super().__init__()
        self.path = args.test_dataset_path if test else args.train_dataset_path
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.beatmap_files = beatmap_files
        self.test = test
        self.shared = shared
        self.sample_weights = None

        if os.path.exists(self.args.sample_weights):
            # Load the sample weights csv to a dictionary
            with open(self.args.sample_weights, "r") as f:
                self.sample_weights = {int(line.split(",")[0]): float(line.split(",")[1]) for line in f.readlines()}
                # Normalize the weights so the mean is 1
                mean = sum(self.sample_weights.values()) / len(self.sample_weights)
                self.sample_weights = {k: v / mean for k, v in self.sample_weights.items()}

    def _get_beatmap_files(self) -> list[Path]:
        if self.beatmap_files is not None:
            return self.beatmap_files

        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            for beatmap_file in os.listdir(
                    os.path.join(self.path, track_name, "beatmaps"),
            ):
                beatmap_files.append(
                    Path(
                        os.path.join(
                            self.path,
                            track_name,
                            "beatmaps",
                            beatmap_file,
                        )
                    ),
                )

        return beatmap_files

    def __iter__(self):
        beatmap_files = self._get_beatmap_files()

        if not self.test:
            random.shuffle(beatmap_files)

        if self.args.cycle_length > 1 and not self.test:
            return InterleavingBeatmapDatasetIterable(
                beatmap_files,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(beatmap_files).__iter__()

    def _iterable_factory(self, beatmap_files: list[Path]):
        return BeatmapDatasetIterable(
            beatmap_files,
            self.args,
            self.parser,
            self.tokenizer,
            self.test,
            self.shared,
            self.sample_weights,
        )


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
            self,
            beatmap_files: list[Path],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(np.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                beatmap_files[
                i * per_worker: min(len(beatmap_files), (i + 1) * per_worker)
                ]
            ).__iter__()
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
        raise StopIteration


class BeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "args",
        "parser",
        "tokenizer",
        "test",
        "shared",
        "frame_seq_len",
        "min_pre_token_len",
        "pre_token_len",
        "class_dropout_prob",
        "diff_dropout_prob",
        "add_pre_tokens",
        "add_empty_sequences",
        "sample_weights",
    )

    def __init__(
            self,
            beatmap_files: list[Path],
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            test: bool,
            shared: Namespace,
            sample_weights: dict[int, float] = None,
    ):
        self.beatmap_files = beatmap_files
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.sample_weights = sample_weights

    def __iter__(self):
        return self._get_next_beatmaps()

    @staticmethod
    def _load_metadata(track_path: Path) -> dict:
        metadata_file = track_path / "metadata.json"
        with open(metadata_file) as f:
            return json.load(f)

    @staticmethod
    def _get_difficulty(metadata: dict, beatmap_name: str):
        return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["0"]

    def _get_next_beatmaps(self) -> dict:
        for beatmap_path in self.beatmap_files:
            metadata = self._load_metadata(beatmap_path.parents[1])

            if self.args.min_difficulty > 0 and self._get_difficulty(metadata, beatmap_path.stem) < self.args.min_difficulty:
                continue

            for sample in self._get_next_beatmap(beatmap_path):
                yield sample

    def _get_next_beatmap(self, beatmap_path: Path) -> dict:
        osu_beatmap = Beatmap.from_path(beatmap_path)
        events = self.parser.parse(osu_beatmap)
        tokens = tokenize_events(events, self.tokenizer)
        sequences, labels = create_sequences(tokens, self.args.src_seq_len, self.tokenizer)

        if self.sample_weights is not None:
            # Get the weight for the current beatmap
            weight = max(self.sample_weights.get(osu_beatmap.beatmap_id, 1.0), 0.1)

            for sequence, label in zip(sequences, labels):
                yield {
                    "input_ids": sequence,
                    "labels": label,
                    "sample_weights": weight,
                }
        else:
            for sequence, label in zip(sequences, labels):
                yield {
                    "input_ids": sequence,
                    "labels": label,
                }
