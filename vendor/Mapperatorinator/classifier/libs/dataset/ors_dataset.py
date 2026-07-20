from __future__ import annotations

import json
import os
import random
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig
from slider import Beatmap
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file
from .osu_parser import OsuParser
from ..tokenizer import Event, EventType, Tokenizer

OSZ_FILE_EXTENSION = ".osz"
AUDIO_FILE_NAME = "audio.mp3"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100


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
    )

    def __init__(
            self,
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            beatmap_files: Optional[list[Path]] = None,
            test: bool = False,
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

    def _get_track_paths(self) -> list[Path]:
        track_paths = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
            track_paths.append(Path(os.path.join(self.path, track_name)))
        return track_paths

    def __iter__(self):
        beatmap_files = self._get_track_paths() if self.args.per_track else self._get_beatmap_files()

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
        "frame_seq_len",
        "pre_token_len",
        "add_empty_sequences",
    )

    def __init__(
            self,
            beatmap_files: list[Path],
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            test: bool,
    ):
        self.beatmap_files = beatmap_files
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.frame_seq_len = args.src_seq_len - 1

    def _get_frames(self, samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Segment audio samples into frames.

        Each frame has `frame_size` audio samples.
        It will also calculate and return the time of each audio frame, in miliseconds.

        Args:
            samples: Audio time-series.

        Returns:
            frames: Audio frames.
            frame_times: Audio frame times.
        """
        samples = np.pad(samples, [0, self.args.hop_length - len(samples) % self.args.hop_length])
        frames = np.reshape(samples, (-1, self.args.hop_length))
        frames_per_milisecond = (
                self.args.sample_rate / self.args.hop_length / MILISECONDS_PER_SECOND
        )
        frame_times = np.arange(len(frames)) / frames_per_milisecond
        return frames, frame_times

    def _create_sequences(
            self,
            frames: npt.NDArray,
            frame_times: npt.NDArray,
            context: dict,
            extra_data: Optional[dict] = None,
    ) -> list[dict[str, int | npt.NDArray | list[Event]]]:
        """Create frame and token sequences for training/testing.

        Args:
            frames: Audio frames.

        Returns:
            A list of source and target sequences.
        """

        def get_event_indices(events2: list[Event], event_times2: list[int]) -> tuple[list[int], list[int]]:
            if len(events2) == 0:
                return [], []

            # Corresponding start event index for every audio frame.
            start_indices = []
            event_index = 0

            for current_time in frame_times:
                while event_index < len(events2) and event_times2[event_index] < current_time:
                    event_index += 1
                start_indices.append(event_index)

            # Corresponding end event index for every audio frame.
            end_indices = start_indices[1:] + [len(events2)]

            return start_indices, end_indices

        start_indices, end_indices = get_event_indices(context["events"], context["event_times"])

        sequences = []
        n_frames = len(frames)
        offset = random.randint(0, self.frame_seq_len)
        # Divide audio frames into splits
        for frame_start_idx in range(offset, n_frames, self.frame_seq_len):
            frame_end_idx = min(frame_start_idx + self.frame_seq_len, n_frames)

            def slice_events(context, frame_start_idx, frame_end_idx):
                if len(context["events"]) == 0:
                    return []
                event_start_idx = start_indices[frame_start_idx]
                event_end_idx = end_indices[frame_end_idx - 1]
                return context["events"][event_start_idx:event_end_idx]

            def slice_context(context, frame_start_idx, frame_end_idx):
                return {"events": slice_events(context, frame_start_idx, frame_end_idx)}

            # Create the sequence
            sequence = {
                           "time": frame_times[frame_start_idx],
                           "frames": frames[frame_start_idx:frame_end_idx],
                           "context": slice_context(context, frame_start_idx, frame_end_idx),
                       } | extra_data

            sequences.append(sequence)

        return sequences

    def _normalize_time_shifts(self, sequence: dict) -> dict:
        """Make all time shifts in the sequence relative to the start time of the sequence,
        and normalize time values.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with trimmed time shifts.
        """

        def process(events: list[Event], start_time) -> list[Event] | tuple[list[Event], int]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    # We cant modify the event objects themselves because that will affect subsequent sequences
                    events[i] = Event(EventType.TIME_SHIFT, int((event.value - start_time) * STEPS_PER_MILLISECOND))

            return events

        start_time = sequence["time"]
        del sequence["time"]

        sequence["context"]["events"] = process(sequence["context"]["events"], start_time)

        return sequence

    def _tokenize_sequence(self, sequence: dict) -> dict:
        """Tokenize the event sequence.

        Begin token sequence with `[SOS]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with tokenized events.
        """
        context = sequence["context"]
        tokens = torch.empty(len(context["events"]), dtype=torch.long)
        for i, event in enumerate(context["events"]):
            tokens[i] = self.tokenizer.encode(event)
        context["tokens"] = tokens

        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> dict:
        """Pad token sequence to a fixed length and split decoder input and labels.

        Pad with `[PAD]` tokens until `tgt_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a. decoder ground truth.

        Prefix the token sequence with the pre_tokens sequence.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        # Count reducible tokens, pre_tokens and context tokens
        num_tokens = len(sequence["context"]["tokens"])

        # Trim tokens to target sequence length
        # n + padding = tgt_seq_len
        n = min(self.args.tgt_seq_len, num_tokens)
        si = 0

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=torch.long)

        tokens = sequence["context"]["tokens"]

        input_tokens[si:si + n] = tokens[:n]

        # Randomize some input tokens
        def randomize_tokens(tokens):
            offset = torch.randint(low=-self.args.timing_random_offset, high=self.args.timing_random_offset + 1,
                                   size=tokens.shape)
            return torch.where((self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) & (
                    tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                               torch.clamp(tokens + offset,
                                           self.tokenizer.event_start[EventType.TIME_SHIFT],
                                           self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                               tokens)

        if self.args.timing_random_offset > 0:
            input_tokens[si:si + n] = randomize_tokens(input_tokens[si:si + n])

        sequence["decoder_input_ids"] = input_tokens
        sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id

        del sequence["context"]

        return sequence

    def _pad_frame_sequence(self, sequence: dict) -> dict:
        """Pad frame sequence with zeros until `frame_seq_len`.

        Frame sequence can be further processed into Mel spectrogram frames,
        which is the input to the transformer encoder.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded frames.
        """
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)

        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded_frames = torch.zeros(
                self.frame_seq_len,
                frames.shape[-1],
                dtype=frames.dtype,
                device=frames.device,
            )
            padded_frames[:n] = frames[:n]
            sequence["frames"] = torch.flatten(padded_frames)
        else:
            sequence["frames"] = torch.flatten(frames)

        return sequence

    def __iter__(self):
        return self._get_next_tracks() if self.args.per_track else self._get_next_beatmaps()

    @staticmethod
    def _load_metadata(track_path: Path) -> dict:
        metadata_file = track_path / "metadata.json"
        with open(metadata_file) as f:
            return json.load(f)

    def _get_difficulty(self, metadata: dict, beatmap_name: str, speed: float = 1.0, beatmap: Beatmap = None) -> float:
        if beatmap is not None and (all(e == 1.5 for e in self.args.dt_augment_range) or speed not in [1.0, 1.5]):
            return beatmap.stars(speed_scale=speed)

        if speed == 1.5:
            return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["64"]
        return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["0"]

    @staticmethod
    def _get_idx(metadata: dict, beatmap_name: str):
        return metadata["Beatmaps"][beatmap_name]["Index"]

    def _get_speed_augment(self):
        mi, ma = self.args.dt_augment_range
        return random.random() * (ma - mi) + mi if random.random() < self.args.dt_augment_prob else 1.0

    def _get_next_beatmaps(self) -> dict:
        for beatmap_path in self.beatmap_files:
            metadata = self._load_metadata(beatmap_path.parents[1])

            if self.args.min_difficulty > 0 and self._get_difficulty(metadata,
                                                                     beatmap_path.stem) < self.args.min_difficulty:
                continue

            speed = self._get_speed_augment()
            audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
            audio_samples = load_audio_file(audio_path, self.args.sample_rate, speed)

            for sample in self._get_next_beatmap(audio_samples, beatmap_path, speed):
                yield sample

    def _get_next_tracks(self) -> dict:
        for track_path in self.beatmap_files:
            metadata = self._load_metadata(track_path)

            if self.args.min_difficulty > 0 and all(self._get_difficulty(metadata, beatmap_name)
                                                    < self.args.min_difficulty for beatmap_name in
                                                    metadata["Beatmaps"]):
                continue

            speed = self._get_speed_augment()
            audio_path = track_path / list(track_path.glob('audio.*'))[0]
            audio_samples = load_audio_file(audio_path, self.args.sample_rate, speed)

            for beatmap_name in metadata["Beatmaps"]:
                beatmap_path = (track_path / "beatmaps" / beatmap_name).with_suffix(".osu")

                if self.args.min_difficulty > 0 and self._get_difficulty(metadata,
                                                                         beatmap_name) < self.args.min_difficulty:
                    continue

                for sample in self._get_next_beatmap(audio_samples, beatmap_path, speed):
                    yield sample

    def _get_next_beatmap(self, audio_samples, beatmap_path: Path, speed: float) -> dict:
        frames, frame_times = self._get_frames(audio_samples)
        osu_beatmap = Beatmap.from_path(beatmap_path)

        if osu_beatmap.beatmap_id not in self.tokenizer.beatmap_mapper:
            return

        extra_data = {
            "labels": self.tokenizer.mapper_idx[self.tokenizer.beatmap_mapper[osu_beatmap.beatmap_id]],
        }

        flip_x, flip_y = False, False
        if self.args.augment_flip:
            flip_x, flip_y = random.random() < 0.5, random.random() < 0.5
        
        events, event_times = self.parser.parse(osu_beatmap, speed, flip_x, flip_y)
        in_context = {"events": events, "event_times": event_times}

        sequences = self._create_sequences(
            frames,
            frame_times,
            in_context,
            extra_data,
        )

        for sequence in sequences:
            sequence = self._normalize_time_shifts(sequence)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            yield sequence
