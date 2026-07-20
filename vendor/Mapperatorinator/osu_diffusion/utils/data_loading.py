import json
import math
import os.path
import pickle
import random
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

import torch
import tqdm
from slider import Position
from slider.beatmap import Beatmap
from slider.beatmap import HitObject
from slider.beatmap import Slider
from slider.beatmap import Spinner
from slider.curve import Catmull
from slider.curve import Linear
from slider.curve import MultiBezier
from slider.curve import Perfect
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import IterableDataset

from .positional_embedding import timestep_embedding
from .tokenizer import Tokenizer
from ..config import DataConfig

playfield_size = torch.tensor((512, 384))
feature_size = 19


def create_datapoint(time: timedelta, pos: Position, datatype: int) -> torch.Tensor:
    features = torch.zeros(19)
    features[0] = max(-256, min(768, pos.x))
    features[1] = max(-256, min(640, pos.y))
    features[2] = time.total_seconds() * 1000
    features[datatype + 3] = 1

    return features


def repeat_type(repeat: int) -> int:
    if repeat < 4:
        return repeat - 1
    elif repeat % 2 == 0:
        return 3
    else:
        return 4


def append_control_points(
    datapoints: list[torch.Tensor],
    slider: Slider,
    datatype: int,
    duration: timedelta,
):
    control_point_count = len(slider.curve.points)

    for i in range(1, control_point_count - 1):
        time = slider.time + i / (control_point_count - 1) * duration
        pos = slider.curve.points[i]
        datapoints.append(create_datapoint(time, pos, datatype))


def get_data(hitobj: HitObject) -> torch.Tensor:
    if isinstance(hitobj, Slider) and len(hitobj.curve.points) < max(16, min(128, int(hitobj.length * 0.5))):
        datapoints = [
            create_datapoint(
                hitobj.time,
                hitobj.position,
                5 if hitobj.new_combo else 4,
            ),
        ]

        assert hitobj.repeat >= 1
        duration: timedelta = (hitobj.end_time - hitobj.time) / hitobj.repeat

        if isinstance(hitobj.curve, Linear):
            append_control_points(datapoints, hitobj, 9, duration)
        elif isinstance(hitobj.curve, Catmull):
            append_control_points(datapoints, hitobj, 8, duration)
        elif isinstance(hitobj.curve, Perfect):
            append_control_points(datapoints, hitobj, 7, duration)
        elif isinstance(hitobj.curve, MultiBezier):
            control_point_count = len(hitobj.curve.points)

            for i in range(1, control_point_count - 1):
                time = hitobj.time + i / (control_point_count - 1) * duration
                pos = hitobj.curve.points[i]

                if pos == hitobj.curve.points[i + 1]:
                    datapoints.append(create_datapoint(time, pos, 9))
                elif pos != hitobj.curve.points[i - 1]:
                    datapoints.append(create_datapoint(time, pos, 6))

        datapoints.append(
            create_datapoint(hitobj.time + duration, hitobj.curve.points[-1], 10),
        )

        slider_end_pos = hitobj.curve(1)
        datapoints.append(
            create_datapoint(
                hitobj.end_time,
                slider_end_pos,
                11 + repeat_type(hitobj.repeat),
            ),
        )

        return torch.stack(datapoints, 0)

    if isinstance(hitobj, Spinner):
        return torch.stack(
            (
                create_datapoint(hitobj.time, hitobj.position, 2),
                create_datapoint(hitobj.end_time, hitobj.position, 3),
            ),
            0,
        )

    return create_datapoint(
        hitobj.time,
        hitobj.position,
        1 if hitobj.new_combo else 0,
    ).unsqueeze(0)


def beatmap_to_sequence(beatmap: Beatmap) -> torch.Tensor:
    # Get the hit objects
    hit_objects = beatmap.hit_objects(stacking=False)
    data_chunks = [get_data(ho) for ho in hit_objects]

    sequence = torch.concatenate(data_chunks, 0)
    sequence = torch.swapaxes(sequence, 0, 1)

    return sequence.float()


def random_flip(seq: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        seq[0] = 512 - seq[0]
    if random.random() < 0.5:
        seq[1] = 384 - seq[1]
    return seq


def calc_distances(seq: torch.Tensor) -> torch.Tensor:
    offset = torch.roll(seq[:2, :], 1, 1)
    offset[0, 0] = 256
    offset[1, 0] = 192
    seq_d = torch.linalg.vector_norm(seq[:2, :] - offset, ord=2, dim=0)
    return seq_d


def split_and_process_sequence(
        seq: torch.Tensor,
        double_time: bool = False,
        distance_std: float = 0.0,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    seq_d = calc_distances(seq)

    # Augment distances to add noise
    if distance_std > 0:
        seq_d *= torch.pow(2, torch.normal(0, distance_std, seq_d.shape))

    # Augment and normalize positions for diffusion
    seq_x = random_flip(seq[:2, :]) / playfield_size.unsqueeze(1) * 2 - 1

    seq_o = seq[2, :]
    # Augment the time vector with random speed change
    if double_time:
        seq_o /= 1.5
    # Obscure the absolute time by normalizing to zero and adding a random offset between zero and the max period
    # We do this to make sure the offset embedding utilizes the full range of values, which is also the case when sampling the model
    seq_o = seq_o - seq_o[0] + random.random() * 1000000

    seq_c = torch.concatenate(
        [
            timestep_embedding(seq_o * 0.1, 128).T,
            timestep_embedding(seq_d, 128).T,
            seq[3:, :],
        ],
        0,
    )

    return (seq_x, seq_c), seq.shape[1]


def split_and_process_sequence_no_augment(
    seq: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    seq_d = calc_distances(seq)
    # Augment and normalize positions for diffusion
    seq_x = seq[:2, :] / playfield_size.to(seq.device).unsqueeze(1) * 2 - 1
    seq_o = seq[2, :]
    seq_c = torch.concatenate(
        [
            timestep_embedding(seq_o * 0.1, 128).T,
            timestep_embedding(seq_d, 128).T,
            seq[3:, :],
        ],
        0,
    )

    return (seq_x, seq_c), seq.shape[1]


def window_split_sequence(seq, s, e):
    seq_x, seq_c = seq
    x = seq_x[:, s:e]
    c = seq_c[:, s:e]

    return x, c


def load_metadata(track_path: Path) -> dict:
    metadata_file = track_path / "metadata.json"
    with open(metadata_file) as f:
        return json.load(f)


def get_difficulty(metadata: dict, beatmap_name: str, double_time: bool = False):
    if double_time:
        return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["64"]
    return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["0"]


def get_class_vector(
        args: DataConfig,
        tokenizer: Tokenizer,
        beatmap: Beatmap,
        beatmap_name: str,
        metadata: dict,
        double_time: bool = False,
) -> torch.Tensor:
    class_vector = torch.zeros(tokenizer.num_tokens)
    beatmap_id = beatmap.beatmap_id
    if args.beatmap_class:
        if random.random() < args.class_dropout_prob:
            class_vector[tokenizer.style_unk] = 1
        else:
            class_vector[tokenizer.encode_style(beatmap_id)] = 1
    if args.difficulty_class:
        if random.random() < args.diff_dropout_prob:
            class_vector[tokenizer.diff_unk] = 1
        else:
            difficulty = get_difficulty(metadata, beatmap_name, double_time)
            class_vector[tokenizer.encode_diff(difficulty)] = 1
    if args.mapper_class:
        if random.random() < args.mapper_dropout_prob:
            class_vector[tokenizer.mapper_unk] = 1
        else:
            class_vector[tokenizer.encode_mapper(beatmap_id)] = 1
    if args.descriptor_class:
        if random.random() < args.descriptor_dropout_prob:
            class_vector[tokenizer.descriptor_unk] = 1
        else:
            for idx in tokenizer.encode_descriptor(beatmap_id):
                class_vector[idx] = 1
    if args.circle_size_class:
        if random.random() < args.cs_dropout_prob:
            class_vector[tokenizer.cs_unk] = 1
        else:
            class_vector[tokenizer.encode_cs(beatmap.circle_size)] = 1
    return class_vector


class BeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "beatmap_idx",
        "seq_len",
        "stride",
        "index",
        "current_class",
        "current_seq",
        "current_seq_len",
        "seq_index",
        "args",
        "tokenizer",
    )

    def __init__(
        self,
        beatmap_files: list[Path],
        args: DataConfig,
        tokenizer: Tokenizer,
    ):
        self.beatmap_files = beatmap_files
        self.seq_len = args.seq_len
        self.stride = args.stride
        self.args = args
        self.tokenizer = tokenizer
        self.index = 0
        self.current_class = None
        self.current_seq = None
        self.current_seq_len = -1
        self.seq_index = 0

    def __iter__(self) -> "BeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, torch.Tensor]:
        while (
            self.current_seq is None
            or self.seq_index + self.seq_len > self.current_seq_len
        ):
            if self.index >= len(self.beatmap_files):
                raise StopIteration

            # Load the beatmap from file
            beatmap_path = self.beatmap_files[self.index]
            beatmap = Beatmap.from_path(beatmap_path)
            metadata = load_metadata(beatmap_path.parents[1])

            double_time = random.random() < self.args.double_time_prob
            self.current_class = get_class_vector(
                self.args,
                self.tokenizer,
                beatmap,
                beatmap_path.stem,
                metadata,
                double_time
            )
            self.current_seq, self.current_seq_len = split_and_process_sequence(beatmap_to_sequence(beatmap), double_time, self.args.distance_std)
            self.seq_index = random.randint(0, self.stride - 1)
            self.index += 1

        # Return the preprocessed hit objects as a sequence of overlapping windows
        window = window_split_sequence(
            self.current_seq,
            self.seq_index,
            self.seq_index + self.seq_len,
        )
        self.seq_index += self.stride
        return window, self.current_class


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
        self,
        beatmap_files: list[Path],
        iterable_factory: Callable,
        cycle_length: int,
    ):
        per_worker = int(math.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                beatmap_files[
                    i * per_worker: min(len(beatmap_files), (i + 1) * per_worker)
                ]
            )
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


class BeatmapDataset(IterableDataset):
    def __init__(
        self,
        args: DataConfig,
        tokenizer: Tokenizer,
        beatmap_files: Optional[list[Path]] = None,
    ):
        super(BeatmapDataset).__init__()
        self.args = args
        self.path = args.train_dataset_path
        self.start = args.start
        self.end = args.end
        self.tokenizer = tokenizer
        self.beatmap_files = beatmap_files

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

    def __iter__(self) -> InterleavingBeatmapDatasetIterable | BeatmapDatasetIterable:
        beatmap_files = self._get_beatmap_files()

        if self.args.shuffle:
            random.shuffle(beatmap_files)

        if self.args.cycle_length > 1:
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
            self.tokenizer,
        )


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def get_beatmap_idx(name) -> dict[int, int]:
    p = Path(__file__).with_name(name)
    with p.open("rb") as f:
        beatmap_idx = pickle.load(f)
    return beatmap_idx


def get_beatmap_files(name: str, data_path: str) -> list[PurePosixPath]:
    p = Path(name)
    with p.open("rb") as f:
        relative_beatmap_files = pickle.load(f)
    beatmap_files = [PurePosixPath(data_path, *PureWindowsPath(f).parts) for f in relative_beatmap_files]
    return beatmap_files


class CachedDataset(Dataset):
    __slots__ = "cached_data"

    def __init__(self, cached_data):
        self.cached_data = cached_data

    def __getitem__(self, index):
        return self.cached_data[index]

    def __len__(self):
        return len(self.cached_data)


def cache_dataset(
        out_path: str,
        args: DataConfig,
        tokenizer: Tokenizer,
        beatmap_files: Optional[list[str]] = None,
):
    dataset = BeatmapDataset(
        args=args.data,
        tokenizer=tokenizer,
        beatmap_files=beatmap_files,
    )

    print("Caching dataset...")
    cached_data = []
    for datum in tqdm.tqdm(dataset):
        cached_data.append(datum)

    torch.save(cached_data, out_path)


def get_cached_data_loader(
        data_path: str,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
):
    cached_data = torch.load(data_path)
    dataset = CachedDataset(cached_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        shuffle=shuffle,
    )

    return dataloader


def get_data_loader(
        args: DataConfig,
        tokenizer: Tokenizer,
        pin_memory: bool = False,
        drop_last: bool = False,
        beatmap_files: Optional[list[str]] = None,
        num_processes: int = 1,
) -> DataLoader:
    dataset = BeatmapDataset(
        args=args.data,
        tokenizer=tokenizer,
        beatmap_files=beatmap_files,
    )

    batch_size = args.optim.batch_size // args.optim.grad_acc // num_processes

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=args.dataloader.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=args.dataloader.num_workers > 0,
    )

    return dataloader
