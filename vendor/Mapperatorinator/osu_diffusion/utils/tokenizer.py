import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from osu_diffusion.config import DiffusionTrainConfig


class Tokenizer:
    __slots__ = [
        "num_classes",
        "num_diff_classes",
        "max_difficulty",
        "beatmap_idx",
        "mapper_idx",
        "beatmap_mapper",
        "num_mapper_classes",
        "beatmap_descriptors",
        "descriptor_idx",
        "num_descriptor_classes",
        "num_cs_classes",
    ]

    def __init__(self, args: DiffusionTrainConfig = None):
        """Fixed vocabulary tokenizer."""
        self.beatmap_idx: dict[int, int] = {}  # beatmap_id -> beatmap_idx
        self.num_classes = 0
        self.num_diff_classes = 0
        self.max_difficulty = 0
        self.beatmap_mapper: dict[int, int] = {}  # beatmap_id -> mapper_id
        self.mapper_idx: dict[int, int] = {}  # mapper_id -> mapper_idx
        self.num_mapper_classes = 0
        self.beatmap_descriptors: dict[int, list[int]] = {}  # beatmap_id -> [descriptor_idx]
        self.descriptor_idx: dict[str, int] = {}  # descriptor_name -> descriptor_idx
        self.num_descriptor_classes = 0
        self.num_cs_classes = 0

        if args is not None:
            if args.data.beatmap_class:
                self._init_beatmap_idx(args)

            if args.data.difficulty_class:
                self.num_diff_classes = args.data.num_diff_classes
                self.max_difficulty = args.data.max_diff

            if args.data.mapper_class:
                self._init_mapper_idx(args)

            if args.data.descriptor_class:
                self._init_descriptor_idx(args)

            if args.data.circle_size_class:
                self.num_cs_classes = args.data.num_cs_classes

    def encode_style(self, beatmap_id: int) -> int:
        """Converts beatmap id into token id."""
        return self.beatmap_idx.get(beatmap_id, self.num_classes - 1)

    @property
    def style_unk(self) -> int:
        """Gets the unknown style value token id."""
        return self.num_classes - 1

    def encode_diff(self, diff: float) -> int:
        """Converts difficulty value into token id."""
        return self.num_classes + np.clip(int(diff * (self.num_diff_classes - 2) / self.max_difficulty), 0, self.num_diff_classes - 2)

    @property
    def diff_unk(self) -> int:
        """Gets the unknown difficulty value token id."""
        return self.num_classes + self.num_diff_classes - 1

    def encode_mapper(self, beatmap_id: int) -> int:
        """Converts beatmap id into token id."""
        user_id = self.beatmap_mapper.get(beatmap_id, -1)
        return self.encode_mapper_id(user_id)

    def encode_mapper_id(self, user_id: int) -> int:
        """Converts user id into token id."""
        mapper_idx = self.mapper_idx.get(user_id, self.num_mapper_classes - 1)
        return self.num_classes + self.num_diff_classes + mapper_idx

    @property
    def mapper_unk(self) -> int:
        """Gets the unknown mapper value token id."""
        return self.num_classes + self.num_diff_classes + self.num_mapper_classes - 1

    def encode_descriptor(self, beatmap_id: int) -> list[int]:
        """Converts beatmap id into token ids."""
        return [self.encode_descriptor_idx(descriptor_idx) for descriptor_idx in self.beatmap_descriptors.get(beatmap_id, [self.num_descriptor_classes - 1])]

    def encode_descriptor_name(self, descriptor: str) -> int:
        """Converts descriptor into token id."""
        descriptor_idx = self.descriptor_idx.get(descriptor, self.num_descriptor_classes)
        return self.encode_descriptor_idx(descriptor_idx)

    def encode_descriptor_idx(self, descriptor_idx: int) -> int:
        """Converts descriptor idx into token id."""
        return self.num_classes + self.num_diff_classes + self.num_mapper_classes + descriptor_idx

    @property
    def descriptor_unk(self) -> int:
        """Gets the unknown descriptor value token id."""
        return self.num_classes + self.num_diff_classes + self.num_mapper_classes + self.num_descriptor_classes - 1

    def encode_cs(self, cs: float) -> int:
        """Converts circle size value into token id."""
        return (self.num_classes + self.num_diff_classes + self.num_mapper_classes + self.num_descriptor_classes
                + np.clip(int(cs * (self.num_cs_classes - 2) / 10), 0, self.num_cs_classes - 2))

    @property
    def cs_unk(self) -> int:
        """Gets the unknown circle size value token id."""
        return (self.num_classes + self.num_diff_classes + self.num_mapper_classes + self.num_descriptor_classes
                + self.num_cs_classes - 1)

    @property
    def num_tokens(self) -> int:
        """Gets the number of tokens."""
        return (self.num_classes + self.num_diff_classes + self.num_mapper_classes + self.num_descriptor_classes
                + self.num_cs_classes)

    def _init_beatmap_idx(self, args: DiffusionTrainConfig) -> None:
        """Initializes and caches the beatmap index."""
        if args is None or "train_dataset_path" not in args.data:
            return

        path = Path(args.data.train_dataset_path)
        cache_path = path / "beatmap_idx.pickle"

        if cache_path.exists():
            with open(path / "beatmap_idx.pickle", "rb") as f:
                self.beatmap_idx = pickle.load(f)
            self.num_classes = max(self.beatmap_idx.values()) + 2
            return

        print("Caching beatmap index...")
        highest_index = -1

        for track in tqdm(path.iterdir()):
            if not track.is_dir():
                continue
            metadata_file = track / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            for beatmap_name in metadata["Beatmaps"]:
                beatmap_metadata = metadata["Beatmaps"][beatmap_name]
                index = beatmap_metadata["Index"]
                self.beatmap_idx[beatmap_metadata["BeatmapId"]] = index
                highest_index = max(highest_index, index)

        self.num_classes = highest_index + 2

        with open(cache_path, "wb") as f:
            pickle.dump(self.beatmap_idx, f)

    def _init_mapper_idx(self, args):
        """"Indexes beatmap mappers and mapper idx."""
        if args is None or "mappers_path" not in args.data:
            raise ValueError("mappers_path not found in args")

        path = Path(args.data.mappers_path)

        if not path.exists():
            raise ValueError(f"mappers_path {path} not found")

        # Load JSON data from file
        with open(path, 'r') as file:
            data = json.load(file)

        # Populate beatmap_mapper
        for item in data:
            self.beatmap_mapper[item['id']] = item['user_id']

        # Get unique user_ids from beatmap_mapper values
        unique_user_ids = list(set(self.beatmap_mapper.values()))

        # Create mapper_idx
        self.mapper_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.num_mapper_classes = len(unique_user_ids) + 1

    def _init_descriptor_idx(self, args):
        """"Indexes beatmap descriptors and descriptor idx."""
        if args is None or "descriptors_path" not in args.data:
            raise ValueError("descriptors_path not found in args")

        path = Path(args.data.descriptors_path)

        if not path.exists():
            raise ValueError(f"descriptors_path {path} not found")

        # The descriptors file is a CSV file with the following format:
        # beatmap_id,descriptor_name
        with open(path, 'r') as file:
            data = file.readlines()

        # Populate descriptor_idx
        for line in data:
            _, descriptor_name = line.strip().split(',')
            if descriptor_name not in self.descriptor_idx:
                self.descriptor_idx[descriptor_name] = len(self.descriptor_idx)

        # Populate beatmap_descriptors
        for line in data:
            beatmap_id_str, descriptor_name = line.strip().split(',')
            beatmap_id = int(beatmap_id_str)
            descriptor_idx = self.descriptor_idx[descriptor_name]
            if beatmap_id not in self.beatmap_descriptors:
                self.beatmap_descriptors[beatmap_id] = []
            self.beatmap_descriptors[beatmap_id].append(descriptor_idx)

        self.num_descriptor_classes = len(self.descriptor_idx) + 1

    def state_dict(self):
        return {
            "beatmap_idx": self.beatmap_idx,
            "num_classes": self.num_classes,
            "num_diff_classes": self.num_diff_classes,
            "max_difficulty": self.max_difficulty,
            "beatmap_mapper": self.beatmap_mapper,
            "mapper_idx": self.mapper_idx,
            "num_mapper_classes": self.num_mapper_classes,
            "beatmap_descriptors": self.beatmap_descriptors,
            "descriptor_idx": self.descriptor_idx,
            "num_descriptor_classes": self.num_descriptor_classes,
            "num_cs_classes": self.num_cs_classes,
        }

    def load_state_dict(self, state_dict):
        self.beatmap_idx = state_dict["beatmap_idx"]
        self.num_classes = state_dict["num_classes"]
        self.num_diff_classes = state_dict["num_diff_classes"]
        self.max_difficulty = state_dict["max_difficulty"]
        if "beatmap_mapper" in state_dict:
            self.beatmap_mapper = state_dict["beatmap_mapper"]
        if "mapper_idx" in state_dict:
            self.mapper_idx = state_dict["mapper_idx"]
        if "num_mapper_classes" in state_dict:
            self.num_mapper_classes = state_dict["num_mapper_classes"]
        if "beatmap_descriptors" in state_dict:
            self.beatmap_descriptors = state_dict["beatmap_descriptors"]
        if "descriptor_idx" in state_dict:
            self.descriptor_idx = state_dict["descriptor_idx"]
        if "num_descriptor_classes" in state_dict:
            self.num_descriptor_classes = state_dict["num_descriptor_classes"]
        if "num_cs_classes" in state_dict:
            self.num_cs_classes = state_dict["num_cs_classes"]
