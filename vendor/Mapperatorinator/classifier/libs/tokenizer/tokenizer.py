import json
from pathlib import Path

from omegaconf import DictConfig

from .event import Event, EventType, EventRange

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Tokenizer:
    __slots__ = [
        "offset",
        "event_ranges",
        "input_event_ranges",
        "num_classes",
        "num_diff_classes",
        "max_difficulty",
        "event_range",
        "event_start",
        "event_end",
        "vocab_size",
        "beatmap_idx",
        "mapper_idx",
        "beatmap_mapper",
        "num_mapper_classes",
        "beatmap_descriptors",
        "descriptor_idx",
        "num_descriptor_classes",
        "num_cs_classes",
    ]

    def __init__(self, args: DictConfig = None):
        """Fixed vocabulary tokenizer."""
        self.offset = 1
        self.event_ranges: list[EventRange] = [
            EventRange(EventType.TIME_SHIFT, 0, 1024),
            EventRange(EventType.SNAPPING, 0, 16),
            EventRange(EventType.DISTANCE, 0, 640),
        ]
        self.num_classes = 0
        self.beatmap_mapper: dict[int, int] = {}  # beatmap_id -> mapper_id
        self.mapper_idx: dict[int, int] = {}  # mapper_id -> mapper_idx

        if args is not None:
            miliseconds_per_sequence = ((args.data.src_seq_len - 1) * args.model.spectrogram.hop_length *
                                        MILISECONDS_PER_SECOND / args.model.spectrogram.sample_rate)
            max_time_shift = int(miliseconds_per_sequence / MILISECONDS_PER_STEP)
            min_time_shift = 0

            self.event_ranges = [
                EventRange(EventType.TIME_SHIFT, min_time_shift, max_time_shift),
                EventRange(EventType.SNAPPING, 0, 16),
            ]

            self._init_mapper_idx(args)

            if args.data.add_distances:
                self.event_ranges.append(EventRange(EventType.DISTANCE, 0, 640))

            if args.data.add_positions:
                p = args.data.position_precision
                x_min, x_max, y_min, y_max = args.data.position_range
                x_min, x_max, y_min, y_max = x_min // p, x_max // p, y_min // p, y_max // p

                if args.data.position_split_axes:
                    self.event_ranges.append(EventRange(EventType.POS_X, x_min, x_max))
                    self.event_ranges.append(EventRange(EventType.POS_Y, y_min, y_max))
                else:
                    x_count = x_max - x_min + 1
                    y_count = y_max - y_min + 1
                    self.event_ranges.append(EventRange(EventType.POS, 0, x_count * y_count - 1))

        self.event_ranges: list[EventRange] = self.event_ranges + [
            EventRange(EventType.NEW_COMBO, 0, 0),
            EventRange(EventType.HITSOUND, 0, 2 ** 3 * 3 * 3),
            EventRange(EventType.VOLUME, 0, 100),
            EventRange(EventType.CIRCLE, 0, 0),
            EventRange(EventType.SPINNER, 0, 0),
            EventRange(EventType.SPINNER_END, 0, 0),
            EventRange(EventType.SLIDER_HEAD, 0, 0),
            EventRange(EventType.BEZIER_ANCHOR, 0, 0),
            EventRange(EventType.PERFECT_ANCHOR, 0, 0),
            EventRange(EventType.CATMULL_ANCHOR, 0, 0),
            EventRange(EventType.RED_ANCHOR, 0, 0),
            EventRange(EventType.LAST_ANCHOR, 0, 0),
            EventRange(EventType.SLIDER_END, 0, 0),
            EventRange(EventType.BEAT, 0, 0),
            EventRange(EventType.MEASURE, 0, 0),
        ]

        if args is not None and args.data.add_timing_points:
            self.event_ranges.append(EventRange(EventType.TIMING_POINT, 0, 0))

        self.event_range: dict[EventType, EventRange] = {er.type: er for er in self.event_ranges}

        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self.offset
        for er in self.event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset

        self.vocab_size: int = self.offset + sum(
            er.max_value - er.min_value + 1 for er in self.event_ranges
        )

    @property
    def pad_id(self) -> int:
        """[PAD] token for padding."""
        return 0

    def decode(self, token_id: int) -> Event:
        """Converts token ids into Event objects."""
        offset = self.offset
        for er in self.event_ranges:
            if offset <= token_id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + token_id - offset)
            offset += er.max_value - er.min_value + 1
        for er in self.input_event_ranges:
            if offset <= token_id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + token_id - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"id {token_id} is not mapped to any event")

    def encode(self, event: Event) -> int:
        """Converts Event objects into token ids."""
        if event.type not in self.event_range:
            raise ValueError(f"unknown event type: {event.type}")

        er = self.event_range[event.type]
        offset = self.event_start[event.type]

        if not er.min_value <= event.value <= er.max_value:
            raise ValueError(
                f"event value {event.value} is not within range "
                f"[{er.min_value}, {er.max_value}] for event type {event.type}"
            )

        return offset + event.value - er.min_value

    def event_type_range(self, event_type: EventType) -> tuple[int, int]:
        """Get the token id range of each Event type."""
        if event_type not in self.event_range:
            raise ValueError(f"unknown event type: {event_type}")

        er = self.event_range[event_type]
        offset = self.event_start[event_type]
        return offset, offset + (er.max_value - er.min_value)

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
        self.num_classes = len(unique_user_ids)

    def state_dict(self):
        return {
            "offset": self.offset,
            "event_ranges": self.event_ranges,
            "num_classes": self.num_classes,
            "event_range": self.event_range,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "vocab_size": self.vocab_size,
            "beatmap_mapper": self.beatmap_mapper,
            "mapper_idx": self.mapper_idx,
        }

    def load_state_dict(self, state_dict):
        self.offset = state_dict["offset"]
        self.event_ranges = state_dict["event_ranges"]
        self.num_classes = state_dict["num_classes"]
        self.event_range = state_dict["event_range"]
        self.event_start = state_dict["event_start"]
        self.event_end = state_dict["event_end"]
        self.vocab_size = state_dict["vocab_size"]
        self.beatmap_mapper = state_dict["beatmap_mapper"]
        self.mapper_idx = state_dict["mapper_idx"]
