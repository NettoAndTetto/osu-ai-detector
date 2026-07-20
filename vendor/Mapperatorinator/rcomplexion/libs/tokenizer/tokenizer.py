from omegaconf import DictConfig

from .event import Event, EventType, EventRange


class Tokenizer:
    __slots__ = [
        "_offset",
        "event_ranges",
        "input_event_ranges",
        "event_range",
        "event_start",
        "event_end",
        "vocab_size_out",
        "vocab_size_in",
    ]

    def __init__(self, args: DictConfig):
        """Fixed vocabulary tokenizer."""
        self._offset = 1

        self.event_ranges: list[EventRange] = [
            EventRange(EventType.TIME_SHIFT, 0, int((args.data.max_time - args.data.min_time) * args.data.time_resolution)),
        ]

        self.input_event_ranges: list[EventRange] = [
            EventRange(EventType.CIRCLE, 0, 0),
            EventRange(EventType.SPINNER, 0, 0),
            EventRange(EventType.SPINNER_END, 0, 0),
            EventRange(EventType.SLIDER_HEAD, 0, 0),
            EventRange(EventType.LAST_ANCHOR, 0, 0),
            EventRange(EventType.SLIDER_END, 0, 0),
        ]

        self.event_range: dict[EventType, EventRange] = {er.type: er for er in self.event_ranges} | {er.type: er for er in self.input_event_ranges}

        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self._offset
        for er in self.event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset
        for er in self.input_event_ranges:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset

        self.vocab_size_out: int = self._offset + sum(
            er.max_value - er.min_value + 1 for er in self.event_ranges
        )
        self.vocab_size_in: int = self.vocab_size_out + sum(
            er.max_value - er.min_value + 1 for er in self.input_event_ranges
        )

    @property
    def pad_id(self) -> int:
        """[PAD] token for padding."""
        return 0

    def decode(self, token_id: int) -> Event:
        """Converts token ids into Event objects."""
        offset = self._offset
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

    def state_dict(self):
        return {
            "event_ranges": self.event_ranges,
            "input_event_ranges": self.input_event_ranges,
            "event_range": self.event_range,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "vocab_size_out": self.vocab_size_out,
            "vocab_size_in": self.vocab_size_in,
        }

    def load_state_dict(self, state_dict):
        self.event_ranges = state_dict["event_ranges"]
        self.input_event_ranges = state_dict["input_event_ranges"]
        self.event_range = state_dict["event_range"]
        self.event_start = state_dict["event_start"]
        self.event_end = state_dict["event_end"]
        self.vocab_size_out = state_dict["vocab_size_out"]
        self.vocab_size_in = state_dict["vocab_size_in"]
