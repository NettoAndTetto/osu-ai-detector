from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    LAST_ANCHOR = "last_anchor"
    SLIDER_END = "slider_end"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"
