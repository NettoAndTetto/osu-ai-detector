import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment

import numpy.typing as npt

from ..tokenizer import Event, EventType

MILISECONDS_PER_SECOND = 1000


def load_audio_file(file: Path, sample_rate: int, speed: float = 1.0) -> npt.NDArray:
    """Load an audio file as a numpy time-series array

    The signals are resampled, converted to mono channel, and normalized.

    Args:
        file: Path to audio file.
        sample_rate: Sample rate to resample the audio.
        speed: Speed multiplier for the audio.

    Returns:
        samples: Audio time series.
    """
    file = Path(file)
    audio = AudioSegment.from_file(file, format=file.suffix[1:])
    audio.frame_rate = int(audio.frame_rate * speed)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples *= 1.0 / np.max(np.abs(samples))
    return samples


def update_event_times(
        events: list[Event],
        event_times: list[int],
        end_time: Optional[float] = None,
        types_first: bool = False
) -> None:
    """Extends the event times list with the times of the new events if the event list is longer than the event times list.

    Args:
        events: List of events.
        event_times: List of event times.
        end_time: End time of the events, for interpolation.
        types_first: If True, the type token is at the start of the group before the timeshift token.
    """
    non_timed_events = [
        EventType.BEZIER_ANCHOR,
        EventType.PERFECT_ANCHOR,
        EventType.CATMULL_ANCHOR,
        EventType.RED_ANCHOR,
    ]
    timed_events = [
        EventType.CIRCLE,
        EventType.SPINNER,
        EventType.SPINNER_END,
        EventType.SLIDER_HEAD,
        EventType.LAST_ANCHOR,
        EventType.SLIDER_END,
        EventType.BEAT,
        EventType.MEASURE,
    ]

    start_index = len(event_times)
    end_index = len(events)
    current_time = 0 if len(event_times) == 0 else event_times[-1]
    for i in range(start_index, end_index):
        if types_first:
            if i + 1 < end_index and events[i + 1].type == EventType.TIME_SHIFT:
                current_time = events[i + 1].value
        elif events[i].type == EventType.TIME_SHIFT:
            current_time = events[i].value
        event_times.append(current_time)

    # Interpolate time for control point events
    interpolate = False
    if types_first:
        # Start-T-D-CP-D-CP-D-LCP-T-D-End-T-D
        # 1-----1-1-1--1-1--1-7---7-7-9---9-9
        # 1-----1-1-3--3-5--5-7---7-7-9---9-9
        index = range(start_index, end_index)
        current_time = 0 if len(event_times) == 0 else event_times[-1]
    else:
        # T-D-Start-D-CP-D-CP-T-D-LCP-T-D-End
        # 1-1-1-----1-1--1-1--7-7--7--9-9-9--
        # 1-1-1-----3-3--5-5--7-7--7--9-9-9--
        index = range(end_index - 1, start_index - 1, -1)
        current_time = end_time if end_time is not None else event_times[-1]
    for i in index:
        event = events[i]

        if event.type in timed_events:
            interpolate = False

        if event.type in non_timed_events:
            interpolate = True

        if not interpolate:
            current_time = event_times[i]
            continue

        if event.type not in non_timed_events:
            event_times[i] = current_time
            continue

        # Find the time of the first timed event and the number of control points between
        j = i
        step = 1 if types_first else -1
        count = 0
        other_time = current_time
        while 0 <= j < len(events):
            event2 = events[j]
            if event2.type == EventType.TIME_SHIFT:
                other_time = event_times[j]
                break
            if event2.type in non_timed_events:
                count += 1
            j += step
        if j < 0:
            other_time = 0
        if j >= len(events):
            other_time = end_time if end_time is not None else event_times[-1]

        # Interpolate the time
        current_time = int((current_time - other_time) / (count + 1) * count + other_time)
        event_times[i] = current_time


def merge_events(events1: list[Event], event_times1: list[int], events2: list[Event], event_times2: list[int]) -> tuple[list[Event], list[int]]:
    """Merge two lists of events in a time sorted manner. Assumes both lists are sorted by time.

    Args:
        events1: List of events.
        event_times1: List of event times.
        events2: List of events.
        event_times2: List of event times.

    Returns:
        merged_events: Merged list of events.
        merged_event_times: Merged list of event times.
    """
    merged_events = []
    merged_event_times = []
    i = 0
    j = 0

    while i < len(events1) and j < len(events2):
        t1 = event_times1[i]
        t2 = event_times2[j]

        if t1 <= t2:
            merged_events.append(events1[i])
            merged_event_times.append(t1)
            i += 1
        else:
            merged_events.append(events2[j])
            merged_event_times.append(t2)
            j += 1

    merged_events.extend(events1[i:])
    merged_events.extend(events2[j:])
    merged_event_times.extend(event_times1[i:])
    merged_event_times.extend(event_times2[j:])
    return merged_events, merged_event_times


def remove_events_of_type(events: list[Event], event_times: list[int], event_types: list[EventType]) -> tuple[list[Event], list[int]]:
    """Remove all events of a specific type from a list of events.

    Args:
        events: List of events.
        event_types: Types of event to remove.

    Returns:
        filtered_events: Filtered list of events.
    """
    new_events = []
    new_event_times = []
    for event, time in zip(events, event_times):
        if event.type not in event_types:
            new_events.append(event)
            new_event_times.append(time)
    return new_events, new_event_times


def speed_events(events: list[Event], event_times: list[int], speed: float) -> tuple[list[Event], list[int]]:
    """Change the speed of a list of events.

    Args:
        events: List of events.
        event_times: List of event times
        speed: Speed multiplier.

    Returns:
        sped_events: Sped up list of events.
    """
    sped_events = []
    for event in events:
        if event.type == EventType.TIME_SHIFT:
            event.value = int(event.value / speed)
        sped_events.append(event)

    sped_event_times = []
    for t in event_times:
        sped_event_times.append(int(t / speed))

    return sped_events, sped_event_times


@dataclasses.dataclass
class Group:
    event_type: EventType = None
    time: int = 0
    distance: int = None
    x: float = None
    y: float = None
    new_combo: bool = False
    hitsounds: list[int] = dataclasses.field(default_factory=list)
    samplesets: list[int] = dataclasses.field(default_factory=list)
    additions: list[int] = dataclasses.field(default_factory=list)
    volumes: list[int] = dataclasses.field(default_factory=list)


type_events = [
    EventType.CIRCLE,
    EventType.SPINNER,
    EventType.SPINNER_END,
    EventType.SLIDER_HEAD,
    EventType.BEZIER_ANCHOR,
    EventType.PERFECT_ANCHOR,
    EventType.CATMULL_ANCHOR,
    EventType.RED_ANCHOR,
    EventType.LAST_ANCHOR,
    EventType.SLIDER_END,
    EventType.BEAT,
    EventType.MEASURE,
]


def get_groups(
        events: list[Event],
        *,
        event_times: Optional[list[int]] = None,
        types_first: bool = False
) -> list[Group]:
    groups = []
    group = Group()
    for i, event in enumerate(events):
        if event.type == EventType.TIME_SHIFT:
            group.time = event.value
        elif event.type == EventType.DISTANCE:
            group.distance = event.value
        elif event.type == EventType.POS_X:
            group.x = event.value
        elif event.type == EventType.POS_Y:
            group.y = event.value
        elif event.type == EventType.NEW_COMBO:
            group.new_combo = True
        elif event.type == EventType.HITSOUND:
            group.hitsounds.append((event.value % 8) * 2)
            group.samplesets.append(((event.value // 8) % 3) + 1)
            group.additions.append(((event.value // 24) % 3) + 1)
        elif event.type == EventType.VOLUME:
            group.volumes.append(event.value)
        elif event.type in type_events:
            if types_first:
                if group.event_type is not None:
                    groups.append(group)
                    group = Group()
                group.event_type = event.type
                if event_times is not None:
                    group.time = event_times[i]
            else:
                group.event_type = event.type
                if event_times is not None:
                    group.time = event_times[i]
                groups.append(group)
                group = Group()

    if group.event_type is not None:
        groups.append(group)

    return groups


def get_group_indices(events: list[Event], types_first: bool = False) -> list[list[int]]:
    groups = []
    indices = []
    for i, event in enumerate(events):
        indices.append(i)
        if event.type in type_events:
            if types_first:
                if len(indices) > 1:
                    groups.append(indices[:-1])
                    indices = [indices[-1]]
            else:
                groups.append(indices)
                indices = []

    if len(indices) > 0:
        groups.append(indices)

    return groups
