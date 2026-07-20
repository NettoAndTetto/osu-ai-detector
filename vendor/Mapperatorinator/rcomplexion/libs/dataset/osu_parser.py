from __future__ import annotations

from datetime import timedelta

import numpy as np
from omegaconf import DictConfig
from slider import Beatmap, Circle, Slider, Spinner

from ..tokenizer import Event, EventType


class OsuParser:
    def __init__(self, args: DictConfig) -> None:
        self.time_resolution = args.time_resolution
        self.min_time = args.min_time
        self.max_timeshift = int((args.max_time - args.min_time) * self.time_resolution)

    def parse(self, beatmap: Beatmap) -> list[Event]:
        # noinspection PyUnresolvedReferences
        """Parse an .osu beatmap.

        Each hit object is parsed into a list of Event objects, in order of its
        appearance in the beatmap. In other words, in ascending order of time.

        Args:
            beatmap: Beatmap object parsed from an .osu file.

        Returns:
            events: List of Event object lists.

        Example::
            >>> beatmap = [
                "64,80,11000,1,0",
                "100,100,16000,2,0,B|200:200|250:200|250:200|300:150,2"
            ]
            >>> events = parse(beatmap)
            >>> print(events)
            [
                Event(EventType.TIME_SHIFT, 11000), Event(EventType.DISTANCE, 36), Event(EventType.CIRCLE),
                Event(EventType.TIME_SHIFT, 16000), Event(EventType.DISTANCE, 42), Event(EventType.SLIDER_HEAD),
                Event(EventType.TIME_SHIFT, 16500), Event(EventType.DISTANCE, 141), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 17000), Event(EventType.DISTANCE, 50), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 17500), Event(EventType.DISTANCE, 10), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 18000), Event(EventType.DISTANCE, 64), Event(EventType.LAST _ANCHOR),
                Event(EventType.TIME_SHIFT, 20000), Event(EventType.DISTANCE, 11), Event(EventType.SLIDER_END)
            ]
        """
        hit_objects = beatmap.hit_objects(spinners=False, stacking=False)
        last_time = timedelta(seconds=-100)
        events = []

        for hit_object in hit_objects:
            if isinstance(hit_object, Circle):
                last_time = self._parse_circle(hit_object, events, last_time, beatmap)
            elif isinstance(hit_object, Slider):
                last_time = self._parse_slider(hit_object, events, last_time, beatmap)
            elif isinstance(hit_object, Spinner):
                # last_time = self._parse_spinner(hit_object, events, last_time, beatmap)
                pass

        return events

    @staticmethod
    def beats_between(time: timedelta, last_time: timedelta, beatmap: Beatmap) -> float:
        """Get the milliseconds per beat at the given time.

        Parameters
        ----------
        time : datetime.timedelta
            The time to look up the bpm for.
        beatmap : slider.beatmap.Beatmap
            The beatmap to look up the bpm in.

        Returns
        -------
        float
            The msb at the given time.
        """
        current_time = time
        beats = 0
        for tp in reversed(beatmap.timing_points):
            if tp.offset < current_time and tp.ms_per_beat > 0:
                if tp.offset <= last_time:
                    beats += (current_time - last_time).total_seconds() * 1000 / tp.ms_per_beat
                    break
                beats += (current_time - tp.offset).total_seconds() * 1000 / tp.ms_per_beat
                current_time = tp.offset

        return beats

    def _clip_time(self, time: timedelta, last_time: timedelta, beatmap: Beatmap) -> int:
        """Clip time to valid range."""
        ms_delta = time.total_seconds() * 1000 - last_time.total_seconds() * 1000
        return np.clip(int(round((ms_delta - self.min_time) * self.time_resolution)), 0, self.max_timeshift)

    def _parse_circle(self, circle: Circle, events: list[Event], last_time: timedelta, beatmap: Beatmap) -> timedelta:
        """Parse a circle hit object.

        Args:
            circle: Circle object.
            events: List of events to add to.
            last_time: Last time of the hit objects.

        Returns:
            last_time: Time of the circle.
        """
        time = circle.time
        timeshift = self._clip_time(time, last_time, beatmap)

        events.append(Event(EventType.TIME_SHIFT, timeshift))
        events.append(Event(EventType.CIRCLE))

        return time

    def _parse_slider(self, slider: Slider, events: list[Event], last_time: timedelta, beatmap: Beatmap) -> timedelta:
        """Parse a slider hit object.

        Args:
            slider: Slider object.
            events: List of events to add to.
            last_time: Last time of the hit objects.

        Returns:
            last_time: Last time of the slider.
        """

        time = slider.time
        timeshift = self._clip_time(time, last_time, beatmap)
        last_time = time

        events.append(Event(EventType.TIME_SHIFT, timeshift))
        events.append(Event(EventType.SLIDER_HEAD))

        duration: timedelta = (slider.end_time - slider.time) / slider.repeat
        time = slider.time + duration
        timeshift = self._clip_time(time, last_time, beatmap)

        events.append(Event(EventType.TIME_SHIFT, timeshift))
        events.append(Event(EventType.LAST_ANCHOR))

        time = slider.end_time
        timeshift = self._clip_time(time, last_time, beatmap)

        events.append(Event(EventType.TIME_SHIFT, timeshift))
        events.append(Event(EventType.SLIDER_END))

        return last_time

    def _parse_spinner(self, spinner: Spinner, events: list[Event], last_time: timedelta, beatmap: Beatmap) -> timedelta:
        """Parse a spinner hit object.

        Args:
            spinner: Spinner object.
            events: List of events to add to.
            last_time: Last time of the hit objects.

        Returns:
            last_time: Last time of the spinner.
        """
        time = spinner.time
        timeshift = self._clip_time(time, last_time, beatmap)
        last_time = time

        events.append(Event(EventType.TIME_SHIFT, timeshift))
        events.append(Event(EventType.SPINNER))

        time = spinner.end_time
        timeshift = self._clip_time(time, last_time, beatmap)

        events.append(Event(EventType.TIME_SHIFT, timeshift))
        events.append(Event(EventType.SPINNER_END))

        return last_time
