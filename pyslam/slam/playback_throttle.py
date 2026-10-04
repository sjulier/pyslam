"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import math
from collections import deque

from pyslam.config_parameters import Parameters


class KeyframeDemand:
    """
    What tracking records at each keyframe decision, for the playback throttle.

    - Weak tracking: the frame matched fewer than weak_ratio of the map points that its reference
      keyframe tracks. When local mapping cannot keep up, new keyframes and map points arrive too
      late and this happens in a growing share of the frames, until tracking is lost. The fraction
      of weak frames among the recent ones is the throttle's signal.
    - Suppressed keyframe requests: tracking only asks for a new keyframe when local mapping is idle;
      when it is busy the request is not raised at all. Their share is reported at the end of a run,
      but it is not used as a signal: local mapping is busy for about half of the requests on a
      machine that tracks every frame, too.
    """

    def __init__(self, window=None, weak_ratio=None):
        window = Parameters.kPlaybackThrottleWindow if window is None else window
        self.weak_ratio = (
            Parameters.kPlaybackThrottleWeakTrackingRatio if weak_ratio is None else weak_ratio
        )
        self._weak = deque(maxlen=window)  # for each recent frame: True if tracking was weak
        self.num_frames = 0
        self.num_weak = 0
        self.num_wanted = 0
        self.num_suppressed = 0

    def record(
        self,
        wanted_if_idle: bool,
        is_local_mapping_idle: bool,
        num_tracked_points=None,
        num_ref_tracked_points=None,
    ):
        if wanted_if_idle:
            self.num_wanted += 1
            self.num_suppressed += int(not is_local_mapping_idle)
        if num_tracked_points is not None and num_ref_tracked_points:
            weak = num_tracked_points < self.weak_ratio * num_ref_tracked_points
            self._weak.append(weak)
            self.num_frames += 1
            self.num_weak += int(weak)

    def num_samples(self):
        return len(self._weak)

    def reset_window(self):
        """Forget the recent samples (the totals are kept): they were taken at another playback speed."""
        self._weak.clear()

    def weak_fraction(self):
        """Fraction of the frames in the window in which tracking was weak (0 with no samples)."""
        return sum(self._weak) / len(self._weak) if self._weak else 0.0

    def total_weak_fraction(self):
        return self.num_weak / self.num_frames if self.num_frames > 0 else 0.0

    def total_suppressed_fraction(self):
        return self.num_suppressed / self.num_wanted if self.num_wanted > 0 else 0.0


def is_throttle_enabled(throttle=False, no_throttle=False, headless=False):
    """Whether main_slam.py throttles the playback: on by default with windows (drawing them takes CPU
    time from tracking and local mapping), off by default headless; --throttle / --no-throttle decide."""
    if no_throttle or not Parameters.kLocalMappingOnSeparateThread:
        return False
    return bool(
        throttle
        or Parameters.kPlaybackThrottle
        or (not headless and Parameters.kPlaybackThrottleWithGui)
    )


class PlaybackThrottle:
    """
    Adapts the playback speed of a dataset to what local mapping can keep up with.

    The speed is relative to the camera's frame rate. It never exceeds max_speed (the speed the user
    asked for; 0 or inf = no limit). When tracking was weak in more than the high fraction of the
    recent frames, the speed is reduced; when it was weak in less than the low fraction, the speed
    goes back up towards max_speed.
    """

    def __init__(
        self,
        max_speed=1.0,
        enabled=True,
        high=None,
        low=None,
        decrease=None,
        increase=None,
        min_speed=None,
        update_period=None,
        min_samples=None,
        warmup_frames=None,
    ):
        self.max_speed = math.inf if (max_speed is None or max_speed <= 0) else float(max_speed)
        self.enabled = enabled
        self.high = Parameters.kPlaybackThrottleHighWeakFraction if high is None else high
        self.low = Parameters.kPlaybackThrottleLowWeakFraction if low is None else low
        self.decrease = Parameters.kPlaybackThrottleDecreaseFactor if decrease is None else decrease
        self.increase = Parameters.kPlaybackThrottleIncreaseFactor if increase is None else increase
        self.min_speed = Parameters.kPlaybackThrottleMinSpeed if min_speed is None else min_speed
        self.update_period = (
            Parameters.kPlaybackThrottleUpdatePeriod if update_period is None else update_period
        )
        self.min_samples = Parameters.kPlaybackThrottleMinSamples if min_samples is None else min_samples
        self.warmup_frames = (
            Parameters.kPlaybackThrottleWarmupFrames if warmup_frames is None else warmup_frames
        )

        self.speed = self.max_speed  # current speed limit (inf = no wait)
        self.measured_speed = None  # actual playback speed, smoothed
        self.num_frames = 0
        self.num_decreases = 0
        self.lowest_speed = self.max_speed
        self.last_fraction = 0.0  # the weak-tracking fraction that the last decision used

    def observe_frame(self, frame_duration, elapsed):
        """Record the actual duration of one loop iteration (processing + wait) for a frame of the given duration."""
        if frame_duration is None or frame_duration <= 0 or elapsed <= 0:
            return
        speed = frame_duration / elapsed
        self.measured_speed = (
            speed if self.measured_speed is None else 0.8 * self.measured_speed + 0.2 * speed
        )

    def update(self, demand: KeyframeDemand):
        """Call once per frame. Returns True when the speed has changed."""
        self.num_frames += 1
        if not self.enabled or demand is None:
            return False
        # Tracking is weak by nature while the map is being initialised: ignore the first frames,
        # and drop what the window recorded during them.
        if self.num_frames <= self.warmup_frames:
            if self.num_frames == self.warmup_frames:
                demand.reset_window()
            return False
        if self.num_frames % self.update_period != 0:
            return False
        # The window only holds frames played at the current speed (it is emptied at each change):
        # wait until there are enough of them. Slowing down needs less evidence than speeding up.
        num_samples = demand.num_samples()
        if num_samples < self.min_samples:
            return False

        fraction = demand.weak_fraction()
        self.last_fraction = fraction
        old_speed = self.speed
        if fraction > self.high:
            # start from the speed actually reached: the limit may be far above it (e.g. no limit)
            base = self.speed
            if self.measured_speed is not None:
                base = min(base, self.measured_speed)
            self.speed = max(self.min_speed, base * self.decrease)
        elif (
            fraction < self.low
            and self.speed < self.max_speed
            and num_samples >= 2 * self.min_samples
        ):
            self.speed = min(self.max_speed, self.speed * self.increase)

        if self.speed != old_speed:
            demand.reset_window()
            if self.speed < old_speed:
                self.num_decreases += 1
            self.lowest_speed = min(self.lowest_speed, self.speed)
            return True
        return False

    def wait_time(self, frame_duration, processing_duration):
        """Time to sleep after processing a frame, to respect the current speed."""
        if frame_duration is None or frame_duration <= 0 or math.isinf(self.speed):
            return 0.0
        return max(0.0, frame_duration / self.speed - processing_duration)

    def speed_str(self, speed=None):
        speed = self.speed if speed is None else speed
        return "unlimited" if math.isinf(speed) else f"{speed:.2f}x"
