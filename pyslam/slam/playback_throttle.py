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
    Recent keyframe demand, recorded by tracking at each keyframe decision.

    Tracking only asks for a new keyframe when local mapping is idle. When local mapping is busy, the
    request is not raised at all: a "suppressed" request. The fraction of suppressed requests among
    the recent frames that wanted a keyframe tells how far local mapping is behind the frame rate.
    """

    def __init__(self, window=None):
        window = Parameters.kPlaybackThrottleWindow if window is None else window
        self._wanted = deque(maxlen=window)  # for each frame that wanted a keyframe: True if suppressed
        self.num_wanted = 0
        self.num_suppressed = 0

    def record(self, wanted_if_idle: bool, is_local_mapping_idle: bool):
        if not wanted_if_idle:
            return
        suppressed = not is_local_mapping_idle
        self._wanted.append(suppressed)
        self.num_wanted += 1
        self.num_suppressed += int(suppressed)

    def num_samples(self):
        return len(self._wanted)

    def reset_window(self):
        """Forget the recent samples (the totals are kept): they were taken at another playback speed."""
        self._wanted.clear()

    def suppressed_fraction(self):
        """Fraction of suppressed requests in the window (0 with no samples)."""
        return sum(self._wanted) / len(self._wanted) if self._wanted else 0.0

    def total_suppressed_fraction(self):
        return self.num_suppressed / self.num_wanted if self.num_wanted > 0 else 0.0


class PlaybackThrottle:
    """
    Adapts the playback speed of a dataset to what local mapping can keep up with.

    The speed is relative to the camera's frame rate. It never exceeds max_speed (the speed the user
    asked for; 0 or inf = no limit). When the suppressed fraction of the recent keyframe requests is
    above the high threshold, the speed is reduced; when it is below the low threshold, the speed
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
    ):
        self.max_speed = math.inf if (max_speed is None or max_speed <= 0) else float(max_speed)
        self.enabled = enabled
        self.high = Parameters.kPlaybackThrottleHighSuppressedFraction if high is None else high
        self.low = Parameters.kPlaybackThrottleLowSuppressedFraction if low is None else low
        self.decrease = Parameters.kPlaybackThrottleDecreaseFactor if decrease is None else decrease
        self.increase = Parameters.kPlaybackThrottleIncreaseFactor if increase is None else increase
        self.min_speed = Parameters.kPlaybackThrottleMinSpeed if min_speed is None else min_speed
        self.update_period = (
            Parameters.kPlaybackThrottleUpdatePeriod if update_period is None else update_period
        )
        self.min_samples = Parameters.kPlaybackThrottleMinSamples if min_samples is None else min_samples

        self.speed = self.max_speed  # current speed limit (inf = no wait)
        self.measured_speed = None  # actual playback speed, smoothed
        self.num_frames = 0
        self.num_decreases = 0
        self.lowest_speed = self.max_speed

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
        if self.num_frames % self.update_period != 0:
            return False
        # The window only holds requests made at the current speed (it is emptied at each change):
        # wait until there are enough of them. Slowing down needs less evidence than speeding up.
        num_samples = demand.num_samples()
        if num_samples < self.min_samples:
            return False

        fraction = demand.suppressed_fraction()
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
