"""Tests of the playback throttle (pyslam/slam/playback_throttle.py). Run with: pytest test/slam/test_playback_throttle.py"""

import math
import sys

sys.path.append("./")
sys.path.append("../../")

from pyslam.config_parameters import Parameters
from pyslam.slam.playback_throttle import KeyframeDemand, PlaybackThrottle, is_throttle_enabled

NUM_REF = 200  # map points tracked by the reference keyframe
STRONG, WEAK = 180, 60  # points matched by a frame: above / below half of NUM_REF


def feed(demand, throttle, num_frames, weak_percent, frame_duration=0.1, processing=0.02):
    """Play num_frames frames, of which weak_percent % (evenly spread) track weakly."""
    changes = 0
    for i in range(num_frames):
        weak = (i * weak_percent) % 100 < weak_percent
        demand.record(True, True, WEAK if weak else STRONG, NUM_REF)
        changes += int(throttle.update(demand))
        wait = throttle.wait_time(frame_duration, processing)
        throttle.observe_frame(frame_duration, processing + wait)
    return changes


def make(max_speed, **kwargs):
    params = dict(high=0.25, low=0.10, decrease=0.8, increase=1.1, min_speed=0.1, update_period=10, min_samples=10, warmup_frames=0)
    params.update(kwargs)
    return KeyframeDemand(window=30, weak_ratio=0.5), PlaybackThrottle(max_speed=max_speed, **params)


def test_demand_records_weak_frames_and_suppressed_requests():
    demand = KeyframeDemand(window=4, weak_ratio=0.5)
    assert demand.num_samples() == 0 and demand.weak_fraction() == 0.0
    for tracked in (180, 60, 99, 100):  # weak: fewer than 0.5 * 200 points
        demand.record(True, True, tracked, NUM_REF)
    assert demand.weak_fraction() == 0.5
    for _ in range(4):  # the window slides
        demand.record(False, True, 180, NUM_REF)
    assert demand.weak_fraction() == 0.0
    assert demand.num_frames == 8 and demand.num_weak == 2 and demand.total_weak_fraction() == 0.25
    # suppressed requests: counted among the frames that wanted a keyframe, whatever the tracking
    demand.record(True, False, 180, NUM_REF)
    demand.record(False, False, 180, NUM_REF)  # no keyframe wanted: not a request
    assert demand.num_wanted == 5 and demand.num_suppressed == 1
    # a frame without the numbers of tracked points is not a sample
    demand.record(True, True)
    demand.record(True, True, 10, 0)
    assert demand.num_frames == 10


def test_busy_local_mapping_alone_does_not_slow_down():
    # Local mapping is busy at every request, but tracking stays strong: a machine that keeps up.
    demand, throttle = make(1.0)
    for _ in range(300):
        demand.record(True, False, STRONG, NUM_REF)
        assert not throttle.update(demand)
    assert throttle.speed == 1.0 and demand.total_suppressed_fraction() == 1.0


def test_no_change_when_tracking_is_strong():
    demand, throttle = make(1.0)
    assert feed(demand, throttle, 300, weak_percent=0) == 0
    assert throttle.speed == 1.0 and throttle.num_decreases == 0


def test_slows_down_when_tracking_is_weak():
    demand, throttle = make(1.0)
    feed(demand, throttle, 300, weak_percent=50)
    assert throttle.speed < 1.0
    assert throttle.speed >= throttle.min_speed
    assert throttle.num_decreases > 0
    assert throttle.last_fraction > 0.25  # the fraction that the last decision used


def test_never_below_min_speed():
    demand, throttle = make(1.0, min_speed=0.25)
    feed(demand, throttle, 3000, weak_percent=50)
    assert throttle.speed == 0.25


def test_recovers_up_to_max_speed_only():
    demand, throttle = make(2.0)
    feed(demand, throttle, 300, weak_percent=50)
    assert throttle.speed < 2.0
    feed(demand, throttle, 2000, weak_percent=0)  # tracking is strong again
    assert throttle.speed == 2.0


def test_unlimited_speed_slows_down_from_the_measured_speed():
    demand, throttle = make(0)  # 0 = no limit
    assert math.isinf(throttle.speed) and throttle.wait_time(0.1, 0.02) == 0.0
    feed(demand, throttle, 60, weak_percent=50)  # frames take 0.02 s: measured speed 5x
    assert not math.isinf(throttle.speed)
    assert throttle.speed <= 5.0 * 0.8 + 1e-6
    assert throttle.wait_time(0.1, 0.02) > 0.0


def test_disabled_throttle_never_changes_speed():
    demand = KeyframeDemand(window=30)
    throttle = PlaybackThrottle(max_speed=1.0, enabled=False)
    assert feed(demand, throttle, 300, weak_percent=50) == 0
    assert throttle.speed == 1.0


def test_window_is_emptied_at_each_change():
    demand, throttle = make(1.0)
    feed(demand, throttle, 9, weak_percent=50)
    assert throttle.num_decreases == 0  # not enough samples yet (min_samples = 10)
    feed(demand, throttle, 1, weak_percent=100)
    assert throttle.num_decreases == 1
    assert demand.num_samples() == 0  # the samples taken at the previous speed are not used again
    feed(demand, throttle, 9, weak_percent=50)
    assert throttle.num_decreases == 1  # it waits for min_samples new frames
    assert demand.num_frames == 19  # the totals are kept


def test_between_thresholds_is_stable():
    demand, throttle = make(1.0)
    feed(demand, throttle, 300, weak_percent=20)  # between low (10%) and high (25%)
    assert throttle.speed == 1.0


def test_start_up_is_ignored():
    demand, throttle = make(1.0, warmup_frames=50)
    feed(demand, throttle, 50, weak_percent=100)  # weak while the map is initialised
    assert throttle.num_decreases == 0
    assert demand.num_samples() == 0  # the start-up frames are dropped from the window
    feed(demand, throttle, 300, weak_percent=0)
    assert throttle.speed == 1.0 and throttle.num_decreases == 0


def test_default_parameters_floor_and_quick_recovery():
    # with the defaults: never below kPlaybackThrottleMinSpeed, and back to full speed within
    # 200 frames once tracking is strong again (KITTI 06 on a Mac with windows took 240 frames to
    # climb from 0.31x to 0.67x with the earlier x1.1 steps)
    demand = KeyframeDemand()
    throttle = PlaybackThrottle(max_speed=1.0)
    feed(demand, throttle, Parameters.kPlaybackThrottleWarmupFrames, weak_percent=0)
    feed(demand, throttle, 1000, weak_percent=50)
    assert throttle.speed == Parameters.kPlaybackThrottleMinSpeed
    feed(demand, throttle, 200, weak_percent=0)
    assert throttle.speed == 1.0


def test_enabled_by_default_with_windows_only():
    assert is_throttle_enabled(headless=False)
    assert not is_throttle_enabled(headless=True)
    assert is_throttle_enabled(throttle=True, headless=True)
    assert not is_throttle_enabled(no_throttle=True, headless=False)
    assert not is_throttle_enabled(throttle=True, no_throttle=True)
