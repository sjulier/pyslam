"""Tests of the playback throttle (pyslam/slam/playback_throttle.py). Run with: pytest test/slam/test_playback_throttle.py"""

import math
import sys

sys.path.append("./")
sys.path.append("../../")

from pyslam.slam.playback_throttle import KeyframeDemand, PlaybackThrottle


def feed(demand, throttle, num_frames, suppressed_every, frame_duration=0.1, processing=0.02):
    """Play num_frames frames that all want a keyframe; every suppressed_every-th request is NOT suppressed."""
    changes = 0
    for i in range(num_frames):
        demand.record(True, is_local_mapping_idle=(i % suppressed_every == 0))
        changes += int(throttle.update(demand))
        wait = throttle.wait_time(frame_duration, processing)
        throttle.observe_frame(frame_duration, processing + wait)
    return changes


def make(max_speed, **kwargs):
    params = dict(high=0.5, low=0.35, decrease=0.8, increase=1.1, min_speed=0.1, update_period=10, min_samples=10)
    params.update(kwargs)
    return KeyframeDemand(window=30), PlaybackThrottle(max_speed=max_speed, **params)


def test_demand_counts_only_wanted_frames():
    demand = KeyframeDemand(window=4)
    demand.record(False, False)  # no keyframe wanted: ignored
    assert demand.num_samples() == 0 and demand.suppressed_fraction() == 0.0
    for idle in (True, False, False, True):
        demand.record(True, idle)
    assert demand.suppressed_fraction() == 0.5
    for _ in range(4):  # the window slides
        demand.record(True, True)
    assert demand.suppressed_fraction() == 0.0
    assert demand.num_wanted == 8 and demand.num_suppressed == 2


def test_no_change_when_local_mapping_keeps_up():
    demand, throttle = make(1.0)
    assert feed(demand, throttle, 300, suppressed_every=1) == 0  # never suppressed
    assert throttle.speed == 1.0 and throttle.num_decreases == 0


def test_slows_down_when_requests_are_suppressed():
    demand, throttle = make(1.0)
    feed(demand, throttle, 300, suppressed_every=10)  # 90% suppressed
    assert throttle.speed < 1.0
    assert throttle.speed >= throttle.min_speed
    assert throttle.num_decreases > 0


def test_never_below_min_speed():
    demand, throttle = make(1.0, min_speed=0.25)
    feed(demand, throttle, 3000, suppressed_every=10)
    assert throttle.speed == 0.25


def test_recovers_up_to_max_speed_only():
    demand, throttle = make(2.0)
    feed(demand, throttle, 300, suppressed_every=10)
    assert throttle.speed < 2.0
    feed(demand, throttle, 2000, suppressed_every=1)  # local mapping keeps up again
    assert throttle.speed == 2.0


def test_unlimited_speed_slows_down_from_the_measured_speed():
    demand, throttle = make(0)  # 0 = no limit
    assert math.isinf(throttle.speed) and throttle.wait_time(0.1, 0.02) == 0.0
    feed(demand, throttle, 60, suppressed_every=10)  # frames take 0.02 s: measured speed 5x
    assert not math.isinf(throttle.speed)
    assert throttle.speed <= 5.0 * 0.8 + 1e-6
    assert throttle.wait_time(0.1, 0.02) > 0.0


def test_disabled_throttle_never_changes_speed():
    demand = KeyframeDemand(window=30)
    throttle = PlaybackThrottle(max_speed=1.0, enabled=False)
    assert feed(demand, throttle, 300, suppressed_every=10) == 0
    assert throttle.speed == 1.0


def test_window_is_emptied_at_each_change():
    demand, throttle = make(1.0)
    feed(demand, throttle, 9, suppressed_every=10)
    assert throttle.num_decreases == 0  # not enough samples yet (min_samples = 10)
    feed(demand, throttle, 1, suppressed_every=10)
    assert throttle.num_decreases == 1
    assert demand.num_samples() == 0  # the samples taken at the previous speed are not used again
    feed(demand, throttle, 9, suppressed_every=10)
    assert throttle.num_decreases == 1  # it waits for min_samples new requests
    assert demand.num_wanted == 19  # the totals are kept


def test_between_thresholds_is_stable():
    demand, throttle = make(1.0)
    # 40% suppressed: between low (35%) and high (50%)
    for i in range(300):
        demand.record(True, is_local_mapping_idle=(i % 5 >= 2))
        throttle.update(demand)
    assert throttle.speed == 1.0
