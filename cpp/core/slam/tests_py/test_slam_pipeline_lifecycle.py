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
*
* Native frontend lifecycle and lock-order regression tests.
*
* Run in a subprocess with a timeout, since a lock-order regression can deadlock:
*   timeout 120 python cpp/core/slam/tests_py/test_slam_pipeline_lifecycle.py
"""

import os
import gc
import sys
import threading
import tempfile
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
os.environ["PYSLAM_USE_CPP"] = "1"

# these tests are about cpp_core's Slam, Tracking and LocalMapping: the C++ pipeline, which is off by
# default (Parameters.USE_CPP_PIPELINE)
os.environ.setdefault("PYSLAM_USE_CPP_PIPELINE", "1")
import pyslam.config  # Library paths before importing the native core.
from pyslam.config_parameters import Parameters
from pyslam.io.dataset_types import SensorType
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.slam import Slam
from pyslam.slam.slam_commons import SlamState
from compare_cpp_python_trajectory import _make_camera, _texture, _frame


class PipelineLifecycleTests(unittest.TestCase):
    def make_slam(self, threaded):
        Parameters.kUseLoopClosing = False
        Parameters.kLocalMappingOnSeparateThread = threaded
        Parameters.kDoVolumetricIntegration = False
        Parameters.kDoSparseSemanticMappingAndSegmentation = False
        Parameters.kUseLargeWindowBA = False
        Parameters.kUseMotionBlurDection = False
        tracker = dict(FeatureTrackerConfigs.ORB2, deterministic=True, num_features=800)
        slam = Slam(_make_camera(True), tracker, sensor_type=SensorType.RGBD, headless=True)
        self.addCleanup(slam.quit)
        return slam

    def initialize(self, slam):
        texture = _texture()
        for i in range(2):
            image, depth = _frame(texture, i)
            slam.track(image, None, depth, i, float(i))
        self.assertEqual(slam.tracking.state, SlamState.OK)
        return texture

    def test_immediate_start_quit_and_reset_after_quit(self):
        slam = self.make_slam(True)
        for _ in range(20):
            slam.local_mapping.quit()
            slam.local_mapping.start()
        slam.local_mapping.quit()
        slam.local_mapping.request_reset()

    def test_keyframe_demand_matches_python(self):
        # main_slam.py's playback throttle reads tracking.kf_demand in both pipelines
        import random
        import cpp_core
        from pyslam.slam.playback_throttle import KeyframeDemand

        cpp_demand = cpp_core.KeyframeDemand(window=7, weak_ratio=0.5)
        py_demand = KeyframeDemand(window=7, weak_ratio=0.5)
        rng = random.Random(0)
        for step in range(200):
            args = (rng.random() < 0.5, rng.random() < 0.5, rng.randint(0, 100), rng.randint(0, 100))
            cpp_demand.record(*args)
            py_demand.record(*args)
            if step % 37 == 36:
                cpp_demand.reset_window()
                py_demand.reset_window()
            self.assertEqual(cpp_demand.num_samples(), py_demand.num_samples())
            self.assertAlmostEqual(cpp_demand.weak_fraction(), py_demand.weak_fraction())
        self.assertAlmostEqual(cpp_demand.total_weak_fraction(), py_demand.total_weak_fraction())
        self.assertAlmostEqual(cpp_demand.total_suppressed_fraction(), py_demand.total_suppressed_fraction())
        for name in ("num_frames", "num_weak", "num_wanted", "num_suppressed"):
            self.assertEqual(getattr(cpp_demand, name), getattr(py_demand, name), name)

    def test_relative_pose_anchor_scale_matches_python(self):
        # #93: the C++ final trajectory rescales the relative poses with the same formula
        import numpy as np
        import cpp_core
        from pyslam.slam.relative_pose_scale import RelativePoseAnchors

        class _KeyFrame:  # what RelativePoseAnchors.scale() reads
            def __init__(self, position):
                self.position = position

            def is_bad(self):
                return False

            def Ow(self):
                return self.position

        rng = np.random.default_rng(0)
        for n in (0, 1, 2, 3, 5, 10):
            for scale in (0.5, 1.0, 2.2):
                then = rng.normal(size=(n, 3))
                now = then * scale + rng.normal(size=3) + 1e-3 * rng.normal(size=(n, 3))
                anchors = RelativePoseAnchors.__new__(RelativePoseAnchors)
                anchors.keyframes = [_KeyFrame(p) for p in now]
                anchors.positions = then
                expected = anchors.scale() if n else 1.0
                got = cpp_core.relative_pose_anchor_scale_change(list(then), list(now))
                self.assertAlmostEqual(got, expected, places=9, msg=f"n={n} scale={scale}")
        same = [np.zeros(3)] * 4  # no spread: no scale
        self.assertEqual(cpp_core.relative_pose_anchor_scale_change(same, same), 1.0)

    def test_tracking_records_keyframe_demand(self):
        from pyslam.slam.playback_throttle import PlaybackThrottle

        slam = self.make_slam(True)
        demand = slam.tracking.kf_demand
        self.assertIsNotNone(demand)
        self.assertEqual(demand.weak_ratio, Parameters.kPlaybackThrottleWeakTrackingRatio)
        texture = self.initialize(slam)
        for i in range(2, 12):
            image, depth = _frame(texture, i)
            slam.track(image, None, depth, i, float(i))
        self.assertGreater(demand.num_frames, 0)
        throttle = PlaybackThrottle(max_speed=1.0, enabled=True, warmup_frames=0, update_period=1, min_samples=1)
        throttle.update(demand)  # the throttle's interface works on the C++ object

    def test_reset_while_stopped(self):
        slam = self.make_slam(True)
        self.initialize(slam)
        slam.local_mapping.request_stop()
        deadline = time.monotonic() + 3
        while not slam.local_mapping.is_stopped() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(slam.local_mapping.is_stopped())
        slam.reset_session()
        self.assertEqual(slam.tracking.state, SlamState.NO_IMAGES_YET)
        slam.local_mapping.release()
        self.initialize(slam)

    def test_track_releases_gil_when_waiting_for_map(self):
        slam = self.make_slam(False)
        texture = self.initialize(slam)
        locked = threading.Event()

        def hold_map():
            with slam.map.update_lock:
                locked.set()
                time.sleep(0.2)

        worker = threading.Thread(target=hold_map)
        worker.start()
        self.assertTrue(locked.wait(3))
        image, depth = _frame(texture, 2)
        slam.track(image, None, depth, 2, 2.0)
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())

    def test_relocalize_releases_map_lock(self):
        slam = self.make_slam(False)
        texture = self.initialize(slam)

        class LoopClosing:
            def is_closing(self):
                return False

            def relocalize(self, frame, image):
                def inspect_map():
                    with slam.map.update_lock:
                        return slam.map.num_keyframes()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    self.num_keyframes = pool.submit(inspect_map).result(timeout=3)
                return False

            def quit(self):
                pass

        collaborator = LoopClosing()
        slam.loop_closing = collaborator
        slam.set_tracking_state(SlamState.LOST)
        image, depth = _frame(texture, 2)
        slam.track(image, None, depth, 2, 2.0)
        self.assertEqual(collaborator.num_keyframes, 2)

    def test_missing_state_does_not_stop_mapper(self):
        slam = self.make_slam(True)
        with self.assertRaises(RuntimeError):
            slam.load_system_state("/tmp/pyslam-nonexistent-review-state")
        self.assertTrue(slam.local_mapping.is_running())

    def test_quit_releases_python_collaborator_cycle(self):
        slam = self.make_slam(False)

        class Collaborator:
            def __init__(self, owner):
                self.slam = owner

            def quit(self):
                pass

        collaborator = Collaborator(slam)
        reference = weakref.ref(collaborator)
        slam.loop_closing = collaborator
        del collaborator
        slam.quit()
        gc.collect()
        self.assertIsNone(reference())

    def test_stop_drains_queued_keyframes(self):
        slam = self.make_slam(False)
        self.initialize(slam)
        mapper = slam.local_mapping
        mapper.push_keyframe(slam.tracking.kf_ref)
        mapper.request_stop()
        mapper.step()
        self.assertEqual(mapper.queue_size(), 0)
        self.assertTrue(mapper.is_idle())
        mapper.step()
        self.assertTrue(mapper.is_stopped())
        # Single-thread callers do not have a worker that can acknowledge reset.
        mapper.request_reset()

    def test_threaded_tracking_snapshots_and_save_load(self):
        slam = self.make_slam(True)
        texture = self.initialize(slam)
        stop = threading.Event()
        failures = []

        def inspect_snapshots():
            try:
                while not stop.is_set():
                    tracking = slam.tracking
                    _ = tracking.f_cur, tracking.kf_ref, tracking.predicted_pose
                    _ = tracking.cur_R, tracking.cur_t, tracking.descriptor_distance_sigma
                    time.sleep(0.001)
            except Exception as error:
                failures.append(error)

        reader = threading.Thread(target=inspect_snapshots)
        reader.start()
        try:
            for i in range(2, 20):
                image, depth = _frame(texture, i)
                slam.track(image, None, depth, i, float(i))
        finally:
            stop.set()
            reader.join(timeout=3)
        self.assertFalse(reader.is_alive())
        self.assertFalse(failures)
        self.assertEqual(slam.tracking.state, SlamState.OK)
        deadline = time.monotonic() + 10
        while (slam.local_mapping.queue_size() or not slam.local_mapping.is_idle()) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(slam.local_mapping.queue_size(), 0)
        self.assertTrue(slam.local_mapping.is_idle())
        with tempfile.TemporaryDirectory(prefix="pyslam-review-state-") as path:
            slam.save_system_state(path)
            counts = slam.map.num_keyframes(), slam.map.num_points()
            slam.load_system_state(path)
            self.assertEqual(counts, (slam.map.num_keyframes(), slam.map.num_points()))
            self.assertTrue(slam.local_mapping.is_running())


if __name__ == "__main__":
    unittest.main()
