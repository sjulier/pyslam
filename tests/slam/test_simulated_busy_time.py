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
* Single-thread mode's simulated mapper (LocalMapping.charge_sim_step): each inline step is charged
* its measured duration (scaled), or a fixed time when kLocalMappingSimulatedBusyTimeFixed is set.
"""

import unittest

import pyslam.config  # noqa: F401  (native library paths)
from pyslam.config_parameters import Parameters
from pyslam.slam.local_mapping import LocalMapping

NAMES = (
    "kLocalMappingOnSeparateThread",
    "kLocalMappingSimulateBusyTimeInSingleThread",
    "kLocalMappingSimulatedBusyTimeScale",
    "kLocalMappingSimulatedBusyTimeFixed",
)


def simulated_mapper():
    """A LocalMapping with only the simulated-mapper state (no SLAM system, no thread)."""
    lm = LocalMapping.__new__(LocalMapping)
    lm.sim_now = None
    lm.sim_step_intervals = []
    return lm


class TestSimulatedBusyTime(unittest.TestCase):
    def setUp(self):
        self.saved = {name: getattr(Parameters, name) for name in NAMES}
        Parameters.kLocalMappingOnSeparateThread = False
        Parameters.kLocalMappingSimulateBusyTimeInSingleThread = True
        Parameters.kLocalMappingSimulatedBusyTimeScale = 1.0
        Parameters.kLocalMappingSimulatedBusyTimeFixed = None

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(Parameters, name, value)

    def test_measured_duration_is_scaled(self):
        Parameters.kLocalMappingSimulatedBusyTimeScale = 2.0
        lm = simulated_mapper()
        lm.set_sim_time(10.0)
        lm.charge_sim_step(0.05)
        self.assertEqual(lm.sim_step_intervals, [(10.0, 10.1)])
        self.assertFalse(lm.is_idle_for_keyframe_decision())

    def test_fixed_time_replaces_the_measured_duration(self):
        Parameters.kLocalMappingSimulatedBusyTimeScale = 2.0  # not applied to the fixed time
        Parameters.kLocalMappingSimulatedBusyTimeFixed = 0.08
        lm = simulated_mapper()
        lm.set_sim_time(10.0)
        lm.charge_sim_step(0.9)  # however long the step really took
        lm.charge_sim_step(0.001)
        # serial: the second step starts when the first one finishes
        self.assertEqual(len(lm.sim_step_intervals), 2)
        (s0, f0), (s1, f1) = lm.sim_step_intervals
        self.assertAlmostEqual(s0, 10.0)
        self.assertAlmostEqual(f0, 10.08)
        self.assertAlmostEqual(s1, 10.08)
        self.assertAlmostEqual(f1, 10.16)
        self.assertEqual(lm.queue_size_for_keyframe_decision(), 1)  # the second one has not started
        lm.set_sim_time(10.12)  # the first one is done, the second one is running
        self.assertEqual(lm.queue_size_for_keyframe_decision(), 0)
        self.assertFalse(lm.is_idle_for_keyframe_decision())
        lm.set_sim_time(10.2)  # both done
        self.assertTrue(lm.is_idle_for_keyframe_decision())

    def test_same_input_same_schedule_with_fixed_time(self):
        """With a fixed time, the schedule depends only on the timestamps, not on the step durations."""
        Parameters.kLocalMappingSimulatedBusyTimeFixed = 0.05

        def schedule(durations):
            lm = simulated_mapper()
            out = []
            for i, d in enumerate(durations):
                lm.set_sim_time(i * 0.04)
                out.append(lm.is_idle_for_keyframe_decision())
                lm.charge_sim_step(d)
            return out

        self.assertEqual(schedule([0.01, 0.3, 0.02, 0.5, 0.01]), schedule([0.2, 0.001, 0.4, 0.03, 0.6]))


if __name__ == "__main__":
    unittest.main()
