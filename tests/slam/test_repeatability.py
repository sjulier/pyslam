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
* The same input gives the same output: the synthetic sequence is tracked twice, each time in a
* fresh process (objects then sit at different addresses, and ids restart), and the two runs
* must agree bit for bit: per-frame state and pose, keyframe and map point counts, the final
* keyframe poses and map point positions. Uses the core PYSLAM_USE_CPP selects.
*
* Single-thread mode with local mapping charged no simulated time, so the keyframe decisions do
* not depend on how fast the machine is.
"""

import hashlib
import os
import subprocess
import sys
import unittest

# Out and back along the plane, so later frames see (and fuse into) points made earlier.
FRAME_IDS = list(range(0, 30)) + list(range(30, -1, -1))


def run_once():
    """Track the sequence in this process and return a digest of everything it produced."""
    import numpy as np
    from slam_test_support import make_slam, track_sequence

    from pyslam.config_parameters import Parameters

    Parameters.kLocalMappingSimulateBusyTimeInSingleThread = False

    class _Case:  # make_slam registers its cleanup on a test case
        cleanups = []

        def addCleanup(self, fn):
            self.cleanups.append(fn)

    case = _Case()
    slam = make_slam(case)
    try:
        records = track_sequence(slam, FRAME_IDS)
        h = hashlib.sha256()
        for r in records:
            h.update(f"{r['state']} {r['num_keyframes']} {r['num_points']}".encode())
            if r["position"] is not None:
                h.update(np.ascontiguousarray(r["position"], dtype=np.float64).tobytes())
        for kf in sorted(slam.map.get_keyframes(), key=lambda k: k.id):
            h.update(f"kf {kf.id} {kf.is_bad()}".encode())
            h.update(np.ascontiguousarray(kf.Tcw(), dtype=np.float64).tobytes())
        for p in sorted(slam.map.get_points(), key=lambda p: p.id):
            h.update(f"p {p.id} {p.num_observations()}".encode())
            h.update(np.ascontiguousarray(p.pt(), dtype=np.float64).tobytes())
        summary = f"{records[-1]['num_keyframes']} keyframes, {records[-1]['num_points']} points"
        return h.hexdigest()[:16], summary
    finally:
        for fn in case.cleanups:
            fn()


class TestRepeatability(unittest.TestCase):
    def run_in_subprocess(self):
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--run-once"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        lines = [line for line in out.stdout.splitlines() if line.startswith("DIGEST ")]
        self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-2000:])
        self.assertTrue(lines, out.stdout[-2000:] + out.stderr[-2000:])
        return lines[-1]

    def test_two_runs_give_identical_results(self):
        first = self.run_in_subprocess()
        second = self.run_in_subprocess()
        self.assertEqual(first, second)


if __name__ == "__main__":
    if "--run-once" in sys.argv:
        digest, summary = run_once()
        print(f"DIGEST {digest} ({summary})", flush=True)
        os._exit(0)  # skip interpreter teardown of the native threads
    unittest.main()
