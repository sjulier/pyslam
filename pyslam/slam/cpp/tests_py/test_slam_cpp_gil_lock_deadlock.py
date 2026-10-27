#!/usr/bin/env python3
"""
Regression test: freeing a NumPy-backed cv::Mat in a thread without the GIL must not wait for the GIL.

The deadlock it guards against (docs/TROUBLESHOOTING.md, "SLAM stops part-way, every thread idle"):
- local mapping releases the GIL in local_BA() and, holding LocalMapBase::_lock, frees the last reference to a
  keyframe whose descriptors are a NumPy array imported without a copy (cpp/casters/opencv_type_casters.h);
- tracking holds the GIL and waits for _lock in local_map.is_empty().
If the cv::Mat allocator waits for the GIL there, neither thread can go on.

Here a worker thread plays local mapping (LocalCovisibilityMap.update() releases the GIL and drops the previous
keyframe under _lock) and the main thread plays tracking (is_empty() in a loop, holding the GIL). The pair runs in a
child process, so a deadlock cannot hang this test: the child is killed after the time limit and the test fails.
"""
import os
import subprocess
import sys
import threading
import time

kNumUpdates = 300
kTimeLimitSeconds = 30  # about 1-3 s without the deadlock


def child():
    import faulthandler

    import numpy as np

    from pyslam.slam.cpp import cpp_module

    # if it freezes, print every thread's stack before the parent kills this process
    faulthandler.dump_traceback_later(kTimeLimitSeconds - 5)

    local_map = cpp_module.Map().local_map
    go = threading.Event()

    def local_mapping():
        go.wait()
        for _ in range(kNumUpdates):
            frame = cpp_module.Frame(None)
            frame.des = np.zeros((500, 32), np.uint8)  # imported without a copy, as the extractor's output
            kf = cpp_module.KeyFrame(frame)  # shares frame.des (Frame::copy_from)
            del frame
            local_map.update(kf)  # GIL released; the previous keyframe is freed under the local map's lock
            del kf

    t = threading.Thread(target=local_mapping)
    t.start()
    go.set()
    while t.is_alive():
        local_map.is_empty()  # holds the GIL while it waits for the local map's lock
    t.join()
    faulthandler.cancel_dump_traceback_later()


def main():
    start = time.time()
    try:
        result = subprocess.run([sys.executable, __file__, "--child"], timeout=kTimeLimitSeconds)
    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: froze for {kTimeLimitSeconds} s (GIL <-> LocalMapBase::_lock deadlock)")
        return 1
    if result.returncode != 0:
        print(f"❌ FAILED: the child exited with code {result.returncode}")
        return 1
    print(f"✅ PASSED: {kNumUpdates} local map updates against a GIL-holding is_empty() loop in "
          f"{time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        sys.exit(main())
