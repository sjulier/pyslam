#!/usr/bin/env python3
"""
Regression test: a keyframe freed in a thread without the GIL must not DECREF its g_des there.

KeyFrame::g_des is a Python object owned by the C++ keyframe: loop closing sets it on every keyframe it processes
(loop_closing.py, keyframe.g_des = detection_output.g_des_vec). Local mapping can free the last reference to a
culled keyframe with the GIL released (LocalCovisibilityMap.update(), as in test_slam_cpp_gil_lock_deadlock.py).
A DECREF there races with the threads that hold the GIL: the reference count goes wrong (the object is freed too
early, or never), and freeing it runs Python's allocator without the GIL. ~KeyFrame now leaves the DECREF to the
main thread (cpp/casters/deferred_py_decrefs.h).

A worker thread frees keyframes that all share one g_des object, with the GIL released, while the main thread keeps
taking and dropping references to that object. Once both are done, the object's reference count must be back to
where it started. The pair runs in a child process with a time limit, so a crash or a freeze fails the test.
"""
import os
import subprocess
import sys
import threading
import time

kNumKeyframes = 300
kTimeLimitSeconds = 60


def child():
    import faulthandler

    from pyslam.slam.cpp import cpp_module

    faulthandler.dump_traceback_later(kTimeLimitSeconds - 5)

    g_des = [(i, 0.5) for i in range(100)]  # like the BoW vector that loop closing stores
    baseline = sys.getrefcount(g_des)

    local_map = cpp_module.Map().local_map
    keyframes = []
    for _ in range(kNumKeyframes):
        kf = cpp_module.KeyFrame(cpp_module.Frame(None))
        kf.g_des = g_des
        keyframes.append(kf)
    del kf

    go = threading.Event()

    def local_mapping():
        go.wait()
        while keyframes:
            kf = keyframes.pop()
            local_map.update(kf)  # GIL released; the previous keyframe is freed under the local map's lock
            del kf
        local_map.update(cpp_module.KeyFrame(cpp_module.Frame(None)))  # frees the last one too

    t = threading.Thread(target=local_mapping)
    t.start()
    go.set()
    while t.is_alive():
        for _ in range(100):
            ref = g_des  # INCREF / DECREF with the GIL, racing with any DECREF without it
            del ref
    t.join()

    # the deferred DECREFs run in this thread at the interpreter's next checks
    deadline = time.time() + 5
    while sys.getrefcount(g_des) != baseline and time.time() < deadline:
        time.sleep(0.01)
    faulthandler.cancel_dump_traceback_later()
    count = sys.getrefcount(g_des)
    if count != baseline:
        print(f"reference count of g_des: {count}, expected {baseline}")
        sys.exit(2)


def main():
    start = time.time()
    try:
        result = subprocess.run([sys.executable, __file__, "--child"], timeout=kTimeLimitSeconds)
    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: froze for {kTimeLimitSeconds} s")
        return 1
    if result.returncode != 0:
        print(f"❌ FAILED: the child exited with code {result.returncode} (KeyFrame::g_des freed without the GIL?)")
        return 1
    print(f"✅ PASSED: {kNumKeyframes} keyframes with a shared g_des freed without the GIL, reference count "
          f"intact, in {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        sys.exit(main())
