"""Run one command and record its wall time, exit code and peak memory; catch a stall with its stacks.

    python dev/benchmark/run_monitored.py OUT_PREFIX [--timeout S] [--stall S] -- command args...

Writes OUT_PREFIX.log (the command's stdout and stderr) and OUT_PREFIX.run.json:
  wall_s           wall time, including any time the machine slept
  slept_s          time the machine slept (seen as a gap in this script's 1 s loop): not counted
                   as run time for the timeout, nor as silence for the stall check
  returncode       the command's exit code (-9: killed here, after a timeout or a stall)
  peak_rss_gb      the peak, sampled once a second, of the summed RSS of the process and its children
  peak_footprint_gb  the same for macOS's phys_footprint (Activity Monitor's "Memory"); None elsewhere.
                   On macOS compare this, not RSS: RSS swings by +-0.3 GB with memory pressure.
  timed_out, stalled_at_s
A stall is a log that has not grown for --stall seconds. Then, before killing the process tree:
  - macOS: `sample` records the native stacks of every thread of every Python process (OUT_PREFIX.sample.PID.txt)
  - Linux, macOS: SIGUSR1 to main_slam.py, whose faulthandler writes every thread's Python stack to the log
"""

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

import psutil

kTick = 1.0  # seconds between samples
kSleepGap = 15.0  # a longer gap between two ticks means the machine slept

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None


def phys_footprint(pid):
    """macOS phys_footprint of a process (rusage_info_v2.ri_phys_footprint), in bytes; 0 if unreadable."""
    buf = (ctypes.c_uint64 * 32)()  # rusage_info_v2: a 16-byte uuid, then uint64 fields
    if _libproc.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
        return 0
    # after the uuid (2 words): user, system, pkg_idle_wkups, interrupt_wkups, pageins, wired, resident, phys_footprint
    return buf[2 + 7]


def process_tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return [proc]


def kill_tree(proc):
    for q in reversed(process_tree(proc)):
        try:
            q.kill()
        except psutil.Error:
            pass


def capture_stacks(proc, out):
    tree = process_tree(proc)
    if sys.platform == "darwin":
        for q in tree:
            try:
                if "python" in q.name().lower():
                    subprocess.run(["/usr/bin/sample", str(q.pid), "3", "-file", f"{out}.sample.{q.pid}.txt"],
                                   capture_output=True, timeout=60)
            except (psutil.Error, subprocess.SubprocessError, OSError):
                pass
    if hasattr(signal, "SIGUSR1"):
        for q in tree:
            try:
                if any("main_slam" in part for part in q.cmdline()):
                    q.send_signal(signal.SIGUSR1)
            except psutil.Error:
                pass
        time.sleep(10)  # time for faulthandler to write


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description="see the module docstring")
    parser.add_argument("out")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds of run time (default 900)")
    parser.add_argument("--stall", type=float, default=120.0, help="seconds of silence in the log (default 120)")
    args = parser.parse_args(sys.argv[1:split])
    cmd = sys.argv[split + 1:]

    t0 = time.time()
    peak_rss = peak_fp = 0
    timed_out, stalled = False, None
    slept = 0.0
    with open(args.out + ".log", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        proc = psutil.Process(p.pid)
        last_size, last_growth, last_tick = 0, time.time(), time.time()
        while p.poll() is None:
            gap = time.time() - last_tick
            if gap > kSleepGap:
                slept += gap
                last_growth += gap
            last_tick = time.time()
            tree = process_tree(proc)
            try:
                peak_rss = max(peak_rss, sum(q.memory_info().rss for q in tree))
            except psutil.Error:
                pass
            if _libproc is not None:
                peak_fp = max(peak_fp, sum(phys_footprint(q.pid) for q in tree))
            size = os.path.getsize(args.out + ".log")
            if size != last_size:
                last_size, last_growth = size, time.time()
            elif time.time() - last_growth > args.stall:
                stalled = time.time() - t0
                capture_stacks(proc, args.out)
                kill_tree(proc)
                break
            if time.time() - t0 - slept > args.timeout:
                timed_out = True
                kill_tree(proc)
                break
            time.sleep(kTick)
        p.wait()
    result = dict(cmd=cmd, wall_s=time.time() - t0, slept_s=slept, returncode=p.returncode,
                  peak_rss_gb=peak_rss / 2**30, peak_footprint_gb=peak_fp / 2**30 if _libproc else None,
                  timed_out=timed_out, stalled_at_s=stalled)
    with open(args.out + ".run.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "cmd"}), flush=True)


if __name__ == "__main__":
    main()
