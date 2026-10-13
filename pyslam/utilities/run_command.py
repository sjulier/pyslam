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

import os
import shutil
import signal
import subprocess
import psutil
import time

# The return code of a command stopped by the stall check of run_command_async()
kStalledReturnCode = -999
# How often the stall check looks at the command, in seconds
kStallCheckInterval = 10
# Below this share of one CPU core, on average since the last sign of progress, the command is idle
kStallMaxCpuShare = 0.05
# A command has stalled when it is idle, and no file was written, for stall_factor times its mean time
# per frame (from its progress file), but at least kStallMinSeconds; before it has a mean (the first
# frames: loading the models), for kStallStartupSeconds
kStallMinSeconds = 60
kStallStartupSeconds = 600


def run_command_sync(command, debug=False):
    """runs command and returns the output."""
    if debug:
        print("$ {}".format(command))
    p = subprocess.Popen(command, shell=True, executable="/bin/bash")
    return p


def _latest_file_time(folder):
    """The latest modification time of a file in a folder and its subfolders (0 if none)."""
    latest = 0.0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass  # removed meanwhile
    return latest


def _process_tree(process):
    try:
        root = psutil.Process(process.pid)
        return [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _cpu_seconds(processes):
    total = 0.0
    for p in processes:
        try:
            times = p.cpu_times()
            total += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def mean_frame_time(progress_file):
    """The mean time per frame of a run, from the progress file main_slam.py writes ("<frames>
    <time of the first frame> <time of the latest>"), or None before it has two frames."""
    try:
        with open(progress_file) as f:
            num_frames, first_time, last_time = f.read().split()[:3]
        num_frames, first_time, last_time = int(num_frames), float(first_time), float(last_time)
    except (OSError, ValueError):
        return None
    return (last_time - first_time) / (num_frames - 1) if num_frames >= 2 else None


def stall_limit(progress_file, stall_factor):
    """How long a run may be idle, writing no file, before it counts as stalled (seconds)."""
    mean = mean_frame_time(progress_file) if progress_file else None
    if mean is None:
        return kStallStartupSeconds
    return max(kStallMinSeconds, stall_factor * mean)


def _wait_unless_stalled(process, watch_folder, stall_factor, progress_file):
    """Wait for a command to finish. Return False when it does, and the idle time (seconds) when it
    stalls: no file in watch_folder changed (no new frame, no log line) and its processes used almost no
    CPU for longer than stall_limit() (a deadlock: every thread waits)."""
    file_time = _latest_file_time(watch_folder)
    cpu = _cpu_seconds(_process_tree(process))
    mark_time, mark_cpu = time.time(), cpu
    while True:
        try:
            process.wait(timeout=kStallCheckInterval)
            return False
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        new_file_time = _latest_file_time(watch_folder)
        cpu = _cpu_seconds(_process_tree(process))
        if cpu < mark_cpu:  # a process ended: its CPU time is no longer counted
            mark_cpu = cpu
        busy = cpu - mark_cpu > kStallMaxCpuShare * (now - mark_time) + 1.0
        if new_file_time != file_time or busy:
            file_time = new_file_time
            mark_time, mark_cpu = now, cpu
        elif now - mark_time > stall_limit(progress_file, stall_factor):
            return now - mark_time


def _dump_and_kill(process, dump_path):
    """Write the threads' stacks of a stalled command, then kill it with all its processes. The Python
    stacks go to its own log: main_slam.py dumps them on SIGUSR1 (Linux, macOS). With py-spy installed,
    the native stacks go to dump_path too."""
    processes = _process_tree(process)
    if hasattr(signal, "SIGUSR1"):
        for p in processes:
            try:
                if any("main_slam.py" in arg for arg in p.cmdline()):
                    p.send_signal(signal.SIGUSR1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(2)
    py_spy = shutil.which("py-spy")
    if py_spy and dump_path:
        with open(dump_path, "w") as f:
            for p in processes:
                try:
                    out = subprocess.run(
                        [py_spy, "dump", "--native", "--pid", str(p.pid)],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    f.write(f"==== pid {p.pid}\n{out.stdout}{out.stderr}\n")
                except (OSError, subprocess.TimeoutExpired) as e:
                    f.write(f"==== pid {p.pid}: py-spy failed: {e}\n")
    for p in processes:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait()


def run_command_async(
    command,
    output_log_filename=None,
    verbose=False,
    msg=None,
    stall_factor=None,
    stall_watch_folder=None,
):
    """Run a command and wait for it. With an output log file and stall_factor, a command which stalls
    (see _wait_unless_stalled(); stall_watch_folder defaults to the log's folder) is killed after its
    threads' stacks are dumped, and kStalledReturnCode is returned. The command gets the path of its
    progress file in PYSLAM_PROGRESS_FILE (main_slam.py writes it)."""
    # Redirect the standard output and standard error output if specifying an output file
    redirected_std_out = subprocess.PIPE
    redirected_std_err = subprocess.PIPE
    if output_log_filename:
        output_log_file = open(output_log_filename, "wb")
        redirected_std_out = output_log_file
        redirected_std_err = output_log_file
    if verbose:
        if msg:
            print(f"[{msg}] command: $ {command}, log file: {output_log_filename}")
        else:
            print("$ {}".format(command))
    check_stall = bool(output_log_filename and stall_factor)
    env = None
    if check_stall:
        watch_folder = stall_watch_folder or os.path.dirname(os.path.abspath(output_log_filename))
        progress_file = os.path.join(watch_folder, "progress.txt")
        if os.path.exists(progress_file):
            os.remove(progress_file)  # from an earlier attempt
        env = dict(os.environ, PYSLAM_PROGRESS_FILE=progress_file)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=redirected_std_out,
        stderr=redirected_std_err,
        text=True,  # Use text=True to get text output
        env=env,
    )
    if not check_stall:
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    idle_seconds = _wait_unless_stalled(process, watch_folder, stall_factor, progress_file)
    if idle_seconds:
        mean = mean_frame_time(progress_file)
        dump_path = os.path.join(watch_folder, "stall_dump.txt")
        _dump_and_kill(process, dump_path)
    output_log_file.close()
    if not idle_seconds:
        return process.returncode, None, None
    with open(output_log_filename, "a") as f:
        f.write(
            f"\nERROR: stalled: no new frame, no file written and no CPU used for {idle_seconds:.0f} s"
            + (f" ({idle_seconds / mean:.0f} times the mean time per frame, {mean:.2f} s)" if mean else "")
            + ", so the run was stopped (a deadlock, see docs/TROUBLESHOOTING.md; the threads' stacks "
            "are above"
            + (f" and in {dump_path}" if os.path.exists(dump_path) else "")
            + ")\n"
        )
    return kStalledReturnCode, None, None


def kill_processes_matching_cmd(keywords, time_sleep=2.0):
    """
    Kill processes whose command line contains all the specified keywords (substring match).

    Args:
        keywords (list of str): List of strings that must all appear in the cmdline string.
    """
    print(f"[INFO] Searching for processes matching: {keywords}")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue
            cmd_str = " ".join(cmdline)
            if all(word in cmd_str for word in keywords):
                print(f"[WARN] Killing process {proc.pid} matching: {cmd_str}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(time_sleep)
