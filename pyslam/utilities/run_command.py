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

import faulthandler
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
# Windows has no SIGUSR1, so a stalled run cannot be asked for its threads' stacks. There the run
# (main_slam.py, see StallStackTimer) writes them itself to this file, next to its progress file,
# once it has had no new frame for this share of its stall limit; run_command_async() then moves
# them into its log.
kStallStacksFilename = "stall_stacks.txt"
kStallStacksShare = 0.5


def run_command_sync(command, debug=False):
    """runs command and returns the output."""
    if debug:
        print("$ {}".format(command))
    p = subprocess.Popen(command, shell=True, executable="/bin/bash")
    return p


def _latest_file_time(folder):
    """The latest modification time of a file in a folder and its subfolders (0 if none), except the
    stacks a stalling run writes itself (StallStackTimer)."""
    latest = 0.0
    for root, _, files in os.walk(folder):
        for name in files:
            if name == kStallStacksFilename:
                continue
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


class StallStackTimer:
    """For a run which cannot be sent SIGUSR1 (Windows): faulthandler writes every thread's stack to
    kStallStacksFilename once the run has had no new frame for kStallStacksShare of its stall limit.
    Its own thread needs no GIL, so this works in a deadlock too. restart() after each frame starts
    the timer again, and empties the file when an earlier pause, which has ended, filled it."""

    def __init__(self, progress_file, stall_factor):
        self.progress_file = progress_file
        self.stall_factor = stall_factor
        path = os.path.join(os.path.dirname(progress_file), kStallStacksFilename)
        self.file = open(path, "wb", buffering=0)  # faulthandler writes to its descriptor
        self.restart()

    @staticmethod
    def create_if_needed(progress_file):
        """A timer for a run of run_command_async() with its stall check on, on a system without
        SIGUSR1; None otherwise."""
        stall_factor = float(os.environ.get("PYSLAM_STALL_FACTOR") or 0)
        if not progress_file or not stall_factor or hasattr(signal, "SIGUSR1"):
            return None
        return StallStackTimer(progress_file, stall_factor)

    def restart(self):
        fd = self.file.fileno()
        if os.fstat(fd).st_size:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
        seconds = kStallStacksShare * stall_limit(self.progress_file, self.stall_factor)
        faulthandler.dump_traceback_later(seconds, file=self.file)


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
    stacks go to its own log: main_slam.py dumps them on SIGUSR1 (Linux, macOS). Windows has no
    SIGUSR1: there main_slam.py has written them to kStallStacksFilename already (StallStackTimer).
    With py-spy installed, the native stacks go to dump_path too. Return whether SIGUSR1 was
    sent."""
    processes = _process_tree(process)
    signalled = False
    if hasattr(signal, "SIGUSR1"):
        for p in processes:
            try:
                if any("main_slam.py" in arg for arg in p.cmdline()):
                    p.send_signal(signal.SIGUSR1)
                    signalled = True
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
    return signalled


def _move_stall_stacks_to_log(stacks_path, log_path, after_seconds):
    """Append the stacks a run wrote itself (StallStackTimer) to its log, and remove their file.
    Return whether there were any."""
    try:
        with open(stacks_path, "rb") as f:
            stacks = f.read()
        os.remove(stacks_path)
    except OSError:
        return False
    if not stacks.strip():
        return False
    with open(log_path, "ab") as f:
        header = "\n---- every thread's stack, written by the run itself after "
        header += f"{after_seconds:.0f} s without a new frame ----\n"
        f.write(header.encode() + stacks)
    return True


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
    progress file in PYSLAM_PROGRESS_FILE (main_slam.py writes it), and stall_factor in
    PYSLAM_STALL_FACTOR (for StallStackTimer)."""
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
        stacks_path = os.path.join(watch_folder, kStallStacksFilename)
        for path in (progress_file, stacks_path):
            if os.path.exists(path):
                os.remove(path)  # from an earlier attempt
        env = dict(
            os.environ, PYSLAM_PROGRESS_FILE=progress_file, PYSLAM_STALL_FACTOR=str(stall_factor)
        )
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
    if not idle_seconds:
        output_log_file.close()
        if os.path.exists(stacks_path):
            os.remove(stacks_path)  # written during a pause, or after the last frame
        return process.returncode, None, None
    mean = mean_frame_time(progress_file)
    dump_path = os.path.join(watch_folder, "stall_dump.txt")
    signalled = _dump_and_kill(process, dump_path)
    output_log_file.close()
    after = kStallStacksShare * stall_limit(progress_file, stall_factor)
    in_log = signalled or _move_stall_stacks_to_log(stacks_path, output_log_filename, after)
    where = ["above"] if in_log else []
    if os.path.exists(dump_path):
        where.append(f"in {dump_path}")
    with open(output_log_filename, "a") as f:
        f.write(
            f"\nERROR: stalled: no new frame, no file written and no CPU used for {idle_seconds:.0f} s"
            + (f" ({idle_seconds / mean:.0f} times the mean time per frame, {mean:.2f} s)" if mean else "")
            + ", so the run was stopped (a deadlock, see docs/TROUBLESHOOTING.md; "
            + (
                "the threads' stacks are " + " and ".join(where)
                if where
                else "the threads' stacks were not written; with py-spy installed they go to "
                + dump_path
            )
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
