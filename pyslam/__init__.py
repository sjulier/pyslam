import os
import sys

# OpenMP: idle worker threads must not spin. conda-forge torch uses LLVM's OpenMP runtime (with MKL on
# Linux), whose threads busy-wait for 200 ms after each parallel region and starve pySLAM's own threads:
# on Linux, tracking went from 0.018 to 0.277 s per frame (track lost). KMP_BLOCKTIME is read only by
# LLVM's runtime; GNU OpenMP (pip torch on Linux) ignores it, and is slower with OMP_WAIT_POLICY=passive.
# The OpenMP runtime reads it once, when torch/numpy are first imported, so the main_*.py scripts
# import pyslam first. A value set in the environment takes precedence.
os.environ.setdefault("KMP_BLOCKTIME", "0")

if sys.platform == "win32":
    # Windows has no RPATH: tell the loader where the DLLs of the native modules built in the source tree
    # are (GTSAM's, which cpp_core, gtsam and gtsam_factors link)
    _gtsam_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "thirdparty", "gtsam_local", "install", "bin")
    if os.path.isdir(_gtsam_bin):
        os.add_dll_directory(_gtsam_bin)

if sys.platform == "darwin":
    # Some models use ops that Apple MPS does not implement (e.g. torchvision's deform_conv2d in ALIKED):
    # let torch run just those ops on the CPU. This must be set before torch is imported.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _keep_running_on_ctrl_z():
    """Ctrl+Z in the terminal (SIGTSTP) is ignored by pySLAM's main scripts and their worker processes,
    with a message. Under `pixi run` it suspended SLAM but not pixi, so the windows froze and the
    terminal did not come back; and a run suspended half-way (some of its processes stopped, some
    not) does not resume cleanly anyway. The main script sets PYSLAM_IGNORE_CTRL_Z for its workers,
    which import pyslam again when they are spawned (macOS); forked workers inherit the handler."""
    import signal
    import threading

    if not hasattr(signal, "SIGTSTP") or threading.current_thread() is not threading.main_thread():
        return
    is_main_script = os.path.basename(sys.argv[0] if sys.argv else "").startswith("main_")
    if not (is_main_script or os.environ.get("PYSLAM_IGNORE_CTRL_Z") == "1"):
        return  # pyslam used as a library: leave the program's signals alone
    if os.environ.get("PYSLAM_IGNORE_CTRL_Z") == "0":
        return  # opt out: PYSLAM_IGNORE_CTRL_Z=0
    os.environ["PYSLAM_IGNORE_CTRL_Z"] = "1"
    # the first process (the main script) says so once; its workers, forked or spawned, stay silent
    os.environ.setdefault("PYSLAM_CTRL_Z_MESSAGE_PID", str(os.getpid()))

    last_message = [0.0]

    def on_ctrl_z(signum, frame):
        import time

        # one message per key press: under `pixi run` the signal arrives twice (from the terminal, and
        # forwarded by pixi)
        if (
            os.environ.get("PYSLAM_CTRL_Z_MESSAGE_PID") == str(os.getpid())
            and time.monotonic() - last_message[0] > 1.0
        ):
            last_message[0] = time.monotonic()
            sys.stderr.write(
                "\npySLAM: Ctrl+Z is ignored (it would freeze the windows, not stop the run). "
                "To stop, press q in a window, or Ctrl+C here.\n"
            )
            sys.stderr.flush()

    try:
        signal.signal(signal.SIGTSTP, on_ctrl_z)
    except (ValueError, OSError):
        pass


_keep_running_on_ctrl_z()
