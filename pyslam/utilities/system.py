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

import sys
import os
import numpy as np
import logging
from termcolor import colored
import cv2

import threading
import logging
from logging.handlers import QueueHandler, QueueListener

# (not torch.multiprocessing: see pyslam/utilities/logging.py)
import multiprocessing as mp


from pathlib import Path
import gdown
import requests  # Use requests for general HTTP downloads
from tqdm import tqdm  # Import tqdm for progress bars

import shutil
import tempfile
import atexit
import time
import signal
import traceback
from typing import Optional


def getchar():
    print("press enter to continue:")
    a = input("").split(" ")[0]
    print(a)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False


class MissingImport:
    """Stand-in returned by import_from() when an optional import fails.

    It is falsy like the None returned before, but it keeps the cause: calling it or accessing
    an attribute (e.g. SomeFeature2D(...) or SomeModel.from_pretrained(...)) raises a RuntimeError
    that names the component, the cause and what to do, instead of a later
    "'NoneType' object is not callable".
    """

    def __init__(self, module, name, error):
        self._module = module
        self._name = name
        self._error = error

    def __bool__(self):
        return False

    def __repr__(self):
        return f"MissingImport({self._module}.{self._name})"

    def message(self):
        cause = f"{type(self._error).__name__}: {self._error}"
        missing = getattr(self._error, "name", None) or ""  # set by ModuleNotFoundError
        tf_names = ("tensorflow", "tensorflow_hub", "tf_slim", "keras")
        if missing.split(".")[0] in tf_names or "tensorflow" in str(self._error).lower():
            hint = "It needs TensorFlow, which is not part of the core installation."
        elif isinstance(self._error, ModuleNotFoundError):
            hint = (
                f"The Python module '{missing}' is missing. Learned features and matchers need "
                "scripts/install_git_modules.sh (third-party code, patches and model weights), and "
                "some components scripts/install_thirdparty.sh; otherwise install the package."
            )
        else:
            hint = "Install or fix the component, or select another one in the configuration."
        return (
            f"{self._name} is not available: cannot import it from {self._module} ({cause})\n"
            f"{hint} See also docs/TROUBLESHOOTING.md."
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(self.message())

    def __getattr__(self, attr):
        if attr.startswith("__"):  # keep copy/pickle/inspection working
            raise AttributeError(attr)
        raise RuntimeError(self.message())


def is_main_process():
    """True in the main process, False in multiprocessing workers. A spawned worker (macOS) re-imports
    the modules before multiprocessing.parent_process() is set, but its name is set already."""
    import multiprocessing

    return (
        multiprocessing.parent_process() is None
        and multiprocessing.current_process().name == "MainProcess"
    )


def import_native_module(name, build_hint):
    """Import a pySLAM native (C++/pybind11) module, or raise ModuleNotFoundError with how to build it.

    When the module is not built, Python may instead import its thirdparty source folder (on sys.path)
    as an empty namespace package, which then fails later with a confusing AttributeError: treat that
    as not built too.
    """
    import importlib

    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        module = None
    if module is None or getattr(module, "__file__", None) is None:
        raise ModuleNotFoundError(f"The native module '{name}' is not built: {build_hint}", name=name)
    return module


# This function check and exec 'from module import name' and directly return the 'name'.'method'.
# The method is used to directly return a 'method' of 'name' (i.e. 'module'.'name'.'method')
# N.B.: if a method is needed you CAN'T
#   from module import name.method
# since method is an attribute of name!
# If the import fails, it returns a falsy MissingImport that raises a descriptive error when used.
def import_from(module, name, method=None, debug=False):
    from .logging import Printer

    try:
        imported_module = __import__(module, fromlist=[name])
        imported_name = getattr(imported_module, name)
        if method is None:
            return imported_name
        else:
            return getattr(imported_name, method)
    except Exception as e:
        if method is not None:
            name = name + "." + method
        Printer.orange(
            "WARNING: cannot import "
            + name
            + " from "
            + module
            + f" ({type(e).__name__}: {e}), check the file docs/TROUBLESHOOTING.md"
        )
        if debug:
            Printer.orange(traceback.format_exc())
        return MissingImport(module, name, e)


class _LazyImport:
    """Stand-in returned by import_from_lazy(): runs import_from() on first use (call, attribute
    access or truth test) and then behaves like the imported object (or its MissingImport)."""

    def __init__(self, module, name, method=None):
        self._lazy_args = (module, name, method)
        self._lazy_target = None
        self._lazy_done = False

    def _resolve(self):
        if not self._lazy_done:
            module, name, method = self._lazy_args
            self._lazy_target = import_from(module, name, method)
            self._lazy_done = True
        return self._lazy_target

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr):
        if attr.startswith("_lazy"):
            raise AttributeError(attr)
        return getattr(self._resolve(), attr)

    def __bool__(self):
        return bool(self._resolve())

    def __repr__(self):
        module, name, _ = self._lazy_args
        return f"<lazy {module}.{name}: {self._lazy_target!r}>" if self._lazy_done else f"<lazy {module}.{name}>"


def import_from_lazy(module, name, method=None):
    """Like import_from(), but the import happens on first use. Use it for optional components, so
    that they are imported (and their warnings printed) only if the configuration selects them."""
    return _LazyImport(module, name, method)


def lazy_module(name):
    """Return module `name` imported lazily (importlib.util.LazyLoader): the module code runs on the
    first attribute access. For heavy optional dependencies used only in some code paths."""
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def get_opencv_version():
    opencv_major = int(cv2.__version__.split(".")[0])
    opencv_minor = int(cv2.__version__.split(".")[1])
    opencv_build = int(cv2.__version__.split(".")[2])
    return (opencv_major, opencv_minor, opencv_build)


def is_opencv_version_greater_equal(a, b, c):
    opencv_version = get_opencv_version()
    return (
        opencv_version[0] * 1000 + opencv_version[1] * 100 + opencv_version[2]
        >= a * 1000 + b * 100 + c
    )


def check_if_main_thread(message=""):
    if threading.current_thread() is threading.main_thread():
        print(f"This is the main thread. {message}")
        return True
    else:
        print(f"This is NOT the main thread. {message}")
        return False


# Set the limit of open files. This is useful when using multiprocessing and socket management
# returns the error: OSError: [Errno 24] Too many open files.
def set_rlimit():
    import resource

    # Check the current soft and hard limits
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Set the new limit
    new_soft_limit = 4096
    if new_soft_limit > soft_limit:
        print(f"set_rlimit(): Current soft limit: {soft_limit}, hard limit: {hard_limit}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))

        # Confirm the change
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"set_rlimit(): Updated soft limit: {soft_limit}, hard limit: {hard_limit}")


# To fix this issue under linux: https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found
def locally_configure_qt_environment():
    ci_and_not_headless = False  # Default value in case the import fails
    try:
        from cv2.version import ci_build, headless

        ci_and_not_headless = ci_build and not headless
    except ImportError:
        pass  # Handle the case where cv2.version does not exist

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)  # Remove if exists
        os.environ.pop("QT_QPA_FONTDIR", None)  # Remove if exists


def force_kill_all_and_exit(code=0, verbose=True):
    """
    Force kill all remaining processes and exit the program.
    """
    # print("[!] Force shutdown initiated.")

    # Log active threads (excluding main)
    active_threads = [t for t in threading.enumerate() if t != threading.main_thread()]
    if active_threads:
        if verbose:
            print(f"[!] Active threads: {[t.name for t in active_threads]}")

    # Attempt to stop threads (cannot forcibly kill threads in Python)
    for t in active_threads:
        if verbose:
            print(f"[!] Thread {t.name} is still running and cannot be force-killed.")

    # Terminate all active multiprocessing children
    active_children = mp.active_children()
    if active_children:
        if verbose:
            print(f"[!] Active child processes: {active_children}")
    for p in active_children:
        try:
            if verbose:
                print(f"[!] Terminating process PID {p.pid}...")
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                if verbose:
                    print(f"[!] Killing stubborn process PID {p.pid}...")
                os.kill(p.pid, signal.SIGKILL)
        except Exception as e:
            if verbose:
                print(f"[!] Failed to terminate process PID {p.pid}: {e}")
            traceback.print_exc()

    # Wait briefly to allow processes to shut down
    time.sleep(0.5)

    if verbose:
        print("[✓] All processes attempted to terminate. Exiting.")
        print("[!] Note: Remaining threads (if any) will be terminated with the process.")

    # Force exit immediately - this bypasses all Python cleanup and thread cleanup
    # Note: The threads reported above are just warnings - os._exit() will terminate
    # the entire process regardless of any remaining threads. The process will exit
    # immediately even if QueueFeederThread or other threads are still running.
    # os._exit() skips the interpreter's shutdown, which is what flushes the output buffers: when the
    # output goes to a file or a pipe (block-buffered), the end of it would be lost
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

    os._exit(code)  # Bypass cleanup and exit immediately
