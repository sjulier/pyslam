"""
The TensorFlow worker: runs in the pixi environment `tf` (or `tf-cpu`) and serves the TensorFlow-based models to
pySLAM (pyslam/workers/tf_client.py starts it).

    pixi run -e tf python -m pyslam.workers.tf_worker --address /tmp/.../tf.sock

The authentication key comes from the environment variable PYSLAM_WORKER_AUTHKEY (hex). The worker
serves one connection and exits when it closes, so a pySLAM run that ends or crashes leaves no worker
behind holding GPU memory.

Requests (see protocol.py for the encoding):
    {"op": "create", "class": name, "args": [...], "kwargs": {...}} -> {"id": n, "attrs": {...}}
    {"op": "call", "id": n, "method": name, "args": [...], "kwargs": {...}} -> {"result": ...}
    {"op": "ping"} -> {"pong": True}
Errors come back as {"error": message, "traceback": text}.
"""

import argparse
import os
import shutil
import signal
import socket
import sys
import threading
import traceback
from multiprocessing.connection import Listener

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# By default TensorFlow takes almost all the GPU memory when it starts: take it as needed, so that the
# GPU stays shared with pySLAM's own (torch) models
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# TensorFlow Hub (HDC-DELF) caches its models in the temporary directory by default, where an interrupted
# download leaves an empty folder that fails every later load, and macOS clears it from time to time
os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "pyslam", "tfhub_modules"))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pyslam.workers import protocol  # noqa: E402


def provide_pkg_resources():
    """tensorflow_hub (used by HDC-DELF) checks TensorFlow's version with pkg_resources.parse_version
    when it is imported; pkg_resources is gone from setuptools >= 81 (the tf environment has a newer
    one). Provide that function, from packaging, if pkg_resources is missing."""
    import importlib.util

    if importlib.util.find_spec("pkg_resources") is not None:
        return
    import types

    from packaging.version import parse

    sys.modules["pkg_resources"] = types.ModuleType("pkg_resources")
    sys.modules["pkg_resources"].parse_version = parse


provide_pkg_resources()

kConnectTimeout = 120  # [s] from listening to pySLAM's connection (pySLAM connects at once)

# The classes the worker may create: name -> (module, library path to set first (config_libs.yaml))
CLASSES = {
    "DelfFeature2D": ("pyslam.local_features.feature_delf", None),
    "LfNetFeature2D": ("pyslam.local_features.feature_lfnet", None),
    "ContextDescFeature2D": ("pyslam.local_features.feature_contextdesc", None),
    "GeodescFeature2D": ("pyslam.local_features.feature_geodesc", None),
    "HDCDELF": ("feature_extraction.feature_extractor_holistic", "vpr"),
}


def public_attributes(obj):
    """The plain attributes of an object (numbers, strings, booleans), which the client mirrors."""
    attrs = {}
    for name, value in vars(obj).items():
        if not name.startswith("_") and isinstance(value, (bool, int, float, str)):
            attrs[name] = value
    return attrs


kDownloadTimeout = 60  # [s] without data, a model download fails instead of waiting forever


def remove_stale_hub_locks():
    """Removes TensorFlow Hub download locks left by a process of this machine that is gone (e.g. a
    download cut by sleep or a kill). The lock file holds "<hostname>.<pid>.<uid>"; TensorFlow Hub
    would otherwise wait for 10 minutes before taking it over."""
    cache = os.environ.get("TFHUB_CACHE_DIR")
    if not cache or not os.path.isdir(cache):
        return
    host = socket.gethostname()
    for name in os.listdir(cache):
        if not name.endswith(".lock"):
            continue
        lock = os.path.join(cache, name)
        try:
            with open(lock) as f:
                owner_host, pid, uid = f.read().strip().rsplit(".", 2)
            if owner_host != host:
                continue
            try:
                os.kill(int(pid), 0)
                continue  # its download is still running
            except ProcessLookupError:
                pass
            except PermissionError:
                continue  # another user's process
            print(f"tf_worker: removing the stale download lock {lock}", flush=True)
            shutil.rmtree(f"{lock[:-len('.lock')]}.{uid}.tmp", ignore_errors=True)
            os.remove(lock)
        except (OSError, ValueError):
            pass


def create(name, args, kwargs):
    if name not in CLASSES:
        raise ValueError(f"tf_worker: unknown class {name} (known: {', '.join(CLASSES)})")
    module_name, lib = CLASSES[name]
    import importlib

    if lib is not None:
        import pyslam.config as config

        config.cfg.set_lib(lib, prepend=True)
    cls = getattr(importlib.import_module(module_name), name)
    # A model may download its weights when it is created: no wait without end on a dead connection
    # (e.g. after the machine slept), and no wait on a lock that a killed download left. The timeout
    # applies to the sockets created meanwhile only, not to the connection with pySLAM.
    remove_stale_hub_locks()
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(kDownloadTimeout)
    try:
        return cls(*args, **kwargs)
    finally:
        socket.setdefaulttimeout(previous)


def serve(conn):
    objects = {}
    while True:
        try:
            msg = protocol.recv(conn)
        except (EOFError, OSError):
            return  # the client has gone: exit
        try:
            op = msg.get("op")
            if op == "ping":
                reply = {"pong": True}
            elif op == "create":
                obj = create(msg["class"], msg.get("args", []), msg.get("kwargs", {}))
                objects[len(objects) + 1] = obj
                reply = {"id": len(objects), "attrs": public_attributes(obj)}
            elif op == "call":
                obj = objects[msg["id"]]
                result = getattr(obj, msg["method"])(*msg.get("args", []), **msg.get("kwargs", {}))
                reply = {"result": result}
            else:
                raise ValueError(f"tf_worker: unknown request {op!r}")
        except Exception as e:  # noqa: BLE001  (reported to the client, which raises it there)
            reply = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        try:
            protocol.send(conn, reply)
        except (EOFError, OSError):
            return


def remove_socket(address):
    """Removes the socket and its private directory (made by tf_client.py for this worker)."""
    for remove, path in ((os.unlink, address), (os.rmdir, os.path.dirname(address))):
        try:
            remove(path)
        except OSError:
            pass  # already removed (or the directory is not empty)


def main():
    parser = argparse.ArgumentParser(description="pySLAM's TensorFlow worker (run in the pixi environment tf or tf-cpu)")
    parser.add_argument("--address", required=True, help="the Unix socket to listen on")
    args = parser.parse_args()
    authkey = bytes.fromhex(os.environ.get("PYSLAM_WORKER_AUTHKEY", ""))
    if not authkey:
        sys.exit("tf_worker: PYSLAM_WORKER_AUTHKEY is not set")

    # The worker owns the socket and its directory: it removes them however it ends (pySLAM may end with
    # os._exit or a signal, which skip the client's clean-up). It exits when the connection closes, so it
    # does not need to die with pySLAM's signals: tf_client.py starts it in a session of its own.
    def on_signal(signum, frame):
        print(f"tf_worker: signal {signum}, exiting", flush=True)
        remove_socket(args.address)
        os._exit(0)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)
    connected = threading.Event()

    def watchdog():  # the client that started the worker died before connecting
        if not connected.wait(timeout=kConnectTimeout):
            print(f"tf_worker: no connection within {kConnectTimeout} s, exiting", flush=True)
            remove_socket(args.address)
            os._exit(1)

    try:
        with Listener(args.address, family="AF_UNIX", authkey=authkey) as listener:
            print(f"tf_worker: listening on {args.address}", flush=True)
            threading.Thread(target=watchdog, daemon=True).start()
            with listener.accept() as conn:
                connected.set()
                serve(conn)
        print("tf_worker: connection closed, exiting", flush=True)
    finally:
        remove_socket(args.address)
    os._exit(0)  # without TensorFlow's teardown, which takes seconds


if __name__ == "__main__":
    main()
