"""
pySLAM's side of the TensorFlow worker (tf_worker.py).

In an environment without TensorFlow (pySLAM's own pixi environments), the TensorFlow-based classes are
replaced by RemoteObject proxies: creating one starts the worker in the pixi environment `tf` or `tf-cpu` (once per
process) and creates the real object there; calling a method sends its arguments (images, keypoints)
over a Unix socket and returns the result. With TensorFlow installed in the current environment, the
classes are used directly, as before.

Environment variables:
    PYSLAM_TF_ENVIRONMENT   the pixi environment of the worker (default: tf-cpu from the CPU ladder, else tf)
    PYSLAM_TF_WORKER        connect to a running worker at this Unix socket path instead of starting one
    PYSLAM_WORKER_AUTHKEY   the authentication key (hex) of that worker
"""

import atexit
import importlib.util
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from functools import partial
from multiprocessing.connection import Client

from pyslam.workers import protocol

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

kWorkerStartTimeout = 600  # [s] the first start may install the pixi environment tf
kFolderPrefix = "pyslam-tf-"  # the private folders of the workers' sockets, in the temporary directory


def has_tensorflow():
    return importlib.util.find_spec("tensorflow") is not None


def find_pixi():
    pixi = os.environ.get("PIXI_EXE") or shutil.which("pixi")
    if not pixi and os.path.exists(os.path.expanduser("~/.pixi/bin/pixi")):
        pixi = os.path.expanduser("~/.pixi/bin/pixi")
    return pixi


def tf_environment():
    """The pixi environment of the worker: tf-cpu from the CPU ladder (default-cpu, ...), else tf."""
    env = os.environ.get("PYSLAM_TF_ENVIRONMENT")
    if env:
        return env
    return "tf-cpu" if os.environ.get("PIXI_ENVIRONMENT_NAME", "").endswith("-cpu") else "tf"


def sweep_stale_folders(min_age=kWorkerStartTimeout):
    """Removes the socket folders of workers that are gone (e.g. killed with SIGKILL, or a machine
    that went down): folders older than min_age whose socket does not accept connections."""
    tmp = tempfile.gettempdir()
    try:
        names = [n for n in os.listdir(tmp) if n.startswith(kFolderPrefix)]
    except OSError:
        return
    for name in names:
        folder = os.path.join(tmp, name)
        try:
            st = os.stat(folder)
            if st.st_uid != os.getuid() or time.time() - st.st_mtime < min_age:
                continue
            sock = os.path.join(folder, "tf.sock")
            if os.path.exists(sock):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    s.connect(sock)
                    continue  # a live worker
                except OSError:
                    pass
                finally:
                    s.close()
            shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


class WorkerError(RuntimeError):
    pass


class TfWorkerClient:
    """A connection to a TensorFlow worker, started by this process unless PYSLAM_TF_WORKER is set."""

    def __init__(self):
        self.process = None
        self.tmpdir = None
        self.lock = threading.Lock()
        address = os.environ.get("PYSLAM_TF_WORKER")
        if address:
            authkey = bytes.fromhex(os.environ.get("PYSLAM_WORKER_AUTHKEY", ""))
            self.conn = Client(address, family="AF_UNIX", authkey=authkey)
        else:
            self.conn = self._start()
        atexit.register(self.close)

    def _start(self):
        pixi = find_pixi()
        if pixi is None:
            raise WorkerError("the TensorFlow worker needs pixi (it runs in the pixi environment tf)")
        sweep_stale_folders()
        self.tmpdir = tempfile.mkdtemp(prefix=kFolderPrefix)
        address = os.path.join(self.tmpdir, "tf.sock")
        authkey = os.urandom(32)
        env = dict(os.environ, PYSLAM_WORKER_AUTHKEY=authkey.hex())
        manifest = os.environ.get("PIXI_PROJECT_MANIFEST", os.path.join(ROOT, "pixi.toml"))
        environment = tf_environment()
        cmd = [pixi, "run", "--manifest-path", manifest, "-e", environment,
               "python", "-m", "pyslam.workers.tf_worker", "--address", address]
        print(f"TensorFlow worker: starting it in the pixi environment {environment} ...", flush=True)
        # in a session of its own, so that a Ctrl-C or a signal to pySLAM's process group does not kill it
        # before it removes its socket: it exits when the connection closes (tf_worker.py)
        self.process = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True)
        time_start = time.time()
        while not os.path.exists(address):
            if self.process.poll() is not None:
                shutil.rmtree(self.tmpdir, ignore_errors=True)
                raise WorkerError(
                    f"the TensorFlow worker stopped (exit code {self.process.returncode}): "
                    f"is the pixi environment {environment} installed? (pixi run models-tf installs it and the models)"
                )
            if time.time() - time_start > kWorkerStartTimeout:
                self.process.kill()
                raise WorkerError(f"the TensorFlow worker did not start within {kWorkerStartTimeout} s")
            time.sleep(0.1)
        print(f"TensorFlow worker: ready ({time.time() - time_start:.1f} s)", flush=True)
        return Client(address, family="AF_UNIX", authkey=authkey)

    def request(self, msg):
        with self.lock:
            protocol.send(self.conn, msg)
            reply = protocol.recv(self.conn)
        if "error" in reply:
            raise WorkerError(f"TensorFlow worker: {reply['error']}\n{reply.get('traceback', '')}")
        return reply

    def close(self):
        conn, self.conn = getattr(self, "conn", None), None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=10)  # it exits when the connection closes
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None


_client = None
_client_lock = threading.Lock()


def get_client():
    """The worker of this process, started on first use. A process started with spawn (e.g. the loop
    detecting process) gets its own."""
    global _client
    with _client_lock:
        if _client is None:
            _client = TfWorkerClient()
        return _client


class RemoteObject:
    """An object living in the TensorFlow worker: its plain attributes are mirrored, its methods are
    called remotely."""

    def __init__(self, class_name, *args, **kwargs):
        self._class_name = class_name
        self._client = get_client()
        reply = self._client.request({"op": "create", "class": class_name, "args": list(args), "kwargs": kwargs})
        self._id = reply["id"]
        self.__dict__.update(reply["attrs"])

    def _call(self, method, *args, **kwargs):
        msg = {"op": "call", "id": self._id, "method": method, "args": list(args), "kwargs": kwargs}
        return self._client.request(msg)["result"]

    def __getattr__(self, name):  # only for names not found on the proxy: remote methods
        if name.startswith("_"):
            raise AttributeError(name)
        return partial(self._call, name)

    def __repr__(self):
        return f"<{self._class_name} in the TensorFlow worker>"


def tf_class(module, name):
    """The class `name` of `module` if TensorFlow is installed here, else a factory of RemoteObject
    proxies for it (same call signature)."""
    if has_tensorflow():
        from pyslam.utilities.system import import_from_lazy

        return import_from_lazy(module, name)
    return partial(RemoteObject, name)
