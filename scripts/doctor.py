#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Check that this machine and this pySLAM checkout are ready to run: the operating system and memory,
the Python environment, the GPU PyTorch can use, every native module (built, loadable, one pybind11
ABI), the ORB vocabulary and the bundled test video. With --run it also runs main_slam.py --headless
on the KITTI 06 video and reports the frames tracked and the trajectory error.

Each check prints one line: OK, WARN, FAIL or INFO, then what was found or what to do. Every import
runs in its own process, so a module that crashes is reported instead of stopping the check. Nothing
on the machine is changed (--run writes to results/ and logs/, as main_slam.py does).

usage: pixi run doctor [--run] [--json FILE]       (or: python scripts/doctor.py, in the pySLAM environment)
Exits with 0 when nothing failed (warnings are allowed), 1 otherwise. Include the output when asking for help.
"""
import argparse
import glob
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GB = 1024**3
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Report:
    def __init__(self):
        self.items = []

    def section(self, title):
        print(f"\n== {title}", flush=True)

    def add(self, status, name, detail=""):
        self.items.append({"status": status, "name": name, "detail": detail})
        print(f"  {status:<4}  {name}" + (f": {detail}" if detail else ""), flush=True)

    def count(self, status):
        return sum(i["status"] == status for i in self.items)

    def failed(self):
        return [i for i in self.items if i["status"] == "FAIL"]


def run_python(code, timeout=180, isolated=False):
    """Run code in a fresh interpreter from the checkout, with the checkout first on the import path.
    Returns (ok, detail): the last stdout line when it succeeds, the last stderr line (the exception)
    when it fails; warnings go to stderr, so they never displace the result. isolated=True (python -I)
    ignores PYTHONPATH and the current directory, to see what the environment itself provides."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable] + (["-I"] if isolated else []) + ["-c", code]
    try:
        p = subprocess.run(cmd, cwd=ROOT_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} s"
    out = [l for l in (ANSI.sub("", l).strip() for l in p.stdout.splitlines()) if l]
    err = [l for l in (ANSI.sub("", l).strip() for l in p.stderr.splitlines()) if l]
    if p.returncode < 0:
        return False, f"crashed (signal {-p.returncode})" + (f": {err[-1]}" if err else "")
    if p.returncode == 0:
        return True, out[-1] if out else ""
    return False, err[-1] if err else (out[-1] if out else f"exit code {p.returncode}")


def import_built_module(name):
    """Import a native module and return its file name. thirdparty/ is on sys.path, so an unbuilt
    module's source folder imports as an empty namespace package (no __file__): that counts as not
    built, as in pyslam.utilities.system.import_native_module."""
    import importlib
    import os

    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        module = None
    if module is None or getattr(module, "__file__", None) is None:
        raise ModuleNotFoundError(f"the native module '{name}' is not built")
    return os.path.basename(module.__file__)


# Run before each native-module check in the child process: pySLAM's Config puts the lib folders on
# sys.path (and prints while doing so), then import_built_module as defined above.
CHILD_PREFIX = textwrap.dedent("""
    import os, sys
    sys.stdout = open(os.devnull, "w")
    import pyslam.config as config
    cfg = getattr(config, "cfg", None) or config.Config()
    sys.stdout = sys.__stdout__
    """) + inspect.getsource(import_built_module)


def is_wsl():
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def memory_total_and_available():
    if sys.platform == "darwin":
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        return total, None
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            info[key] = int(value.split()[0]) * 1024
    return info["MemTotal"], info.get("MemAvailable")


def git_commit():
    try:
        p = subprocess.run(["git", "-C", ROOT_DIR, "log", "-1", "--format=%h %cs"], capture_output=True, text=True)
        return p.stdout.strip()
    except OSError:
        return ""


def check_machine(r):
    r.section("Machine")
    system, arch = platform.system(), platform.machine()
    if system == "Darwin":
        version = platform.mac_ver()[0]
        major = int(version.split(".")[0]) if version else 0
        ok = major >= 14 and arch == "arm64"
        r.add("OK" if ok else "FAIL", "macOS", f"{version} on {arch}" + ("" if ok else " (pySLAM needs macOS 14 or later on Apple silicon)"))
    elif system == "Linux":
        distro = ""
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                fields = dict(l.rstrip().split("=", 1) for l in f if "=" in l)
            distro = fields.get("PRETTY_NAME", "").strip('"')
        ok = arch == "x86_64"
        r.add("OK" if ok else "WARN", "WSL2" if is_wsl() else "Linux",
              f"{distro} on {arch}" + ("" if ok else " (the course setup is tested on x86-64)"))
    else:
        r.add("FAIL", "operating system", f"{system}: pySLAM runs on macOS, Linux and WSL2 (Ubuntu inside Windows)")

    total, available = memory_total_and_available()
    detail = f"{total / GB:.0f} GB" + (f" ({available / GB:.0f} GB free)" if available else "")
    status = "OK"
    if total < 12 * GB:
        status = "WARN"
        detail += " - building GTSAM needs about 12 GB of free memory (PYSLAM_BUILD_JOBS=1 builds one file at a time)"
        if is_wsl():
            detail += "; WSL2 gets half of the PC's memory by default: set memory= and swap= in %UserProfile%\\.wslconfig, then wsl --shutdown"
    elif is_wsl():
        detail += " (WSL2 gets half of the PC's memory by default; memory= in %UserProfile%\\.wslconfig changes it)"
    r.add(status, "memory", detail)
    r.add("INFO", "CPU cores", str(os.cpu_count()))
    free = shutil.disk_usage(ROOT_DIR).free
    r.add("OK" if free >= 10 * GB else "WARN", "free disk",
          f"{free / GB:.0f} GB on the checkout's disk" + ("" if free >= 10 * GB else " - keep about 10 GB free for datasets and results"))


def check_environment(r):
    r.section("Python environment")
    pixi, pixi_env = os.environ.get("PIXI_PROJECT_NAME"), os.environ.get("PIXI_ENVIRONMENT_NAME")
    conda, venv = os.environ.get("CONDA_DEFAULT_ENV"), os.environ.get("VIRTUAL_ENV")
    if pixi:
        env = f"pixi environment '{pixi_env or 'default'}' of {pixi}"
    elif conda and conda != "base":
        env = f"conda environment {conda}"
    elif venv:
        env = f"venv {venv}"
    else:
        env = None
    r.add("OK" if env else "WARN", "environment",
          (env or "none active") + f" ({sys.executable})" + ("" if env else " - run `pixi run doctor`, or activate the pySLAM environment first"))
    version = platform.python_version()
    r.add("OK" if version.startswith("3.11.") else "WARN", "Python", version + ("" if version.startswith("3.11.") else " (pySLAM is set up for 3.11)"))
    commit = git_commit()
    r.add("OK", "checkout", f"{ROOT_DIR} ({commit or 'not a git checkout'})")
    # pySLAM is installed in editable mode, so the environment's pyslam package points into the checkout it belongs to
    ok, out = run_python("import pyslam, os; print(os.path.dirname(os.path.dirname(os.path.abspath(pyslam.__file__))))",
                         timeout=60, isolated=True)
    if not ok:
        r.add("WARN", "pyslam package", "the active Python has no pyslam package: is this the pySLAM environment? (the checks below use this checkout anyway)")
    elif os.path.realpath(out) != os.path.realpath(ROOT_DIR):
        r.add("WARN", "pyslam package", f"the active Python imports pyslam from {out}, not from this checkout: its native modules may not match")
    else:
        r.add("OK", "pyslam package", "installed from this checkout (editable)")
    return commit


def check_gpu(r):
    r.section("GPU (PyTorch)")
    code = textwrap.dedent("""
        import json, torch
        info = {"torch": torch.__version__, "kind": "cpu"}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            # a kernel that runs is the real test: torch raises 'no kernel image' when this build cannot
            # run on the GPU (the arch list alone is not enough: sm_89 runs sm_86 kernels)
            x = torch.ones(256, 256, device="cuda"); (x @ x).sum().item()
            info.update(kind="cuda", name=p.name, memory_gb=p.total_memory / 2**30, cuda=torch.version.cuda,
                        arch=f"sm_{p.major}{p.minor}")
        elif torch.backends.mps.is_available():
            x = torch.ones(256, 256, device="mps"); (x @ x).sum().item()
            info.update(kind="mps", name="Apple GPU (MPS)")
        print(json.dumps(info))
        """)
    ok, out = run_python(code)
    info = None
    if ok:
        try:
            info = json.loads(out)
        except ValueError:
            pass
    if info is None:
        r.add("FAIL", "torch", out)
        return
    if info["kind"] == "cuda":
        r.add("OK", "torch device", f"CUDA {info['cuda']}: {info['name']}, {info['memory_gb']:.0f} GB, {info['arch']} (torch {info['torch']})")
    elif info["kind"] == "mps":
        r.add("OK", "torch device", f"Apple GPU (MPS), torch {info['torch']}")
    else:
        r.add("WARN", "torch device", f"CPU only (torch {info['torch']}): learned features and depth models will be slow")


def abi_summary(returncode, lines):
    """(status, one-line detail) from scripts/check_pybind11_abi.py's exit code and output lines."""
    if returncode == 0:
        last = lines[-1] if lines else "OK"
        found = re.search(r"OK: (\d+) module", last)
        # the script also counts GTSAM's own module, so a bare environment still reports one or two;
        # a built default level has 15 or more
        if found and int(found.group(1)) < 10:
            return "WARN", f"only {found.group(1)} module(s) built: run `pixi run build`"
        return "OK", last
    if returncode == 2:
        return "FAIL", "no built module found: run `pixi run build`"
    counts = "; ".join(l.strip() for l in lines if "module(s)" in l)
    return "FAIL", f"built with different pybind11 ABIs ({counts}): rebuild them with the bundled thirdparty/pybind11"


NATIVE_MODULES = [
    # level, name, child code (after CHILD_PREFIX), the task that builds it
    ("required", "OpenCV", "import cv2; print(cv2.__version__)", ""),
    ("required", "GTSAM + gtsam_factors", "import gtsam; print(import_built_module('gtsam_factors'))", "pixi run build-gtsam"),
    ("required", "g2o", "print(import_built_module('g2o'))", "pixi run build-g2o"),
    ("required", "ORB features", "print(import_built_module('orbslam2_features'))", "pixi run build-orbslam2-features"),
    ("required", "DBoW3", "cfg.set_lib('pydbow3'); print(import_built_module('pydbow3'))", "pixi run build-dbow3"),
    ("optional", "DBoW2", "cfg.set_lib('pydbow2'); print(import_built_module('pydbow2'))", "pixi run build-dbow2"),
    ("optional", "iBoW", "cfg.set_lib('pyibow'); print(import_built_module('pyibow'))", "pixi run build-ibow"),
    ("required", "C++ utilities (cpp/lib)", "print(import_built_module('pnpsolver') + ', ' + import_built_module('sim3solver'))", "pixi run build-cpp-utils"),
    ("required", "C++ core", "import pyslam.slam.cpp; from pyslam.config_parameters import Parameters; "
                             "print(import_built_module('cpp_core') + f' (USE_CPP_CORE={Parameters.USE_CPP_CORE})')", "pixi run build-cpp-core"),
    ("optional", "3D viewer (pypangolin)", "print(import_built_module('pypangolin'))", "pixi run build-pangolin"),
    ("required", "pyslam.slam", "import pyslam.slam.slam; print('imports')", "pixi run build"),
]


def check_native_modules(r):
    r.section("Native modules")
    p = subprocess.run([sys.executable, os.path.join(ROOT_DIR, "scripts", "check_pybind11_abi.py")],
                       cwd=ROOT_DIR, capture_output=True, text=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    status, detail = abi_summary(p.returncode, lines)
    r.add(status, "pybind11 ABI", detail)
    for level, name, code, build in NATIVE_MODULES:
        ok, out = run_python(CHILD_PREFIX + "\n" + code)
        if not ok and build:
            if "C++ core is not built" in out and os.environ.get("PIXI_PROJECT_NAME"):
                # the core's own message gives the pre-pixi commands, which stop outside pixi now
                out = f"the C++ core is not built: {build}"
            elif "the native module '" in out or "No module named" in out:
                out += f": {build}"
        r.add("OK" if ok else ("FAIL" if level == "required" else "WARN"), name, out)


def check_data(r):
    r.section("Data")
    # the default loop detector's vocabulary: the text file on macOS, the DBoW3 binary elsewhere
    # (pyslam.loop_closing.loop_detector_vocabulary), with the size pySLAM expects for a complete download
    ok, out = run_python(textwrap.dedent("""
        import json, platform
        from pyslam.loop_closing import loop_detector_vocabulary as v
        name = "ORBvoc.txt" if platform.system() == "Darwin" else "ORBvoc.dbow3"
        print(json.dumps({"path": v.kDataFolder + "/" + name, "size": v.kVocabularyFileSizes.get(name)}))
        """), timeout=120)
    try:
        vocab = json.loads(out) if ok else None
    except ValueError:
        vocab = None
    if vocab is None:
        vocab = {"path": os.path.join(ROOT_DIR, "data", "ORBvoc.txt" if sys.platform == "darwin" else "ORBvoc.dbow3"), "size": None}
    path, expected = vocab["path"], vocab["size"]
    name = os.path.basename(path)
    if not os.path.exists(path):
        r.add("WARN", "ORB vocabulary", f"{name} missing: run `pixi run vocabulary` (about 100 MB) before the first SLAM run")
    elif expected and os.path.getsize(path) != expected:
        r.add("FAIL", "ORB vocabulary", f"{name} is {os.path.getsize(path) / 1e6:.0f} MB, expected {expected / 1e6:.0f} MB: an interrupted download; delete it and run `pixi run vocabulary`")
    else:
        r.add("OK", "ORB vocabulary", f"{name} ({os.path.getsize(path) / 1e6:.0f} MB)")
    video = os.path.join(ROOT_DIR, "data", "videos", "kitti06", "video_color.mp4")
    r.add("OK" if os.path.exists(video) else "FAIL", "KITTI 06 test video",
          "present" if os.path.exists(video) else f"missing: {video} (the checkout is incomplete)")


def parse_metrics_info(text):
    """The key: value lines of results/metrics_*/other_metrics_info.txt as a dict of strings."""
    info = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            info[key.strip()] = value.strip()
    return info


def run_slam(r, log_path):
    """The guide's own end-to-end check: main_slam.py --headless on the bundled KITTI 06 video."""
    r.section("SLAM run (KITTI 06, headless)")
    t0 = time.time()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        try:
            p = subprocess.run([sys.executable, "main_slam.py", "--headless"], cwd=ROOT_DIR, stdout=log,
                               stderr=subprocess.STDOUT, text=True, timeout=1200)
        except subprocess.TimeoutExpired:
            r.add("FAIL", "main_slam.py --headless", f"timed out after 1200 s (log: {log_path})")
            return
    seconds = time.time() - t0
    # the run writes results/metrics_<date>/; take the folder this run created
    dirs = [d for d in glob.glob(os.path.join(ROOT_DIR, "results", "metrics_*")) if os.path.getmtime(d) >= t0 - 1]
    if p.returncode != 0 or not dirs:
        r.add("FAIL", "main_slam.py --headless",
              f"exit code {p.returncode} after {seconds:.0f} s" + ("" if dirs else ", no metrics written") + f" (log: {log_path})")
        return
    folder = max(dirs, key=os.path.getmtime)
    rmse, frames = None, {}
    try:
        with open(os.path.join(folder, "plot", "stats_final.json")) as f:
            rmse = json.load(f)["rmse"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open(os.path.join(folder, "other_metrics_info.txt")) as f:
            frames = parse_metrics_info(f.read())
    except OSError:
        pass
    try:
        percent_lost = float(frames.get("percent_lost", "nan"))
    except ValueError:
        percent_lost = float("nan")
    detail = (f"{frames.get('num_processed_frames')}/{frames.get('num_total_frames')} frames, "
              f"{frames.get('num_lost_frames')} lost ({percent_lost:.1f}%), {seconds:.0f} s, "
              + (f"ATE RMSE {rmse:.1f} m (it varies from run to run: 11-21 m in our tests)" if rmse is not None else "no ATE in stats_final.json")
              + f" (log: {log_path})")
    # a relocalisation or two is normal (runs differ); more than 5% lost or an ATE far outside the usual range is not
    r.add("OK" if percent_lost <= 5 and rmse is not None and rmse <= 30 else "WARN", "main_slam.py --headless", detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run", action="store_true", help="also run main_slam.py --headless on the KITTI 06 video (about 2 min)")
    parser.add_argument("--json", metavar="FILE", help="also write the results to this JSON file")
    args = parser.parse_args()

    r = Report()
    print(f"pySLAM doctor - {platform.node()} - {datetime.now():%Y-%m-%d %H:%M}", flush=True)
    check_machine(r)
    commit = check_environment(r)
    check_gpu(r)
    check_native_modules(r)
    check_data(r)
    if args.run:
        run_slam(r, os.path.join(ROOT_DIR, "logs", "doctor_slam.log"))

    failed = r.failed()
    print(f"\n{'READY' if not failed else 'NOT READY'}: {len(failed)} failed, {r.count('WARN')} warnings", flush=True)
    for item in failed:
        print(f"  - {item['name']}: {item['detail']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"host": platform.node(), "time": datetime.now().isoformat(timespec="seconds"),
                       "platform": platform.platform(), "python": platform.python_version(),
                       "checkout": ROOT_DIR, "commit": commit, "ready": not failed, "items": r.items}, f, indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
