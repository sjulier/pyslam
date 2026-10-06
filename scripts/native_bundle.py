#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Prebuilt native modules ("native bundles"): pack them on a build machine, fetch them on another.

pySLAM's native modules take from 15 minutes to over an hour to build. The pixi lock file gives every
machine the same compiler and libraries, so the modules built on one machine run on another one of the
same platform, provided that
  - they were built for a common CPU baseline (PYSLAM_MARCH=x86-64-v3, i.e. AVX2), not -march=native;
  - their library search paths are relative to the checkout (done by `pack`).

The modules come in two parts, each with its own bundle:
  prereq  the libraries pySLAM builds on, which rarely change: GTSAM (+ gtsam_factors), g2o, Pangolin,
          DBoW2/3, iBoW. Almost all of the build time. Also carries what is needed to build the pyslam
          part against them (GTSAM's headers and CMake files, g2o's static libraries).
  pyslam  pySLAM's own C++ code: the ORB-SLAM2 features, the C++ utilities (cpp/) and the C++ core
          (pyslam/slam/cpp). A few minutes to build.
A change to pySLAM's C++ code changes only the pyslam part's key: `pixi run build` then installs the
prereq bundle and builds the pyslam part from source. A change to the prereq part (or to pixi.lock)
changes both keys, since the pyslam part is built against it.

usage:
  python scripts/native_bundle.py key [PART]     print the keys and the file names of the bundles for this checkout
  python scripts/native_bundle.py pack [PART]    pack the built modules (build machine; needs patchelf on Linux)
  python scripts/native_bundle.py fetch [PART]   download and unpack the bundle for this checkout, then check it
                                                 (exit code 0: installed; 2: no bundle for this checkout or machine)
  python scripts/native_bundle.py remove [PART]  remove the files a fetched bundle installed
  python scripts/native_bundle.py stale [PART]   remove the CMake build folders configured in the other
                                                 ladder (gpu/cpu), before building from source
PART is prereq or pyslam; without it, both (in that order; `remove`: pyslam first).

environment:
  PYSLAM_NATIVE_URL   where bundles are downloaded from (a URL or a local folder)
  PYSLAM_NATIVE_DIR   where `pack` writes the bundle (default: the checkout)
"""
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_URL = "https://github.com/sjulier/pyslam/releases/download/native-bundles"

# The CPU baseline of the bundles, and the CPU flags (/proc/cpuinfo) a machine needs for it
BUNDLE_MARCH = "x86-64-v3"
X86_64_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}

# The two parts. For each, relative to the checkout:
#   globs     the files of its bundle
#   sources   what it is built from: a change to any of these changes its key
#   required  the python extension modules that every build produces (the libraries differ per platform)
#   folders   folders of the bundle that python or a build adds files to: removed as a whole
PARTS = {
    "prereq": {
        "globs": [
            # GTSAM, all of its install: libraries, python package, and the headers and CMake files that
            # the pyslam part is built against
            "thirdparty/gtsam_local/install/**/*",
            "thirdparty/gtsam_factors/lib/*.so",
            "thirdparty/gtsam_factors/include/**/*",  # headers copied there by its build
            "thirdparty/g2opy/lib/*.so*",
            "thirdparty/g2opy/lib/*.dylib",
            "thirdparty/g2opy/lib/*.a",  # the C++ core links g2o statically
            "thirdparty/g2opy/build/g2o/config.h",
            "thirdparty/pangolin/pypangolin*.so",
            "thirdparty/pangolin/build/src/libpangolin.so*",
            "thirdparty/pangolin/build/src/libpangolin*.dylib",
            "thirdparty/pydbow2/lib/*.so",
            "thirdparty/pydbow2/modules/dbow2/lib/*.so*",
            "thirdparty/pydbow2/modules/dbow2/lib/*.dylib",
            "thirdparty/pydbow3/lib/*.so",
            "thirdparty/pyibow/lib/*.so",
            # Windows: python modules are .pyd; g2o's static libraries are .lib (GTSAM's DLLs, import
            # libraries and CMake files are under thirdparty/gtsam_local/install, above)
            "thirdparty/gtsam_factors/lib/*.pyd",
            "thirdparty/g2opy/lib/*.pyd",
            "thirdparty/g2opy/lib/g2o_*.lib",
            "thirdparty/pangolin/pypangolin*.pyd",
            "thirdparty/pydbow2/lib/*.pyd",
            "thirdparty/pydbow3/lib/*.pyd",
            "thirdparty/pyibow/lib/*.pyd",
        ],
        "sources": [
            "pixi.lock",
            "scripts/install_gtsam.sh",
            "cmake",
            "thirdparty/gtsam_factors",
            "thirdparty/g2opy",
            "thirdparty/pangolin",
            "thirdparty/pangolin.patch",
            "thirdparty/g2opy.patch",
            "thirdparty/pybind11",
            "thirdparty/pydbow2",
            "thirdparty/pydbow3",
            "thirdparty/pyibow",
        ],
        "required": [
            "thirdparty/gtsam_local/install/python/gtsam/gtsam*.so",
            "thirdparty/gtsam_factors/lib/*.so",
            "thirdparty/g2opy/lib/g2o*.so",
            "thirdparty/pangolin/pypangolin*.so",
            "thirdparty/pydbow2/lib/*.so",
            "thirdparty/pydbow3/lib/*.so",
            "thirdparty/pyibow/lib/*.so",
        ],
        "folders": ["thirdparty/gtsam_local/install"],
        "build_roots": ["thirdparty/gtsam_local", "thirdparty/gtsam_factors", "thirdparty/g2opy",
                        "thirdparty/pangolin", "thirdparty/pydbow2", "thirdparty/pydbow3", "thirdparty/pyibow"],
    },
    "pyslam": {
        "globs": [
            "thirdparty/orbslam2_features/lib/*.so",
            "cpp/lib/*.so",
            "pyslam/slam/cpp/lib/*.so",
            "thirdparty/orbslam2_features/lib/*.pyd",
            "cpp/lib/*.pyd",
            "pyslam/slam/cpp/lib/*.pyd",
        ],
        "sources": [
            "build_cpp_core.sh",
            "cpp",
            "pyslam/slam/cpp",
            "thirdparty/orbslam2_features",
        ],
        "required": [
            "thirdparty/orbslam2_features/lib/*.so",
            "cpp/lib/*.so",
            "pyslam/slam/cpp/lib/*.so",
        ],
        "folders": [],
        "build_roots": ["thirdparty/orbslam2_features", "cpp", "pyslam/slam/cpp"],
    },
}
PART_NAMES = ["prereq", "pyslam"]

# Windows: the modules are built by scripts/build_windows.py (part of the key there) with /arch:AVX2
WINDOWS_BUILD_SCRIPT = "scripts/build_windows.py"
WINDOWS_MARCH = "avx2"


def required_globs(part):
    """The python extension modules of a part, with this platform's suffix (.pyd on Windows)."""
    globs = PARTS[part]["required"]
    if platform_name() == "win-64":
        globs = [g[:-len(".so")] + ".pyd" for g in globs]
    return globs


def bundle_march():
    return {"linux-64": BUNDLE_MARCH, "win-64": WINDOWS_MARCH}.get(platform_name(), "default")


def manifest_path(part):
    return os.path.join(ROOT, f"thirdparty/.native_bundle_{part}.json")


# The manifest of the one-part bundles (format 1): removed when a two-part bundle is installed
OLD_MANIFEST = "thirdparty/.native_bundle.json"

# Absolute paths of the build machine in the text files of a bundle (CMake files, config.h) are packed
# as this placeholder and replaced by the checkout's path when the bundle is unpacked
ROOT_PLACEHOLDER = "@PYSLAM_NATIVE_BUNDLE_ROOT@"

# The -march value of the native modules in the checkout: the build scripts use it when PYSLAM_MARCH
# is not set, so that a module rebuilt after a bundle was installed matches the bundle's
MARCH_FILE = "thirdparty/.pyslam_march"

# Part of the key: increase it when the layout of the bundles changes
BUNDLE_FORMAT = 2


def log(msg):
    print(f"[native bundle] {msg}", flush=True)


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)


def environment_name():
    return os.environ.get("PIXI_ENVIRONMENT_NAME", "default")


def ladder():
    """The modules are built once per ladder of levels: 'gpu' (default, depth, ...) or 'cpu' (*-cpu)."""
    return "cpu" if environment_name().endswith("-cpu") else "gpu"


def build_environment():
    """The environment whose libraries the modules are linked to (the first level of the ladder)."""
    return "default-cpu" if ladder() == "cpu" else "default"


def platform_name():
    system, machine = platform.system(), platform.machine()
    if system == "Linux" and machine == "x86_64":
        return "linux-64"
    if system == "Darwin" and machine == "arm64":
        return "osx-arm64"
    if system == "Windows" and machine in ("AMD64", "x86_64"):
        return "win-64"
    return None


def source_key(part):
    """A key for what a part's native modules are built from, or None (and why) if it cannot be computed."""
    if git("rev-parse", "--git-dir").returncode != 0:
        return None, "not a git checkout"
    # local changes to the sources mean that the modules of a bundle would not match them
    sources = PARTS[part]["sources"] + ([WINDOWS_BUILD_SCRIPT] if platform_name() == "win-64" else [])
    status = git("status", "--porcelain", "--untracked-files=no", "--", *sources)
    if status.returncode != 0:
        return None, "git status failed"
    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    # submodules that are checked out at another commit, or with local changes, show up as modified too
    if changed:
        return None, "local changes in " + ", ".join(sorted(changed)[:4])
    h = hashlib.sha256()
    if part == "pyslam":
        # built against the prereq part: a new prereq part means a new pyslam part
        prereq, why = source_key("prereq")
        if prereq is None:
            return None, why
        h.update(f"prereq={prereq}\n".encode())
    for path in PARTS[part]["sources"]:
        r = git("rev-parse", f"HEAD:{path}")
        if r.returncode != 0:
            continue  # a path that does not exist in this version
        h.update(f"{path}={r.stdout.strip()}\n".encode())
    if platform_name() == "win-64":
        r = git("rev-parse", f"HEAD:{WINDOWS_BUILD_SCRIPT}")
        h.update(f"{WINDOWS_BUILD_SCRIPT}={r.stdout.strip()}\n".encode())
    h.update(f"march={bundle_march()}\nformat={BUNDLE_FORMAT}\n".encode())
    return h.hexdigest()[:16], None


def bundle_name(part, key):
    prefix = "pyslam-cpp" if part == "pyslam" else "pyslam-prereq"
    return f"{prefix}-{platform_name()}-{ladder()}-{key}.tar.xz"


def bundle_files(part):
    files = []
    for pattern in PARTS[part]["globs"]:
        for path in glob.glob(os.path.join(ROOT, pattern), recursive=True):
            if (os.path.isfile(path) or os.path.islink(path)) and "__pycache__" not in path:
                files.append(os.path.relpath(path, ROOT))
    return sorted(set(files))


def built_modules(part):
    """The python extension modules of a part that are in the checkout."""
    return [g for g in required_globs(part) if glob.glob(os.path.join(ROOT, g), recursive=True)]


def read_manifest(part):
    try:
        with open(manifest_path(part)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_elf(path):
    if os.path.islink(path):
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def is_macho(path):
    if os.path.islink(path):
        return False
    with open(path, "rb") as f:
        return f.read(4) in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------
def find_patchelf():
    exe = shutil.which("patchelf")
    if exe:
        return [exe]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "exec", "--spec", "patchelf", "patchelf"]
    sys.exit("patchelf not found (install it, or pixi, to pack a bundle)")


def relative_rpath(path, rpath):
    """Replace the entries of an RPATH that point into the checkout by $ORIGIN-relative ones."""
    origin = os.path.dirname(os.path.join(ROOT, path))
    entries = []
    for entry in rpath.split(":"):
        if not entry:
            continue
        if entry.startswith("$ORIGIN"):
            new = entry
        else:
            real = os.path.normpath(entry)
            if not (real == ROOT or real.startswith(ROOT + os.sep)):
                continue  # a path of the build machine outside the checkout
            if not os.path.isdir(real):
                continue  # e.g. a build folder without libraries, or a file given as a folder
            rel = os.path.relpath(real, origin)
            new = "$ORIGIN" if rel == "." else "$ORIGIN/" + rel
        if new not in entries:
            entries.append(new)
    return ":".join(entries)


def run_tool(*args):
    r = subprocess.run(list(args), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pack: {' '.join(args[:3])} ... failed: {r.stderr.strip()}")
    return r.stdout


def in_checkout(path):
    """The path relative to the checkout if it is inside it (also through a link such as /tmp), else None."""
    real, root = os.path.realpath(path), os.path.realpath(ROOT)
    if real == root or real.startswith(root + os.sep):
        return os.path.relpath(real, root)
    return None


def relocate_macho(path, staged):
    """
    Make a Mach-O file of the bundle independent of the build machine's paths (macOS):
    absolute search paths (LC_RPATH) and absolute references to libraries inside the checkout become
    relative to the file (@loader_path), then the file is signed again (ad hoc): on Apple silicon a
    modified binary without a valid signature is killed when it is loaded.
    """
    origin = os.path.dirname(os.path.realpath(os.path.join(ROOT, path)))

    def loader_relative(target_rel):
        rel = os.path.relpath(os.path.join(os.path.realpath(ROOT), target_rel), origin)
        return "@loader_path" if rel == "." else "@loader_path/" + rel

    # 1. search paths
    rpaths, lines = [], run_tool("otool", "-l", staged).splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for following in lines[i + 1:i + 4]:
                if following.strip().startswith("path "):
                    rpaths.append(following.strip()[5:].rsplit(" (offset", 1)[0])
    kept = set(r for r in rpaths if r.startswith("@"))
    for rpath in rpaths:
        if rpath.startswith("@"):
            continue
        rel = in_checkout(rpath)
        new = loader_relative(rel) if rel is not None and os.path.isdir(rpath) else None
        if new is None or new in kept:
            run_tool("install_name_tool", "-delete_rpath", rpath, staged)
        else:
            run_tool("install_name_tool", "-rpath", rpath, new, staged)
            kept.add(new)
    # 2. the file's own name, and its references to libraries inside the checkout
    own = run_tool("otool", "-D", staged).splitlines()[1:]
    if own and not own[0].startswith("@") and in_checkout(own[0]) is not None:
        run_tool("install_name_tool", "-id", "@rpath/" + os.path.basename(own[0]), staged)
    for line in run_tool("otool", "-L", staged).splitlines()[1:]:
        dep = line.strip().rsplit(" (compatibility", 1)[0]
        if dep.startswith("@") or (own and dep == own[0]):
            continue
        rel = in_checkout(dep)
        if rel is not None:
            run_tool("install_name_tool", "-change", dep, loader_relative(rel), staged)
    # 3. sign again
    run_tool("codesign", "-f", "-s", "-", staged)


def roots():
    """The checkout's path as the build tools may have written it (also through a link)."""
    candidates = {ROOT, os.path.realpath(ROOT)}
    # the shell's spelling of the current folder, when the checkout is reached through a link
    pwd = os.environ.get("PWD", "")
    if pwd and os.path.realpath(pwd) == os.path.realpath(ROOT):
        candidates.add(pwd)
    # macOS: /tmp, /var and /etc are links to /private/...; CMake may have recorded either spelling
    for root in list(candidates):
        if root.startswith("/private/"):
            candidates.add(root[len("/private"):])
        elif root.split("/")[1:2] in (["tmp"], ["var"], ["etc"]):
            candidates.add("/private" + root)
    if os.sep == "\\":
        candidates |= {root.replace("\\", "/") for root in candidates}
    return sorted(candidates, key=len, reverse=True)


def placehold_root(staged):
    """Replace the checkout's path in a text file by ROOT_PLACEHOLDER. True if there was one."""
    with open(staged, "rb") as f:
        data = f.read()
    if b"\0" in data[:8192]:
        return False  # binary
    new = data
    for root in roots():
        new = new.replace(root.encode(), ROOT_PLACEHOLDER.encode())
    if new == data:
        return False
    with open(staged, "wb") as f:
        f.write(new)
    return True


def pack(part):
    if platform_name() is None:
        sys.exit("pack: only linux-64, osx-arm64 and win-64 are supported")
    # on macOS (Apple silicon) the modules are not built with -march: every machine has the same baseline
    if platform_name() == "linux-64" and os.environ.get("PYSLAM_MARCH") != BUNDLE_MARCH:
        sys.exit(f"pack: build the modules with PYSLAM_MARCH={BUNDLE_MARCH} and set it for this command too")
    key, why = source_key(part)
    if key is None:
        sys.exit(f"pack {part}: no key for this checkout ({why}): commit the changes first")
    files = bundle_files(part)
    missing = [g for g in required_globs(part) if not glob.glob(os.path.join(ROOT, g), recursive=True)]
    if missing:
        sys.exit(f"pack {part}: nothing built for " + ", ".join(missing))

    patchelf = find_patchelf() if platform_name() == "linux-64" else None
    stage = os.path.join(ROOT, ".native_bundle_stage")
    shutil.rmtree(stage, ignore_errors=True)
    rooted = []
    for path in files:
        src, dst = os.path.join(ROOT, path), os.path.join(stage, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
            continue
        shutil.copy2(src, dst)
        if is_macho(src):
            relocate_macho(path, dst)
        elif is_elf(src):
            r = subprocess.run(patchelf + ["--print-rpath", dst], capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"pack: patchelf failed on {path}: {r.stderr.strip()}")
            new = relative_rpath(path, r.stdout.strip())
            # --force-rpath keeps DT_RPATH (as built), which also applies to the libraries' dependencies
            r = subprocess.run(patchelf + ["--force-rpath", "--set-rpath", new, dst], capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"pack: patchelf failed on {path}: {r.stderr.strip()}")
        elif placehold_root(dst):
            rooted.append(path)
    manifest = {"part": part, "key": key, "march": bundle_march(),
                "platform": platform_name(), "ladder": ladder(), "build_environment": build_environment(),
                "files": files, "rooted": rooted}
    rel_manifest = os.path.relpath(manifest_path(part), ROOT)
    os.makedirs(os.path.join(stage, os.path.dirname(rel_manifest)), exist_ok=True)
    with open(os.path.join(stage, rel_manifest), "w") as f:
        json.dump(manifest, f, indent=1)

    out_dir = os.environ.get("PYSLAM_NATIVE_DIR", ROOT)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, bundle_name(part, key))
    with tarfile.open(out, "w:xz") as tar:
        for path in files + [rel_manifest]:
            tar.add(os.path.join(stage, path), arcname=path)
    shutil.rmtree(stage)
    with open(out + ".sha256", "w") as f:
        f.write(f"{sha256_of(out)}  {os.path.basename(out)}\n")
    log(f"packed {len(files)} files into {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------
def cpu_supports_baseline():
    if platform_name() == "win-64":
        import ctypes

        PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
        return [] if ctypes.windll.kernel32.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) else ["avx2"]
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return sorted(X86_64_V3_FLAGS - flags)
    except OSError:
        pass
    return ["(cannot read /proc/cpuinfo)"]


def download(url, dest, attempts=5):
    """Download url to dest (a local folder is accepted in place of a URL)."""
    if "://" not in url:
        if not os.path.exists(url):
            return False
        shutil.copyfile(url, dest)
        return True
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pyslam-native-bundle"})
            with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as f:
                shutil.copyfileobj(response, f, 1 << 20)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            log(f"download failed ({e}): attempt {attempt}/{attempts}")
        except Exception as e:  # noqa: BLE001  (timeouts, connection errors)
            log(f"download failed ({e}): attempt {attempt}/{attempts}")
        time.sleep(3)
    return False


def remove_files(paths, folders=()):
    for path in paths:
        full = os.path.join(ROOT, path)
        if os.path.lexists(full):
            os.remove(full)
    for folder in folders:
        shutil.rmtree(os.path.join(ROOT, folder), ignore_errors=True)


def remove_old_bundle():
    """Remove a one-part bundle (format 1), if one is installed."""
    path = os.path.join(ROOT, OLD_MANIFEST)
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            files = json.load(f).get("files", [])
    except (OSError, ValueError):
        files = []
    remove_files(files, ["thirdparty/gtsam_local/install/python"])
    os.remove(path)
    log("removed the files of an older (one-part) bundle")


def remove(part, quiet=False):
    manifest = read_manifest(part)
    if manifest is None:
        if not quiet:
            log(f"no fetched {part} bundle to remove")
        return
    remove_files(manifest.get("files", []), PARTS[part]["folders"])
    os.remove(manifest_path(part))
    if part == "prereq" and os.path.exists(os.path.join(ROOT, MARCH_FILE)):
        os.remove(os.path.join(ROOT, MARCH_FILE))
    log(f"removed the {len(manifest.get('files', []))} files of {part} bundle {manifest.get('key')}")


def no_bundle(part, reason):
    log(f"no prebuilt {part} modules for this checkout: {reason}")
    log(f"the {part} modules will be built from source")
    return 2


def fetch(part):
    if platform_name() is None:
        return no_bundle(part, f"no bundles for {platform.system()} {platform.machine()}")
    lacking = cpu_supports_baseline() if platform_name() in ("linux-64", "win-64") else []
    if lacking:
        return no_bundle(part, f"this CPU lacks {', '.join(lacking)} (the bundles are built for {bundle_march()})")
    remove_old_bundle()
    key, why = source_key(part)
    installed = read_manifest(part)
    if key is None:
        if installed is not None:
            remove(part, quiet=True)  # its modules do not match the changed sources: rebuild them
        return no_bundle(part, why)

    if installed is not None:
        if installed.get("key") == key and installed.get("ladder") == ladder() and all(
            os.path.lexists(os.path.join(ROOT, p)) for p in installed.get("files", [])
        ):
            log(f"{part} bundle {key} is already installed")
            return 0
        remove(part, quiet=True)  # another version: its files do not match this checkout

    if part == "pyslam":
        # the pyslam part's modules are linked to the prereq bundle's libraries, built for the bundles' CPU
        # baseline: on top of prereq modules built from source (maybe for another one) build them too
        prereq = read_manifest("prereq")
        if prereq is None or prereq.get("key") != source_key("prereq")[0]:
            return no_bundle(part, "the prereq modules were built from source")
    # modules built from source: the prereq ones are kept (rebuilding them takes up to an hour); the
    # pyslam ones (e.g. left from a change that was then undone) are replaced by the bundle, if there is one
    replace = False
    if built_modules(part):
        if part == "prereq":
            return no_bundle(part, "modules built from source are already there (run ./clean.sh to replace them)")
        replace = True

    name = bundle_name(part, key)
    base = os.environ.get("PYSLAM_NATIVE_URL", DEFAULT_URL).rstrip("/")
    archive = os.path.join(ROOT, name + ".download")
    log(f"looking for {name}")
    try:
        if not download(f"{base}/{name}.sha256", archive + ".sha256", attempts=2):
            return no_bundle(part, "there is none for this version of pySLAM on this platform")
        with open(archive + ".sha256") as f:
            expected = f.read().split()[0]
        if not download(f"{base}/{name}", archive):
            return no_bundle(part, "the download failed")
        if sha256_of(archive) != expected:
            return no_bundle(part, "the downloaded file is corrupt (checksum mismatch)")
        if replace:
            log("replacing the pyslam modules built from source")
            remove_files(bundle_files(part))
        log(f"unpacking {os.path.getsize(archive) / 1e6:.1f} MB")
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                target = os.path.normpath(os.path.join(ROOT, member.name))
                if not target.startswith(ROOT + os.sep) or member.isdev():
                    return no_bundle(part, f"unexpected path in the bundle: {member.name}")
            tar.extractall(ROOT, members=members)
    finally:
        for path in (archive, archive + ".sha256"):
            if os.path.exists(path):
                os.remove(path)

    # the build machine's path in text files (CMake files, config.h) becomes this checkout's
    manifest = read_manifest(part) or {}
    for path in manifest.get("rooted", []):
        full = os.path.join(ROOT, path)
        with open(full, "rb") as f:
            data = f.read()
        with open(full, "wb") as f:
            f.write(data.replace(ROOT_PLACEHOLDER.encode(), ROOT.replace("\\", "/").encode()))

    # the bundle's modules must load in this environment: check them before relying on them
    check = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_native_modules.py"), "--present"])
    if check.returncode != 0:
        remove(part, quiet=True)
        return no_bundle(part, "its modules do not load on this machine")
    if part == "prereq" and platform_name() == "linux-64":
        with open(os.path.join(ROOT, MARCH_FILE), "w") as f:
            f.write(BUNDLE_MARCH + "\n")
    log(f"installed the prebuilt {part} modules (bundle {key})")
    return 0


def stale(part):
    """
    Remove the CMake build folders of a part that were configured in the other ladder: their caches
    name the other environment's compiler and libraries, and a build there would mix the two.
    """
    current = ladder()
    for top in PARTS[part]["build_roots"]:
        for folder, dirs, files in os.walk(os.path.join(ROOT, top)):
            depth = os.path.relpath(folder, os.path.join(ROOT, top)).count(os.sep)
            if depth >= 3 or os.path.basename(folder) == "_deps":
                dirs[:] = []
            if "CMakeCache.txt" not in files:
                continue
            dirs[:] = []
            try:
                with open(os.path.join(folder, "CMakeCache.txt"), errors="replace") as f:
                    envs = set(re.findall(r"/\.pixi/envs/([^/\s]+)/", f.read()))
            except OSError:
                continue
            others = sorted(e for e in envs if ("cpu" if e.endswith("-cpu") else "gpu") != current)
            if others:
                log(f"removing {os.path.relpath(folder, ROOT)}: configured in the {others[0]} environment")
                shutil.rmtree(folder, ignore_errors=True)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    parts = sys.argv[2:] or PART_NAMES
    unknown = [p for p in parts if p not in PARTS]
    if unknown or command not in ("key", "pack", "fetch", "remove", "stale"):
        print(__doc__)
        sys.exit(1)
    if command == "key":
        for part in parts:
            key, why = source_key(part)
            print(f"{part}: key {key or 'none (' + why + ')'}" + (f", file {bundle_name(part, key)}" if key else ""))
    elif command == "pack":
        for part in parts:
            pack(part)
    elif command == "fetch":
        sys.exit(max(fetch(part) for part in parts))
    elif command == "remove":
        for part in reversed(parts):
            remove(part)
    elif command == "stale":
        for part in parts:
            stale(part)


if __name__ == "__main__":
    main()
