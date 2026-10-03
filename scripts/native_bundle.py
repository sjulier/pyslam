#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Prebuilt native modules ("native bundle"): pack them on a build machine, fetch them on another.

pySLAM's native modules (GTSAM, g2o, Pangolin, DBoW2/3, iBoW, the ORB-SLAM2 features, the C++ utilities
and the C++ core) take from 15 minutes to over an hour to build. The pixi lock file gives every machine
the same compiler and libraries, so the modules built on one machine run on another one of the same
platform, provided that
  - they were built for a common CPU baseline (PYSLAM_MARCH=x86-64-v3, i.e. AVX2), not -march=native;
  - their library search paths are relative to the checkout (done by `pack`).

usage:
  python scripts/native_bundle.py key      print the key and the file name of the bundle for this checkout
  python scripts/native_bundle.py pack     pack the built modules into <file name> (build machine; needs patchelf)
  python scripts/native_bundle.py fetch    download and unpack the bundle for this checkout, then check it
                                           (exit code 0: installed; 2: no bundle for this checkout or machine)
  python scripts/native_bundle.py remove   remove the files a fetched bundle installed

The bundle's name contains a key computed from pixi.lock and from the sources of the native modules:
after any change to them the key changes, no bundle is found and the modules are built from source.

environment:
  PYSLAM_NATIVE_URL   where bundles are downloaded from (a URL or a local folder)
  PYSLAM_NATIVE_DIR   where `pack` writes the bundle (default: the checkout)
"""
import glob
import hashlib
import json
import os
import platform
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

# The files of the native modules, relative to the checkout
BUNDLE_GLOBS = [
    "thirdparty/gtsam_local/install/lib/*.so*",
    "thirdparty/gtsam_local/install/lib/*.dylib",
    "thirdparty/gtsam_local/install/python/**/*",
    "thirdparty/gtsam_factors/lib/*.so",
    "thirdparty/g2opy/lib/*.so*",
    "thirdparty/g2opy/lib/*.dylib",
    "thirdparty/pangolin/pypangolin*.so",
    "thirdparty/pangolin/build/src/libpangolin.so*",
    "thirdparty/pangolin/build/src/libpangolin*.dylib",
    "thirdparty/pydbow2/lib/*.so",
    "thirdparty/pydbow2/modules/dbow2/lib/*.so*",
    "thirdparty/pydbow2/modules/dbow2/lib/*.dylib",
    "thirdparty/pydbow3/lib/*.so",
    "thirdparty/pyibow/lib/*.so",
    "thirdparty/orbslam2_features/lib/*.so",
    "cpp/lib/*.so",
    "pyslam/slam/cpp/lib/*.so",
]

# What the modules are built from: a change to any of these changes the key
SOURCE_PATHS = [
    "pixi.lock",
    "scripts/install_gtsam.sh",
    "cpp",
    "pyslam/slam/cpp",
    "thirdparty/gtsam_factors",
    "thirdparty/g2opy",
    "thirdparty/pangolin",
    "thirdparty/pangolin.patch",
    "thirdparty/g2opy.patch",
    "thirdparty/pybind11",
    "thirdparty/pydbow2",
    "thirdparty/pydbow3",
    "thirdparty/pyibow",
    "thirdparty/orbslam2_features",
]

# The python extension modules that every build produces (the libraries next to them differ per platform)
REQUIRED_GLOBS = [
    "thirdparty/gtsam_local/install/python/gtsam/gtsam*.so",
    "thirdparty/gtsam_factors/lib/*.so",
    "thirdparty/g2opy/lib/g2o*.so",
    "thirdparty/pangolin/pypangolin*.so",
    "thirdparty/pydbow2/lib/*.so",
    "thirdparty/pydbow3/lib/*.so",
    "thirdparty/pyibow/lib/*.so",
    "thirdparty/orbslam2_features/lib/*.so",
    "cpp/lib/*.so",
    "pyslam/slam/cpp/lib/*.so",
]

MANIFEST = "thirdparty/.native_bundle.json"

# The -march value of the native modules in the checkout: the build scripts use it when PYSLAM_MARCH
# is not set, so that a module rebuilt after a bundle was installed matches the bundle's
MARCH_FILE = "thirdparty/.pyslam_march"

# Part of the key: increase it when the layout of the bundles changes
BUNDLE_FORMAT = 1

# A folder of the bundle that python adds files to (__pycache__): removed as a whole
BUNDLE_FOLDERS = ["thirdparty/gtsam_local/install/python"]


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
    return None


def source_key():
    """A key for what the native modules are built from, or None if it cannot be computed."""
    if git("rev-parse", "--git-dir").returncode != 0:
        return None, "not a git checkout"
    # local changes to the sources mean that the modules of a bundle would not match them
    status = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCE_PATHS)
    if status.returncode != 0:
        return None, "git status failed"
    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    # submodules that are checked out at another commit, or with local changes, show up as modified too
    if changed:
        return None, "local changes in " + ", ".join(sorted(changed)[:4])
    h = hashlib.sha256()
    for path in SOURCE_PATHS:
        r = git("rev-parse", f"HEAD:{path}")
        if r.returncode != 0:
            continue  # a path that does not exist in this version
        h.update(f"{path}={r.stdout.strip()}\n".encode())
    march = BUNDLE_MARCH if platform_name() == "linux-64" else "default"
    h.update(f"march={march}\nformat={BUNDLE_FORMAT}\n".encode())
    return h.hexdigest()[:16], None


def bundle_name(key):
    return f"pyslam-native-{platform_name()}-{ladder()}-{key}.tar.xz"


def bundle_files():
    files = []
    for pattern in BUNDLE_GLOBS:
        for path in glob.glob(os.path.join(ROOT, pattern), recursive=True):
            if (os.path.isfile(path) or os.path.islink(path)) and "__pycache__" not in path:
                files.append(os.path.relpath(path, ROOT))
    return sorted(set(files))


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


def pack():
    if platform_name() not in ("linux-64", "osx-arm64"):
        sys.exit("pack: only linux-64 and osx-arm64 are supported")
    # on macOS (Apple silicon) the modules are not built with -march: every machine has the same baseline
    if platform_name() == "linux-64" and os.environ.get("PYSLAM_MARCH") != BUNDLE_MARCH:
        sys.exit(f"pack: build the modules with PYSLAM_MARCH={BUNDLE_MARCH} and set it for this command too")
    key, why = source_key()
    if key is None:
        sys.exit(f"pack: no key for this checkout ({why}): commit the changes first")
    files = bundle_files()
    missing = [g for g in REQUIRED_GLOBS if not glob.glob(os.path.join(ROOT, g), recursive=True)]
    if missing:
        sys.exit("pack: nothing built for " + ", ".join(missing))

    patchelf = find_patchelf() if platform_name() == "linux-64" else None
    stage = os.path.join(ROOT, ".native_bundle_stage")
    shutil.rmtree(stage, ignore_errors=True)
    manifest = {"key": key, "march": BUNDLE_MARCH if platform_name() == "linux-64" else "default", "platform": platform_name(), "ladder": ladder(),
                "build_environment": build_environment(), "files": files}
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
    os.makedirs(os.path.join(stage, os.path.dirname(MANIFEST)), exist_ok=True)
    with open(os.path.join(stage, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)

    out_dir = os.environ.get("PYSLAM_NATIVE_DIR", ROOT)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, bundle_name(key))
    with tarfile.open(out, "w:xz") as tar:
        for path in files + [MANIFEST]:
            tar.add(os.path.join(stage, path), arcname=path)
    shutil.rmtree(stage)
    with open(out + ".sha256", "w") as f:
        f.write(f"{sha256_of(out)}  {os.path.basename(out)}\n")
    log(f"packed {len(files)} files into {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------
def cpu_supports_baseline():
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


def remove(quiet=False):
    manifest_path = os.path.join(ROOT, MANIFEST)
    if not os.path.exists(manifest_path):
        if not quiet:
            log("no fetched bundle to remove")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    for path in manifest.get("files", []):
        full = os.path.join(ROOT, path)
        if os.path.lexists(full):
            os.remove(full)
    for folder in BUNDLE_FOLDERS:
        shutil.rmtree(os.path.join(ROOT, folder), ignore_errors=True)
    for path in (manifest_path, os.path.join(ROOT, MARCH_FILE)):
        if os.path.exists(path):
            os.remove(path)
    log(f"removed the {len(manifest.get('files', []))} files of bundle {manifest.get('key')}")


def no_bundle(reason):
    log(f"no prebuilt modules for this checkout: {reason}")
    log("the modules will be built from source")
    return 2


def fetch():
    if platform_name() is None:
        return no_bundle(f"no bundles for {platform.system()} {platform.machine()}")
    lacking = cpu_supports_baseline() if platform_name() == "linux-64" else []
    if lacking:
        return no_bundle(f"this CPU lacks {', '.join(lacking)} (the bundles are built for {BUNDLE_MARCH})")
    key, why = source_key()
    if key is None:
        return no_bundle(why)

    manifest_path = os.path.join(ROOT, MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            installed = json.load(f)
        if installed.get("key") == key and installed.get("ladder") == ladder() and all(
            os.path.lexists(os.path.join(ROOT, p)) for p in installed.get("files", [])
        ):
            if platform_name() == "linux-64" and not os.path.exists(os.path.join(ROOT, MARCH_FILE)):
                with open(os.path.join(ROOT, MARCH_FILE), "w") as f:  # installed by an older version
                    f.write(BUNDLE_MARCH + "\n")
            log(f"bundle {key} is already installed")
            return 0
        remove(quiet=True)  # another version: its files do not match this checkout

    built = [g for g in REQUIRED_GLOBS if glob.glob(os.path.join(ROOT, g), recursive=True)]
    if built:
        return no_bundle("native modules built from source are already there (run ./clean.sh to replace them)")

    name = bundle_name(key)
    base = os.environ.get("PYSLAM_NATIVE_URL", DEFAULT_URL).rstrip("/")
    archive = os.path.join(ROOT, name + ".download")
    log(f"looking for {name}")
    try:
        if not download(f"{base}/{name}.sha256", archive + ".sha256", attempts=2):
            return no_bundle("there is none for this version of pySLAM on this platform")
        with open(archive + ".sha256") as f:
            expected = f.read().split()[0]
        if not download(f"{base}/{name}", archive):
            return no_bundle("the download failed")
        if sha256_of(archive) != expected:
            return no_bundle("the downloaded file is corrupt (checksum mismatch)")
        log(f"unpacking {os.path.getsize(archive) / 1e6:.1f} MB")
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                target = os.path.normpath(os.path.join(ROOT, member.name))
                if not target.startswith(ROOT + os.sep) or member.isdev():
                    return no_bundle(f"unexpected path in the bundle: {member.name}")
            tar.extractall(ROOT, members=members)
    finally:
        for path in (archive, archive + ".sha256"):
            if os.path.exists(path):
                os.remove(path)

    # the bundle's modules must load in this environment: check them before relying on them
    check = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_native_modules.py")])
    if check.returncode != 0:
        remove(quiet=True)
        return no_bundle("its modules do not load on this machine")
    if platform_name() == "linux-64":
        with open(os.path.join(ROOT, MARCH_FILE), "w") as f:
            f.write(BUNDLE_MARCH + "\n")
    log(f"installed the prebuilt native modules (bundle {key})")
    return 0


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "key":
        key, why = source_key()
        print(f"key: {key or 'none (' + why + ')'}")
        if key:
            print(f"file: {bundle_name(key)}")
    elif command == "pack":
        pack()
    elif command == "fetch":
        sys.exit(fetch())
    elif command == "remove":
        remove()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
