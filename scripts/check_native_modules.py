#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Check that pySLAM's native modules load in the current environment: every Python extension module of
the native build is imported, each in its own process (so that one crash does not hide the others).
Exit code 0 if they all load.

A module that is missing also counts as a failure, unless --present is given (check only the modules
that are there: used while the prebuilt modules are installed one part at a time).

usage: python scripts/check_native_modules.py [-v] [--present]
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODULE_GLOBS = [
    "thirdparty/gtsam_local/install/python/gtsam/gtsam*.so",
    "thirdparty/gtsam_factors/lib/*.so",
    "thirdparty/g2opy/lib/g2o*.cpython*.so",
    "thirdparty/pangolin/pypangolin*.so",
    "thirdparty/pydbow2/lib/*.so",
    "thirdparty/pydbow3/lib/*.so",
    "thirdparty/pyibow/lib/*.so",
    "thirdparty/orbslam2_features/lib/*.so",
    "cpp/lib/*.so",
    "pyslam/slam/cpp/lib/*.cpython*.so",
]


def main():
    verbose = "-v" in sys.argv
    modules, missing = [], []
    for pattern in MODULE_GLOBS:
        found = sorted(glob.glob(os.path.join(ROOT, pattern)))
        modules += found
        if not found:
            missing.append(pattern)
    if not modules:
        print("check_native_modules: no native modules found (run `pixi run build`)")
        return 1
    failed = []
    if "--present" not in sys.argv:
        for pattern in missing:
            print(f"  {pattern:22s} MISSING (not built)")
            failed.append(pattern)
    for path in modules:
        name = os.path.basename(path).split(".")[0]
        folder = os.path.dirname(path)
        if name == "gtsam":  # a package: import it by its name, from the folder above
            folder = os.path.dirname(folder)
        # gtsam first: gtsam_factors and cpp_core use its types
        code = f"import sys; sys.path.insert(0, {folder!r}); import gtsam; import {name}"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0
        if verbose or not ok:
            last = (r.stderr.strip().splitlines() or [f"exit code {r.returncode}"])[-1]
            print(f"  {name:22s} {'OK' if ok else 'FAIL  ' + last[:150]}")
        if not ok:
            failed.append(name)
    if failed:
        print(f"check_native_modules: {len(failed)} of {len(modules) + len(missing) * ('--present' not in sys.argv)} "
              f"native modules are missing or do not load: {', '.join(failed)} (run `pixi run build`)")
        return 1
    print(f"OK: {len(modules)} native modules load")
    return 0


if __name__ == "__main__":
    sys.exit(main())
