#!/usr/bin/env python3
"""
* This file is part of PYSLAM
*
* Check that all pySLAM Python modules built with pybind11 share one pybind11 internals ABI.

pybind11 modules can exchange C++ types (e.g. a gtsam_factors factor added to a gtsam graph, a
cpp_core object passed to g2o) only if they were built with the same pybind11 internals ABI. The
ABI is identified by a string compiled into each module, e.g.
    __pybind11_internals_v11_gcc_libstdcpp_cxxabi1018__
(internals version, compiler family, standard library, C++ ABI). This script collects that string
from every built pySLAM module (and GTSAM's python module, whose types gtsam_factors extends) and
fails if more than one is found.

Usage:  python scripts/check_pybind11_abi.py [--verbose]
Exit code: 0 if all modules agree, 1 if they differ, 2 if no module was found.
"""

import argparse
import glob
import importlib.util
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Built pySLAM modules (relative to the repo root). Third-party torch extensions (detectron2, ...)
# are built with torch's own pybind11 and do not share types with these, so they are not listed.
MODULE_GLOBS = [
    "pyslam/slam/cpp/lib/cpp_core*.so",
    "cpp/lib/*.so",
    "thirdparty/g2opy/lib/*.so",
    "thirdparty/orbslam2_features/lib/*.so",
    "thirdparty/pangolin/pypangolin*.so",
    "thirdparty/pydbow2/lib/*.so",
    "thirdparty/pydbow3/lib/*.so",
    "thirdparty/pyibow/lib/*.so",
    "thirdparty/ros2_pybindings/lib/*.so",
    "thirdparty/gtsam_factors/lib/*.so",
]

if os.name == "nt":  # python extension modules are .pyd on Windows, without "cpython" in the name
    MODULE_GLOBS = [g.replace(".cpython*.so", "*.pyd").replace(".so", ".pyd") for g in MODULE_GLOBS]

INTERNALS_RE = re.compile(rb"__pybind11_internals_v\d+[A-Za-z0-9_]*?__")


def find_gtsam_modules():
    spec = importlib.util.find_spec("gtsam")
    if spec is None or not spec.submodule_search_locations:
        return []
    found = []
    for location in spec.submodule_search_locations:
        found += glob.glob(os.path.join(location, "*.pyd" if os.name == "nt" else "*.so"))
    return found


def internals_ids(path):
    with open(path, "rb") as f:
        data = f.read()
    return sorted({m.decode() for m in INTERNALS_RE.findall(data)})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--verbose", action="store_true", help="list every module")
    args = parser.parse_args()

    modules = []
    for pattern in MODULE_GLOBS:
        modules += sorted(glob.glob(os.path.join(ROOT, pattern)))
    modules += find_gtsam_modules()
    if not modules:
        print("check_pybind11_abi: no built pybind11 module found")
        return 2

    by_id = {}
    for path in modules:
        ids = internals_ids(path)
        key = ", ".join(ids) if ids else "(no pybind11 internals string)"
        by_id.setdefault(key, []).append(os.path.relpath(path, ROOT) if path.startswith(ROOT) else path)

    for key, paths in sorted(by_id.items()):
        print(f"{key}: {len(paths)} module(s)")
        if args.verbose or len(by_id) > 1:
            for p in paths:
                print(f"    {p}")

    with_ids = [k for k in by_id if not k.startswith("(")]
    if len(with_ids) > 1:
        print("ERROR: pySLAM modules were built with different pybind11 internals ABIs (see above);")
        print("       they cannot share C++ types. Rebuild them with the bundled thirdparty/pybind11.")
        return 1
    print(f"OK: {len(modules)} module(s) share the pybind11 internals ABI {with_ids[0] if with_ids else '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
