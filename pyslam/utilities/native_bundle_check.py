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


# A warning when pySLAM's prebuilt native modules are older than the checkout: after `git pull` (or a new
# course version) `pixi run build` installs the matching ones, and without it the old modules keep running
# (on macOS an old vocabulary loader took 7 minutes per run, which looked like a hang). The check is
# scripts/native_bundle.py's: it needs no native module, takes about 0.1 s, and says nothing about modules
# built from source.

import importlib.util
import os
import sys

kRootFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def outdated_native_parts():
    """[(part, installed key, key)] for the installed bundles that do not match this checkout; [] if they
    match, if the modules were built from source, or if the check cannot be made (no git, ...)."""
    path = os.path.join(kRootFolder, "scripts", "native_bundle.py")
    try:
        spec = importlib.util.spec_from_file_location("pyslam_native_bundle", path)
        native_bundle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_bundle)
        return native_bundle.outdated_parts()
    except Exception:
        return []


def warn_if_native_modules_outdated():
    """Print one line if the installed prebuilt native modules are not the ones for this checkout."""
    outdated = outdated_native_parts()
    if outdated:
        parts = " and ".join(part for part, _, _ in outdated)
        print(
            f"\033[93mWARNING: pySLAM's prebuilt C++ modules ({parts}) are from an older version than this "
            f"checkout: run `pixi run build` to install the matching ones.\033[0m",
            file=sys.stderr,
            flush=True,
        )
