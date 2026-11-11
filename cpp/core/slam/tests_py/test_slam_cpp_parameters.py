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
*
* cpp_core's Parameters follow the Python ones (pyslam/config_parameters.py):
* - every field is copied from Python at Slam init (sync_runtime_from_python), from a name that
*   exists in Python;
* - the two compile-time constants (kMinDepth, kStereoMatchingShowMatchedPoints) have the Python value.
"""

import math
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ["PYSLAM_USE_CPP"] = "1"

import pyslam.config  # library paths before importing the native core
from pyslam.config_parameters import Parameters

HEADER = ROOT / "cpp/core/slam/config_parameters.h"
SOURCE = ROOT / "cpp/core/slam/config_parameters.cpp"


def _strip_comments(text):
    return re.sub(r"//[^\n]*", "", text)


# the Python modules the C++ Parameters read module-level constants from (sync_runtime_from_python)
_MODULE_VARS = {
    "map": "pyslam.slam.map",
    "tracking_core": "pyslam.slam.tracking_core",
    "local_mapping_core": "pyslam.slam.local_mapping_core",
}
# derived in sync_runtime_from_python, as in optimizer_g2o.py
_DERIVED = {"kThHuberMono", "kThHuberStereo"}


def _reads():
    """(python object name, field) for every read in sync_runtime_from_python."""
    source = SOURCE.read_text()
    body = source[source.index("void Parameters::sync_runtime_from_python()") :]
    reads = re.findall(r'read_(?:attr|number)\((\w+),\s*"(k\w+)"', body)
    reads += [("params", n) for n in re.findall(r'py::hasattr\(params, "(k\w+)"\)', body)]
    return reads


class ParametersFollowPythonTests(unittest.TestCase):
    def test_compile_time_constants(self):
        # kMinDepth: camera_models uses it too; kStereoMatchingShowMatchedPoints: debug windows, which
        # need OpenCV highgui. Every other field is copied from Python at Slam init.
        declared = re.findall(r"static constexpr [\w:]+ (k\w+)\s*=", _strip_comments(HEADER.read_text()))
        self.assertEqual(sorted(declared), ["kMinDepth", "kStereoMatchingShowMatchedPoints"])

    def test_every_field_is_copied_from_python(self):
        header = _strip_comments(HEADER.read_text())
        fields = set(re.findall(r"static (?!constexpr)[\w:<>]+ (k\w+);", header))
        copied = {name for _, name in _reads()} | _DERIVED
        # set by feature_tracker_shared.py through the binding, when the feature tracker is created
        copied.add("kFeatureMatchDefaultRatioTest")
        self.assertEqual(sorted(fields - copied), [])

    def test_every_read_exists_in_python(self):
        # a read of a name that Python does not have keeps the C++ default without a word
        import importlib

        for owner, name in _reads():
            with self.subTest(name=name):
                if owner == "params":
                    self.assertTrue(hasattr(Parameters, name), f"Parameters.{name}")
                else:
                    module = importlib.import_module(_MODULE_VARS[owner])
                    self.assertTrue(hasattr(module, name), f"{module.__name__}.{name}")

    def test_kMinDepth_matches_python(self):
        self.assertEqual(Parameters.kMinDepth, 1e-2)  # the C++ value (config_parameters.h)
        self.assertFalse(Parameters.kStereoMatchingShowMatchedPoints)

    def test_huber_thresholds_match_optimizer_g2o(self):
        self.assertAlmostEqual(math.sqrt(Parameters.kChi2Mono), 2.4476519360399265)
        self.assertAlmostEqual(math.sqrt(Parameters.kChi2Stereo), 2.7955321496988725)

    def test_file_constants_match_python(self):
        # tuning constants that the C++ files define for themselves, as the Python modules do:
        # these do not follow Python at runtime, so their values must agree
        import importlib

        table = {
            "tracking.h": "pyslam.slam.tracking",
            "initializer.h": "pyslam.slam.initializer",
            "local_mapping.cpp": "pyslam.slam.local_mapping",
            "optimizer_gtsam.h": "pyslam.slam.optimizer_gtsam",
        }
        for file_name, module_name in table.items():
            text = _strip_comments((ROOT / "cpp/core/slam" / file_name).read_text())
            module = importlib.import_module(module_name)
            found = re.findall(r"(?:static )?(?:constexpr|const) [\w:]+ (k[A-Z]\w*)\s*=\s*([-+\d.eE]+)f?;", text)
            self.assertTrue(found, file_name)
            for name, value in found:
                with self.subTest(file=file_name, name=name):
                    self.assertTrue(hasattr(module, name), f"{module_name}.{name}")
                    self.assertAlmostEqual(float(value), float(getattr(module, name)))


if __name__ == "__main__":
    unittest.main()
