"""Tests of the pure helpers of scripts/doctor.py. Run with: pytest test/scripts/test_doctor.py"""

import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("doctor", os.path.join(ROOT, "scripts", "doctor.py"))
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


def test_import_built_module_rejects_a_source_folder(tmp_path, monkeypatch):
    (tmp_path / "unbuilt_native").mkdir()  # a thirdparty source folder without its .so
    (tmp_path / "built_native.py").write_text("x = 1\n")  # a module with a file, like a built .so
    monkeypatch.syspath_prepend(str(tmp_path))
    assert doctor.import_built_module("built_native") == "built_native.py"
    with pytest.raises(ModuleNotFoundError, match="not built"):
        doctor.import_built_module("unbuilt_native")  # imports as an empty namespace package
    with pytest.raises(ModuleNotFoundError, match="not built"):
        doctor.import_built_module("no_such_module_anywhere")


def test_parse_metrics_info():
    text = "num_total_frames: 1101\nnum_processed_frames: 1101\nnum_lost_frames: 1\npercent_lost: 0.09\n"
    info = doctor.parse_metrics_info(text)
    assert info["num_total_frames"] == "1101" and info["num_lost_frames"] == "1" and info["percent_lost"] == "0.09"
    assert doctor.parse_metrics_info("") == {}


def test_abi_summary():
    status, detail = doctor.abi_summary(0, ["__abi__: 15 module(s)", "OK: 15 module(s) share the pybind11 internals ABI __abi__"])
    assert status == "OK" and detail.startswith("OK: 15")
    status, detail = doctor.abi_summary(0, ["__abi__: 1 module(s)", "OK: 1 module(s) share the pybind11 internals ABI __abi__"])
    assert status == "WARN" and "only 1 module" in detail  # a bare environment: GTSAM's own module and nothing built
    status, detail = doctor.abi_summary(1, ["__a__: 13 module(s)", "    /x/g2o.so", "__b__: 2 module(s)",
                                            "ERROR: pySLAM modules were built with different pybind11 internals ABIs (see above);",
                                            "       they cannot share C++ types."])
    assert status == "FAIL" and "__a__: 13 module(s)" in detail and "__b__: 2 module(s)" in detail
    status, detail = doctor.abi_summary(2, ["check_pybind11_abi: no built pybind11 module found"])
    assert status == "FAIL" and "no built module" in detail
