"""Tests of the pure helpers of scripts/native_bundle.py. Run with: pytest tests/scripts/test_native_bundle.py"""

import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("native_bundle", os.path.join(ROOT, "scripts", "native_bundle.py"))
native_bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_bundle)

XCODE = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk"
XCODE_SPACE = "/Applications/Xcode 2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.sdk"
CLT = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk"
UNVERSIONED = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"


def placehold(tmp_path, text):
    staged = tmp_path / "GTSAM-exports.cmake"
    staged.write_text(text)
    changed = native_bundle.placehold_root(str(staged))
    return changed, staged.read_text()


def test_macos_sdk_libraries_become_names(tmp_path):
    text = (f'  INTERFACE_LINK_LIBRARIES "{XCODE}/usr/lib/libm.tbd"\n'
            f'  INTERFACE_LINK_LIBRARIES "{XCODE_SPACE}/usr/lib/libm.tbd"\n'
            f'  INTERFACE_LINK_LIBRARIES "tbb;{CLT}/usr/lib/libm.tbd;{UNVERSIONED}/usr/lib/libc++.tbd"\n'
            f'set(LIBM {XCODE}/usr/lib/libm.tbd)\n')
    changed, out = placehold(tmp_path, text)
    assert changed
    assert out == ('  INTERFACE_LINK_LIBRARIES "m"\n'
                   '  INTERFACE_LINK_LIBRARIES "m"\n'
                   '  INTERFACE_LINK_LIBRARIES "tbb;m;c++"\n'
                   'set(LIBM m)\n')


def test_other_paths_are_untouched(tmp_path):
    text = ('  INTERFACE_LINK_LIBRARIES "/usr/lib/x86_64-linux-gnu/libm.so;/opt/conda/lib/libtbb.so"\n'
            f'  INTERFACE_INCLUDE_DIRECTORIES "{XCODE}/usr/include"\n')
    changed, out = placehold(tmp_path, text)
    assert not changed
    assert out == text


def test_checkout_path_becomes_the_placeholder(tmp_path):
    changed, out = placehold(tmp_path, f'set(GTSAM_DIR "{native_bundle.ROOT}/thirdparty/gtsam_local")\n')
    assert changed
    assert out == f'set(GTSAM_DIR "{native_bundle.ROOT_PLACEHOLDER}/thirdparty/gtsam_local")\n'
