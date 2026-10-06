#!/usr/bin/env python
"""`pixi run build` on Windows (win-64): get pySLAM's native modules, in two parts as on Linux and macOS
(scripts/native_bundle.py, scripts/pixi_build.sh):

    prereq  GTSAM, g2o, Pangolin, DBoW2/3, iBoW
    pyslam  pySLAM's own C++ code: the ORB features, the C++ utilities (cpp/) and the C++ core

Each part is installed prebuilt when there is a bundle for this checkout and machine, and built from
source otherwise. Building needs Microsoft's C++ compiler (Visual Studio 2022 or its Build Tools, with
"Desktop development with C++"): the pixi environment finds it but cannot provide it.
Set PYSLAM_NATIVE_BUNDLE=0 to always build from source.

    pixi run -e default-win build                                     # both parts
    pixi run -e default-win python scripts/build_windows.py MODULE...  # build these modules from source

The Linux and macOS builds use the build.sh scripts; this is their counterpart for Windows, where there
is no bash and no make. Each module is configured with CMake and built with Ninja in <module>/build.
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = os.environ.get("CONDA_PREFIX", "")
LIBRARY = os.path.join(PREFIX, "Library")  # where conda puts the C++ libraries on Windows

# MSVC: C++ exceptions, large object files, AVX2 (the baseline of the Linux bundles, x86-64-v3)
# They go in through CFLAGS/CXXFLAGS, which CMake puts in front of its own defaults (/DWIN32 /D_WINDOWS ...)
MSVC_C_FLAGS = "/bigobj /arch:AVX2 /DNOMINMAX /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267 /wd4251"
MSVC_CXX_FLAGS = "/permissive- /EHsc " + MSVC_C_FLAGS

COMMON = [
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    f"-DCMAKE_PREFIX_PATH={LIBRARY}",
    f"-DPython3_EXECUTABLE={sys.executable}",
    f"-DPython_EXECUTABLE={sys.executable}",
    f"-DPYTHON_EXECUTABLE={sys.executable}",
    "-DPYSLAM_MARCH=x86-64-v3",
]

SUITESPARSE = [
    f"-DCHOLMOD_INCLUDE_DIR={LIBRARY}/include/suitesparse",
    f"-DCSPARSE_INCLUDE_DIR={LIBRARY}/include/suitesparse",
]

# name: (source folder, extra CMake options)
MODULES = {
    "orbslam2_features": ("thirdparty/orbslam2_features", []),
    # lib/ as on Linux and macOS (g2o's own default on Windows is bin/): pySLAM and the C++ core look there
    "g2o": ("thirdparty/g2opy", SUITESPARSE + [f"-Dg2o_LIBRARY_OUTPUT_DIRECTORY={ROOT}/thirdparty/g2opy/lib"]),
    # DBoW3 itself, installed where the python module's CMake file looks for it; then the module
    "dbow3_lib": ("thirdparty/pydbow3/modules/dbow3", [f"-DCMAKE_INSTALL_PREFIX={ROOT}/thirdparty/pydbow3/modules/dbow3/install", "-DLIB_INSTALL_DIR=lib"]),
    "dbow3": ("thirdparty/pydbow3", []),
    # GTSAM 4.3.0, cloned by fetch_gtsam(), with the options of scripts/install_gtsam.sh
    "gtsam": ("thirdparty/gtsam_local", [
        f"-DCMAKE_INSTALL_PREFIX={ROOT}/thirdparty/gtsam_local/install",
        "-DGTSAM_USE_SYSTEM_EIGEN=ON", "-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF",
        "-DGTSAM_BUILD_PYTHON=ON",
        "-DGTSAM_BUILD_TESTS=OFF", "-DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF", "-DGTSAM_BUILD_TIMING_ALWAYS=OFF",
        "-DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF", "-DGTSAM_PYTHON_VERSION=3.11",
        "-DCMAKE_DISABLE_FIND_PACKAGE_Ceres=ON",
        "-DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF", "-DGTSAM_USE_BOOST_FEATURES=OFF",
        "-DGTSAM_BUILD_WITH_WERROR=OFF", "-DGTSAM_BUILD_UNSTABLE=OFF",
    ]),
    "gtsam_factors": ("thirdparty/gtsam_factors", []),
    # DBoW2 and iBoW (with its index, obindex2): the libraries, then their python modules
    "dbow2_lib": ("thirdparty/pydbow2/modules/dbow2", []),
    "dbow2": ("thirdparty/pydbow2", []),
    "obindex2": ("thirdparty/pyibow/modules/obindex2/lib", []),
    "ibow_lcd": ("thirdparty/pyibow/modules/ibow-lcd", []),
    "ibow": ("thirdparty/pyibow", []),
    # Pangolin and its python module (the 3D viewer), with the environment's GLEW, libpng and libjpeg and
    # the DLL C runtime (Pangolin's defaults on Windows: download them, static runtime)
    "pangolin": ("thirdparty/pangolin", [
        "-DBUILD_PANGOLIN_LIBREALSENSE=OFF", "-DBUILD_PANGOLIN_LIBREALSENSE2=OFF", "-DBUILD_PANGOLIN_OPENNI=OFF",
        "-DBUILD_PANGOLIN_OPENNI2=OFF", "-DBUILD_PANGOLIN_FFMPEG=OFF", "-DBUILD_PANGOLIN_LIBOPENEXR=OFF",
        "-DBUILD_EXTERN_GLEW=OFF", "-DBUILD_EXTERN_LIBPNG=OFF", "-DBUILD_EXTERN_LIBJPEG=OFF",
        "-DMSVC_USE_STATIC_CRT=OFF", "-DBUILD_TESTS=OFF", "-DBUILD_TOOLS=OFF", "-DBUILD_EXAMPLES=OFF",
    ]),
    "cpp_utils": ("cpp", []),
    "cpp_core": ("pyslam/slam/cpp", []),
}


def run(cmd, cwd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


GTSAM_TAG = "4.3.0"


def fetch_gtsam():
    dst = os.path.join(ROOT, "thirdparty", "gtsam_local")
    if not os.path.exists(os.path.join(dst, "CMakeLists.txt")):
        cmd = ["git", "clone", "--depth", "1", "--branch", GTSAM_TAG, "-c", "core.symlinks=false",
               "https://github.com/borglab/gtsam.git", dst]
        if run(cmd, ROOT) != 0:
            sys.exit("ERROR: gtsam: could not fetch GTSAM " + GTSAM_TAG)


def build(name):
    if name == "gtsam":
        fetch_gtsam()
    if name == "cpp_core":
        # the C++ core includes the GTSAM factors' headers from there (as thirdparty/gtsam_factors/build.sh does)
        src_dir = os.path.join(ROOT, "thirdparty", "gtsam_factors")
        dst_dir = os.path.join(src_dir, "include", "gtsam_factors")
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith(".h"):
                shutil.copy2(os.path.join(src_dir, f), dst_dir)
    folder, options = MODULES[name]
    src = os.path.join(ROOT, folder)
    build_dir = os.path.join(src, "build")
    os.makedirs(build_dir, exist_ok=True)
    if run(["cmake", "-S", src, "-B", build_dir, *COMMON, *options], src) != 0:
        sys.exit(f"ERROR: {name}: CMake configure failed")
    jobs = os.environ.get("PYSLAM_BUILD_JOBS", "")
    cmd = ["cmake", "--build", build_dir, "--config", "Release"] + (["-j", jobs] if jobs else [])
    cmd += ["--", "-k", "0"] if os.environ.get("PYSLAM_KEEP_GOING") else []
    if run(cmd, src) != 0:
        sys.exit(f"ERROR: {name}: build failed")
    if any(o.startswith("-DCMAKE_INSTALL_PREFIX=") for o in options):
        if run(["cmake", "--install", build_dir], src) != 0:
            sys.exit(f"ERROR: {name}: install failed")
    if name == "gtsam":
        # the python package, in the source tree (as scripts/install_gtsam.sh does under pixi)
        target = os.path.join(ROOT, "thirdparty", "gtsam_local", "install", "python")
        shutil.rmtree(target, ignore_errors=True)
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--target", target, "."]
        if run(cmd, os.path.join(build_dir, "python")) != 0:
            sys.exit("ERROR: gtsam: installing the python package failed")
    print(f"OK: {name}", flush=True)


# The modules of each part of the native build, in build order
PART_MODULES = {
    "prereq": ["gtsam", "gtsam_factors", "g2o", "pangolin", "dbow3_lib", "dbow3", "dbow2_lib", "dbow2",
               "obindex2", "ibow_lcd", "ibow"],
    "pyslam": ["orbslam2_features", "cpp_utils", "cpp_core"],
}


def python_script(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", args[0]), *args[1:]], cwd=ROOT).returncode


def require_compiler(part):
    if shutil.which("cl") is None:
        sys.exit(
            f"ERROR: there are no prebuilt {part} modules for this checkout and machine (see the messages above), and\n"
            "       Microsoft's C++ compiler (cl.exe) was not found to build them: install Visual Studio 2022 or its\n"
            "       Build Tools with \"Desktop development with C++\", then run the same command again."
        )


def build_parts():
    """`pixi run build`: each part prebuilt if there is a bundle, built from source otherwise."""
    for part, modules in PART_MODULES.items():
        rc = 2
        if os.environ.get("PYSLAM_NATIVE_BUNDLE", "1") != "0":
            rc = python_script("native_bundle.py", "fetch", part)
        if rc != 0:
            require_compiler(part)
            python_script("native_bundle.py", "stale", part)
            for name in modules:
                build(name)
    if python_script("check_pybind11_abi.py") != 0:
        sys.exit(1)
    sys.exit(python_script("download_vocabulary.py"))


if __name__ == "__main__":
    if os.name != "nt":
        sys.exit("build_windows.py is for Windows: on Linux and macOS run `pixi run build`")
    if not PREFIX:
        sys.exit("ERROR: run inside pySLAM's pixi environment: pixi run -e default-win python scripts/build_windows.py")
    os.environ["CFLAGS"] = (os.environ.get("CFLAGS", "") + " " + MSVC_C_FLAGS).strip()
    os.environ["CXXFLAGS"] = (os.environ.get("CXXFLAGS", "") + " " + MSVC_CXX_FLAGS).strip()
    if len(sys.argv) == 1:
        build_parts()
    names = sys.argv[1:]
    for n in names:
        if n not in MODULES:
            sys.exit(f"ERROR: unknown module {n}: choose from {', '.join(MODULES)}")
    for n in names:
        build(n)
