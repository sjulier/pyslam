#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."
SCRIPTS_DIR="$ROOT_DIR/scripts"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"


# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    #echo "Conda is not installed"
    CONDA_INSTALLED=false
fi

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
    echo "Pixi environment detected: $PIXI_PROJECT_NAME"

    source "$SCRIPTS_DIR/pixi_python_config.sh"
else
    PIXI_ACTIVATED=false
fi

# ====================================================

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ubuntu_version=$(lsb_release -rs | cut -d. -f1)
else
    ubuntu_version=""
fi

# Check if CC is set and available, otherwise use default gcc
if command -v "$CC" &> /dev/null; then
    gcc_version=$($CC -dumpversion | cut -d. -f1)
elif command -v gcc &> /dev/null; then
    gcc_version=$(gcc -dumpversion | cut -d. -f1)
else
    print_red "Error: No C compiler found. Please install gcc or a equivalent compiler."
fi
echo "gcc_version: $gcc_version"

if [[ "$CONDA_INSTALLED" == true && "$ubuntu_version" == "20" && "$gcc_version" == 11 ]]; then
    print_blue "Setting GCC and G++ to version 9"
    export CC=/usr/bin/gcc-9
    export CXX=/usr/bin/g++-9
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++"

    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

print_blue '================================================'
print_blue "Installing gtsam from source"
print_blue '================================================'

# Prefer the active conda/venv python (set by install_all_conda.sh via pyenv-activate.sh)
if [[ -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_EXE="$CONDA_PREFIX/bin/python"
else
    PYTHON_EXE=$(which python)
fi
echo "Using PYTHON_EXE: $PYTHON_EXE"

PYTHON_VERSION=$($PYTHON_EXE -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

# GTSAM tag to build. thirdparty/gtsam_factors and the C++ core
# (optimizer_gtsam.cpp) are written against this version, so they must be updated together.
GTSAM_TAG="4.3.0"

# Parallel jobs, limited by the available memory (see get_build_jobs in bash_utils.sh).
NUM_CORES=$(get_build_jobs)
echo "Building with $NUM_CORES parallel jobs (set PYSLAM_BUILD_JOBS to change)"


WITH_MARCH_NATIVE=ON
if [[ "$OSTYPE" == darwin* ]]; then
    WITH_MARCH_NATIVE=OFF
fi
# PYSLAM_MARCH: the CPU to compile for (the value of -march; default: native, this machine's CPU).
# GTSAM and all of pySLAM's native modules must use the same value (see the CMakeLists.txt of the
# modules). With another value than native, GTSAM gets it through the compiler flags.
if [[ -n "$PYSLAM_MARCH" && "$PYSLAM_MARCH" != "native" && "$OSTYPE" != darwin* ]]; then
    WITH_MARCH_NATIVE=OFF
    export CFLAGS="$CFLAGS -march=$PYSLAM_MARCH"
    export CXXFLAGS="$CXXFLAGS -march=$PYSLAM_MARCH"
    echo "PYSLAM_MARCH: $PYSLAM_MARCH"
fi
echo "WITH_MARCH_NATIVE: $WITH_MARCH_NATIVE"

# gtwrap (GTSAM Python bindings) needs pyparsing at build time, and the python install generates
# type stubs with pybind11-stubgen (see python/requirements.txt in GTSAM)
ensure_python_package "$PYTHON_EXE" "pyparsing>=3.2.5" pyparsing || exit 1
ensure_python_package "$PYTHON_EXE" "pybind11-stubgen>=2.5.1" pybind11_stubgen || exit 1

cd thirdparty
# A checkout of a different GTSAM version (e.g. from an older pySLAM) is replaced.
if [ -d gtsam_local ]; then
    LOCAL_GTSAM_TAG=$(git -C gtsam_local describe --tags --exact-match 2>/dev/null)
    if [[ "$LOCAL_GTSAM_TAG" != "$GTSAM_TAG" ]]; then
        print_yellow "thirdparty/gtsam_local is GTSAM '${LOCAL_GTSAM_TAG:-unknown}', not $GTSAM_TAG: removing it to rebuild"
        rm -rf gtsam_local
    fi
fi
if [ ! -d gtsam_local ]; then
    # Remove a partial checkout on failure, so that the next run starts from scratch.
    # Shallow clone of just the tag: the full history is large and slow to fetch.
    if ! git clone --depth 1 --branch $GTSAM_TAG https://github.com/borglab/gtsam.git gtsam_local; then
        print_red "Error: failed to fetch GTSAM $GTSAM_TAG"
        rm -rf gtsam_local
        exit 1
    fi
fi
cd gtsam_local
make_buid_dir
GTSAM_INSTALL_DIR="$(pwd -P)/install"
GTSAM_CONFIG_FILE="install/lib/cmake/GTSAM/GTSAMConfig.cmake"
TARGET_GTSAM_LIB="install/lib/libgtsam.so"
if [[ "$OSTYPE" == darwin* ]]; then
    TARGET_GTSAM_LIB="install/lib/libgtsam.dylib"
fi

# Print the (physical) directory of the libgtsam that a gtsam python extension module loads.
function linked_libgtsam_dir(){
    local lib
    if [[ "$OSTYPE" == darwin* ]]; then
        lib=$(otool -L "$1" 2>/dev/null | awk '/libgtsam\./ {print $1; exit}')
        # GTSAM >= 4.3 uses @rpath install names: resolve them like dyld, trying the
        # module's LC_RPATH entries in order (with @loader_path expanded).
        if [[ "$lib" == @rpath/* ]]; then
            local rpath candidate
            while read -r rpath; do
                candidate="${rpath/@loader_path/$(dirname "$1")}/${lib#@rpath/}"
                if [[ -f "$candidate" ]]; then lib="$candidate"; break; fi
            done < <(otool -l "$1" 2>/dev/null | awk '/cmd LC_RPATH/ {getline; getline; print $2}')
        fi
    else
        lib=$(ldd "$1" 2>/dev/null | awk '/libgtsam\.so/ {print $3; exit}')
    fi
    if [[ -n "$lib" && -d "$(dirname "$lib")" ]]; then
        (cd "$(dirname "$lib")" && pwd -P)
    fi
}

# NOTE: gtsam has some issues when compiling with march=native option!
# https://groups.google.com/g/gtsam-users/c/jdySXchYVQg
# https://bitbucket.org/gtborg/gtsam/issues/414/compiling-with-march-native-results-in 
GTSAM_OPTIONS="-DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=$WITH_MARCH_NATIVE -DGTSAM_BUILD_PYTHON=ON -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF" 
if [[ "$version" == *"24.04"* ]] ; then
    # Ubuntu 24.04 requires CMake 3.22 or higher
    GTSAM_OPTIONS+=" -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
fi
# Pin the interpreter: with GTSAM_PYTHON_VERSION set, GTSAM skips its own Python lookup and
# the wrapper (pybind11) uses PYTHON_EXECUTABLE.
GTSAM_OPTIONS+=" -DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF -DGTSAM_PYTHON_VERSION=$PYTHON_VERSION"
GTSAM_OPTIONS+=" -DPYTHON_EXECUTABLE=$PYTHON_EXE -DPython_EXECUTABLE=$PYTHON_EXE -DPython3_EXECUTABLE=$PYTHON_EXE"
if [[ "$OSTYPE" == darwin* ]]; then
    GTSAM_OPTIONS+=" -DGTSAM_WITH_TBB=OFF"
fi
# GTSAM looks for Ceres only for its tests/timing (disabled here). Skip the lookup: a system
# Ceres (e.g. Ubuntu's libceres-dev) fails to configure when its glog is not found, e.g. in conda.
GTSAM_OPTIONS+=" -DCMAKE_DISABLE_FIND_PACKAGE_Ceres=ON"
# pyslam does not use GTSAM's Boost features (serialization, timers, ...), so build without Boost
# and avoid depending on a full Boost installation.
GTSAM_OPTIONS+=" -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF -DGTSAM_USE_BOOST_FEATURES=OFF"
# Do not turn warnings into errors: newer compilers (e.g. GCC 15 with AVX-512 and Eigen) emit
# warnings such as -Wmaybe-uninitialized that would otherwise fail the build.
GTSAM_OPTIONS+=" -DGTSAM_BUILD_WITH_WERROR=OFF"
# pyslam does not use gtsam_unstable: skip it (and its python module) to shorten the build
GTSAM_OPTIONS+=" -DGTSAM_BUILD_UNSTABLE=OFF"
# Build with the install paths (install names on macOS, RPATH on Linux), so that the python
# module installed from the build tree loads the installed libgtsam (see above).
GTSAM_OPTIONS+=" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=$GTSAM_INSTALL_DIR/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"
# The effective GTSAM configuration is recorded next to the installed library: rebuild (from a fresh
# build dir) when it differs, e.g. after pulling changed GTSAM_OPTIONS onto an existing install.
GTSAM_CONFIG_STAMP_FILE="$GTSAM_INSTALL_DIR/.pyslam_gtsam_options"
GTSAM_CONFIG_STAMP=$(echo "$GTSAM_TAG -DCMAKE_BUILD_TYPE=Release $GTSAM_OPTIONS $EXTERNAL_OPTIONS $MAC_OPTIONS" | xargs)

# The gtsam python module must load the *installed* libgtsam, the same one that gtsam_factors
# and the C++ core link. The module is pip-installed from the build tree (see below), so the build must
# use the install paths (CMAKE_BUILD_WITH_INSTALL_RPATH below); otherwise the build-tree libgtsam
# is loaded as well and two copies of GTSAM end up in the same process.
# Rebuild if the library is missing or the build-tree python module was built the old way.
BUILD_GTSAM_PY_MODULE=$(ls build/python/gtsam/gtsam*.so 2>/dev/null | head -1)
NEED_GTSAM_BUILD=false
if [[ ! -f "$TARGET_GTSAM_LIB" || ! -f "$GTSAM_CONFIG_FILE" || -z "$BUILD_GTSAM_PY_MODULE" ]]; then
    NEED_GTSAM_BUILD=true
elif [[ "$(linked_libgtsam_dir "$BUILD_GTSAM_PY_MODULE")" != "$GTSAM_INSTALL_DIR/lib" ]]; then
    echo "The gtsam python module in build/ does not link $GTSAM_INSTALL_DIR/lib: rebuilding GTSAM"
    NEED_GTSAM_BUILD=true
elif [[ "$(cat "$GTSAM_CONFIG_STAMP_FILE" 2>/dev/null)" != "$GTSAM_CONFIG_STAMP" ]]; then
    echo "The GTSAM configuration changed (or was not recorded): rebuilding GTSAM"
    NEED_GTSAM_BUILD=true
fi

if [[ "$NEED_GTSAM_BUILD" == true ]]; then
    # start from scratch: a changed configuration must not reuse the old CMake cache or leave stale
    # files (e.g. a Boost-linked GTSAMConfig or libgtsam_unstable) in install/
    rm -rf build install && mkdir build && cd build || exit 1
    echo GTSAM_OPTIONS: $GTSAM_OPTIONS
    cmake .. -DCMAKE_INSTALL_PREFIX="$GTSAM_INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release $GTSAM_OPTIONS $EXTERNAL_OPTIONS $MAC_OPTIONS || { print_red "Error: GTSAM cmake configure failed"; exit 1; }
	make -j $NUM_CORES || { print_red "Error: GTSAM build failed"; exit 1; }
    make install || { print_red "Error: GTSAM install failed"; exit 1; }
    echo "$GTSAM_CONFIG_STAMP" > "$GTSAM_CONFIG_STAMP_FILE"
    cd ..
fi

# Install the gtsam python package unless this version is already there and loads the installed
# libgtsam. This runs even when the C++ library is already built, so that a recreated python
# environment gets the package back. A different gtsam (e.g. a pip wheel) is replaced.
# - Under pixi it is installed in the source tree ($GTSAM_INSTALL_DIR/python), which pixi.toml puts on
#   PYTHONPATH and config_libs.yaml on pySLAM's path: every level's environment then finds it, not
#   only the one the build ran in.
# - Otherwise it is installed into $PYTHON_EXE's environment.
GTSAM_PY_TARGET_DIR=""
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    GTSAM_PY_TARGET_DIR="$GTSAM_INSTALL_DIR/python"
    export PYTHONPATH="$GTSAM_PY_TARGET_DIR${PYTHONPATH:+:$PYTHONPATH}"
    # a copy in the environment (from an older build) would shadow or duplicate the one in the tree
    if PYTHONPATH= $PYTHON_EXE -c "import importlib.metadata as m; m.version('gtsam')" &>/dev/null; then
        echo "Removing the gtsam python package from the environment (it is installed in the source tree now)"
        PYTHONPATH= $PYTHON_EXE -m pip uninstall -y gtsam || { print_red "Error: could not remove gtsam from the environment"; exit 1; }
    fi
fi
function installed_gtsam_py_module(){
    $PYTHON_EXE -c "import gtsam, glob, os; print(glob.glob(os.path.join(os.path.dirname(gtsam.__file__), 'gtsam*.so'))[0])" 2>/dev/null
}
INSTALLED_GTSAM_PY_VERSION=$($PYTHON_EXE -c "import gtsam, importlib.metadata as m; print(m.version('gtsam'))" 2>/dev/null)
INSTALLED_GTSAM_PY_LIB_DIR=$(linked_libgtsam_dir "$(installed_gtsam_py_module)")
if [[ "$NEED_GTSAM_BUILD" == true || "$INSTALLED_GTSAM_PY_VERSION" != "$GTSAM_TAG" || "$INSTALLED_GTSAM_PY_LIB_DIR" != "$GTSAM_INSTALL_DIR/lib" ]]; then
    echo "Installing gtsam python package (found: '${INSTALLED_GTSAM_PY_VERSION:-none}' linking '${INSTALLED_GTSAM_PY_LIB_DIR:-none}', expected: $GTSAM_TAG linking $GTSAM_INSTALL_DIR/lib)"
    # Build the module (and type stubs), then pip-install it into $PYTHON_EXE. `make python-install`
    # is not used: it adds `pip install --user` outside a virtualenv (e.g. in a conda env), which
    # installs into ~/.local and shadows gtsam for every python of the same version.
    if [[ -n "$GTSAM_PY_TARGET_DIR" ]]; then
        ( cd build && make -j $NUM_CORES python-stubs && cd python && rm -rf "$GTSAM_PY_TARGET_DIR" \
            && $PYTHON_EXE -m pip install --no-deps --target "$GTSAM_PY_TARGET_DIR" . ) || { print_red "Error: GTSAM python install failed"; exit 1; }
    else
        ( cd build && make -j $NUM_CORES python-stubs && cd python && $PYTHON_EXE -m pip install . ) || { print_red "Error: GTSAM python install failed"; exit 1; }
    fi
fi
if ! $PYTHON_EXE -c "import gtsam" ; then
    print_red "Error: 'import gtsam' fails with $PYTHON_EXE"
    exit 1
fi
INSTALLED_GTSAM_PY_LIB_DIR=$(linked_libgtsam_dir "$(installed_gtsam_py_module)")
if [[ "$INSTALLED_GTSAM_PY_LIB_DIR" != "$GTSAM_INSTALL_DIR/lib" ]]; then
    print_red "Error: the gtsam python module loads libgtsam from '$INSTALLED_GTSAM_PY_LIB_DIR' instead of $GTSAM_INSTALL_DIR/lib"
    exit 1
fi

echo current folder: $(pwd)

cd "$ROOT_DIR"

print_blue '================================================'
print_blue "Building gtsam_factors"
print_blue '================================================'

cd thirdparty
cd gtsam_factors
./build.sh $EXTERNAL_OPTIONS -DWITH_MARCH_NATIVE=$WITH_MARCH_NATIVE || { print_red "Error: gtsam_factors build failed"; exit 1; }

cd "$ROOT_DIR"

cd "$STARTING_DIR"