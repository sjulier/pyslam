#!/usr/bin/env bash

# pySLAM is built inside its pixi environment: run `pixi run build` (or a build-* task) in the
# repository's root folder. Outside pixi this script would use another compiler and other libraries
# than the rest of the build. PYSLAM_ALLOW_NON_PIXI=1 lets it run anyway (e.g. the legacy conda setup).
if [[ -z "$PIXI_PROJECT_NAME" && -z "$PYSLAM_ALLOW_NON_PIXI" ]]; then
    echo "ERROR: $(basename "$(dirname "$(readlink -f "$0")")")/$(basename "$0") must run inside pySLAM's pixi environment:" >&2
    echo "       run 'pixi run build' in the pySLAM folder (or start 'pixi shell' there first)." >&2
    echo "       To build in another environment anyway, set PYSLAM_ALLOW_NON_PIXI=1." >&2
    exit 1
fi

# Parallel jobs, limited by the available memory (see get_build_jobs in bash_utils.sh)
NUM_JOBS=$( . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../bash_utils.sh" >/dev/null 2>&1 && get_build_jobs )
NUM_JOBS=${NUM_JOBS:-4}
echo "Building with $NUM_JOBS parallel jobs (set PYSLAM_BUILD_JOBS to change)"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

PYTHON_EXE=${Python3_EXECUTABLE:-${PYTHON_EXE:-$(which python3)}}

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++"

    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"
echo "Using Python executable: $PYTHON_EXE"

# ====================================================

cd modules/dbow3
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}/modules/dbow3/install $EXTERNAL_OPTIONS $MAC_OPTIONS
make -j "$NUM_JOBS"
make install

cd ${SCRIPT_DIR}
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$PYTHON_EXE $EXTERNAL_OPTIONS $MAC_OPTIONS
make -j "$NUM_JOBS"
