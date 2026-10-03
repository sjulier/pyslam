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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
SCRIPTS_DIR="$SCRIPT_DIR/../../scripts"

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
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
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
# if [[ -d "$OpenCV_DIR" ]]; then
#     EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
# fi 

# check if WITH_MARCH_NATIVE is not set
if [[ -z "$WITH_MARCH_NATIVE" ]]; then
    WITH_MARCH_NATIVE=ON
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DWITH_MARCH_NATIVE=$WITH_MARCH_NATIVE"
fi

EXTERNAL_OPTIONS+=" -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

make_dir build
cd build
cmake .. $EXTERNAL_OPTIONS || exit 1
make -j 4 || exit 1

cd ..

# ====================================================

if [[ ! -d "$SCRIPT_DIR/include/gtsam_factors" ]]; then
    mkdir -p "$SCRIPT_DIR/include/gtsam_factors"
fi

# copy header files to include folder
cp *.h "$SCRIPT_DIR/include/gtsam_factors"
