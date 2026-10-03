#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this script allows to build the C++ core of pySLAM

# pySLAM is built inside its pixi environment: run `pixi run build` (or a build-* task) in the
# repository's root folder. Outside pixi this script would use another compiler and other libraries
# than the rest of the build. PYSLAM_ALLOW_NON_PIXI=1 lets it run anyway (e.g. the legacy conda setup).
if [[ -z "$PIXI_PROJECT_NAME" && -z "$PYSLAM_ALLOW_NON_PIXI" ]]; then
    echo "ERROR: $(basename "$(dirname "$(readlink -f "$0")")")/$(basename "$0") must run inside pySLAM's pixi environment:" >&2
    echo "       run 'pixi run build' in the pySLAM folder (or start 'pixi shell' there first)." >&2
    echo "       To build in another environment anyway, set PYSLAM_ALLOW_NON_PIXI=1." >&2
    exit 1
fi

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_"
SCRIPTS_DIR="$ROOT_DIR/scripts"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

#set -e

print_blue '================================================'
print_blue "Building pySLAM C++ core"
print_blue '================================================'

cd "$ROOT_DIR/pyslam/slam/cpp"
# Remove the previously built module first: if this build fails, a stale cpp_core from older
# sources must not stay importable and look like a successful build.
rm -f lib/cpp_core*.so lib/cpp_core*.pyd
./build.sh || { print_red "ERROR: building the pySLAM C++ core failed (see the messages above)"; cd "$STARTING_DIR"; exit 1; }
if ! compgen -G "lib/cpp_core*.so" >/dev/null && ! compgen -G "lib/cpp_core*.pyd" >/dev/null; then
    print_red "ERROR: the build finished but lib/cpp_core*.so was not produced"
    cd "$STARTING_DIR"
    exit 1
fi

cd "$STARTING_DIR"