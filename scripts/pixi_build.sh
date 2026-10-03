#!/usr/bin/env bash
# `pixi run build`: get pySLAM's native modules, in two parts (scripts/native_bundle.py):
#   prereq  GTSAM, g2o, Pangolin, DBoW2/3, iBoW: prebuilt if there is a bundle for this checkout and
#           machine, otherwise built from source (15 minutes to over an hour)
#   pyslam  pySLAM's own C++ code: prebuilt likewise, otherwise built from source against the prereq
#           part (a few minutes). After a change to that code only this part is rebuilt.
# Set PYSLAM_NATIVE_BUNDLE=0 to always build from source.
SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR_/.."
cd "$ROOT_DIR" || exit 1

PIXI_="${PIXI_EXE:-pixi}"
ENV_="${PIXI_ENVIRONMENT_NAME:-default}"

for part in prereq pyslam; do
    rc=2
    if [[ "${PYSLAM_NATIVE_BUNDLE:-1}" != "0" ]]; then
        python scripts/native_bundle.py fetch "$part"
        rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        task=build-prerequisites
        [[ $part == pyslam ]] && task=build-pyslam
        "$PIXI_" run -e "$ENV_" "$task" || exit 1
    fi
done
python scripts/check_pybind11_abi.py && python scripts/download_vocabulary.py
