#!/usr/bin/env bash
# `pixi run build`: get pySLAM's native modules, prebuilt if there is a bundle for this checkout and
# machine (scripts/native_bundle.py), otherwise built from source (15 minutes to over an hour).
# Set PYSLAM_NATIVE_BUNDLE=0 to always build from source.
SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR_/.."
cd "$ROOT_DIR" || exit 1

rc=2
if [[ "${PYSLAM_NATIVE_BUNDLE:-1}" != "0" ]]; then
    python scripts/native_bundle.py fetch
    rc=$?
fi
if [[ $rc -ne 0 ]]; then
    "${PIXI_EXE:-pixi}" run -e "${PIXI_ENVIRONMENT_NAME:-default}" build-from-source || exit 1
    exit 0
fi
python scripts/check_pybind11_abi.py && python scripts/download_vocabulary.py
