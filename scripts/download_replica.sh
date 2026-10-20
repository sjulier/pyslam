#!/usr/bin/env bash

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."
cd "$ROOT_DIR"

mkdir -p datasets
cd datasets

wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
# a fresh Ubuntu (e.g. WSL2) has no unzip; Python's zipfile is always there
if command -v unzip >/dev/null 2>&1; then
    unzip Replica.zip
else
    python3 -m zipfile -e Replica.zip .
fi
mv Replica replica