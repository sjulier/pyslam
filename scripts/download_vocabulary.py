#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Download the vocabulary of the default loop detector (DBoW3 with the ORB vocabulary) at install
time, so that the first SLAM run does not have to. An interrupted download is resumed, and a
stalled one is restarted (see gdrive_download_with_retry in pyslam/utilities/file_management.py).

usage: python scripts/download_vocabulary.py [LOOP_DETECTOR_CONFIG ...]   (default: DBOW3)
       e.g. python scripts/download_vocabulary.py DBOW3 DBOW2
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import pyslam  # noqa: E402,F401  (first: sets environment variables that must precede numpy/torch)
from pyslam.config import Config  # noqa: E402,F401
from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs  # noqa: E402


def main():
    names = sys.argv[1:] or ["DBOW3"]
    failed = []
    for name in names:
        config = getattr(LoopDetectorConfigs, name, None)
        vocabulary = config.get("vocabulary_data") if isinstance(config, dict) else None
        if vocabulary is None:
            print(f"{name}: no vocabulary to download")
            continue
        try:
            vocabulary.check_download()
            size_mb = os.path.getsize(vocabulary.vocab_file_path) / 1e6
            print(f"{name}: {vocabulary.vocab_file_path} ({size_mb:.0f} MB)")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"{name}: FAILED: {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
