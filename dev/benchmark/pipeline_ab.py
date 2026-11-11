"""A/B runs of main_slam.py: the Python and C++ SLAM pipelines, over cells of speed and sequence.

    pixi run python dev/benchmark/pipeline_ab.py OUT_DIR --runs 3 --cells 0:0:tum-mono 1:0:tum-mono \\
        --tum ~/datasets/tum/rgbd_dataset_freiburg1_xyz [-- extra main_slam.py options]

A cell is PIPELINE:SPEED:SEQUENCE:
  PIPELINE  0 = Python pipeline, 1 = C++ pipeline (PYSLAM_USE_CPP_PIPELINE)
  SPEED     main_slam.py --speed (0 = as fast as possible)
  SEQUENCE  tum-mono, tum-rgbd (--tum DIR; tum-rgbd needs DIR/associations_rgbd.txt, made with
            `python scripts/associate.py rgb.txt depth.txt > associations_rgbd.txt` in DIR), or
            config (the dataset of config.yaml, e.g. the KITTI 06 video)
Each run is headless, with --no-throttle unless the extra options say --throttle, and a
--timing-report. The rounds are interleaved (every cell once, then again), so that a drift in the
machine's state hits all cells alike. Each run goes through run_monitored.py (peak memory, stall
stacks, sleep detection). On macOS the runs are wrapped in `caffeinate -i -s`: idle sleep suspends
a run and looks like a hang. Then summarize.py OUT_DIR prints the table.
"""

import argparse
import os
import shutil
import subprocess
import sys

kHere = os.path.dirname(os.path.abspath(__file__))
kRoot = os.path.abspath(os.path.join(kHere, "..", ".."))
kSequences = ("tum-mono", "tum-rgbd", "config")


def sequence_options(sequence, tum):
    if sequence == "config":
        return []
    if not tum:
        sys.exit(f"{sequence}: give --tum DIR")
    options = ["--tum", tum, "--settings", "settings/TUM1.yaml"]
    if sequence == "tum-rgbd":
        return options + ["--associations", "associations_rgbd.txt", "--sensor", "rgbd"]
    return options + ["--sensor", "mono"]


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cells", nargs="+", required=True, metavar="PIPELINE:SPEED:SEQUENCE")
    parser.add_argument("--tum", help="the TUM sequence folder of the tum-* cells")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--stall", type=float, default=120.0)
    args = parser.parse_args(argv)

    cells = []
    for cell in args.cells:
        pipeline, speed, sequence = cell.split(":")
        if pipeline not in ("0", "1") or sequence not in kSequences:
            sys.exit(f"bad cell {cell}: PIPELINE is 0 or 1, SEQUENCE one of {', '.join(kSequences)}")
        cells.append((pipeline, speed, sequence))
    throttle = [] if "--throttle" in extra else ["--no-throttle"]
    keep_awake = ["caffeinate", "-i", "-s"] if sys.platform == "darwin" and shutil.which("caffeinate") else []
    os.makedirs(args.out_dir, exist_ok=True)

    for run in range(1, args.runs + 1):
        for pipeline, speed, sequence in cells:
            tag = f"pipe{pipeline}_speed{speed}_{sequence}_run{run}"
            out = os.path.join(args.out_dir, tag)
            print(f"{tag}: ", end="", flush=True)
            command = keep_awake + [
                sys.executable, os.path.join(kHere, "run_monitored.py"), out,
                "--timeout", str(args.timeout), "--stall", str(args.stall), "--",
                sys.executable, "main_slam.py", "--headless", "--speed", speed, *throttle,
                "--timing-report", out + ".timing.json", *sequence_options(sequence, args.tum), *extra,
            ]
            env = dict(os.environ, PYSLAM_USE_CPP_PIPELINE=pipeline)
            subprocess.run(command, cwd=kRoot, env=env)


if __name__ == "__main__":
    main()
