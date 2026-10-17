#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
The trajectory error (ATE rmse) of every SLAM run in a results folder, computed twice: after fitting
the estimated trajectory to the ground truth with a rotation and a translation only (without scale),
and with a scale as well (with scale), and the scale which that second fit found (fitted scale).

main_slam.py reports one of the two: with scale for a monocular run, whose map has no scale, and
without scale for RGB-D and stereo runs, whose maps claim metres. Computing both puts runs of
different kinds on the same footing. Every run (main_slam.py, or each run of main_slam_evaluation.py)
saves its estimated poses, and the ground-truth poses they were matched with, in
plot/trajectory_final.json, which this script finds.

usage: python scripts/trajectory_errors.py results/eval_<date>       (every run of an evaluation)
       python scripts/trajectory_errors.py results/metrics_<date>    (one run of main_slam.py)
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
from evo.core import metrics
from evo.core.trajectory import PosePath3D


def ate(gt, est, with_scale):
    """The ATE rmse of est against gt after fitting est to gt (with or without scale), and the scale."""
    est = copy.deepcopy(est)
    _, _, scale = est.align(gt, correct_scale=with_scale)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est))
    return ape.get_statistic(metrics.StatisticsType.rmse), scale


def main():
    parser = argparse.ArgumentParser(
        description="The ATE rmse of every SLAM run in a results folder, without and with a scale fitted "
        "to the ground truth, and the scale fitted",
        epilog="example: python scripts/trajectory_errors.py results/eval_2026_10_17-14_26_32",
    )
    parser.add_argument("folder", help="a results folder: results/eval_<date> or results/metrics_<date>")
    args = parser.parse_args()

    folder = args.folder.rstrip("/\\")
    paths = sorted(glob.glob(os.path.join(folder, "**", "trajectory_final.json"), recursive=True))
    if not paths:
        sys.exit(f"no plot/trajectory_final.json in {folder}: give a results/eval_<date> or results/metrics_<date> folder")
    for path in paths:
        with open(path) as f:
            run = json.load(f)
        est = PosePath3D(poses_se3=[np.array(T) for T in run["trajectory_est"]])
        gt = PosePath3D(poses_se3=[np.array(T) for T in run["trajectory_gt"]])
        rigid, _ = ate(gt, est, with_scale=False)
        similar, scale = ate(gt, est, with_scale=True)
        # the run's folder: <preset>/<dataset>/iteration_<n> in an evaluation, the folder itself for one run
        name = os.path.relpath(os.path.dirname(os.path.dirname(path)), folder).replace(os.sep, "/")
        if name == ".":
            name = os.path.basename(os.path.abspath(folder))
        print(f"{name}: {rigid:.3f} m without scale; {similar:.3f} m with scale (fitted scale {scale:.2f})")


if __name__ == "__main__":
    main()
