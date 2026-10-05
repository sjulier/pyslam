#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import pyslam  # first: sets the OpenMP thread settings before numpy/torch are imported
import sys
import pyslam.config as config

import argparse
import csv
import re
import numpy as np
import math

import os
import subprocess
from datetime import datetime
import shutil

import yaml
import json
import concurrent.futures
from pathlib import Path

from pyslam.evaluation.slam_evaluation_manager import SlamEvaluationManager

try:
    import hjson
except ImportError:
    print(
        "hjson not installed. Please install it with: pip install hjson"
    )  # why hjson? because it allows comments in the json file and it does not complain about the trailing commas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kSettingsFolder = kRootFolder + "/settings"
kResultsFolder = kRootFolder + "/results"
kEvaluationFolder = kRootFolder + "/pyslam/evaluation"


date_time_now = datetime.now()
date_time_now_string = date_time_now.strftime("%Y_%m_%d-%H_%M_%S")
eval_path_prefix = kResultsFolder + "/eval_" + date_time_now_string

config_dir_path = os.path.abspath(os.path.join(kEvaluationFolder, "configs"))
default_config_file_path = os.path.join(config_dir_path, "evaluation_video.json")
default_template_config_file_path = os.path.abspath(
    os.path.join(config_dir_path, "config.template.yaml")
)


def feature_config_names():
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

    return sorted(
        name
        for name, value in vars(FeatureTrackerConfigs).items()
        if isinstance(value, dict) and not name.startswith("_")
    )


def feature_preset(name):
    """The preset that runs SLAM with the features `name` (a FeatureTrackerConfigs entry)."""
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.local_features.feature_tracker import FeatureTrackerTypes
    from pyslam.loop_closing.loop_detector_configs import loop_detector_name_for_features

    feature_config = getattr(FeatureTrackerConfigs, name)
    if feature_config.get("tracker_type") == FeatureTrackerTypes.LK:
        sys.exit(f"--features {name}: SLAM needs features with descriptors (the LK_* trackers have none)")
    settings = {"FeatureTrackerConfig.name": name}
    # the same rule as main_slam.py --features: with descriptors that are not ORB, CosPlace when it is
    # installed, else DBoW3 on ORB descriptors of its own
    loop_name = loop_detector_name_for_features(feature_config.get("descriptor_type"))
    if loop_name is not None:
        settings["LoopDetectionConfig.name"] = loop_name
    return {"name": name, "custom_parameters": {"settings": settings}}


def tum_sensor_type(associations_path):
    """rgbd if the list of the frames of a TUM sequence has depth images (4 columns), mono otherwise."""
    try:
        with open(associations_path) as f:
            lines = [line.split() for line in f if line.strip() and not line.lstrip().startswith("#")]
    except OSError:
        return "mono"
    return "rgbd" if lines and len(lines[0]) >= 4 else "mono"


def config_overrides_from_args(args, argparser):
    """The entries of the evaluation configuration that the command line replaces."""
    overrides = {}
    if sum(bool(sequence) for sequence in (args.video, args.images, args.tum)) > 1:
        argparser.error("give only one of --video, --images and --tum")
    if args.video or args.images or args.tum:
        if not args.settings:
            argparser.error("--video, --images and --tum need the camera settings file: --settings")
        dataset = {"settings_path": os.path.abspath(args.settings)}
        sensor_type = "mono"
        if args.tum:
            # a sequence in the layout of the TUM RGB-D datasets: the names of the images and their
            # timestamps come from the list of the frames, as for the TUM datasets themselves
            tum_path = os.path.abspath(os.path.expanduser(args.tum)).rstrip("/")
            overrides["dataset_type"] = "TUM_DATASET"
            overrides["dataset_base_path"] = os.path.dirname(tum_path)
            dataset["name"] = os.path.basename(tum_path)
            associations = args.associations
            if associations is None:
                has_associations = os.path.exists(os.path.join(tum_path, "associations.txt"))
                associations = "associations.txt" if has_associations else "rgb.txt"
            dataset["associations"] = associations
            sensor_type = args.sensor or tum_sensor_type(os.path.join(tum_path, associations))
        elif args.video:
            video_path = os.path.abspath(os.path.expanduser(args.video))
            overrides["dataset_type"] = "VIDEO_DATASET"
            overrides["dataset_base_path"] = os.path.dirname(video_path)
            dataset["name"] = os.path.basename(video_path)
            if args.timestamps:
                dataset["timestamps"] = args.timestamps
        else:
            images_path = os.path.abspath(os.path.expanduser(args.images))
            overrides["dataset_type"] = "FOLDER_DATASET"
            overrides["dataset_base_path"] = images_path
            dataset["name"] = os.path.basename(images_path.rstrip("/"))
            dataset["pattern"] = args.pattern
            dataset["fps"] = args.fps
        if args.groundtruth:
            dataset["groundtruth_file"] = args.groundtruth
        overrides["datasets"] = [dataset]
        overrides["sensor_type"] = sensor_type
        overrides["output_path"] = "video" if args.video else "images" if args.images else "tum"
    elif args.settings or args.groundtruth or args.timestamps:
        argparser.error("--settings, --groundtruth and --timestamps go with --video, --images or --tum")
    if args.features:
        unknown = [name for name in args.features if name not in feature_config_names()]
        if unknown:
            print(f"Unknown feature configuration(s): {' '.join(unknown)}. Valid names:", file=sys.stderr)
            print("    " + " ".join(feature_config_names()), file=sys.stderr)
            sys.exit(2)
        overrides["presets"] = [feature_preset(name) for name in args.features]
    if args.runs is not None:
        overrides["number_of_runs_per_dataset"] = args.runs
    if args.jobs is not None:
        overrides["num_threads"] = args.jobs
    main_slam_options = []
    if args.throttle:
        main_slam_options.append("--throttle")
    if args.speed is not None:
        main_slam_options.append(f"--speed {args.speed}")
    if main_slam_options:
        overrides["main_slam_options"] = " ".join(main_slam_options)
    return overrides


# You can use this script for three objectives:
# 1. Run the default evaluation: SLAM on the KITTI 06 video that comes with pySLAM, a few times with
#   each set of features, and a table of the trajectory errors (configs/evaluation_video.json).
#   $ ./main_slam_evaluation.py
#   Choose the features, the number of runs, and a video or image folder of your own:
#   $ ./main_slam_evaluation.py --features ORB2 SUPERPOINT --runs 5
#   $ ./main_slam_evaluation.py --video my/video.mp4 --settings settings/MY_CAMERA.yaml
#   $ ./main_slam_evaluation.py --images my/images --pattern "*.jpg" --fps 30 --settings settings/MY_CAMERA.yaml
#   $ ./main_slam_evaluation.py --tum my/tum_sequence --settings settings/MY_CAMERA.yaml
# 2. Run an evaluation by calling main_slam.py on a set of datasets with a set of presets.
#   $ ./main_slam_evaluation.py -c pyslam/evaluation/configs/evaluation_<my awesome dataset>.json
# 3. If needed, you can also create a report from an existing data folder without running an evaluation (a bit hacky, for instance, after having merged the results of different evaluations).
#   $ ./main_slam_evaluation.py --just-create-report -o <path to results folder containing evaluation*.json>
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Run SLAM (main_slam.py, without windows) several times on one or more datasets, "
        "with one or more presets (e.g. features), and create a table of the results. By default: "
        "the KITTI 06 video that comes with pySLAM. The options below replace parts of the "
        "configuration file.",
    )
    argparser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default=default_config_file_path,
        help="Path of the input configuration file for the evaluations "
        "(default: pyslam/evaluation/configs/evaluation_video.json).",
    )
    argparser.add_argument(
        "-t",
        "--template-config-file",
        type=str,
        default=default_template_config_file_path,
        help="Path of the template configuration file for the evaulations.",
    )
    sequence = argparser.add_argument_group("your own sequence, instead of the datasets of the configuration file")
    sequence.add_argument("--video", help="a video file (monocular)")
    sequence.add_argument("--images", help="a folder of images, read in the order of their names (monocular)")
    sequence.add_argument(
        "--tum",
        metavar="FOLDER",
        help="a sequence in the layout of the TUM RGB-D datasets, read as those are: the images, their "
        "timestamps and their order from the list of the frames (associations.txt, or rgb.txt if "
        "there is none), the ground truth from groundtruth.txt (timestamp tx ty tz qx qy qz qw)",
    )
    sequence.add_argument(
        "--associations",
        help="with --tum: name of the list of the frames in the folder of the sequence",
    )
    sequence.add_argument(
        "--sensor",
        choices=["mono", "rgbd"],
        help="with --tum: default rgbd if the list of the frames has depth images, mono otherwise",
    )
    sequence.add_argument("--pattern", default="*.png", help="with --images: which files (default '*.png')")
    sequence.add_argument("--fps", type=float, default=10, help="with --images: frames per second (default 10)")
    sequence.add_argument(
        "--settings",
        help="the camera settings file of the sequence (calibration, see the files in settings/): "
        "required with --video, --images and --tum",
    )
    sequence.add_argument(
        "--groundtruth",
        help="name of a ground truth file in the folder of the sequence. With --video and --images: "
        "one line per frame, timestamp x y z qx qy qz qw scale; without it there is no trajectory "
        "error and the table has only the percentage of frames where tracking was lost. With --tum: "
        "default groundtruth.txt",
    )
    sequence.add_argument(
        "--timestamps",
        help="with --video: name of a file in the folder of the video with one timestamp [s] per frame",
    )
    runs = argparser.add_argument_group("what is run")
    runs.add_argument(
        "--features",
        nargs="+",
        metavar="NAME",
        help="one preset per feature configuration, e.g. --features ORB2 ROOT_SIFT SUPERPOINT "
        "(FeatureTrackerConfigs entries: `main_feature_matching.py --list` lists them and what "
        "installs their models)",
    )
    runs.add_argument("--runs", type=int, help="number of runs of each preset on each dataset")
    runs.add_argument(
        "--jobs",
        type=int,
        help="number of runs at the same time (results may get worse with more than one)",
    )
    runs.add_argument(
        "--throttle",
        action="store_true",
        help="slow the playback down when tracking gets weak (main_slam.py --throttle): for features "
        "that cannot keep up with the camera's frame rate, which otherwise lose track",
    )
    runs.add_argument(
        "--speed",
        type=float,
        help="playback speed relative to the camera's frame rate (main_slam.py --speed), e.g. 0.5",
    )
    #
    argparser.add_argument(
        "--just-create-report", action="store_true", help="Create a report from the results folder"
    )
    argparser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="/home/luigi/Work/slam_wss/pyslam-master-new/results/eval_2025_04_13-21_16_14/tum",
        help="Path of the output folder",
    )
    args = argparser.parse_args()
    config_overrides = config_overrides_from_args(args, argparser)

    if args.just_create_report:
        # just create a report from an existing data folder
        assert args.output_path, "Please specify the output path"
        folder = Path(args.output_path)
        config_json_file = json_file = next(folder.glob("evaluation*.json"), None)
        evaluation_manager = SlamEvaluationManager(
            str(config_json_file),
            args.template_config_file,
            just_create_report=args.just_create_report,
        )
        print(f"Creating a report from the results folder: {args.output_path}")
        evaluation_manager.output_path = args.output_path
    else:
        # run the evaluation
        evaluation_manager = SlamEvaluationManager(
            args.config_file,
            args.template_config_file,
            just_create_report=args.just_create_report,
            config_overrides=config_overrides,
        )
        evaluation_manager.run_evaluation()

    evaluation_manager.create_final_table()

    # the tables (the mean over the runs per dataset and preset; in the rows below it, the average
    # and the standard deviation over all the runs of a preset). The trajectory errors (rmse, max)
    # are over the tracked frames only: percent_tracked says how much of the sequence that is.
    for table_path in sorted(Path(evaluation_manager.output_path).glob("table_*.csv")):
        print(f"\n{table_path.name[len('table_'):-len('.csv')]}:")
        print(table_path.read_text().strip())
    print(f"\nResults: {evaluation_manager.output_path} (report.html, table_*.csv, one folder per run)")

    print("Done.")
