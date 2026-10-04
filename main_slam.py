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
import cv2
import time
import os
import sys
import numpy as np
import json
import threading
import warnings
import multiprocessing
import torch.multiprocessing as mp
import platform

from pyslam.config import Config, kDefaultConfigPath  # , dump_config_to_json

from pyslam.semantics.semantic_mapping_configs import SemanticMappingConfigs
from pyslam.semantics.semantic_eval import evaluate_semantic_mapping

from pyslam.slam.slam import Slam, SlamState
from pyslam.slam import PinholeCamera, USE_CPP
from pyslam.slam.playback_throttle import PlaybackThrottle

from pyslam.viz.slam_plot_drawer import SlamPlotDrawerThread
from pyslam.io.ground_truth import groundtruth_factory, is_valid_groundtruth, need_sim3_alignment
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.io.trajectory_writer import TrajectoryWriter

from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.logging import Printer, LoggerQueue
from pyslam.utilities.system import force_kill_all_and_exit
from pyslam.utilities.img_management import ImgWriter
from pyslam.utilities.evaluation import eval_ate
from pyslam.utilities.geom_trajectory import find_poses_associations
from pyslam.utilities.colors import GlColors
from pyslam.utilities.serialization import SerializableEnumEncoder
from pyslam.utilities.timer import TimerFps
from pyslam.viz.cvimage_thread import CvImageViewer
from pyslam.viz.qimage_thread import QimageViewer

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.local_features.feature_tracker import FeatureTrackerTypes
from pyslam.local_features.feature_types import FeatureDescriptorTypes

from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

from pyslam.depth_estimation.depth_estimator_factory import (
    depth_estimator_factory,
    DepthEstimatorType,
)
from pyslam.utilities.depth import img_from_depth, filter_shadow_points

from pyslam.config_parameters import Parameters

from datetime import datetime
import traceback

import argparse
import glob
import tempfile
import yaml


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.slam.camera import PinholeCamera


datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")


def draw_associated_cameras(viewer3D, assoc_est_poses, assoc_gt_poses, T_gt_est):
    T_est_gt = np.linalg.inv(T_gt_est)
    scale = np.mean([np.linalg.norm(T_est_gt[i, :3]) for i in range(3)])
    R_est_gt = T_est_gt[:3, :3] / scale  # we need a pure rotation to avoid camera scale changes
    assoc_gt_poses_aligned = [np.eye(4) for i in range(len(assoc_gt_poses))]
    for i, assoc_gt_pose in enumerate(assoc_gt_poses):
        assoc_gt_poses_aligned[i][:3, 3] = T_est_gt[:3, :3] @ assoc_gt_pose[:3, 3] + T_est_gt[:3, 3]
        assoc_gt_poses_aligned[i][:3, :3] = R_est_gt @ assoc_gt_pose[:3, :3]
    viewer3D.draw_cameras(
        [assoc_est_poses, assoc_gt_poses_aligned], [GlColors.kCyan, GlColors.kMagenta]
    )


def sequence_config_path(args, parser):
    """The configuration file for the sequence given on the command line (--video, --images or
    --tum, with --settings): a copy of the configuration file (config.yaml, or the one of -c) with
    its dataset replaced, written to a temporary file. Without such a sequence, the file of -c."""
    num_sequences = sum(bool(sequence) for sequence in (args.video, args.images, args.tum))
    if num_sequences == 0:
        if args.settings or args.groundtruth or args.timestamps:
            parser.error("--settings, --groundtruth and --timestamps go with --video, --images or --tum")
        return args.config_path
    if num_sequences > 1:
        parser.error("give only one of --video, --images and --tum")
    if not args.settings:
        parser.error("--video, --images and --tum need the camera settings file: --settings")
    settings_path = os.path.abspath(os.path.expanduser(args.settings))
    if not os.path.isfile(settings_path):
        sys.exit(f"--settings: no file {settings_path}")

    # the ground truth is optional for a video or an image folder (without it there is no trajectory
    # error); a file given with --groundtruth is in the folder of the sequence
    dataset = {"sensor_type": "mono", "settings": settings_path}
    if args.groundtruth:
        dataset["groundtruth_file"] = args.groundtruth
    if args.video:
        video_path = os.path.abspath(os.path.expanduser(args.video))
        if not os.path.isfile(video_path):
            sys.exit(f"--video: no file {video_path}")
        dataset_type = "VIDEO_DATASET"
        dataset.update(type="video", base_path=os.path.dirname(video_path), name=os.path.basename(video_path))
        if args.timestamps:
            dataset["timestamps"] = args.timestamps
    elif args.images:
        images_path = os.path.abspath(os.path.expanduser(args.images))
        if not glob.glob(os.path.join(images_path, args.pattern)):
            sys.exit(f"--images: no images {args.pattern} in {images_path} (see --pattern)")
        dataset_type = "FOLDER_DATASET"
        dataset.update(type="folder", base_path=images_path, name=args.pattern, fps=args.fps)
        if args.timestamps:
            dataset["timestamps"] = args.timestamps
    else:
        # a sequence in the layout of the TUM RGB-D datasets: the images, their timestamps and
        # their order come from the list of the frames, as for the TUM datasets themselves
        tum_path = os.path.abspath(os.path.expanduser(args.tum)).rstrip("/")
        associations = args.associations
        if associations is None:
            has_associations = os.path.exists(os.path.join(tum_path, "associations.txt"))
            associations = "associations.txt" if has_associations else "rgb.txt"
        if not os.path.isfile(os.path.join(tum_path, associations)):
            sys.exit(f"--tum: no {associations} in {tum_path}")
        groundtruth_file = args.groundtruth or "groundtruth.txt"
        if not os.path.isfile(os.path.join(tum_path, groundtruth_file)):
            sys.exit(
                f"--tum: no {groundtruth_file} in {tum_path} (TUM sequences need their ground truth). "
                f"For a sequence without ground truth, use its images: --images {os.path.join(args.tum, 'rgb')}"
            )
        sensor_type = args.sensor
        if sensor_type is None:  # rgbd if the list of the frames has depth images (4 columns)
            with open(os.path.join(tum_path, associations)) as f:
                lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith("#")]
            sensor_type = "rgbd" if lines and len(lines[0]) >= 4 else "mono"
        dataset_type = "TUM_DATASET"
        dataset.update(
            type="tum",
            sensor_type=sensor_type,
            base_path=os.path.dirname(tum_path),
            name=os.path.basename(tum_path),
            associations=associations,
            groundtruth_file=args.groundtruth or "auto",
        )

    with open(args.config_path or kDefaultConfigPath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["DATASET"]["type"] = dataset_type
    config[dataset_type] = dataset
    with tempfile.NamedTemporaryFile(
        "w", prefix="pyslam_config_", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
    return f.name


def draw_end_of_sequence_message(img, text):
    """A copy of `img` with `text` in a shaded strip across the bottom (on two lines if the image is
    too narrow for one)."""
    img = img.copy()
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    margin = 10
    lines = [text]
    if cv2.getTextSize(text, font, scale, thickness)[0][0] > img.shape[1] - 2 * margin:
        words = text.split()
        lines = [" ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2 :])]
    line_height = cv2.getTextSize(text, font, scale, thickness)[0][1] + margin
    strip_height = min(img.shape[0], len(lines) * line_height + margin)
    strip = img[-strip_height:]
    img[-strip_height:] = cv2.addWeighted(strip, 0.3, np.zeros_like(strip), 0.7, 0)
    for i, line in enumerate(lines):
        y = img.shape[0] - strip_height + (i + 1) * line_height
        cv2.putText(img, line, (margin, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pySLAM: SLAM on the dataset of config.yaml, or on a sequence of your own",
        epilog="examples: pixi run slam --features SUPERPOINT; "
        "pixi run slam --video my/video.mp4 --settings settings/MY_CAMERA.yaml; "
        "pixi run slam --images my/images --pattern '*.jpg' --fps 30 --settings settings/MY_CAMERA.yaml",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=None,
        help="Optional path for custom configuration file",
    )
    parser.add_argument(
        "--no_output_date",
        action="store_true",
        help="Do not append date to output directory",
    )
    sequence = parser.add_argument_group(
        "a sequence of your own (instead of the dataset of the configuration file)",
        "the same options as main_slam_evaluation.py; they need --settings",
    )
    sequence.add_argument("--video", help="a video file (monocular)")
    sequence.add_argument("--images", help="a folder of images, read in the order of their names")
    sequence.add_argument(
        "--tum",
        metavar="FOLDER",
        help="a sequence in the layout of the TUM RGB-D datasets, with its ground truth "
        "(groundtruth.txt); the frames come from associations.txt, or from rgb.txt (monocular)",
    )
    sequence.add_argument("--associations", help="with --tum: the list of the frames")
    sequence.add_argument("--sensor", choices=("mono", "rgbd"), help="with --tum: default rgbd if the list of the frames has depth images")
    sequence.add_argument("--pattern", default="*.png", help="with --images: which files (default '*.png')")
    sequence.add_argument("--fps", type=float, default=10, help="with --images: frames per second (default 10)")
    sequence.add_argument(
        "--settings",
        help="the camera settings file of the sequence (calibration, image size, frame rate): "
        "copy one of settings/, e.g. WEBCAM.yaml",
    )
    sequence.add_argument(
        "--groundtruth",
        help="the ground truth file, in the folder of the sequence; with --tum: default groundtruth.txt",
    )
    sequence.add_argument("--timestamps", help="with --video or --images: a file of the timestamps of the frames")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed relative to the camera's frame rate, with or without --headless: "
        "1 = the camera's rate (default), 2 = twice as fast, 0 = as fast as possible. "
        "Feeding frames faster than the camera leaves local mapping less time per frame, "
        "which can make tracking fail. This is the maximum speed: see --throttle.",
    )
    parser.add_argument(
        "--throttle",
        action="store_true",
        help="Slow the playback down below --speed when tracking gets weak because local mapping "
        "cannot keep up with the frames. Off by default: try it if tracking is lost on your machine.",
    )
    parser.add_argument(
        "--no-throttle",
        action="store_true",
        help="Never slow the playback down (the default, unless kPlaybackThrottle is set).",
    )
    parser.add_argument(
        "--features",
        default=None,
        metavar="NAME",
        help="The features: a FeatureTrackerConfigs entry, e.g. ORB2 (the default), ROOT_SIFT, "
        "SUPERPOINT or LIGHTGLUE (`pixi run feature-matching --list` lists them and what installs "
        "their models). It replaces FeatureTrackerConfig.name of the settings file. With features "
        "whose descriptors are not ORB, loop closing computes ORB descriptors of its own "
        "(DBOW3_INDEPENDENT), unless the settings file names a loop detector.",
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=None,
        metavar="N",
        help="The plots over the frames (# matches, chi2 error, timing) show the last N frames "
        f"(default: {Parameters.kPlotSlidingWindowNumFrames}; 0: the whole run). In a plot window, "
        "'+' widens the window, '-' narrows it and '0' shows the whole run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full camera and configuration dumps (JSON)",
    )
    args = parser.parse_args()

    config_path = sequence_config_path(args, parser)  # --video, --images or --tum, if given
    if config_path:
        config = Config(config_path)  # use the custom configuration path file
        if config_path != args.config_path:
            os.remove(config_path)  # the temporary file of sequence_config_path(): it has been read
    else:
        config = Config()

    if args.plot_window is not None:
        Parameters.kPlotSlidingWindowNumFrames = max(0, args.plot_window)

    if args.no_output_date:
        print("Not appending date to output directory")
        datetime_string = None

    dataset = dataset_factory(config)

    if Parameters.kUseDepthEstimatorInFrontEnd and dataset.sensor_type == SensorType.MONOCULAR:
        config.sensor_type = SensorType.RGBD
        dataset.sensor_type = SensorType.RGBD
        dataset.scale_viewer_3d = 0.5

    is_monocular = dataset.sensor_type == SensorType.MONOCULAR
    num_total_frames = dataset.num_frames

    online_trajectory_writer = None
    final_trajectory_writer = None
    trajectory_saving_base_path = None
    if config.trajectory_saving_settings["save_trajectory"]:
        (
            trajectory_online_file_path,
            trajectory_final_file_path,
            trajectory_saving_base_path,
        ) = config.get_trajectory_saving_paths(datetime_string)
        online_trajectory_writer = TrajectoryWriter(
            format_type=config.trajectory_saving_settings["format_type"],
            filename=trajectory_online_file_path,
        )
        final_trajectory_writer = TrajectoryWriter(
            format_type=config.trajectory_saving_settings["format_type"],
            filename=trajectory_final_file_path,
        )
    metrics_save_dir = trajectory_saving_base_path

    # cam_settings needed for CLIO fps fallback (Camera.fps) when no bag is present.
    groundtruth = groundtruth_factory(config.dataset_settings, cam_settings=config.cam_settings)
    has_groundtruth = is_valid_groundtruth(groundtruth)
    # Sim(3) alignment for monocular SLAM or non-metric GT (e.g. CLIO sparse COLMAP).
    eval_ate_correct_scale = is_monocular or (
        has_groundtruth and need_sim3_alignment(groundtruth)
    )

    camera = PinholeCamera(config)
    if args.verbose:
        Printer.green(f"Camera: {json.dumps(camera.to_json(), indent=4, cls=SerializableEnumEncoder)}")
    else:
        cam = camera.to_json()
        Printer.green(
            f"Camera: {cam.get('width')}x{cam.get('height')}, fx={cam.get('fx')} fy={cam.get('fy')} "
            f"cx={cam.get('cx')} cy={cam.get('cy')}, fps={cam.get('fps')}, "
            f"sensor={getattr(dataset.sensor_type, 'name', dataset.sensor_type)} (--verbose for details)"
        )

    # Select your tracker configuration (see the file feature_tracker_configs.py)
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_tracker_config = FeatureTrackerConfigs.ORB2

    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing.
    # LoopDetectorConfigs: DBOW2, DBOW2_INDEPENDENT, DBOW3, DBOW3_INDEPENDENT, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES, MEGALOC  etc.
    # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
    loop_detection_config = LoopDetectorConfigs.DBOW3

    # Select your semantic mapping configuration (see the file semantic_mapping_configs.py). Set it to None to disable semantic mapping.
    semantic_mapping_config = (
        SemanticMappingConfigs.get_config_from_slam_dataset(
            dataset.type, Parameters.kSemanticSegmentationType
        )
        if Parameters.kDoSparseSemanticMappingAndSegmentation
        else None
    )

    # Override the feature tracker and loop detector configuration from the `settings` file
    if (
        config.feature_tracker_config_name is not None
    ):  # Check if we set `FeatureTrackerConfig.name` in the `settings` file
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(
            config.feature_tracker_config_name
        )  # Override the feature tracker configuration from the `settings` file
    if args.features:  # the command line replaces the settings file
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(args.features)
        if feature_tracker_config is None:
            sys.exit(f"--features {args.features}: unknown; `pixi run feature-matching --list` lists the names")
        if feature_tracker_config.get("tracker_type") == FeatureTrackerTypes.LK:
            sys.exit(f"--features {args.features}: SLAM needs features with descriptors (the LK_* trackers have none)")
        # the vocabulary of the default loop detector (DBOW3) is made of ORB descriptors: with other
        # features, use the loop detector that computes ORB descriptors of its own
        if config.loop_detection_config_name is None and feature_tracker_config.get(
            "descriptor_type"
        ) not in (FeatureDescriptorTypes.ORB, FeatureDescriptorTypes.ORB2):
            loop_detection_config = LoopDetectorConfigs.DBOW3_INDEPENDENT
    if (
        config.num_features_to_extract > 0
    ):  # Check if we set `FeatureTrackerConfig.nFeatures` in the `settings` file
        Printer.yellow(
            "Setting feature_tracker_config num_features from settings: ",
            config.num_features_to_extract,
        )
        feature_tracker_config["num_features"] = (
            config.num_features_to_extract
        )  # Override the number of features from the `settings` file
    if (
        config.loop_detection_config_name is not None
    ):  # Check if we set `LoopDetectorConfig.name` in the `settings` file
        loop_detection_config = LoopDetectorConfigs.get_config_from_name(
            config.loop_detection_config_name
        )  # Override the loop detector configuration from the `settings` file
    if (
        config.semantic_mapping_config_name is not None
    ):  # Check if we set `SemanticMappingConfig.name` in the `settings` file. It is recommended to load semantics from the slam dataset name instead
        semantic_mapping_config = SemanticMappingConfigs.get_config_from_name(
            config.semantic_mapping_config_name
        )  # Override the semantic mapping configuration from the `settings` file

    if args.verbose:
        Printer.green(
            "feature_tracker_config: ",
            json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder),
        )
        Printer.green(
            "loop_detection_config: ",
            json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder),
        )
        if Parameters.kDoSparseSemanticMappingAndSegmentation:
            Printer.green(
                "semantic_mapping_config: ",
                json.dumps(semantic_mapping_config, indent=4, cls=SerializableEnumEncoder),
            )
    else:

        def _config_name(cfg, key):
            value = cfg.get(key) if isinstance(cfg, dict) else None
            return getattr(value, "name", value)

        loop_detector_name = (
            _config_name(loop_detection_config, "global_descriptor_type")
            if loop_detection_config is not None
            else "disabled"
        )
        Printer.green(
            f"Features: detector={_config_name(feature_tracker_config, 'detector_type')} "
            f"descriptor={_config_name(feature_tracker_config, 'descriptor_type')} "
            f"num_features={feature_tracker_config.get('num_features')}, "
            f"loop detector: {loop_detector_name}"
            + (
                f", semantic mapping: {_config_name(semantic_mapping_config, 'semantic_mapping_type') or 'on'}"
                if Parameters.kDoSparseSemanticMappingAndSegmentation
                else ""
            )
            + " (--verbose for details)"
        )
    config.feature_tracker_config = feature_tracker_config
    config.loop_detection_config = loop_detection_config
    config.semantic_mapping_config = semantic_mapping_config

    # Select your depth estimator in the front-end (EXPERIMENTAL, WIP)
    depth_estimator = None
    if Parameters.kUseDepthEstimatorInFrontEnd:
        Parameters.kVolumetricIntegrationUseDepthEstimator = False  # Just use this depth estimator in the front-end (This is not a choice, we are imposing it for avoiding computing the depth twice)
        # Select your depth estimator (see the file depth_estimator_factory.py)
        # DEPTH_ANYTHING_V2, DEPTH_ANYTHING_V3, DEPTH_PRO, DEPTH_RAFT_STEREO, DEPTH_SGBM, etc.
        depth_estimator_type = DepthEstimatorType.DEPTH_PRO
        max_depth = 20
        depth_estimator = depth_estimator_factory(
            depth_estimator_type=depth_estimator_type,
            max_depth=max_depth,
            dataset_env_type=dataset.environmentType(),
            camera=camera,
        )
        Printer.green(f"Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}")

    # create SLAM object
    slam = Slam(
        camera,
        feature_tracker_config,
        loop_detection_config,
        semantic_mapping_config,
        dataset.sensorType(),
        environment_type=dataset.environmentType(),
        config=config,
        headless=args.headless,
    )
    slam.set_viewer_scale(dataset.scale_viewer_3d)
    time.sleep(1)  # to show initial messages

    # load system state if requested
    if config.system_state_load:
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = (
            slam.viewer_scale() if slam.viewer_scale() > 0 else 0.1
        )  # 0.1 is the default viewer scale
        print(f"viewer_scale: {viewer_scale}")
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

    # create viewer3D, plot drawer, image writer, and cv image viewer
    if args.headless:
        viewer3D = None
        plot_drawer = None
        cv_image_viewer = None
    else:
        viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
        plot_drawer = SlamPlotDrawerThread(slam, viewer3D)
        img_writer = ImgWriter(font_scale=0.5)
        cv_image_viewer = CvImageViewer()
        if False:
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # to make it resizable if needed

    if viewer3D:
        print(f"Viewer3D scale: {viewer3D.scale}")

    gt_traj3d = None
    gt_poses = None
    gt_timestamps = None
    if has_groundtruth:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        if viewer3D:
            viewer3D.set_gt_trajectory(
                gt_traj3d, gt_timestamps, align_with_scale=eval_ate_correct_scale
            )

    if viewer3D:
        # wait for the viewer3D to be ready
        viewer3D.wait_for_ready()

    timer_main = TimerFps("Main", is_verbose=False)
    timer_main.start()

    do_step = False  # proceed step by step on GUI
    do_reset = False  # reset on GUI
    is_paused = False  # pause/resume on GUI

    # Playback speed: --speed is the maximum; the throttle reduces it when local mapping cannot keep up
    playback_throttle = PlaybackThrottle(
        max_speed=args.speed,
        enabled=(
            (args.throttle or Parameters.kPlaybackThrottle)
            and not args.no_throttle
            and Parameters.kLocalMappingOnSeparateThread
        ),
    )
    is_throttle_hint_shown = False  # the hint about --throttle when tracking is lost
    is_end_message_shown = False  # the message at the end of the sequence (with windows)
    img_draw = None  # the last image drawn in the Camera window
    is_map_save = False  # save map on GUI
    is_bundle_adjust = False  # bundle adjust on GUI
    is_viewer_closed = False  # viewer GUI was closed

    key = None
    key_cv = None

    num_tracking_lost = 0
    num_frames = 0

    img_id = 0  # 210, 340, 400, 770   # you can start from a desired frame id if needed

    try:
        while not is_viewer_closed:

            time_start = time.time()

            img, img_right, depth = None, None, None

            if do_step:
                Printer.orange("do step: ", do_step)

            if do_reset:
                Printer.yellow("do reset: ", do_reset)
                slam.reset()

            if not is_paused or do_step:

                if dataset.is_ok:
                    print("..................................")
                    img = dataset.getImageColor(img_id)
                    depth = dataset.getDepth(img_id)
                    img_right = (
                        dataset.getImageColorRight(img_id)
                        if dataset.sensor_type == SensorType.STEREO
                        else None
                    )

                if img is not None:
                    timestamp = dataset.getTimestamp()  # get current timestamp
                    next_timestamp = dataset.getNextTimestamp()  # get next timestamp
                    frame_duration = (
                        next_timestamp - timestamp
                        if (timestamp is not None and next_timestamp is not None)
                        else -1
                    )

                    print(f"image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}")

                    if img is not None:

                        if depth is None and depth_estimator:
                            depth_prediction, pts3d_prediction = depth_estimator.infer(
                                img, img_right
                            )
                            if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                                depth = filter_shadow_points(depth_prediction)
                            else:
                                depth = depth_prediction

                            if not args.headless:
                                depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                                # cv2.imshow("depth prediction", depth_img)
                                cv_image_viewer.draw(depth_img, "depth prediction")

                        slam.track(img, img_right, depth, img_id, timestamp)  # main SLAM function

                        # 3D display (map display)
                        if viewer3D:
                            viewer3D.draw_slam_map(slam)

                        if not args.headless:
                            is_draw_features_with_radius = viewer3D.is_draw_features_with_radius()
                            img_draw = slam.map.draw_feature_trails(
                                img,
                                with_level_radius=is_draw_features_with_radius,
                                trail_max_length=Parameters.kMaxFeatureTrailLength,
                            )
                            timer_main.refresh()
                            fps = timer_main.get_fps()
                            fps_text = f" fps: {fps:.1f}" if USE_CPP else ""
                            img_writer.write(img_draw, f"id: {img_id} {fps_text}", (20, 20))
                            # 2D display (image display)
                            # cv2.imshow("Camera", img_draw)
                            cv_image_viewer.draw(img_draw, "Camera")

                        # draw 2d plots
                        if plot_drawer:
                            plot_drawer.draw(img_id)

                    if (
                        online_trajectory_writer is not None
                        and slam.tracking.cur_R is not None
                        and slam.tracking.cur_t is not None
                    ):
                        online_trajectory_writer.write_trajectory(
                            slam.tracking.cur_R, slam.tracking.cur_t, timestamp
                        )

                    img_id += 1
                    num_frames += 1
                else:
                    time.sleep(0.1)  # img is None
                    # Printer.yellow("sleeping for 0.1 seconds - img is None")
                    if args.headless:
                        break  # exit from the loop if headless
                    if not is_end_message_shown and not dataset.is_ok:
                        is_end_message_shown = True
                        Printer.green(
                            "End of the sequence. The windows stay open: press 'q' or Esc in the "
                            "Camera window to quit and compute the trajectory error."
                        )
                        if img_draw is not None and cv_image_viewer:
                            # also in the Camera window, on a copy of its last image
                            cv_image_viewer.draw(
                                draw_end_of_sequence_message(
                                    img_draw,
                                    "End of the sequence: press q or Esc to quit and compute the trajectory error",
                                ),
                                "Camera",
                            )

            else:
                time.sleep(0.1)  # pause or do step on GUI
                # Printer.yellow("sleeping for 0.1 seconds - GUI paused")

            # 3D display (map display)
            if viewer3D:
                viewer3D.draw_dense_map(slam)

            if not args.headless:
                # get keys
                key = plot_drawer.get_key() if plot_drawer else None

                # manage SLAM states
                if slam.tracking.state == SlamState.LOST:
                    # key_cv = cv2.waitKey(0) & 0xFF   # wait key for debugging
                    # key_cv = cv2.waitKey(500) & 0xFF
                    key_cv = cv_image_viewer.get_key() if cv_image_viewer else None
                    time.sleep(0.1)
                else:
                    # key_cv = cv2.waitKey(1) & 0xFF
                    key_cv = cv_image_viewer.get_key() if cv_image_viewer else None

            # frames without a pose: tracking is lost, or relocalization is tried after a loss (when
            # it keeps failing, the rest of the sequence is in this state)
            if slam.tracking.state in (SlamState.LOST, SlamState.RELOCALIZE):
                num_tracking_lost += 1
                if (
                    not is_throttle_hint_shown
                    and not playback_throttle.enabled
                    and img is not None
                    and frame_duration > 0
                    and Parameters.kLocalMappingOnSeparateThread
                ):
                    is_throttle_hint_shown = True
                    Printer.yellow(
                        "Tracking is lost. If this happens at the same places in every run, the "
                        "machine may be too slow for the camera's frame rate: try --throttle (it "
                        "slows the playback down when tracking gets weak) or a lower --speed."
                    )

            # manage interface infos
            if is_map_save:
                slam.save_system_state(config.system_state_folder_path)
                dataset.save_info(config.system_state_folder_path)
                groundtruth.save(config.system_state_folder_path)
                Printer.blue("\nuncheck pause checkbox on GUI to continue...\n")

            if is_bundle_adjust:
                slam.bundle_adjust()
                Printer.blue("\nuncheck pause checkbox on GUI to continue...\n")

            if viewer3D:

                if not is_paused and viewer3D.is_paused():  # when a pause is triggered
                    est_poses, timestamps, ids = slam.get_final_trajectory()
                    if has_groundtruth:
                        assoc_timestamps, assoc_est_poses, assoc_gt_poses = find_poses_associations(
                            timestamps, est_poses, gt_timestamps, gt_poses
                        )
                        ape_stats, T_gt_est = eval_ate(
                            poses_est=assoc_est_poses,
                            poses_gt=assoc_gt_poses,
                            frame_ids=ids,
                            curr_frame_id=img_id,
                            is_final=False,
                            is_monocular=eval_ate_correct_scale,
                            save_dir=None,
                        )
                        Printer.green(f"EVO stats: {json.dumps(ape_stats, indent=4)}")
                        # draw_associated_cameras(viewer3D, assoc_est_poses, assoc_gt_poses, T_gt_est)
                    else:
                        Printer.yellow(
                            "Ground truth not available: skipping trajectory evaluation on pause"
                        )

                is_paused = viewer3D.is_paused()
                is_map_save = viewer3D.is_map_save() and is_map_save == False
                is_bundle_adjust = viewer3D.is_bundle_adjust() and is_bundle_adjust == False
                do_step = viewer3D.do_step() and do_step == False
                do_reset = viewer3D.reset() and do_reset == False
                is_viewer_closed = viewer3D.is_closed()

            # Keep the camera's frame period (divided by the playback speed), also in headless mode:
            # tracking alone runs several times faster than the camera, and feeding frames at that rate
            # starves local mapping (e.g. tracking is then lost at the turns of KITTI 06).
            # The speed is --speed (0 = no wait), reduced by the throttle when local mapping cannot
            # keep up with the frames.
            if img is not None and frame_duration > 0:
                if playback_throttle.update(slam.tracking.kf_demand):
                    Printer.yellow(
                        f"Playback speed: {playback_throttle.speed_str()} "
                        f"(tracking was weak in {100 * playback_throttle.last_fraction:.0f}% of the recent frames)"
                    )
                processing_duration = time.time() - time_start
                delta_time_sleep = (
                    playback_throttle.wait_time(frame_duration, processing_duration) - 1e-3
                )  # NOTE: 1e-3 is the cv wait time we use below with cv2.waitKey(1)
                if delta_time_sleep > 1e-3:
                    time.sleep(delta_time_sleep)
                playback_throttle.observe_frame(frame_duration, time.time() - time_start)
                    # Printer.yellow(f"sleeping for {delta_time_sleep} seconds - frame duration > processing duration")

            # press 'q' or ESC for quitting (the viewers' get_key() return the pressed key as a character)
            # also in loop closing's debug windows (similarity matrix, consistency checks)
            key_qimage = QimageViewer.get_instance().get_key() if QimageViewer.is_running() else None
            if key == "q" or key_cv in ("q", "\x1b") or key_qimage in ("q", "\x1b"):
                break

    except KeyboardInterrupt:
        Printer.yellow("\nCTRL+C detected. Shutting down ...\n")
        force_kill_all_and_exit(verbose=False)
        sys.exit(0)

    # exit from the main loop
    if not args.headless:
        Printer.green("pySLAM: shutting down (closing the windows, then saving the trajectory) ...")

    # Close the viewers first, all at once, so that the windows go as soon as the user has asked to
    # quit (they were closed one after the other at the very end, a few seconds later).
    viewers = [v for v in (cv_image_viewer, plot_drawer, viewer3D) if v]
    if QimageViewer.is_running():  # loop closing's debug window (similarity matrix, consistency checks)
        viewers.append(QimageViewer.get_instance())
    viewer_threads = [threading.Thread(target=v.quit, daemon=True) for v in viewers]
    for t in viewer_threads:
        t.start()
    for t in viewer_threads:
        t.join()

    # here we save the online estimated trajectory
    if online_trajectory_writer:
        online_trajectory_writer.close_file()

    # compute metrics on the estimated final trajectory
    try:
        est_poses, timestamps, ids = slam.get_final_trajectory()
        is_final = not dataset.is_ok
        if has_groundtruth:
            assoc_timestamps, assoc_est_poses, assoc_gt_poses = find_poses_associations(
                timestamps, est_poses, gt_timestamps, gt_poses
            )
            ape_stats, T_gt_est = eval_ate(
                poses_est=assoc_est_poses,
                poses_gt=assoc_gt_poses,
                frame_ids=ids,
                curr_frame_id=img_id,
                is_final=is_final,
                is_monocular=eval_ate_correct_scale,
                save_dir=metrics_save_dir,
            )
            Printer.green(f"EVO stats: {json.dumps(ape_stats, indent=4)}")
        else:
            Printer.yellow("Ground truth not available: skipping trajectory evaluation")

        if final_trajectory_writer:
            final_trajectory_writer.write_full_trajectory(est_poses, timestamps)
            final_trajectory_writer.close_file()

        other_metrics_file_path = os.path.join(metrics_save_dir, "other_metrics_info.txt")
        with open(other_metrics_file_path, "w") as f:
            f.write(f"num_total_frames: {num_total_frames}\n")
            f.write(f"num_processed_frames: {num_frames}\n")
            f.write(f"num_lost_frames: {num_tracking_lost}\n")
            f.write(f"percent_lost: {num_tracking_lost/num_total_frames*100:.2f}\n")
            # the frames with a pose in the final trajectory: the trajectory errors are over these only
            f.write(f"num_tracked_frames: {len(est_poses)}\n")
            f.write(f"percent_tracked: {len(est_poses)/num_total_frames*100:.2f}\n")
            f.write(f"playback_max_speed: {playback_throttle.speed_str(playback_throttle.max_speed)}\n")
            f.write(f"playback_lowest_speed: {playback_throttle.speed_str(playback_throttle.lowest_speed)}\n")
            f.write(f"playback_num_slowdowns: {playback_throttle.num_decreases}\n")
            f.write(
                f"percent_weak_tracking_frames: {slam.tracking.kf_demand.total_weak_fraction()*100:.2f}\n"
            )
            f.write(
                f"percent_suppressed_keyframe_requests: {slam.tracking.kf_demand.total_suppressed_fraction()*100:.2f}\n"
            )

        evaluate_semantic_mapping(slam, dataset, metrics_save_dir)

    except Exception as e:
        print("Exception while computing metrics: ", e)
        print(f"traceback: {traceback.format_exc()}")

    Printer.green(
        f"Playback: max speed {playback_throttle.speed_str(playback_throttle.max_speed)}, "
        f"lowest speed {playback_throttle.speed_str(playback_throttle.lowest_speed)}, "
        f"{playback_throttle.num_decreases} slow-downs; "
        f"tracking was weak in {slam.tracking.kf_demand.total_weak_fraction()*100:.0f}% of the frames; "
        f"{slam.tracking.kf_demand.total_suppressed_fraction()*100:.0f}% of the keyframe requests "
        f"found local mapping busy",
        flush=True,
    )

    # Stop SLAM (which stops all processes and shuts down their managers); the viewers are closed above
    slam.quit()

    # Explicitly stop all LoggerQueue instances to prevent shutdown errors
    LoggerQueue.stop_all_instances()

    # Give viewers and logger queues time to clean up
    time.sleep(1.0)

    if args.headless:
        force_kill_all_and_exit(verbose=False)  # just in case when running an evaluation
    else:
        if platform.system() == "Darwin" or mp.get_start_method() == "spawn":
            # HACK: wait (up to 5 s) for the child processes to exit, then kill any that are left
            deadline = time.time() + 5.0
            while multiprocessing.active_children() and time.time() < deadline:
                time.sleep(0.1)
            force_kill_all_and_exit(verbose=True)  # debug
