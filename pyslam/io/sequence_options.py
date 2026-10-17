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

# The command line options which name a sequence of your own (--video, --images or --tum, with
# --settings), shared by main_slam.py and main_semantic_image_segmentation.py. The options are
# turned into a configuration file for dataset_factory(): a copy of config.yaml (or of the file of
# -c) with its dataset replaced.

import glob
import os
import sys
import tempfile

import yaml

from pyslam.config import kDefaultConfigPath


def add_sequence_options(parser, need_settings=True):
    """Add the sequence options to an argparse parser, as an argument group."""
    sequence = parser.add_argument_group(
        "a sequence of your own (instead of the dataset of the configuration file)",
        "the same options as main_slam.py and main_slam_evaluation.py"
        + ("; they need --settings" if need_settings else ""),
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
    sequence.add_argument(
        "--sensor",
        choices=("mono", "rgbd"),
        help="with --tum: default rgbd if the list of the frames has depth images",
    )
    sequence.add_argument("--pattern", default="*.png", help="with --images: which files (default '*.png')")
    sequence.add_argument(
        "--fps", type=float, default=10, help="with --images: frames per second (default 10)"
    )
    sequence.add_argument(
        "--settings",
        help="the camera settings file of the sequence (calibration, image size, frame rate): "
        "copy one of settings/, e.g. WEBCAM.yaml"
        + ("" if need_settings else "; not needed here (the images are used as they are)"),
    )
    sequence.add_argument(
        "--groundtruth",
        help="the ground truth file, in the folder of the sequence; with --tum: default groundtruth.txt",
    )
    sequence.add_argument(
        "--timestamps", help="with --video or --images: a file of the timestamps of the frames"
    )
    return sequence


def has_sequence_options(args):
    """Whether the command line names a sequence of its own."""
    return bool(args.video or args.images or args.tum)


def sequence_config_path(args, parser, default_settings=None):
    """The configuration file for the sequence given on the command line (--video, --images or
    --tum, with --settings): a copy of the configuration file (config.yaml, or the one of -c) with
    its dataset replaced, written to a temporary file. Without such a sequence, the file of -c.
    `default_settings` is the settings file used when --settings is not given (a script which does
    not use the camera calibration); without it, --settings is required."""
    num_sequences = sum(bool(sequence) for sequence in (args.video, args.images, args.tum))
    if num_sequences == 0:
        if args.settings or args.groundtruth or args.timestamps:
            parser.error("--settings, --groundtruth and --timestamps go with --video, --images or --tum")
        return getattr(args, "config_path", None)
    if num_sequences > 1:
        parser.error("give only one of --video, --images and --tum")
    settings = args.settings or default_settings
    if not settings:
        parser.error("--video, --images and --tum need the camera settings file: --settings")
    settings_path = os.path.abspath(os.path.expanduser(settings))
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
        from pyslam.utilities.text_files import open_text

        with open_text(os.path.join(tum_path, associations)) as f:
            lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith("#")]
        has_depth = bool(lines) and len(lines[0]) >= 4
        sensor_type = args.sensor
        if sensor_type is None:  # rgbd if the list of the frames has depth images (4 columns)
            sensor_type = "rgbd" if has_depth else "mono"
        elif sensor_type == "rgbd" and not has_depth:
            sys.exit(
                f"--sensor rgbd: {associations} in {tum_path} lists no depth images. Pair the colour and depth "
                f"images first: pixi run python scripts/associate.py {os.path.join(args.tum, 'rgb.txt')} "
                f"{os.path.join(args.tum, 'depth.txt')} > {os.path.join(args.tum, 'associations.txt')}"
            )
        dataset_type = "TUM_DATASET"
        dataset.update(
            type="tum",
            sensor_type=sensor_type,
            base_path=os.path.dirname(tum_path),
            name=os.path.basename(tum_path),
            associations=associations,
            groundtruth_file=args.groundtruth or "auto",
        )

    with open(getattr(args, "config_path", None) or kDefaultConfigPath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["DATASET"]["type"] = dataset_type
    config[dataset_type] = dataset
    with tempfile.NamedTemporaryFile(
        "w", prefix="pyslam_config_", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
    return f.name
