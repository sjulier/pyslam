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

# The depth of one image, or of every frame of a sequence, predicted by one of pySLAM's depth
# networks: the depth of every pixel in metres, the pixels lifted into 3D with the camera's
# calibration, and a one-line summary (the range of the depth and the time per image).
#
# On a sequence in the layout of the TUM RGB-D datasets whose list of frames (associations.txt) has
# the camera's depth images, every prediction is also scored against them. --save NAME writes the
# predicted depth maps into the folder of the sequence, in the format of the camera's depth images,
# with a list of the frames for main_slam.py --associations (SLAM with the predicted depth). --load
# NAME scores maps saved earlier without running a network, and --keypoints reads the depth at the
# ORB keypoints with depth_at_keypoints() (pyslam/depth_estimation/depth_sampling.py).
#
#   pixi run -e depth depth-prediction --list
#   pixi run -e depth depth-prediction --method DEPTH_ANYTHING_V2 --test kitti
#   pixi run -e depth depth-prediction --method DEPTH_PRO --image photo.jpg
#   pixi run -e depth depth-prediction --method DEPTH_ANYTHING_V2 --tum $S --settings settings/TUM1.yaml --save dav2
#   pixi run -e depth depth-prediction --tum $S --settings settings/TUM1.yaml --load dav2 --keypoints nearest --step 5

import pyslam  # first: sets the OpenMP thread settings before numpy/torch are imported
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)

# The methods (DepthEstimatorType entries): what each gives
kMethods = {
    "DEPTH_ANYTHING_V2": "depth in metres from one image: Depth Anything V2, metric version, with one model for "
    "indoor scenes (fine-tuned on Hypersim) and one for outdoor scenes (Virtual KITTI), each in three sizes "
    "(--size). It does not use the focal length",
    "DEPTH_ANYTHING_V3": "depth in metres from one image: Depth Anything 3, metric model (DA3METRIC-LARGE), "
    "whose output is scaled with the camera's focal length",
    "DEPTH_PRO": "depth in metres from one image: Depth Pro. It uses the camera's focal length, or its own "
    "estimate of it when there is no settings file",
    "DEPTH_SGBM": "depth from a stereo pair: OpenCV's semi-global block matching (here only --test kitti, "
    "which has the right image)",
    "DEPTH_RAFT_STEREO": "depth from a stereo pair: RAFT-Stereo (here only --test kitti)",
    "DEPTH_CRESTEREO_PYTORCH": "depth from a stereo pair: CREStereo (here only --test kitti)",
}
kStereo = ("DEPTH_SGBM", "DEPTH_RAFT_STEREO", "DEPTH_CRESTEREO_PYTORCH")
# A module of the depth level which each method needs (checked before anything is imported)
kNeeds = {"DEPTH_ANYTHING_V3": "omegaconf", "DEPTH_PRO": "pillow_heif", "DEPTH_RAFT_STEREO": "opt_einsum"}
# What `pixi run -e depth models-depth` installs for each method ({dataset} and {encoder}: DA-V2's checkpoint)
kInstalled = {
    "DEPTH_ANYTHING_V2": "thirdparty/depth_anything_v2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth",
    "DEPTH_ANYTHING_V3": "thirdparty/depth_anything_v3",
    "DEPTH_PRO": "thirdparty/ml_depth_pro/checkpoints/depth_pro.pt",
    "DEPTH_RAFT_STEREO": "thirdparty/raft_stereo/models",
    "DEPTH_CRESTEREO_PYTORCH": "thirdparty/crestereo_pytorch/models/crestereo_eth3d.pth",
}
kSizes = {"small": "vits", "base": "vitb", "large": "vitl"}  # --size: DA-V2's encoders

# The images of --test (the same as main_semantic_image_segmentation.py's)
kTests = {
    "kitti": "test/data/kitti06-12-color.png",  # frame 12 of the KITTI 06 video
    "kitti_next": "test/data/kitti06-17-color.png",  # half a second later
    "desk": "data/images/desk_detic.jpeg",
    "churchill": "test/data/churchill/1.ppm",
    "box": "test/data/box_in_scene.png",
    "mars": "test/data/mars1.png",
}
kTestSettings = {"kitti": "settings/KITTI04-12.yaml", "kitti_next": "settings/KITTI04-12.yaml"}
kTestRight = {"kitti": "test/data/kitti06-12-R-color.png"}  # the right image of the stereo pair
kIndoorTests = ("desk", "box")  # DA-V2's indoor model for these, the outdoor model for the others

kAssumedFov = 60.0  # degrees: the horizontal field of view assumed for an image without a settings file
kMaxImageSide = 1600  # larger images (photographs) are reduced to this, to keep the 3D view responsive
kErrorSaturation = 1.5  # Error window: full red at 1.5 times too far, full blue at 1.5 times too near
kMinScoredPixels = 100  # a frame is scored if the camera and the network both have a depth on this many pixels


def main():
    from pyslam.io.sequence_options import add_sequence_options, has_sequence_options

    parser = argparse.ArgumentParser(
        description="The depth of one image, or of every frame of a sequence, predicted by one of pySLAM's "
        "depth networks; scored against the camera's depth when a TUM sequence has it",
        epilog="examples: pixi run -e depth depth-prediction --method DEPTH_ANYTHING_V2 --test kitti; "
        "pixi run -e depth depth-prediction --method DEPTH_ANYTHING_V2 --tum $S --settings settings/TUM1.yaml "
        "--save dav2; pixi run -e depth depth-prediction --tum $S --settings settings/TUM1.yaml --load dav2 "
        "--keypoints nearest --step 5",
    )
    parser.add_argument(
        "--method", default="DEPTH_ANYTHING_V2", metavar="NAME", help="the depth network (default DEPTH_ANYTHING_V2; see --list)"
    )
    parser.add_argument("--test", default="kitti", choices=sorted(kTests), help="the image (default kitti)")
    parser.add_argument("--image", metavar="FILE", help="an image file of your own, instead of --test")
    parser.add_argument(
        "--size", choices=list(kSizes), help="DEPTH_ANYTHING_V2: the size of the model (default large; small and base are faster and less accurate)"
    )
    parser.add_argument(
        "--scene",
        choices=("indoor", "outdoor"),
        help="DEPTH_ANYTHING_V2: the indoor or the outdoor model. Default: indoor for --test desk and box, a "
        "photograph (--image) and a TUM sequence; outdoor for the other test images, a video and a folder of images",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="with one image: how many times to repeat it for the timing (default 3)"
    )
    add_sequence_options(parser, need_settings=True)
    sequence = parser.add_argument_group("with a sequence")
    sequence.add_argument(
        "--save",
        metavar="NAME",
        help="with --tum: write the predicted depth maps into the folder of the sequence: depth_NAME/ (one PNG per "
        "frame, in the format of the camera's depth images, and scores.csv) and associations_NAME.txt (the list "
        "of the frames, for main_slam.py --associations)",
    )
    sequence.add_argument(
        "--load", metavar="NAME", help="with --tum: score the depth maps saved with --save NAME instead of running a network"
    )
    sequence.add_argument(
        "--keypoints",
        choices=("nearest", "bilinear", "median"),
        help="with --tum and the camera's depth: detect ORB keypoints in each frame as SLAM does, read the predicted "
        "and the camera's depth there with depth_at_keypoints() (pyslam/depth_estimation/depth_sampling.py), and score them",
    )
    sequence.add_argument("--patch", type=int, default=5, metavar="N", help="with --keypoints median: the size of the window (default 5)")
    sequence.add_argument("--step", type=int, default=1, metavar="N", help="predict every Nth frame (default 1: every frame)")
    sequence.add_argument("--max-frames", type=int, metavar="N", help="stop after N predicted frames")
    sequence.add_argument("--headless", action="store_true", help="no windows")
    parser.add_argument("--list", action="store_true", help="list the methods")
    args = parser.parse_args()

    if args.list:
        print("Methods (--method), and what each gives:")
        for name, what in kMethods.items():
            print(f"    {name:24s}{what}")
        print("The networks need the depth level: pixi run -e depth depth-prediction ... "
              "(pixi run -e depth models-depth installs their code and weights). --load needs no network.")
        sys.exit(0)

    sequence_given = has_sequence_options(args)
    if args.image and sequence_given:
        parser.error("give --image or a sequence, not both")
    for option in ("save", "load"):
        if getattr(args, option) and not args.tum:
            parser.error(f"--{option} writes and reads inside the folder of a TUM sequence: it goes with --tum")
    if args.save and args.load:
        parser.error("--load scores maps saved earlier; it does not save them again: give --save or --load")
    if args.keypoints and not args.tum:
        parser.error("--keypoints needs the camera's depth: it goes with --tum")
    for option in ("save", "load"):
        name = getattr(args, option)
        if name and (os.sep in name or "/" in name or name.startswith(".")):
            parser.error(f"--{option}: give a name, not a path (the maps go to depth_NAME/ in the folder of the sequence)")
    if args.step < 1:
        parser.error("--step must be 1 or more")
    if args.patch < 1 or args.patch % 2 == 0:
        parser.error("--patch must be an odd number")
    if not sequence_given and (args.groundtruth or args.timestamps):
        parser.error("--groundtruth and --timestamps go with --video, --images or --tum")
    if not args.load:
        check_method(args)

    from pyslam.utilities.logging import Printer

    if sequence_given:
        if not args.settings:
            parser.error("--video, --images and --tum need the camera settings file: --settings")
        run_sequence(args, parser, Printer)
    else:
        run_image(args, Printer)


def check_method(args):
    """Stop with one line if the method is unknown, needs another level, or is not installed."""
    if args.method not in kMethods:
        print(f"Unknown method '{args.method}'. Valid names (see --list):", file=sys.stderr)
        print("    " + " ".join(kMethods), file=sys.stderr)
        sys.exit(2)
    if args.method in kNeeds and importlib.util.find_spec(kNeeds[args.method]) is None:
        sys.exit(f"{args.method} needs the depth level: run this command with `pixi run -e depth depth-prediction ...`.")
    if args.method in kStereo and (args.image or args.video or args.images or args.tum or args.test not in kTestRight):
        sys.exit(f"{args.method} needs a stereo pair: here only --test kitti has one (its right image).")
    if args.size and args.method != "DEPTH_ANYTHING_V2":
        print(f"--size is for DEPTH_ANYTHING_V2 only: {args.method} has one size.")
    if args.method not in kInstalled:
        return  # nothing to install (OpenCV's SGBM)
    if args.method == "DEPTH_ANYTHING_V2":
        # either checkpoint will do here: the scene is decided later
        paths = [kInstalled[args.method].format(dataset=d, encoder=kSizes[args.size or "large"]) for d in ("hypersim", "vkitti")]
    else:
        paths = [kInstalled[args.method]]
    if not any(os.path.exists(os.path.join(kScriptFolder, p)) for p in paths):
        sys.exit(f"{args.method} is not installed ({paths[0]} is missing): run `pixi run -e depth models-depth`.")


class Camera:
    """The calibration which the networks, the 3D view and the depth images need: fx, fy, cx, cy in
    pixels, the image size, the stereo baseline times fx (bf), the factor of the depth images
    (DepthMapFactor) and the number of ORB features, from a settings file (or assumed)."""

    def __init__(self, fx, fy, cx, cy, width, height, bf=0.0, depth_map_factor=None, num_features=None, source=""):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)
        self.bf = float(bf)
        self.b = self.bf / self.fx if self.fx else 0.0
        self.depth_map_factor = depth_map_factor
        self.num_features = num_features
        self.source = source  # where the calibration comes from, for the summary

    @staticmethod
    def from_settings(path):
        from pyslam.config import load_settings_yaml

        full_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(full_path):
            full_path = os.path.join(kScriptFolder, path)
        if not os.path.isfile(full_path):
            sys.exit(f"--settings: no file {path}")
        s = load_settings_yaml(full_path)
        if not s or "Camera.fx" not in s:
            sys.exit(f"--settings: {path} has no camera calibration (Camera.fx, ...)")
        fx, fy = float(s["Camera.fx"]), float(s.get("Camera.fy", s["Camera.fx"]))
        return Camera(
            fx, fy, s.get("Camera.cx", 0.0), s.get("Camera.cy", 0.0), s.get("Camera.width", 0), s.get("Camera.height", 0),
            bf=s.get("Camera.bf", 0.0),
            depth_map_factor=float(s["DepthMapFactor"]) if "DepthMapFactor" in s else None,
            num_features=int(s["FeatureTrackerConfig.nFeatures"]) if "FeatureTrackerConfig.nFeatures" in s else None,
            source=os.path.basename(path),
        )

    @staticmethod
    def assumed(width, height, fx=None, source=None):
        """A camera with its principal point in the middle of the image, and the given focal length,
        or the one of a kAssumedFov field of view."""
        if fx is None:
            fx = 0.5 * width / np.tan(np.radians(kAssumedFov / 2))
            source = source or f"assumed: a {kAssumedFov:.0f} degree field of view, as there are no settings"
        return Camera(fx, fx, width / 2, height / 2, width, height, source=source or "")

    def for_image(self, width, height):
        """The calibration for an image of this size: scaled if the settings are for another size."""
        if not self.width or not self.height or (self.width, self.height) == (width, height):
            return self
        sx, sy = width / self.width, height / self.height
        print(f"The settings are for {self.width}x{self.height} pixels and the image has {width}x{height}: the calibration is scaled to the image")
        return Camera(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height, self.bf * sx,
                      self.depth_map_factor, self.num_features, f"{self.source}, scaled to {width}x{height}")


class Network:
    """One of pySLAM's depth networks (a DepthEstimator), with what the summary says about it."""

    def __init__(self, method, size, indoor, camera):
        import torch
        from pyslam.io.dataset_types import DatasetEnvironmentType

        self.method = method
        self.size = size or "large"
        self.indoor = indoor
        self.camera = camera
        env = DatasetEnvironmentType.INDOOR if indoor else DatasetEnvironmentType.OUTDOOR
        max_depth = 20 if indoor else 80  # the range of the display and clipping (DA-V2's scale does not depend on it)
        start = time.time()
        if method == "DEPTH_ANYTHING_V2":
            from pyslam.depth_estimation.depth_estimator_depth_anything_v2 import DepthEstimatorDepthAnythingV2

            encoder = kSizes[self.size]
            path = os.path.join(kScriptFolder, kInstalled[method].format(dataset="hypersim" if indoor else "vkitti", encoder=encoder))
            if not os.path.isfile(path):
                sys.exit(f"DEPTH_ANYTHING_V2: the {'indoor' if indoor else 'outdoor'} {self.size} model is not installed "
                         f"({os.path.relpath(path, kScriptFolder)}): run `pixi run -e depth models-depth`.")
            self.estimator = DepthEstimatorDepthAnythingV2(camera=camera, max_depth=max_depth, dataset_env_type=env, encoder_name=encoder)
        else:
            from pyslam.depth_estimation.depth_estimator_factory import depth_estimator_factory, DepthEstimatorType

            self.estimator = depth_estimator_factory(
                depth_estimator_type=DepthEstimatorType[method], camera=camera, max_depth=max_depth,
                dataset_env_type=env, precision=torch.float16,
            )
        self.load_time = time.time() - start
        device = getattr(self.estimator, "device", None)
        self.device = getattr(device, "type", None) or (str(device) if device is not None else "cpu")

    def predict(self, img, img_right=None):
        """The depth map of img, in metres, with 0 where the network gives no depth."""
        depth, _ = self.estimator.infer(img, img_right)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape[:2] != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth[~np.isfinite(depth) | (depth < 0)] = 0
        return depth

    def focal_length(self):
        """The focal length the network used, or estimated (Depth Pro), in pixels; None if it uses none."""
        if self.method == "DEPTH_PRO":
            return getattr(self.estimator, "focal_length_px", None)
        if self.method == "DEPTH_ANYTHING_V3" and self.camera is not None:
            return 0.5 * (self.camera.fx + self.camera.fy)
        return None

    def describe(self):
        """What the summary line says about the network, e.g. 'metric, indoor model, large'."""
        if self.method == "DEPTH_ANYTHING_V2":
            return f"metric, {'indoor' if self.indoor else 'outdoor'} model, {self.size}"
        if self.method == "DEPTH_ANYTHING_V3":
            return f"metric, focal length {self.focal_length():.0f} px ({self.camera.source})"
        if self.method == "DEPTH_PRO":
            if self.camera is not None:
                return f"metric, focal length {self.camera.fx:.0f} px ({self.camera.source})"
            return "metric, its own estimate of the focal length"
        return f"stereo, baseline {self.camera.b:.3f} m ({self.camera.source})"


# ---------------------------------------------------------------------------------------------------
# Scores: the predicted depth d against the camera's depth d*, on the pixels where both have one
# ---------------------------------------------------------------------------------------------------


def score(pred, gt):
    """AbsRel and delta<1.25 as the network gives the depth (metric) and after scaling it by its median
    scale median(d*/d) (shape only), and the median scale; None if too few pixels have both depths."""
    both = (gt > 0) & (pred > 0)
    if np.count_nonzero(both) < kMinScoredPixels:
        return None
    d, g = pred[both].astype(np.float64), gt[both].astype(np.float64)
    scale = float(np.median(g / d))
    out = {"median_scale": scale}
    for kind, dd in (("metric", d), ("shape", d * scale)):
        out[f"absrel_{kind}"] = float(np.mean(np.abs(dd - g) / g))
        out[f"delta_{kind}"] = float(np.mean(np.maximum(dd / g, g / dd) < 1.25))
    return out


class KeypointScorer:
    """--keypoints: ORB keypoints detected as SLAM detects them (the settings' number of features), the
    depth read there from the prediction and from the camera with depth_at_keypoints(), and AbsRel at
    the keypoints which have both."""

    def __init__(self, method, patch, num_features):
        import pyslam.config  # noqa: F401  (first: it sets the paths of the native modules the features need)
        from pyslam.local_features.feature_tracker import feature_tracker_factory
        from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
        from pyslam.depth_estimation.depth_sampling import depth_at_keypoints

        config = dict(FeatureTrackerConfigs.ORB2)
        config["num_features"] = num_features
        self.tracker = feature_tracker_factory(**config)
        self.depth_at_keypoints = depth_at_keypoints
        self.method, self.patch, self.num_features = method, patch, num_features
        self.enabled = True

    def describe(self):
        return f"{self.method}" + (f", patch {self.patch}" if self.method == "median" else "")

    def run(self, img, pred, gt, scale, Printer):
        """The keypoints (N x 2), their predicted and camera depths, and the scores; None if
        depth_at_keypoints() gives nothing."""
        if not self.enabled:
            return None
        kps, _ = self.tracker.detectAndCompute(img)  # the colour image, as SLAM gives it
        uv = np.array([kp.pt for kp in kps], dtype=np.float64).reshape(-1, 2)
        dp = self.depth_at_keypoints(pred, uv, method=self.method, patch=self.patch)
        dg = self.depth_at_keypoints(gt, uv, method=self.method, patch=self.patch)
        if dp is None or dg is None:
            Printer.orange(
                f"depth_at_keypoints() returned None for --keypoints {self.method}: the keypoints are skipped "
                "(pyslam/depth_estimation/depth_sampling.py, Lab 03, Activity 6)."
            )
            self.enabled = False
            return None
        dp, dg = np.asarray(dp, dtype=np.float64).reshape(-1), np.asarray(dg, dtype=np.float64).reshape(-1)
        if dp.shape[0] != len(uv) or dg.shape[0] != len(uv):
            Printer.red(
                f"depth_at_keypoints() returned {dp.shape[0]} depths for {len(uv)} keypoints: it must return one depth "
                "per keypoint. The keypoints are skipped."
            )
            self.enabled = False
            return None
        both = np.isfinite(dp) & np.isfinite(dg) & (dp > 0) & (dg > 0)
        out = {"keypoints": len(uv), "keypoints_with_depth": int(np.count_nonzero(both))}
        if np.any(both):
            out["kp_absrel_metric"] = float(np.mean(np.abs(dp[both] - dg[both]) / dg[both]))
            out["kp_absrel_shape"] = float(np.mean(np.abs(dp[both] * scale - dg[both]) / dg[both]))
        return uv, dp, dg, out


# ---------------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------------


def colour_depth(depth, near, far):
    """Near (yellow) to far (dark blue), black where there is no depth."""
    t = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    img = cv2.applyColorMap(np.round((1 - t) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[~(depth > 0)] = 0
    return img


def depth_bar(height, near, far, width=90):
    """A vertical scale bar for colour_depth(), in metres, far at the top."""
    bar = np.full((height, width, 3), 255, np.uint8)
    top, bottom = 12, height - 12
    ramp = np.linspace(far, near, max(bottom - top, 2), dtype=np.float32)[:, None]
    bar[top:bottom, 6:26] = colour_depth(np.repeat(ramp, 20, axis=1), near, far)
    for i in range(5):
        value = far - (far - near) * i / 4
        y = int(round(top + (bottom - top - 1) * i / 4))
        cv2.line(bar, (26, y), (31, y), (0, 0, 0), 1)
        cv2.putText(bar, f"{value:.1f} m", (34, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def error_colours(ratio):
    """BGR colours for ratios d/d*: white when right, red when too far, blue when too near."""
    t = np.clip(np.abs(np.log(ratio)) / np.log(kErrorSaturation), 0, 1)[..., None]
    fade = (255 * (1 - t)).astype(np.uint8)
    full = np.full_like(fade, 255)
    too_far = (ratio > 1)[..., None]
    return np.where(too_far, np.concatenate([fade, fade, full], -1), np.concatenate([full, fade, fade], -1))


def colour_error(pred, gt, scale):
    """Red where the prediction (times scale) is too far, blue where it is too near, white where it is
    right; black where the camera has no depth, grey where the network has none."""
    img = np.zeros(gt.shape + (3,), np.uint8)
    has_gt = gt > 0
    both = has_gt & (pred > 0)
    img[has_gt & ~both] = 128
    img[both] = error_colours(pred[both] * scale / gt[both])
    return img


def error_legend(width, label):
    strip = np.full((44, width, 3), 255, np.uint8)
    x0, x1 = 10, max(width - 10, 12)
    ratios = np.exp(np.linspace(-np.log(kErrorSaturation), np.log(kErrorSaturation), x1 - x0))
    strip[6:20, x0:x1] = error_colours(np.repeat(ratios[None, :], 14, axis=0))
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    cv2.putText(strip, f"too near (/{kErrorSaturation:g})", (x0, 36), font, size, (0, 0, 0), 1, cv2.LINE_AA)
    text = f"too far (x{kErrorSaturation:g})"
    cv2.putText(strip, text, (x1 - cv2.getTextSize(text, font, size, 1)[0][0], 36), font, size, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(strip, label, ((width - cv2.getTextSize(label, font, size, 1)[0][0]) // 2, 36), font, size, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


class Windows:
    """The windows Image, Depth (the depth under the mouse in its title), Error (with the camera's depth)
    and 3D (the pixels lifted into 3D)."""

    def __init__(self, indoor):
        from pyslam.viz.viewer3D import Viewer3D

        self.pred, self.gt, self.title = None, None, ""
        self.metric_error = False  # m: the Error window shows the depth as given, or times the frame's median scale
        self.near = self.far = None
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Depth", self._on_mouse)
        self.viewer3D = Viewer3D(scale=0.1 if indoor else 1.0, title="3D")

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE or self.pred is None:
            return
        h, w = self.pred.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        d = self.pred[y, x]
        text = f"Depth: {d:.2f} m at ({x}, {y})" if d > 0 else f"Depth: none at ({x}, {y})"
        if self.gt is not None:
            g = self.gt[y, x]
            text += f"; the camera: {g:.2f} m" if g > 0 else "; the camera: none"
        cv2.setWindowTitle("Depth", f"{text} ({self.title})")

    def show(self, img, pred, camera, title, gt=None, scale=None, keypoints=None):
        self.pred, self.gt, self.title = pred, gt, title
        if self.near is None:  # the colours of the first image, kept for the whole sequence
            valid = pred[pred > 0]
            self.near, self.far = (float(np.percentile(valid, 1)), float(np.percentile(valid, 99))) if valid.size else (0.0, 1.0)
            if self.far <= self.near:
                self.far = self.near + 1.0
        shown = img.copy()
        if keypoints is not None:
            uv, dp, dg = keypoints
            both = (dp > 0) & (dg > 0)
            colours = np.full((len(uv), 3), 128, np.uint8)  # grey: no depth from the network or the camera
            if np.any(both):
                colours[both] = error_colours(dp[both] * (1.0 if self.metric_error else scale) / dg[both])
            for (u, v), c in zip(uv, colours):
                centre = (int(round(u)), int(round(v)))
                cv2.circle(shown, centre, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(shown, centre, 3, tuple(int(x) for x in c), -1, cv2.LINE_AA)
        cv2.imshow("Image", shown)
        cv2.setWindowTitle("Image", f"Image: {title}")
        cv2.imshow("Depth", np.hstack([colour_depth(pred, self.near, self.far), depth_bar(pred.shape[0], self.near, self.far)]))
        cv2.setWindowTitle("Depth", f"Depth: {title} (the mouse shows the depth of a pixel)")
        if gt is not None and scale is not None:
            factor = 1.0 if self.metric_error else scale
            label = "metric (m: shape only)" if self.metric_error else f"shape only: times {scale:.2f} (m: metric)"
            error = colour_error(pred, gt, factor)
            cv2.imshow("Error", np.vstack([error, error_legend(error.shape[1], label)]))
            cv2.setWindowTitle("Error", f"Error: {title}")
        if self.viewer3D is not None:
            from pyslam.utilities.depth import depth2pointcloud

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.viewer3D.draw_dense_geometry(point_cloud=depth2pointcloud(pred, rgb, camera.fx, camera.fy, camera.cx, camera.cy))

    def closed(self):
        return self.viewer3D is not None and self.viewer3D.is_closed()

    def quit(self):
        if self.viewer3D is not None:
            self.viewer3D.quit()
        cv2.destroyAllWindows()


def key_action(windows, wait_ms):
    """'stop', 'pause', 'metric' or None, from the key pressed in a window (or the 3D window closed)."""
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (ord("q"), 27) or windows.closed():
        return "stop"
    if key == ord("p"):
        return "pause"
    if key == ord("m"):
        return "metric"
    return None


# ---------------------------------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------------------------------


def run_image(args, Printer):
    if args.image:
        path, name = args.image, os.path.basename(args.image)
    else:
        path, name = os.path.join(kScriptFolder, kTests[args.test]), args.test
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read the image {path}")
    img_right = None
    if args.method in kStereo:
        img_right = cv2.imread(os.path.join(kScriptFolder, kTestRight[args.test]))
    h0, w0 = img.shape[:2]
    if max(h0, w0) > kMaxImageSide:
        f = kMaxImageSide / max(h0, w0)
        img = cv2.resize(img, (int(round(w0 * f)), int(round(h0 * f))), interpolation=cv2.INTER_AREA)
        print(f"The image has {w0}x{h0} pixels: it is reduced to {img.shape[1]}x{img.shape[0]}")
    h, w = img.shape[:2]

    # the camera: from --settings, or the settings of the test image, or none
    settings = args.settings or (None if args.image else kTestSettings.get(args.test))
    camera = Camera.from_settings(settings).for_image(w, h) if settings else None
    if args.method in kStereo and not camera.bf:
        camera.bf = 379.8145  # KITTI 06's baseline (0.537 m) times fx, as in the KITTI stereo settings
        camera.b = camera.bf / camera.fx
    indoor = (args.scene == "indoor") if args.scene else (bool(args.image) or args.test in kIndoorTests)
    # DA3 needs a focal length to give metres: without settings, the one of a kAssumedFov field of view.
    # Depth Pro estimates its own when it has no camera.
    network_camera = camera
    if camera is None and args.method == "DEPTH_ANYTHING_V3":
        network_camera = Camera.assumed(w, h)
    network = Network(args.method, args.size, indoor, network_camera)

    start = time.time()
    depth = network.predict(img, img_right)
    first = time.time() - start
    times = []
    for _ in range(max(0, args.runs)):
        start = time.time()
        network.predict(img, img_right)
        times.append(time.time() - start)
    per_image = np.mean(times) if times else first
    valid = depth[depth > 0]
    if valid.size == 0:
        sys.exit(f"{args.method} gave no depth for {name}")

    focal = network.focal_length()
    summary = (
        f"{args.method} on {name} ({w}x{h} pixels): {network.describe()}, on {network.device}; "
        f"{per_image:.2f} s per image"
        + (
            f" (mean of {len(times)} runs after the first, which took {first:.2f} s"
            if len(times) > 1
            else f" (one run after the first, which took {first:.2f} s" if times else " (the first run"
        )
        + f"; {network.load_time:.1f} s to load the model); "
        f"depth {valid.min():.1f} m to {valid.max():.1f} m, median {np.median(valid):.1f} m"
    )
    if args.method == "DEPTH_PRO" and camera is None and focal:
        summary += f"; focal length estimated by Depth Pro: {focal:.0f} px"
        if (w, h) != (w0, h0):
            summary += f" at {w}x{h}, so {focal * w0 / w:.0f} px at the image's own {w0}x{h0}"
    Printer.green(f"\nSummary: {summary}\n")

    if not args.headless:
        # the 3D view: the camera of the settings, or Depth Pro's focal length, or the one assumed
        view_camera = camera or network_camera or (Camera.assumed(w, h, fx=focal) if focal else Camera.assumed(w, h))
        windows = Windows(indoor)
        windows.show(img, depth, view_camera, name)
        print("Press q in the Image or Depth window, or close the 3D window, to stop.")
        while key_action(windows, 50) != "stop":
            pass
        windows.quit()


# ---------------------------------------------------------------------------------------------------
# A sequence
# ---------------------------------------------------------------------------------------------------


class TumFrames:
    """The frames of a sequence in the layout of the TUM RGB-D datasets, read as pySLAM's TUM reader
    reads them: one line per frame in the list (associations.txt: timestamp, colour image, timestamp,
    depth image; or rgb.txt: timestamp, colour image), with paths relative to the folder."""

    def __init__(self, folder, list_name):
        from pyslam.utilities.text_files import open_text

        self.folder, self.list_name = folder, list_name
        path = os.path.join(folder, list_name)
        if not os.path.isfile(path):
            sys.exit(f"--tum: no {list_name} in {folder}")
        with open_text(path) as f:
            self.lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith("#")]
        if not self.lines:
            sys.exit(f"--tum: {path} lists no frames")
        self.has_depth = len(self.lines[0]) >= 4

    def __len__(self):
        return len(self.lines)

    def timestamp(self, i):
        return self.lines[i][0]

    def rgb(self, i):
        return self.lines[i][1]

    def image(self, i):
        return cv2.imread(os.path.join(self.folder, self.lines[i][1]))

    def depth(self, i, factor):
        """The camera's depth of frame i in metres (0: none), or None if the list has no depth images."""
        if not self.has_depth:
            return None
        raw = cv2.imread(os.path.join(self.folder, self.lines[i][3]), cv2.IMREAD_UNCHANGED)
        return None if raw is None else raw.astype(np.float32) / factor


def read_saved_maps(folder, name):
    """--load: {colour image: depth map path} from associations_NAME.txt, and depth_NAME/info.json (or {})."""
    from pyslam.utilities.text_files import open_text

    list_path = os.path.join(folder, f"associations_{name}.txt")
    if not os.path.isfile(list_path):
        sys.exit(f"--load {name}: no associations_{name}.txt in {folder}: write it with --save {name}, or copy it "
                 f"and depth_{name}/ from the member of your group who did")
    with open_text(list_path) as f:
        lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith("#")]
    maps = {l[1]: l[3] for l in lines if len(l) >= 4}
    info_path = os.path.join(folder, f"depth_{name}", "info.json")
    info = {}
    if os.path.isfile(info_path):
        with open(info_path) as f:
            info = json.load(f)
    return maps, info


def run_sequence(args, parser, Printer):
    camera = Camera.from_settings(args.settings)
    tum = None
    dataset = None
    if args.tum:
        folder = os.path.abspath(os.path.expanduser(args.tum)).rstrip("/")
        list_name = args.associations or ("associations.txt" if os.path.isfile(os.path.join(folder, "associations.txt")) else "rgb.txt")
        tum = TumFrames(folder, list_name)
        name = os.path.basename(folder)
        num_total = len(tum)
        default_indoor = True  # as pySLAM's TUM reader
        if (tum.has_depth or args.save or args.load) and not camera.depth_map_factor:
            sys.exit(f"--settings: {args.settings} has no DepthMapFactor, which the depth images need")
        if args.keypoints and not tum.has_depth:
            sys.exit(f"--keypoints needs the camera's depth: {list_name} lists no depth images (see scripts/associate.py)")
    else:
        from pyslam.config import Config
        from pyslam.io.sequence_options import sequence_config_path
        from pyslam.io.dataset_factory import dataset_factory

        config_path = sequence_config_path(args, parser)
        config = Config(config_path)
        os.remove(config_path)  # the temporary file of sequence_config_path(): it has been read
        dataset = dataset_factory(config)
        name = os.path.basename(os.path.normpath(args.video or args.images))
        num_total = getattr(dataset, "num_frames", None)
        default_indoor = False  # as pySLAM's video and image folder readers

    indoor = (args.scene == "indoor") if args.scene else default_indoor
    saved_maps, saved_info = None, {}
    network = None
    if args.load:
        saved_maps, saved_info = read_saved_maps(tum.folder, args.load)
        if args.method != parser.get_default("method") or args.size or args.scene:
            print(f"--load {args.load}: the maps are read, not predicted, so --method, --size and --scene are not used")
    else:
        network = Network(args.method, args.size, indoor, camera)
    keypoint_scorer = KeypointScorer(args.keypoints, args.patch, camera.num_features or 1000) if args.keypoints else None

    save_folder = None
    if args.save:
        save_folder = os.path.join(tum.folder, f"depth_{args.save}")
        if os.path.isdir(save_folder) and os.listdir(save_folder):
            Printer.orange(f"--save {args.save}: depth_{args.save}/ exists: its maps are replaced")
        os.makedirs(save_folder, exist_ok=True)
        if args.step > 1 or args.max_frames:
            Printer.orange(
                f"--save with --step {args.step}" + (f" and --max-frames {args.max_frames}" if args.max_frames else "")
                + f": associations_{args.save}.txt lists only the predicted frames, so SLAM with it would see only those. "
                "Use these maps for the scores, not for SLAM."
            )

    windows = None if args.headless else Windows(indoor)
    what = f"depth_{args.load} (saved maps)" if args.load else args.method
    every = "every frame" if args.step == 1 else f"every {args.step}th frame"
    print(f"{what} on {name}: {every}" + ("" if args.headless else "; in a window, q or Esc stops, p pauses, and m "
          "switches the Error window between shape only and metric"))

    rows = []  # one per predicted frame
    saved = []  # (timestamp, colour image, depth map) of each frame written by --save
    image_size = None
    frame_camera = camera
    paused = False
    stopped = False
    frame_id = -1
    try:
        while not stopped:
            frame_id += 1
            if tum is not None and frame_id >= len(tum):
                break
            if dataset is not None and not dataset.is_ok:
                break
            if args.max_frames is not None and len(rows) >= args.max_frames:
                break
            rgb_name = tum.rgb(frame_id) if tum is not None else None
            if frame_id % args.step != 0 or (saved_maps is not None and rgb_name not in saved_maps):
                if dataset is not None:
                    dataset.getImageColor(frame_id)  # the frames of a video are read in order
                continue
            img = tum.image(frame_id) if tum is not None else dataset.getImageColor(frame_id)
            if img is None:
                if tum is not None:
                    Printer.orange(f"cannot read {rgb_name}: frame {frame_id} skipped")
                    continue
                break
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
                frame_camera = camera.for_image(*image_size)

            seconds = None
            if saved_maps is not None:
                raw = cv2.imread(os.path.join(tum.folder, saved_maps[rgb_name]), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    Printer.orange(f"cannot read {saved_maps[rgb_name]}: frame {frame_id} skipped")
                    continue
                pred = raw.astype(np.float32) / camera.depth_map_factor
            else:
                start = time.time()
                pred = network.predict(img)
                seconds = time.time() - start

            row = {"frame": frame_id, "rgb": rgb_name or "", "time": seconds}
            gt = tum.depth(frame_id, camera.depth_map_factor) if tum is not None else None
            scores = score(pred, gt) if gt is not None else None
            if scores:
                row.update(scores)
            keypoints = None
            if keypoint_scorer is not None and gt is not None:
                result = keypoint_scorer.run(img, pred, gt, scores["median_scale"] if scores else 1.0, Printer)
                if result is not None:
                    uv, dp, dg, kp_scores = result
                    row.update(kp_scores)
                    keypoints = (uv, dp, dg)
            valid = pred[pred > 0]
            row["median_depth"] = float(np.median(valid)) if valid.size else 0.0
            rows.append(row)

            if save_folder is not None:
                depth_name = f"depth_{args.save}/{tum.timestamp(frame_id)}.png"
                # the format of the camera's depth images: 16 bits, metres times DepthMapFactor, 0 for no depth
                scaled = np.round(pred.astype(np.float64) * camera.depth_map_factor)
                cv2.imwrite(os.path.join(tum.folder, depth_name), np.clip(scaled, 0, 65535).astype(np.uint16))
                saved.append((tum.timestamp(frame_id), rgb_name, depth_name))

            if len(rows) == 1 or len(rows) % 50 == 0:
                text = f"frame {frame_id}" + (f"/{num_total}" if num_total else "")
                if seconds is not None:
                    text += f": {seconds:.2f} s"
                if scores:
                    text += (f"; AbsRel {scores['absrel_metric']:.2f} metric, {scores['absrel_shape']:.2f} shape only; "
                             f"median scale {scores['median_scale']:.2f}")
                else:
                    text += f"; median depth {row['median_depth']:.2f} m"
                print(text)

            if windows is not None:
                title = f"{name}, frame {frame_id}"
                view = dict(gt=gt if scores else None, scale=scores["median_scale"] if scores else None, keypoints=keypoints)
                windows.show(img, pred, frame_camera, title, **view)
                while True:
                    action = key_action(windows, 50 if paused else 1)
                    if action == "stop":
                        stopped = True
                        break
                    if action == "pause":
                        paused = not paused
                        Printer.yellow("Paused: p resumes." if paused else "Resumed.")
                    if action == "metric":
                        windows.metric_error = not windows.metric_error
                        windows.show(img, pred, frame_camera, title, **view)
                    if not paused:
                        break
    finally:
        if save_folder is not None:
            write_saved(args, tum, save_folder, saved, rows, network, camera, indoor)

    if not rows:
        sys.exit(f"no frames were read from {args.tum or args.video or args.images}"
                 + (f" (none of them is in associations_{args.load}.txt)" if args.load else ""))
    print_summary(args, name, num_total, image_size, rows, network, saved_info, keypoint_scorer, saved, Printer)
    if windows is not None:
        windows.quit()


def write_saved(args, tum, save_folder, saved, rows, network, camera, indoor):
    """--save: associations_NAME.txt (the frames written, in the order of the list), scores.csv and info.json."""
    with open(os.path.join(tum.folder, f"associations_{args.save}.txt"), "w") as f:
        for timestamp, rgb_name, depth_name in saved:
            f.write(f"{timestamp} {rgb_name} {timestamp} {depth_name}\n")
    columns = ["frame", "rgb", "time", "absrel_metric", "absrel_shape", "delta_metric", "delta_shape", "median_scale"]
    if any("keypoints" in r for r in rows):
        columns += ["keypoints", "keypoints_with_depth", "kp_absrel_metric", "kp_absrel_shape"]
    columns.append("median_depth")
    with open(os.path.join(save_folder, "scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in rows:
            writer.writerow(["" if r.get(c) is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in columns])
    info = {
        "method": network.method,
        "description": network.describe(),
        "device": network.device,
        "settings": os.path.abspath(args.settings),
        "depth_map_factor": camera.depth_map_factor,
        "frames_list": tum.list_name,
        "step": args.step,
        "frames": len(saved),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(save_folder, "info.json"), "w") as f:
        json.dump(info, f, indent=1)


def mean_of(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def print_summary(args, name, num_total, image_size, rows, network, saved_info, keypoint_scorer, saved, Printer):
    n = len(rows)
    frames = f"{n} frames" if not num_total or n == num_total else f"{n} of {num_total} frames"
    if args.step > 1:
        frames += f", every {args.step}"
    size = f", {image_size[0]}x{image_size[1]}" if image_size else ""
    if args.load:
        method = saved_info.get("method")
        head = f"depth_{args.load} on {name} ({frames}{size}): saved maps" + (
            f" of {method} ({saved_info.get('description', '')})" if method else ""
        )
    else:
        times = [r["time"] for r in rows if r.get("time") is not None]
        per_frame = float(np.mean(times[1:])) if len(times) > 1 else float(times[0])
        description = network.describe()
        if network.method == "DEPTH_ANYTHING_V2":
            description = description[len("metric, "):]  # "indoor model, large"
        head = f"{network.method} on {name} ({frames}{size}): {description}, on {network.device}; {per_frame:.2f} s per frame"
    parts = [head]
    if mean_of(rows, "absrel_metric") is not None:
        parts.append(
            f"metric: AbsRel {mean_of(rows, 'absrel_metric'):.2f}, delta<1.25 {mean_of(rows, 'delta_metric'):.2f}, "
            f"median scale {mean_of(rows, 'median_scale'):.2f}"
        )
        parts.append(f"shape only: AbsRel {mean_of(rows, 'absrel_shape'):.2f}, delta<1.25 {mean_of(rows, 'delta_shape'):.2f}")
    else:
        medians = [r["median_depth"] for r in rows]
        parts.append(f"median depth {np.median(medians):.1f} m (from {min(medians):.1f} m to {max(medians):.1f} m over the frames)")
    if keypoint_scorer is not None:
        if mean_of(rows, "kp_absrel_metric") is not None:
            parts.append(
                f"at the keypoints ({keypoint_scorer.describe()}): AbsRel {mean_of(rows, 'kp_absrel_metric'):.3f} metric, "
                f"{mean_of(rows, 'kp_absrel_shape'):.3f} shape only ({mean_of(rows, 'keypoints_with_depth'):.0f} of "
                f"{mean_of(rows, 'keypoints'):.0f} keypoints per frame with both depths)"
            )
        else:
            parts.append(f"no depth at the keypoints: depth_at_keypoints() returned None for {keypoint_scorer.method}")
    if args.save:
        parts.append(f"{len(saved)} depth maps saved to depth_{args.save}")
    Printer.green(f"\nSummary: {'; '.join(parts)}\n")
    if args.save:
        print(f"Written in the folder of the sequence: depth_{args.save}/ (the maps, scores.csv, info.json) and "
              f"associations_{args.save}.txt. SLAM with them: pixi run slam --tum {args.tum} --settings {args.settings} "
              f"--associations associations_{args.save}.txt")


if __name__ == "__main__":
    main()
