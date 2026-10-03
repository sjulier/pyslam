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
import argparse
import select
import sys
import numpy as np
import cv2
import os
import math
import time
import platform

from pyslam.config import Config

from pyslam.slam.visual_odometry import VisualOdometryEducational
from pyslam.slam.visual_odometry_rgbd import (
    VisualOdometryRgbd,
    VisualOdometryRgbdTensor,
)
from pyslam.slam import PinholeCamera

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType

from pyslam.viz.mplot_thread import Mplot2d, Mplot3d
from pyslam.viz.qplot_thread import Qplot2d
from pyslam.viz.rerun_interface import Rerun

from pyslam.local_features.feature_tracker import (
    feature_tracker_factory,
    FeatureTrackerTypes,
)
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.utilities.logging import Printer


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + "/results"


kUseRerun = True
# check rerun does not have issues
if kUseRerun and not Rerun.is_ok:
    kUseRerun = False

"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = True
if platform.system() == "Darwin":
    kUsePangolin = (
        True  # Under mac force pangolin to be used since Mplot3d() has some reliability issues
    )
if kUsePangolin:
    from pyslam.viz.viewer3D import Viewer3D

kUseQplot2d = False
if platform.system() == "Darwin":
    kUseQplot2d = True  # Under mac force the usage of Qtplot2d: It is smoother


def factory_plot2d(*args, **kwargs):
    if kUseRerun:
        return None
    if kUseQplot2d:
        return Qplot2d(*args, **kwargs)
    else:
        return Mplot2d(*args, **kwargs)


def print_summary(vo, num_frames):
    """Frames processed and, with ground truth, the position error of the estimated trajectory."""
    msg = f"Visual odometry: {num_frames} frames"
    est, gt = np.asarray(vo.traj3d_est, dtype=float), np.asarray(vo.traj3d_gt, dtype=float)
    if len(est) > 1 and len(est) == len(gt):
        err = np.linalg.norm(est - gt, axis=1)
        msg += f"; position error vs ground truth: final {err[-1]:.2f} m, RMS {np.sqrt(np.mean(err**2)):.2f} m"
    Printer.green(msg)


def wait_at_end_with_rerun():
    """At the end of the sequence with the Rerun viewer: keep it open until Enter, Ctrl+C or the
    viewer window is closed (there is no other window to press q in)."""
    if not sys.stdin.isatty():
        return
    Printer.green("End of sequence: press Enter (or close the Rerun window, or Ctrl+C) to exit")
    while Rerun.is_viewer_alive():
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if ready:
            sys.stdin.readline()
            return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pySLAM visual odometry on the dataset of config.yaml")
    parser.add_argument(
        "--features",
        default="LK_SHI_TOMASI",
        help="the feature tracker: a FeatureTrackerConfigs entry (default LK_SHI_TOMASI; others: e.g. "
        "LK_FAST, ORB2, SIFT, or learned ones such as SUPERPOINT; see pixi run feature-matching --list)",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Show the results in separate windows (camera, trajectory, 3D viewer, plots) instead of "
        "the Rerun viewer; press q in one of them to exit",
    )
    args = parser.parse_args()
    if args.no_rerun:
        kUseRerun = False

    config = Config()

    dataset = dataset_factory(config)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)

    num_features = 2000  # how many features do you want to detect and track?
    if (
        config.num_features_to_extract > 0
    ):  # override the number of features to extract if we set something in the settings file
        num_features = config.num_features_to_extract

    # select your tracker configuration (see the file feature_tracker_configs.py)
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    tracker_config = FeatureTrackerConfigs.get_config_from_name(args.features)
    if tracker_config is None:
        sys.exit(2)
    tracker_config = dict(tracker_config)
    tracker_config["num_features"] = num_features

    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
        Printer.green("Using VisualOdometryRgbdTensor")
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
        Printer.green("Using VisualOdometryEducational")
    time.sleep(1)  # time to read the message

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5 * traj_img_size)
    draw_scale = 1

    plt3d = None

    viewer3D = None

    is_draw_3d = True
    is_draw_with_rerun = kUseRerun
    if is_draw_with_rerun:
        Rerun.init_vo()
    else:
        if kUsePangolin:
            viewer3D = Viewer3D(scale=dataset.scale_viewer_3d * 10)
        else:
            plt3d = Mplot3d(title="3D trajectory")

    is_draw_err = True
    err_plt = factory_plot2d(xlabel="img id", ylabel="m", title="error")

    is_draw_matched_points = True
    matched_points_plt = factory_plot2d(xlabel="img id", ylabel="# matches", title="# matches")

    img_id = 0
    is_end_of_sequence = False
    is_interrupted = False
    try:
        while True:

            img = None

            if dataset.is_ok:
                timestamp = dataset.getTimestamp()  # get current timestamp
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                img_right = (
                    dataset.getImageColorRight(img_id)
                    if dataset.sensor_type == SensorType.STEREO
                    else None
                )

            if img is not None:

                vo.track(img, img_right, depth, img_id, timestamp)  # main VO function

                if (
                    len(vo.traj3d_est) > 1
                ):  # start drawing from the third image (when everything is initialized and flows in a normal way)

                    x, y, z = vo.traj3d_est[-1]
                    gt_x, gt_y, gt_z = vo.traj3d_gt[-1]

                    if is_draw_traj_img:  # draw 2D trajectory (on the plane xz)
                        draw_x, draw_y = int(
                            draw_scale * x
                        ) + half_traj_img_size, half_traj_img_size - int(draw_scale * z)
                        draw_gt_x, draw_gt_y = int(
                            draw_scale * gt_x
                        ) + half_traj_img_size, half_traj_img_size - int(draw_scale * gt_z)
                        cv2.circle(
                            traj_img,
                            (draw_x, draw_y),
                            1,
                            (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0),
                            1,
                        )  # estimated from green to blue
                        cv2.circle(
                            traj_img, (draw_gt_x, draw_gt_y), 1, (0, 0, 255), 1
                        )  # groundtruth in red
                        # write text on traj_img
                        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                        cv2.putText(
                            traj_img,
                            text,
                            (20, 40),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 255),
                            1,
                            8,
                        )
                        # show

                        if is_draw_with_rerun:
                            Rerun.log_img_seq("trajectory_img/2d", img_id, traj_img)
                        else:
                            cv2.imshow("Trajectory", traj_img)

                    if is_draw_with_rerun:
                        Rerun.log_2d_seq_scalar("trajectory_error/err_x", img_id, math.fabs(gt_x - x))
                        Rerun.log_2d_seq_scalar("trajectory_error/err_y", img_id, math.fabs(gt_y - y))
                        Rerun.log_2d_seq_scalar("trajectory_error/err_z", img_id, math.fabs(gt_z - z))

                        Rerun.log_2d_seq_scalar(
                            "trajectory_stats/num_matches", img_id, vo.num_matched_kps
                        )
                        Rerun.log_2d_seq_scalar("trajectory_stats/num_inliers", img_id, vo.num_inliers)

                        Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, cam, vo.poses[-1])
                        Rerun.log_3d_trajectory(img_id, vo.traj3d_est, "estimated", color=[0, 0, 255])
                        Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, "ground_truth", color=[255, 0, 0])
                    else:
                        if is_draw_3d:  # draw 3d trajectory
                            if kUsePangolin:
                                viewer3D.draw_vo(vo)
                            else:
                                plt3d.draw(vo.traj3d_gt, "ground truth", color="r", marker=".")
                                plt3d.draw(vo.traj3d_est, "estimated", color="g", marker=".")

                        if is_draw_err:  # draw error signals
                            errx = [img_id, math.fabs(gt_x - x)]
                            erry = [img_id, math.fabs(gt_y - y)]
                            errz = [img_id, math.fabs(gt_z - z)]
                            err_plt.draw(errx, "err_x", color="g")
                            err_plt.draw(erry, "err_y", color="b")
                            err_plt.draw(errz, "err_z", color="r")

                        if is_draw_matched_points:
                            matched_kps_signal = [img_id, vo.num_matched_kps]
                            inliers_signal = [img_id, vo.num_inliers]
                            matched_points_plt.draw(matched_kps_signal, "# matches", color="b")
                            matched_points_plt.draw(inliers_signal, "# inliers", color="g")

                # draw camera image
                if not is_draw_with_rerun:
                    cv2.imshow("Camera", vo.draw_img)

            elif not dataset.is_ok:
                if not is_end_of_sequence:
                    is_end_of_sequence = True
                    print_summary(vo, img_id)
                    if is_draw_with_rerun:
                        break  # no window to press q in: wait below
                    Printer.green("End of sequence: press q in one of the windows to exit")
                time.sleep(0.1)
            else:
                time.sleep(0.1)

            # get keys
            key = matched_points_plt.get_key() if matched_points_plt is not None else None
            if key == "" or key is None:
                key = err_plt.get_key() if err_plt is not None else None
            if key == "" or key is None:
                key = plt3d.get_key() if plt3d is not None else None

            # press 'q' to exit!
            key_cv = cv2.waitKey(1) & 0xFF
            if key == "q" or (key_cv == ord("q")):
                break
            if viewer3D and viewer3D.is_closed():
                break
            if is_draw_with_rerun and img_id % 10 == 0 and not Rerun.is_viewer_alive():
                Printer.orange("The Rerun viewer was closed: exiting")
                print_summary(vo, img_id)
                break
            img_id += 1
    except KeyboardInterrupt:
        is_interrupted = True
        print("")
        Printer.orange("Interrupted (Ctrl+C): exiting")
        if not is_end_of_sequence:
            print_summary(vo, img_id)

    if is_draw_with_rerun and is_end_of_sequence and not is_interrupted:
        try:
            wait_at_end_with_rerun()
        except KeyboardInterrupt:
            print("")

    # print('press a key in order to exit...')
    # cv2.waitKey(0)

    if is_draw_traj_img:
        if not os.path.exists(kResultsFolder):
            os.makedirs(kResultsFolder, exist_ok=True)
        print(f"saving {kResultsFolder}/map.png")
        cv2.imwrite(f"{kResultsFolder}/map.png", traj_img)
    if plt3d:
        plt3d.quit()
    if viewer3D:
        viewer3D.quit()
    if err_plt:
        err_plt.quit()
    if matched_points_plt:
        matched_points_plt.quit()

    cv2.destroyAllWindows()
