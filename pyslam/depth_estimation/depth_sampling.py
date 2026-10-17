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

# The depth at the keypoints: SLAM does not use the depth of every pixel, only the depth where it
# detected its keypoints, which sit on corners and edges.
#
# main_depth_prediction.py --keypoints METHOD detects ORB keypoints in every frame, as SLAM does, calls
# depth_at_keypoints() on the predicted depth map and on the camera's depth map, and compares the two.

import numpy as np

from pyslam.utilities.logging import Printer


# The ways of reading the depth at a keypoint (main_depth_prediction.py --keypoints)
kMethods = ("nearest", "bilinear", "median")

_warned = set()  # the methods not written yet whose warning has been printed


def depth_at_keypoints(depth, keypoints, method="nearest", patch=5):
    """The depth at each keypoint, read from a depth map.

    depth: the depth map, a float array of shape (H, W): the depth of every pixel in metres, and 0 where
        there is no depth.
    keypoints: the positions of N keypoints, a float array of shape (N, 2). Row i is (u, v), in pixels:
        u is the column and v the row, so the pixel at a whole (u, v) is depth[v, u]. The positions need
        not be whole numbers. They are inside the image: 0 <= u <= W - 1 and 0 <= v <= H - 1.
    method: how to read the depth at a position which falls between pixels:
        "nearest": the depth of the pixel which the keypoint falls in (the pixel whose centre is nearest);
        "bilinear": the depth interpolated from the four pixels around the keypoint, using only those
            which have a depth (their weights are scaled up to add to 1), or 0 if none of them has one;
        "median": the median of the depths of the pixels in a patch x patch window centred on the pixel
            which the keypoint falls in, using only the pixels which have a depth, or 0 if none has one.
            At the borders of the image, the window is cut to the part inside the image.
    patch: the size of the window of "median", an odd number of pixels.

    Returns a float array of shape (N,): the depth at each keypoint in metres, 0 where no depth can be
    read. As supplied, only "nearest" is written: "bilinear" and "median" return None, with a warning,
    and the keypoints are then skipped.
    """
    keypoints = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    h, w = depth.shape[:2]

    if method == "nearest":
        # The worked example. Pixel i covers the positions from i - 0.5 to i + 0.5, so the pixel which a
        # keypoint falls in is its position rounded to the nearest whole number, and np.clip keeps it
        # inside the image. u is the column and v the row, so the depth is depth[v, u]. Where the map
        # has no depth it holds 0, and so does the result.
        u = np.clip(np.rint(keypoints[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.rint(keypoints[:, 1]).astype(int), 0, h - 1)
        return np.asarray(depth, dtype=np.float64)[v, u]

    # TODO (COMP0227 Lab 03, Activity 6): write "bilinear" and "median". As they are, they return None.
    if method not in _warned:
        _warned.add(method)
        Printer.orange(
            f'depth_at_keypoints() in pyslam/depth_estimation/depth_sampling.py: the method "{method}" is not '
            "written yet (COMP0227 Lab 03, Activity 6), so it returns None"
        )
    return None
