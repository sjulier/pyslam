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

# The prediction step of a tracker. Between two frames, the trackers of main_object_tracking.py
# (ByteTrack and BoT-SORT, from Ultralytics) predict where each tracked box will be, with a Kalman
# filter on the box's centre, size and their velocities, and then match the new detections to the
# predicted boxes.
#
# main_object_tracking.py --predict own calls predict_box() for every track at every frame, in place
# of the tracker's own prediction. The update step, which corrects a track with the detection it was
# matched to, stays the tracker's.

import numpy as np


# The box state of the tracker which is running (main_object_tracking.py sets it):
#   "xyah": ByteTrack's: the centre x and y, the aspect ratio (width over height), the height
#   "xywh": BoT-SORT's: the centre x and y, the width, the height
# followed, in both, by the velocities of the four (pixels, or aspect ratio, per frame).
kBoxState = "xyah"

# The standard deviations of the process noise, as fractions of the box's size (Ultralytics' values)
kStdWeightPosition = 1.0 / 20
kStdWeightVelocity = 1.0 / 160


def process_noise(state):
    """Q: the 8 x 8 covariance of the process noise of one prediction, as the tracker's Kalman filter
    sets it. It is diagonal, and its standard deviations grow with the size of the box: a large box
    (near the camera) moves more pixels per frame than a small one.

    state: the 8-vector of the track before the prediction (see predict_box()).
    """
    h = state[3]
    if kBoxState == "xywh":  # BoT-SORT: the width and the height
        w = state[2]
        std_pos = [kStdWeightPosition * w, kStdWeightPosition * h, kStdWeightPosition * w, kStdWeightPosition * h]
        std_vel = [kStdWeightVelocity * w, kStdWeightVelocity * h, kStdWeightVelocity * w, kStdWeightVelocity * h]
    else:  # ByteTrack: the height, and a small fixed noise on the aspect ratio
        std_pos = [kStdWeightPosition * h, kStdWeightPosition * h, 1e-2, kStdWeightPosition * h]
        std_vel = [kStdWeightVelocity * h, kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h]
    return np.diag(np.square(np.r_[std_pos, std_vel]))


def predict_box(state, covariance, dt=1.0):
    """The prediction step of the Kalman filter of a track: where its box will be in the next frame.

    state: the state of the track, a float array of shape (8,): the centre x, the centre y, the aspect
        ratio (width over height) and the height of the box, in pixels (BoT-SORT: the width in place of
        the aspect ratio), followed by their four velocities, per frame.
    covariance: the covariance of the state, a float array of shape (8, 8).
    dt: the time step, in frames.

    The model is constant velocity: each of the first four components moves by its velocity times dt,
    and the velocities stay as they are. With the 8 x 8 transition matrix F (the identity, with dt
    where row i meets column i + 4, for i = 0 to 3) and the process noise Q = process_noise(state):

        predicted state      = F @ state
        predicted covariance = F @ covariance @ F.T + Q

    Returns (predicted state, predicted covariance), the same shapes as the inputs. As supplied, it
    returns None, and the tracker then uses its own prediction.
    """
    # TODO (COMP0227 Lab 05, Activity 4): write the prediction. As it is, it returns None.
    return None
