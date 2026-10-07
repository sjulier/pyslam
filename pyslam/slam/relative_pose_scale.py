"""
* This file is part of PYSLAM
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

# The scale of the frames' relative poses in the tracking history.
#
# Tracking stores each frame's pose relative to its reference keyframe (as ORB-SLAM2/3 do), and the final
# trajectory is that relative pose times the keyframe's final pose. In monocular SLAM the map's scale
# changes after a frame is tracked: a loop correction applies a Sim3 (KITTI 06: scale ~2.2 near the loop,
# spread along the loop by the essential graph), and bundle adjustment moves it too. The keyframes are
# rescaled; the stored relative translations were not, so the frames between keyframes over- or undershot
# (alternating short and long steps near a loop) and the ATE suffered. ORB-SLAM3 rescales the stored
# relative poses only when the whole map is rescaled by one factor (Tracking::UpdateFrameIMU, IMU
# initialisation and map merging); ORB-SLAM2 never does.
#
# Here, tracking also stores the positions of a few keyframes around the reference keyframe ("anchors")
# when the frame is tracked. When the relative pose is used, the anchors' positions then and now give the
# local scale change, whatever made it (loop correction, global or local BA), and the relative
# translation is scaled by it. If the reference keyframe has been culled since, the frame's pose at
# tracking time is moved by the anchors' similarity transform (then -> now) instead of following the
# parent's Tcp, whose scale is unknown.
# Parameters.kTrajectoryRescaleRelativePoses = False gives the ORB-SLAM2/3 behaviour (to compare ATEs).

import numpy as np

from pyslam.config_parameters import Parameters

kNumAnchorCovisibleKeyFrames = 8  # the reference keyframe, its parent and up to this many covisible keyframes
kMinAnchorSpread = 1e-9  # [map units^2] below it, the anchors cannot give a scale
kMinAnchorsForScale = 3  # with two keyframes (one of them maybe just moved by local BA) the ratio is too noisy


class RelativePoseAnchors:
    """The keyframes around a frame's reference keyframe and their positions when the frame was tracked."""

    __slots__ = ("keyframes", "positions", "Twr")

    def __init__(self, kf_ref, previous=None):
        keyframes = [kf_ref]
        kids = {kf_ref.kid}
        parent = kf_ref.get_parent()
        candidates = ([parent] if parent is not None else []) + list(
            kf_ref.get_best_covisible_keyframes(kNumAnchorCovisibleKeyFrames)
        )
        if len(candidates) < 2 and previous is not None:
            # a keyframe that local mapping has not connected yet (no parent, no covisible keyframes): the
            # previous frame's anchors are around the same place
            candidates += previous.keyframes
        for kf in candidates:
            if kf is not None and kf.kid not in kids and not kf.is_bad():
                keyframes.append(kf)
                kids.add(kf.kid)
        self.keyframes = keyframes
        self.positions = np.array([np.asarray(kf.Ow(), dtype=np.float64).reshape(3) for kf in keyframes])
        self.Twr = np.array(kf_ref.Twc(), dtype=np.float64)  # the reference keyframe's pose then

    def _matched_positions(self):
        """The anchors' positions then and now, for those that are still in the map."""
        idx = [i for i, kf in enumerate(self.keyframes) if not kf.is_bad()]
        now = np.array([np.asarray(self.keyframes[i].Ow(), dtype=np.float64).reshape(3) for i in idx])
        return self.positions[idx], now

    def scale(self):
        """The local scale change since the frame was tracked (1.0 if the anchors cannot tell)."""
        then, now = self._matched_positions()
        if len(then) < kMinAnchorsForScale:
            return 1.0
        spread_then = ((then - then.mean(axis=0)) ** 2).sum()
        if spread_then < kMinAnchorSpread:
            return 1.0
        return float(np.sqrt(((now - now.mean(axis=0)) ** 2).sum() / spread_then))

    def similarity(self):
        """The similarity (s, R, t) that moves the anchors from then to now (Umeyama), or None."""
        then, now = self._matched_positions()
        if len(then) < 3:
            return None
        mu_then, mu_now = then.mean(axis=0), now.mean(axis=0)
        x_then, x_now = then - mu_then, now - mu_now
        var_then = (x_then**2).sum() / len(then)
        if var_then < kMinAnchorSpread:
            return None
        U, D, Vt = np.linalg.svd(x_now.T @ x_then / len(then))
        if D[1] < 1e-6 * D[0]:  # the anchors are on a line: the rotation about it is undetermined
            return None
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        s = float(np.trace(np.diag(D) @ S) / var_then)
        return s, R, mu_now - s * R @ mu_then


def rescaled_relative_pose(Tcr, anchors):
    """The relative pose Tcr (4x4, frame w.r.t. reference keyframe) with its translation in today's scale."""
    if anchors is None or not Parameters.kTrajectoryRescaleRelativePoses:
        return Tcr
    T = np.array(Tcr, dtype=np.float64)
    T[:3, 3] *= anchors.scale()
    return T


def frame_Twc_after_culled_reference(Tcr, anchors):
    """The frame's pose (Twc, 4x4) when its reference keyframe has been culled: its pose at tracking time,
    moved by the anchors' similarity transform. None if the anchors cannot tell (the caller then follows
    the spanning tree, as ORB-SLAM does)."""
    if anchors is None or not Parameters.kTrajectoryRescaleRelativePoses:
        return None
    sim = anchors.similarity()
    if sim is None:
        return None
    s, R, t = sim
    Twc_then = anchors.Twr @ np.linalg.inv(np.array(Tcr, dtype=np.float64))
    Twc = np.eye(4)
    Twc[:3, :3] = R @ Twc_then[:3, :3]
    Twc[:3, 3] = s * R @ Twc_then[:3, 3] + t
    return Twc
