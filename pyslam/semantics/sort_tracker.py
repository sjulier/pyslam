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

# SORT (Bewley, Ge, Ott, Ramos and Upcroft, "Simple Online and Realtime Tracking", ICIP 2016,
# arXiv 1602.00763, Sec. 3), as a tracker of Ultralytics' track mode: main_object_tracking.py
# --tracker SORT adds it to Ultralytics' trackers under tracker_type "sort" (settings/trackers/sort.yaml).
#
# Written from the paper, with the values of the authors' code (github.com/abewley/sort, sort.py):
#   - one Kalman filter per box, state (u, v, s, r, u', v', s'): the centre, the area s and the aspect
#     ratio r = w / h, constant velocity on u, v and s; R = diag(1, 1, 10, 10),
#     P0 = diag(10, 10, 10, 10, 1e4, 1e4, 1e4), Q = diag(1, 1, 1, 1, 0.01, 0.01, 1e-4);
#   - one association per frame: the IoU of every predicted box with every box of score det_thresh or
#     more, an optimal assignment (Hungarian), and the pairs under iou_threshold rejected afterwards;
#   - a new tracker for every box left over; a box is reported only when its tracker was matched in this
#     frame and its run of matches in a row is min_hits or more (or in the first min_hits frames);
#   - a tracker is deleted when the frames since its last match exceed max_age (max_age 1: at its second
#     frame in a row without a box).
# There is no second stage for the low-score boxes, no camera-motion compensation and no appearance.

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """IoU of every box in a (N x 4, x1 y1 x2 y2) with every box in b (M x 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter)


def box_to_z(box):
    """x1, y1, x2, y2 -> the measurement (u, v, s, r)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h])


def x_to_box(x):
    """The state (u, v, s, r, ...) -> x1, y1, x2, y2."""
    w = np.sqrt(max(x[2] * x[3], 1e-9))
    h = x[2] / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class SortKalmanTrack:
    """One tracker of SORT: a Kalman filter on the box, and its counts of matches and misses."""

    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1
    H = np.hstack([np.eye(4), np.zeros((4, 3))])
    R = np.diag([1.0, 1.0, 10.0, 10.0])
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
    P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def __init__(self, track_id, box):
        self.track_id = track_id
        self.x = np.r_[box_to_z(box), 0.0, 0.0, 0.0]
        self.P = self.P0.copy()
        self.since_update = 0  # frames since the last match
        self.hit_streak = 1  # matches in a row (the box that started it is the first)

    def predict(self):
        if self.x[6] + self.x[2] <= 0:  # the area would go below zero
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.since_update > 0:
            self.hit_streak = 0
        self.since_update += 1
        return x_to_box(self.x)

    def update(self, box):
        self.since_update = 0
        self.hit_streak += 1
        y = box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        W = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + W @ y
        self.P = (np.eye(7) - W @ self.H) @ self.P


class SORT:
    """SORT with the interface of Ultralytics' trackers (BYTETracker): update(results, img) takes the
    detections of a frame (Boxes, after .cpu().numpy()) and returns the reported boxes as rows
    [x1, y1, x2, y2, track_id, score, cls, idx], idx the index of the detection. The reported box is the
    detection's own, so that its mask goes with it."""

    def __init__(self, args, frame_rate=30):
        self.det_thresh = float(args.det_thresh)
        self.iou_threshold = float(args.iou_threshold)
        self.min_hits = int(args.min_hits)
        self.max_age = int(args.max_age)
        self.reset()

    def reset(self):
        self.tracks = []
        self.frame_id = 0
        self.num_started = 0  # trackers started
        self.num_held = 0  # matched boxes not reported, as their tracker was still on probation
        self.predicted = {}  # track id -> the predicted box of the current frame

    def update(self, results, img=None, feats=None, **kwargs):
        self.frame_id += 1
        xyxy = np.asarray(results.xyxy, float).reshape(-1, 4)
        conf = np.asarray(results.conf, float).reshape(-1)
        cls = np.asarray(results.cls, float).reshape(-1)
        keep = np.flatnonzero(conf >= self.det_thresh)
        boxes = xyxy[keep]

        predicted = [t.predict() for t in self.tracks]
        self.predicted = {t.track_id: tuple(float(v) for v in p) for t, p in zip(self.tracks, predicted)}
        iou = iou_matrix(predicted, boxes)
        matched = {}  # tracker index -> box index (into boxes)
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matched[r] = c
        for r, c in matched.items():
            self.tracks[r].update(boxes[c])
        box_of = {self.tracks[r].track_id: c for r, c in matched.items()}
        taken = set(matched.values())
        for c in range(len(boxes)):
            if c not in taken:
                self.num_started += 1
                self.tracks.append(SortKalmanTrack(self.num_started, boxes[c]))
                box_of[self.num_started] = c

        out = []
        for t in list(self.tracks):
            if t.since_update < 1:
                if t.hit_streak >= self.min_hits or self.frame_id <= self.min_hits:
                    i = int(keep[box_of[t.track_id]])
                    out.append([*xyxy[i], t.track_id, conf[i], cls[i], i])
                else:
                    self.num_held += 1
            if t.since_update > self.max_age:
                self.tracks.remove(t)
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)
