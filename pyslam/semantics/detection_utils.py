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

# Non-maximum suppression (NMS): a detector predicts many overlapping boxes for each object, and NMS
# keeps one of them. main_object_detection.py --nms own uses nms() below in place of the detector's
# own suppression, one class at a time.
#
# Test nms() without any network (numpy only, so it also runs at the default level):
#
#   pixi run python -m pyslam.semantics.detection_utils

import json
import os

import numpy as np

kTestBoxes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nms_test_boxes.json")


def box_iou(a, b):
    """The intersection over union (IoU) of every box of a with every box of b.

    a: an array of shape (N, 4), or (4,) for one box; b: an array of shape (M, 4), or (4,). Each box
        is x1, y1, x2, y2 in pixels: its left, top, right and bottom edges.

    Returns an array of shape (N, M): the IoU of box i of a with box j of b is in row i, column j. It
    is 0 for boxes which do not overlap, and 1 for identical boxes.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def nms(boxes, scores, iou_threshold=0.7):
    """Non-maximum suppression: which of the boxes to keep.

    boxes: an array of shape (N, 4), each box x1, y1, x2, y2 in pixels (all of one class).
    scores: an array of N scores, one per box.
    iou_threshold: a box is removed if its IoU with a box already kept is above this.

    Repeat: keep the box with the highest score of those left, and remove every box whose IoU with
    it is above iou_threshold; stop when no box is left.

    Returns an array of the indices (into boxes) of the boxes to keep, in order of decreasing score,
    or None (and then the detector's own suppression is used).
    """
    # TODO (COMP0227 Lab 04, Activity 5): write this function. As it is, it returns None.
    return None


def self_test():
    """Run nms() on 30 chair boxes of one Replica room0 image and say whether it kept the right 3."""
    with open(kTestBoxes) as f:
        test = json.load(f)
    boxes = np.array(test["boxes"], dtype=np.float64)
    scores = np.array(test["scores"], dtype=np.float64)
    expected = test["expected_kept_scores"]
    threshold = test["iou_threshold"]
    print(f"{len(boxes)} chair boxes of one image (Replica room0, keyframe 42), IoU threshold {threshold}")
    keep = nms(boxes, scores, threshold)
    if keep is None:
        print("nms() returned None: it is not written yet (Lab 04, Activity 5).")
        return 1
    keep = np.asarray(keep).astype(int).ravel()
    kept_scores = [round(float(scores[i]), 4) for i in keep]
    print(f"nms() kept {len(keep)} boxes, with the scores {', '.join(f'{s:.4f}' for s in kept_scores)}")
    if kept_scores == expected:
        print("Right: these are the 3 boxes that NMS keeps at this threshold, in order of decreasing score.")
        return 0
    if sorted(kept_scores) == sorted(expected):
        print("The right 3 boxes, but not in order of decreasing score.")
    else:
        print(f"Not right: NMS keeps 3 boxes at this threshold, with the scores {', '.join(f'{s:.4f}' for s in expected)}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(self_test())
