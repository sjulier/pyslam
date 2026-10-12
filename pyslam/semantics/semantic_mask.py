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

# A detector mask from a semantic label map: keypoints are only detected where the mask is not
# zero, so the features of the masked classes (people, cars, ...) never enter the map.
#
# main_slam.py --semantic-mask DIR reads the label maps which main_semantic_image_segmentation.py
# --save DIR wrote for the same sequence, and calls mask_from_labels() for every frame.

import bisect
import json
import os

import cv2
import numpy as np

from pyslam.utilities.logging import Printer


# The classes masked by default: the things which move in a street scene (Cityscapes names).
kDefaultMaskedClasses = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def mask_from_labels(labels, class_names, masked_classes, dilation=0):
    """The detector mask for one frame.

    labels: the label map of the frame, an array of shape (height, width) with the class of every
        pixel (an integer).
    class_names: a dict from the labels to the names of the classes, e.g. {0: "road", ...}. Methods
        which only label the objects they find have an extra entry "no object".
    masked_classes: the names of the classes to mask, e.g. ["person", "car"].
    dilation: how many pixels to grow the masked regions by, so that the keypoints on the edges of
        the objects are masked too (0: no growth).

    Returns an array of shape (height, width) and dtype np.uint8 which is 0 where no keypoint
    may be detected and 255 elsewhere, or None for "no mask" (every keypoint is kept).
    """
    # TODO (COMP0227 Lab 02, Part 3): write this function. As it is, it returns None: no mask.
    return None


class SavedLabelMaps:
    """The label maps of a sequence, as main_semantic_image_segmentation.py --save DIR wrote them:
    DIR/info.json (the method, the names of the classes, the frames saved) and DIR/labels_NNNNNN.png
    (the label map of frame NNNNNN). It gives the mask of each frame of a SLAM run."""

    def __init__(self, folder, masked_classes=None, dilation=0):
        self.folder = folder
        info_path = os.path.join(folder, "info.json")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(
                f"--semantic-mask: no info.json in {folder}: it must be a folder written by "
                "main_semantic_image_segmentation.py --save"
            )
        with open(info_path) as f:
            self.info = json.load(f)
        self.class_names = {int(k): v for k, v in self.info["class_names"].items()}
        self.frames = sorted(int(i) for i in self.info["frames"])
        self.masked_classes = list(masked_classes) if masked_classes else list(kDefaultMaskedClasses)
        self.dilation = int(dilation)
        self.num_masks = 0  # frames given a mask
        self.num_frames = 0  # frames asked for one
        self.masked_fraction_sum = 0.0
        self.warned = False
        self._last = (None, None)  # (frame saved, its labels)
        unknown = [c for c in self.masked_classes if c not in self.class_names.values()]
        if unknown:
            Printer.orange(
                f"--mask-classes: {', '.join(unknown)}: not among the classes of "
                f"{self.info['method']} ({', '.join(sorted(self.class_names.values()))})"
            )
        if not self.frames:
            raise ValueError(f"--semantic-mask: no frames in {info_path}")
        Printer.green(
            f"Semantic mask: {self.info['method']} label maps of {len(self.frames)} frames "
            f"(every {self.info.get('step', 1)}) from {folder}; masked classes: "
            f"{', '.join(self.masked_classes)}; dilation: {self.dilation} pixels"
        )

    def labels(self, frame_id):
        """The label map saved for frame_id, or for the last saved frame before it."""
        i = bisect.bisect_right(self.frames, frame_id) - 1
        if i < 0:
            return None
        saved = self.frames[i]
        if self._last[0] != saved:
            path = os.path.join(self.folder, f"labels_{saved:06d}.png")
            labels = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if labels is None:
                Printer.red(f"--semantic-mask: cannot read {path}")
                return None
            self._last = (saved, labels.astype(np.int32))
        return self._last[1]

    def mask(self, frame_id, image_shape):
        """The detector mask of frame_id (uint8, 0 = no keypoints), or None."""
        self.num_frames += 1
        labels = self.labels(frame_id)
        if labels is None:
            return None
        if labels.shape[:2] != image_shape[:2]:
            labels = cv2.resize(labels, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask_from_labels(labels, self.class_names, self.masked_classes, self.dilation)
        if mask is None:
            if not self.warned:
                self.warned = True
                Printer.orange(
                    "mask_from_labels() in pyslam/semantics/semantic_mask.py returned None: no mask. "
                    "Write it (Lab 02, Part 3)."
                )
            return None
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        self.num_masks += 1
        self.masked_fraction_sum += float(np.count_nonzero(mask == 0)) / mask.size
        return mask

    def summary(self):
        if self.num_masks == 0:
            return f"Semantic mask: no frame was masked ({self.num_frames} frames)"
        return (
            f"Semantic mask: {self.num_masks} of {self.num_frames} frames masked; "
            f"{100 * self.masked_fraction_sum / self.num_masks:.1f}% of the pixels masked on average "
            f"(classes: {', '.join(self.masked_classes)}; dilation {self.dilation})"
        )
