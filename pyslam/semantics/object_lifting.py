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

# Objects in the map: main_slam.py --detections DIR reads the detections which
# main_object_detection.py --save DIR wrote for the same sequence, and lifts each of them into the map
# with the pose and the depth of its frame (COMP0227 Lab 04):
#
#   the point of a detection in the camera:  p_c = z K^-1 [u, v, 1]^T
#       (u, v): the centroid of its mask (--lift-depth median), or the centre of its box (centre)
#       z:      the median depth of the mask's pixels (median) or the depth at the box's centre
#               (centre), from the frame's depth image (RGB-D); monocular: the median depth of the map
#               points SLAM tracks inside the mask (in the map's own scale)
#   the point in the map:                    p_w = T_wc p_c
#
# At the end of the run the detections are lifted again with the final trajectory and merged into
# objects (merge_into_objects): in order of score, each point joins the nearest object of its class
# whose centre lies within --merge-radius, and the object's centre becomes the mean of its points;
# otherwise the point starts a new object. The objects are printed and drawn; no file is written.
#
# The functions lift_mask(), map_points_in_camera(), merge_into_objects() and class_colour_rgb() are
# meant to be shared with the tracks of Lab 05 (a detection is a track of one box).

import csv
import json
import os
import zlib

import cv2
import numpy as np

kMinMaskPixels = 50  # pixels with a depth inside the mask (median), else the detection is not lifted
kMinMapPoints = 3  # monocular: map points inside the mask
kTimestampTolerance = 1e-3  # s: a detection belongs to the frame of SLAM with this timestamp

# The 80 classes of COCO, in the order of YOLO's class numbers (as in main_object_detection.py)
kCocoClasses = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# 20 distinct colours (Ultralytics' palette), as hex RGB; a COCO class takes the colour (7 x its number) mod 20,
# so that the classes of one room differ; any other name (a YOLOE prompt) a colour from its hash
kClassPalette = ("FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                 "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7")


def class_colour_bgr(name):
    """The colour of a class, the same in every run and every window (BGR, 0-255), as
    main_object_detection.py draws it."""
    name = str(name)
    i = (7 * kCocoClasses.index(name)) % 20 if name in kCocoClasses else zlib.crc32(name.encode()) % 20
    h = kClassPalette[i]
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def class_colour_rgb(name):
    """class_colour_bgr() as RGB in [0, 1], for the Map Viewer."""
    b, g, r = class_colour_bgr(name)
    return (r / 255.0, g / 255.0, b / 255.0)


def map_points_in_camera(frame):
    """The pixel positions (N x 2) and depths (N) of the map points which SLAM tracks in its current
    frame, from the frame's keypoints, its map points and its pose; (None, None) if they cannot be read."""
    try:
        kps = np.asarray(frame.kps, np.float64).reshape(-1, 2)
        points = frame.get_points() if hasattr(frame, "get_points") else frame.points
        Tcw = np.asarray(frame.Tcw() if callable(getattr(frame, "Tcw", None)) else frame.pose(), np.float64)
    except Exception:
        return None, None
    if points is None:
        return None, None
    uv, zs = [], []
    for i, p in enumerate(points):
        if p is None or i >= len(kps):
            continue
        try:
            if p.is_bad():
                continue
            pw = p.pt() if callable(getattr(p, "pt", None)) else p.pt
        except Exception:
            continue
        pc = Tcw[:3, :3] @ np.asarray(pw, np.float64).reshape(3) + Tcw[:3, 3]
        if pc[2] > 0:
            uv.append(kps[i])
            zs.append(pc[2])
    if not uv:
        return np.zeros((0, 2)), np.zeros(0)
    return np.asarray(uv), np.asarray(zs)


def lift_mask(mask, box, camera, Twc, depth=None, map_uv=None, map_depth=None, how="median"):
    """Lift one detection into the map.

    mask: a boolean image of the detection's pixels (or None: the inner half of its box is used);
    box: x1, y1, x2, y2 in pixels; camera: fx, fy, cx, cy, width, height; Twc: the frame's pose, 4 x 4,
    camera to map; depth: the frame's depth image in metres (RGB-D), or None, and then map_uv and
    map_depth (map_points_in_camera()) give the depth; how: "median" (the median depth under the mask,
    at the mask's centroid) or "centre" (the depth at the box's centre).

    Returns a dict with uv, z, p_c and p_w, or None if no depth can be read there."""
    shape = (int(camera.height), int(camera.width))
    x1, y1, x2, y2 = (float(v) for v in box)
    if mask is None:
        mask = np.zeros(shape, bool)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        mask[max(int(cy - h / 4), 0):int(cy + h / 4) + 1, max(int(cx - w / 4), 0):int(cx + w / 4) + 1] = True
    z = None
    if how == "centre":
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        iu = int(np.clip(round(u), 0, shape[1] - 1))
        iv = int(np.clip(round(v), 0, shape[0] - 1))
        if depth is not None:
            d = float(depth[iv, iu])
            z = d if d > 0 else None
        elif map_uv is not None and len(map_uv):
            # monocular: the map point nearest to the box's centre, if it lies in the box
            inside = (map_uv[:, 0] >= x1) & (map_uv[:, 0] <= x2) & (map_uv[:, 1] >= y1) & (map_uv[:, 1] <= y2)
            if np.any(inside):
                d2 = np.sum((map_uv[inside] - np.array([u, v])) ** 2, axis=1)
                z = float(map_depth[inside][int(np.argmin(d2))])
    else:
        vv, uu = np.nonzero(mask)
        if len(uu) == 0:
            return None
        u, v = float(uu.mean()), float(vv.mean())
        if depth is not None:
            zs = depth[mask]
            zs = zs[zs > 0]
            if len(zs) >= kMinMaskPixels:
                z = float(np.median(zs))
        elif map_uv is not None and len(map_uv):
            pu = np.clip(np.rint(map_uv[:, 0]).astype(int), 0, shape[1] - 1)
            pv = np.clip(np.rint(map_uv[:, 1]).astype(int), 0, shape[0] - 1)
            inside = mask[pv, pu]
            if np.count_nonzero(inside) >= kMinMapPoints:
                z = float(np.median(map_depth[inside]))
    if z is None:
        return None
    Kinv = np.linalg.inv(np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.0]]))
    p_c = z * (Kinv @ np.array([u, v, 1.0]))
    p_w = (np.asarray(Twc, np.float64) @ np.append(p_c, 1.0))[:3]
    return dict(uv=(u, v), z=z, p_c=p_c, p_w=p_w)


def merge_into_objects(items, radius):
    """Greedy merge of points into objects.

    items: dicts with "cls", "centre" (3 values), "order" (the higher first: a detection's score, or a
        track's length) and "weight" (1 for a detection, or a track's number of boxes).

    In order, each item joins the nearest object of its class whose centre lies within radius, and the
    object's centre becomes the weighted mean of its items; otherwise it starts a new object. Returns
    the objects, the largest first: dicts with "cls", "centre", "n" (the sum of the weights) and
    "members" (the indices of the items)."""
    objects = []
    for i in sorted(range(len(items)), key=lambda i: -items[i]["order"]):
        item = items[i]
        centre = np.asarray(item["centre"], np.float64)
        best = None
        for o in objects:
            if o["cls"] != item["cls"]:
                continue
            d = float(np.linalg.norm(o["centre"] - centre))
            if d <= radius and (best is None or d < best[0]):
                best = (d, o)
        if best is None:
            objects.append(dict(cls=item["cls"], centre=centre.copy(), n=item["weight"], members=[i]))
        else:
            o = best[1]
            o["centre"] = (o["centre"] * o["n"] + centre * item["weight"]) / (o["n"] + item["weight"])
            o["n"] += item["weight"]
            o["members"].append(i)
    objects.sort(key=lambda o: -o["n"])
    return objects


class SavedDetections:
    """The detections of main_object_detection.py --save DIR, lifted into the map frame by frame."""

    def __init__(self, folder, merge_radius=0.35, lift_depth="median", min_score=0.5):
        self.folder = folder
        self.merge_radius, self.lift_depth, self.min_score = float(merge_radius), lift_depth, float(min_score)
        path = os.path.join(folder, "detections.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"--detections {folder}: no detections.csv: write it with "
                f"`pixi run -e semantics object-detection ... --save {folder}`"
            )
        self.info = {}
        if os.path.isfile(os.path.join(folder, "info.json")):
            with open(os.path.join(folder, "info.json")) as f:
                self.info = json.load(f)
        self.by_frame = {}  # the detector's frame number -> rows
        self.frame_timestamps = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                frame = int(r["frame"])
                row = dict(frame=frame, timestamp=float(r["timestamp"]), index=int(r["index"]), cls=r["class"],
                           score=float(r["score"]), box=[float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                           mask_pixels=int(r.get("mask_pixels") or 0))
                self.by_frame.setdefault(frame, []).append(row)
                self.frame_timestamps[frame] = row["timestamp"]
        self.num_detections = sum(len(v) for v in self.by_frame.values())
        self._frames = sorted(self.frame_timestamps)
        self._stamps = np.array([self.frame_timestamps[f] for f in self._frames])
        self.lifted = []  # one per lifted detection: SLAM's frame, class, score, the point in the camera and in the map
        self.num_frames_matched = 0
        self.num_used = 0  # detections with a score of at least min_score in the matched frames
        self.num_not_lifted = 0
        self.depth_source = None

    def describe(self):
        what = ""
        if self.info:
            what = f" ({self.info.get('method')}" + (f", prompt \"{self.info['prompt']}\"" if self.info.get("prompt") else "") + ")"
        return f"{self.num_detections} detections in {len(self._frames)} frames from {self.folder}{what}"

    def rows_for(self, timestamp):
        """The detections of the frame of SLAM with this timestamp, and the detector's frame number."""
        if timestamp is None or len(self._stamps) == 0:
            return None, []
        i = int(np.argmin(np.abs(self._stamps - timestamp)))
        if abs(self._stamps[i] - timestamp) > kTimestampTolerance:
            return None, []
        frame = self._frames[i]
        return frame, self.by_frame[frame]

    def _masks(self, frame, shape):
        path = os.path.join(self.folder, "masks", f"{frame:06d}.png")
        if not os.path.isfile(path):
            return None
        index = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if index is None or index.shape[:2] != tuple(shape[:2]):
            return None
        return index

    def lift_frame(self, slam_frame_id, timestamp, camera, Twc, depth=None, frame=None):
        """Lift the detections of one frame of SLAM (those with a score of at least min_score). depth is
        the frame's depth image in metres (RGB-D), or None, and then the depth comes from the map points
        of frame (SLAM's current frame). Twc is the frame's pose (camera to map), 4 x 4. Returns the
        number of detections lifted."""
        det_frame, rows = self.rows_for(timestamp)
        if det_frame is None or Twc is None:
            return 0
        self.num_frames_matched += 1
        shape = (int(camera.height), int(camera.width))
        index = self._masks(det_frame, depth.shape if depth is not None else shape)
        map_uv = map_depth = None
        if depth is None and frame is not None:
            map_uv, map_depth = map_points_in_camera(frame)
        self.depth_source = ("the depth images" if depth is not None else "the map points (monocular: the map's own scale)")
        num = 0
        for r in rows:
            if r["score"] < self.min_score:
                continue
            self.num_used += 1
            mask = (index == r["index"]) if index is not None else None
            lifted = lift_mask(mask, r["box"], camera, Twc, depth=depth, map_uv=map_uv, map_depth=map_depth, how=self.lift_depth)
            if lifted is None:
                self.num_not_lifted += 1
                continue
            lifted.update(frame=int(slam_frame_id), timestamp=float(timestamp), cls=r["cls"], score=r["score"])
            self.lifted.append(lifted)
            num += 1
        return num

    def viewer_groups(self):
        """For the Map Viewer: the lifted points of each class, small, and the centres of the objects
        merged so far (with the poses of the run so far), large: [(N x 3 points, RGB colour, size)]."""
        if not self.lifted:
            return []
        groups = []
        by_class = {}
        for l in self.lifted:
            by_class.setdefault(l["cls"], []).append(l["p_w"])
        for cls, points in by_class.items():
            groups.append((np.asarray(points, np.float64), class_colour_rgb(cls), 3))
        objects = merge_into_objects([dict(cls=l["cls"], centre=l["p_w"], order=l["score"], weight=1) for l in self.lifted],
                                     self.merge_radius)
        for o in objects:
            groups.append((o["centre"].reshape(1, 3), class_colour_rgb(o["cls"]), 12))
        return groups

    def finish(self, poses, frame_ids, unit="m"):
        """Lift the detections again with the final trajectory (poses: T_wc of the frames frame_ids),
        merge them into objects and return (the line to print, the objects). unit: "m", or "map units"
        for a monocular map."""
        final = {int(i): np.asarray(T, np.float64) for i, T in zip(frame_ids, poses)}
        items = []
        frames = set()
        for l in self.lifted:
            T = final.get(l["frame"])
            if T is None:
                continue
            l["p_final"] = (T @ np.append(l["p_c"], 1.0))[:3]
            items.append(dict(cls=l["cls"], centre=l["p_final"], order=l["score"], weight=1))
            frames.add(l["frame"])
        objects = merge_into_objects(items, self.merge_radius)
        counts = {}
        for o in objects:
            counts[o["cls"]] = counts.get(o["cls"], 0) + 1
        listed = ", ".join(f"{c} x{n}" for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        single = sum(1 for o in objects if o["n"] == 1)
        line = (f"Objects: {len(objects)} from {len(items)} detections in {len(frames)} frames (score at least "
                f"{self.min_score:g}; the same class within {self.merge_radius:.2f} {unit})" + (f": {listed}" if listed else "")
                + f"; {single} single detection{'s' if single != 1 else ''}")
        for o in objects:
            P = np.array([items[i]["centre"] for i in o["members"]])
            o["spread"] = float(np.max(np.linalg.norm(P - o["centre"], axis=1))) if len(P) > 1 else 0.0
        return line, objects
