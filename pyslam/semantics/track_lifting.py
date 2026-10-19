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

# Tracks in the map: main_slam.py --tracks DIR reads the tracked boxes which main_object_tracking.py
# --save DIR wrote for the same sequence, and lifts each box of each frame into the map with the pose
# of that frame (every frame, not only the keyframes) and its depth:
#
#   the point of the box in the camera:  p_c = z K^-1 [u, v, 1]^T
#       (u, v): the centroid of the box's mask (the centre of the box if there is no mask)
#       z:      the median depth of the mask's pixels (with RGB-D), or of the map points SLAM
#               tracks inside the mask (monocular: in the map's own, unknown, scale)
#   the point in the map:                p_w = T_wc p_c
#
# A track becomes a path in the map. At the end, the boxes are lifted again with the final trajectory
# (after bundle adjustment and loop closing); a track whose points stay within --static-radius of their
# median is static, and the static tracks of one class whose centres lie within --merge-radius of each
# other are merged into one object. The objects are printed and drawn; no file of them is written.

import csv
import json
import os

import cv2
import numpy as np

kMinMaskPixels = 50  # pixels with a depth inside the mask (as Lecture 03's object map), else the box is not lifted
kMinMapPoints = 3  # monocular: map points inside the mask
kMinLifted = 5  # lifted boxes a track needs to be called static or moving
kTimestampTolerance = 1e-3  # s: a box belongs to the frame of SLAM with this timestamp
kStaticPercentile = 90  # the spread of a track: this percentile of the distances of its points to their median


def id_colour_rgb(track_id):
    """The colour of an identity in main_object_tracking.py's windows, as RGB in [0, 1]."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)


class SavedTracks:
    """The tracks of main_object_tracking.py --save DIR, lifted into the map frame by frame."""

    def __init__(self, folder, static_radius=0.30, merge_radius=0.35):
        self.folder = folder
        self.static_radius, self.merge_radius = float(static_radius), float(merge_radius)
        path = os.path.join(folder, "tracks.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"--tracks {folder}: no tracks.csv: write it with `pixi run -e semantics object-tracking ... --save {folder}`"
            )
        self.info = {}
        if os.path.isfile(os.path.join(folder, "info.json")):
            with open(os.path.join(folder, "info.json")) as f:
                self.info = json.load(f)
        self.by_frame = {}  # the tracker's frame number -> rows
        self.frame_timestamps = {}  # the tracker's frame number -> timestamp
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                frame = int(r["frame"])
                row = dict(frame=frame, timestamp=float(r["timestamp"]), id=int(r["id"]), cls=r["class"],
                           score=float(r["score"]), box=[float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                           mask_pixels=int(r.get("mask_pixels") or 0))
                self.by_frame.setdefault(frame, []).append(row)
                self.frame_timestamps[frame] = row["timestamp"]
        self.num_boxes = sum(len(v) for v in self.by_frame.values())
        self.num_ids = len({r["id"] for v in self.by_frame.values() for r in v})
        self._frames = sorted(self.frame_timestamps)
        self._stamps = np.array([self.frame_timestamps[f] for f in self._frames])
        self.lifted = []  # one per lifted box: frame (SLAM's), id, class, the point in the camera, the point in the map (online)
        self.num_frames_matched = 0
        self.num_not_lifted = 0
        self.depth_source = None

    def describe(self):
        return (f"{self.num_boxes} boxes of {self.num_ids} identities from {self.folder}"
                + (f" ({self.info.get('tracker')} on {self.info.get('method')})" if self.info else ""))

    def rows_for(self, timestamp):
        """The tracked boxes of the frame of SLAM with this timestamp, and the tracker's frame number."""
        if timestamp is None or len(self._stamps) == 0:
            return None, []
        i = int(np.argmin(np.abs(self._stamps - timestamp)))
        if abs(self._stamps[i] - timestamp) > kTimestampTolerance:
            return None, []
        frame = self._frames[i]
        return frame, self.by_frame[frame]

    def _mask(self, frame, shape):
        path = os.path.join(self.folder, "masks", f"{frame:06d}.png")
        if not os.path.isfile(path):
            return None
        ids = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if ids is None or ids.shape[:2] != shape[:2]:
            return None
        return ids

    def lift_frame(self, slam_frame_id, timestamp, camera, Twc, depth=None, frame=None):
        """Lift the boxes of one frame of SLAM: depth is the frame's depth image in metres (RGB-D), or
        None, and then the depth comes from the map points of frame (SLAM's current frame) inside each
        box. Twc is the frame's pose (camera to map), 4 x 4. Returns the number of boxes lifted."""
        tracker_frame, rows = self.rows_for(timestamp)
        if tracker_frame is None or Twc is None:
            return 0
        self.num_frames_matched += 1
        shape = (int(camera.height), int(camera.width))
        ids = self._mask(tracker_frame, depth.shape if depth is not None else shape)
        map_uv = map_depth = None
        if depth is None and frame is not None:
            map_uv, map_depth = map_points_in_camera(frame)
        Kinv = np.linalg.inv(np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.0]]))
        num = 0
        for r in rows:
            x1, y1, x2, y2 = r["box"]
            if ids is not None:
                mask = ids == r["id"]
            else:  # no mask: the inner half of the box
                mask = np.zeros(shape, bool)
                cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                mask[max(int(cy - h / 4), 0):int(cy + h / 4) + 1, max(int(cx - w / 4), 0):int(cx + w / 4) + 1] = True
            vv, uu = np.nonzero(mask)
            if len(uu) == 0:
                self.num_not_lifted += 1
                continue
            u, v = float(uu.mean()), float(vv.mean())
            z = None
            if depth is not None:
                self.depth_source = "the depth images"
                zs = depth[mask]
                zs = zs[zs > 0]
                if len(zs) >= kMinMaskPixels:
                    z = float(np.median(zs))
            elif map_uv is not None and len(map_uv):
                self.depth_source = "the map points (monocular: the map's own scale)"
                pu = np.clip(np.rint(map_uv[:, 0]).astype(int), 0, shape[1] - 1)
                pv = np.clip(np.rint(map_uv[:, 1]).astype(int), 0, shape[0] - 1)
                inside = mask[pv, pu]
                if np.count_nonzero(inside) >= kMinMapPoints:
                    z = float(np.median(map_depth[inside]))
            if z is None:
                self.num_not_lifted += 1
                continue
            p_c = z * (Kinv @ np.array([u, v, 1.0]))
            p_w = (Twc @ np.append(p_c, 1.0))[:3]
            self.lifted.append(dict(frame=int(slam_frame_id), timestamp=float(timestamp), id=r["id"], cls=r["cls"],
                                    uv=(u, v), z=z, p_c=p_c, p_w=p_w))
            num += 1
        return num

    def paths(self):
        """The path of each identity in the map (online poses): [(N x 3 points, RGB colour)]."""
        by_id = {}
        for l in self.lifted:
            by_id.setdefault(l["id"], []).append(l["p_w"])
        return [(np.asarray(p, np.float64), id_colour_rgb(i)) for i, p in by_id.items()]

    def objects(self, poses, frame_ids):
        """Lift the boxes again with the final trajectory (poses: T_wc of the frames frame_ids), split the
        tracks into static and moving, and merge the static ones into objects. Returns the summary line
        and the objects: dicts with the class, the centre, the tracks, the number of boxes and the spread
        (the distance from the centre within which 90 % of the object's points lie). Nothing is written to
        a file: the objects are printed and drawn only."""
        final = {int(i): np.asarray(T, np.float64) for i, T in zip(frame_ids, poses)}
        by_id = {}
        for l in self.lifted:
            T = final.get(l["frame"])
            if T is None:
                continue
            l["p_final"] = (T @ np.append(l["p_c"], 1.0))[:3]
            by_id.setdefault(l["id"], []).append(l)
        tracks = []
        for track_id, ls in by_id.items():
            counts = {}
            for l in ls:
                counts[l["cls"]] = counts.get(l["cls"], 0) + 1
            P = np.array([l["p_final"] for l in ls])
            centre = np.median(P, axis=0)
            spread = float(np.percentile(np.linalg.norm(P - centre, axis=1), kStaticPercentile))
            tracks.append(dict(id=track_id, cls=max(counts, key=counts.get), n=len(ls), centre=centre, spread=spread, points=P))
        used = [t for t in tracks if t["n"] >= kMinLifted]
        static = [t for t in used if t["spread"] < self.static_radius]
        moving = [t for t in used if t["spread"] >= self.static_radius]
        objects = []
        for t in sorted(static, key=lambda t: -t["n"]):  # greedy, the longest tracks first
            best = None
            for o in objects:
                if o["cls"] != t["cls"]:
                    continue
                d = float(np.linalg.norm(o["centre"] - t["centre"]))
                if d <= self.merge_radius and (best is None or d < best[0]):
                    best = (d, o)
            if best is None:
                objects.append(dict(cls=t["cls"], centre=t["centre"].copy(), tracks=[t["id"]], n=t["n"], points=[t["points"]]))
            else:
                o = best[1]
                o["centre"] = (o["centre"] * o["n"] + t["centre"] * t["n"]) / (o["n"] + t["n"])
                o["tracks"].append(t["id"])
                o["n"] += t["n"]
                o["points"].append(t["points"])
        for o in objects:
            P = np.concatenate(o.pop("points"))
            o["spread"] = float(np.percentile(np.linalg.norm(P - o["centre"], axis=1), kStaticPercentile))
        objects.sort(key=lambda o: -o["n"])
        counts = {}
        for o in objects:
            counts[o["cls"]] = counts.get(o["cls"], 0) + 1
        listed = ", ".join(f"{c} x{n}" for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        line = (f"Objects: {len(objects)} from {len(static)} static tracks" + (f": {listed}" if listed else "")
                + f"; moving: {len(moving)} tracks (static: within {self.static_radius:.2f} m of their centre; "
                f"merged: the same class within {self.merge_radius:.2f} m; {len(tracks) - len(used)} tracks with fewer "
                f"than {kMinLifted} lifted boxes left out)")
        return line, objects

    def object_lines(self, objects, metric=True):
        """One line per object, for the terminal: its class, centre, tracks, boxes and spread."""
        unit = "m" if metric else "map units"
        return [
            f"  {o['cls']}: centre ({o['centre'][0]:.2f}, {o['centre'][1]:.2f}, {o['centre'][2]:.2f}) {unit}; "
            f"{len(o['tracks'])} track{'s' if len(o['tracks']) != 1 else ''} ({', '.join(str(i) for i in o['tracks'])}), "
            f"{o['n']} boxes; 90 % of its points within {o['spread']:.2f} {unit}"
            for o in objects
        ]


def class_colour_rgb(name):
    """A colour for each class (the same in every run), RGB in [0, 1], for the objects in the Map Viewer."""
    return id_colour_rgb(sum((i + 1) * ord(c) for i, c in enumerate(name)))


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
