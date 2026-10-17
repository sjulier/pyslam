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

# The scores of a tracker against known tracks (main_object_tracking.py --evaluate DIR): MOTA (the
# CLEAR MOT metrics), IDF1 (the identity metrics) and HOTA, computed as TrackEval computes them
# (github.com/JonathonLuiten/TrackEval: clear.py, identity.py, hota.py), boxes matched by their IoU.
#
# The known tracks are in the layout of the MOTChallenge benchmark:
#   DIR/img1/000001.jpg ...       the frames, in the order of their names
#   DIR/gt/gt.txt                 one line per box: frame, id, left, top, width, height, flag, class, visibility
#                                 (frames from 1; flag 0: ignore the box; class 1: a pedestrian)
#   DIR/gt/labelled_frames.txt    optional: the frames which were labelled, one number per line. Without it
#                                 every frame was labelled; with it, only these frames are scored (a frame
#                                 listed here with no line in gt.txt has no people in it)
#   DIR/seqinfo.ini               optional: name, frameRate, seqLength, imDir, imExt

import configparser
import glob
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

kEps = np.finfo("float").eps
kPedestrian = 1  # the class of the boxes which are scored
kDistractors = (2, 7, 8, 12)  # MOT17's person on vehicle, static person, distractor, reflection
kHotaAlphas = np.arange(0.05, 0.99, 0.05)  # the 19 IoU thresholds HOTA is averaged over


def read_mot_folder(folder):
    """The frames (paths, in order), the known boxes by frame, the scored frames, and the sequence's
    name and frame rate, from a folder in the MOTChallenge layout."""
    info = {}
    ini = os.path.join(folder, "seqinfo.ini")
    if os.path.isfile(ini):
        parser = configparser.ConfigParser()
        parser.read(ini)
        if parser.has_section("Sequence"):
            info = dict(parser["Sequence"])
    image_dir = os.path.join(folder, info.get("imdir", "img1"))
    extensions = [info["imext"]] if "imext" in info else [".jpg", ".png"]
    images = []
    for ext in extensions:
        images += glob.glob(os.path.join(image_dir, "*" + ext))
    images.sort()
    if not images:
        raise FileNotFoundError(f"no images in {image_dir}")
    gt_path = os.path.join(folder, "gt", "gt.txt")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"no gt/gt.txt in {folder}")
    gt = read_mot_boxes(gt_path, with_classes=True)
    labelled_path = os.path.join(folder, "gt", "labelled_frames.txt")
    if os.path.isfile(labelled_path):
        with open(labelled_path) as f:
            scored = sorted({int(l.split()[0]) for l in f if l.strip() and not l.lstrip().startswith("#")})
    else:
        scored = list(range(1, len(images) + 1))
    name = info.get("name", os.path.basename(os.path.normpath(folder)))
    fps = float(info["framerate"]) if "framerate" in info else None
    return images, gt, scored, name, fps


def read_mot_boxes(path, with_classes=False):
    """{frame: rows}, each row (id, left, top, width, height, flag, class), from a MOTChallenge file.
    A tracker's file (with_classes False) gets flag 1 and class 1 on every row."""
    frames = {}
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 6 or line.lstrip().startswith("#"):
                continue
            frame, track_id = int(float(parts[0])), int(float(parts[1]))
            box = [float(v) for v in parts[2:6]]
            flag = int(float(parts[6])) if with_classes and len(parts) > 6 else 1
            cls = int(float(parts[7])) if with_classes and len(parts) > 7 else kPedestrian
            frames.setdefault(frame, []).append((track_id, *box, flag, cls))
    return frames


def box_iou(a, b):
    """The IoU of every box of a (N x 4) with every box of b (M x 4), boxes as left, top, width, height."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(union > kEps, inter / np.maximum(union, kEps), 0.0)


def prepare(gt_frames, tracker_frames, scored_frames):
    """The data of TrackEval's metrics: for each scored frame, the known ids and the tracker's ids
    (renumbered from 0) and the IoU of their boxes. As TrackEval does for MOTChallenge, a tracker box
    matched (IoU >= 0.5) to a known distractor (a person on a vehicle, a static person, a reflection,
    ...) is dropped, and only the known pedestrians whose flag is not 0 are scored."""
    steps = []
    for frame in scored_frames:
        gt_rows = gt_frames.get(frame, [])
        tr_rows = tracker_frames.get(frame, [])
        gt_ids = np.array([r[0] for r in gt_rows], int)
        gt_boxes = np.array([r[1:5] for r in gt_rows], float).reshape(-1, 4)
        gt_flags = np.array([r[5] for r in gt_rows], int)
        gt_classes = np.array([r[6] for r in gt_rows], int)
        tr_ids = np.array([r[0] for r in tr_rows], int)
        tr_boxes = np.array([r[1:5] for r in tr_rows], float).reshape(-1, 4)
        sim = box_iou(gt_boxes, tr_boxes)
        if len(gt_ids) and len(tr_ids):
            scores = sim.copy()
            scores[scores < 0.5 - kEps] = 0
            rows, cols = linear_sum_assignment(-scores)
            matched = scores[rows, cols] > 0 + kEps
            remove = cols[matched][np.isin(gt_classes[rows[matched]], kDistractors)]
            keep = np.setdiff1d(np.arange(len(tr_ids)), remove)
            tr_ids, tr_boxes, sim = tr_ids[keep], tr_boxes[keep], sim[:, keep]
        keep_gt = (gt_flags != 0) & (gt_classes == kPedestrian)
        steps.append((gt_ids[keep_gt], tr_ids, sim[keep_gt]))
    gt_unique = np.unique(np.concatenate([s[0] for s in steps])) if steps else np.array([], int)
    tr_unique = np.unique(np.concatenate([s[1] for s in steps])) if steps else np.array([], int)
    gt_map = {int(v): i for i, v in enumerate(gt_unique)}
    tr_map = {int(v): i for i, v in enumerate(tr_unique)}
    data = [
        (np.array([gt_map[int(v)] for v in g], int), np.array([tr_map[int(v)] for v in t], int), s)
        for g, t, s in steps
    ]
    return data, len(gt_unique), len(tr_unique)


def clear_mot(data, num_gt_ids, threshold=0.5):
    """MOTA and its parts (TrackEval's clear.py): matches with IoU >= threshold, a match of the previous
    frame kept where it is still possible, and an identity switch where a known object is matched to
    another identity than the last one it had."""
    tp = fn = fp = idsw = 0
    prev_tracker_id = np.full(num_gt_ids, np.nan)  # the last identity each known object was matched to
    prev_step_tracker_id = np.full(num_gt_ids, np.nan)  # its identity in the previous frame, if matched
    for gt_ids, tr_ids, sim in data:
        if len(gt_ids) == 0:
            fp += len(tr_ids)
            continue
        if len(tr_ids) == 0:
            fn += len(gt_ids)
            continue
        score = 1000 * (tr_ids[None, :] == prev_step_tracker_id[gt_ids[:, None]]) + sim
        score[sim < threshold - kEps] = 0
        rows, cols = linear_sum_assignment(-score)
        matched = score[rows, cols] > 0 + kEps
        rows, cols = rows[matched], cols[matched]
        matched_gt, matched_tr = gt_ids[rows], tr_ids[cols]
        prev = prev_tracker_id[matched_gt]
        idsw += int(np.sum(~np.isnan(prev) & (matched_tr != prev)))
        prev_tracker_id[matched_gt] = matched_tr
        prev_step_tracker_id[:] = np.nan
        prev_step_tracker_id[matched_gt] = matched_tr
        tp += len(rows)
        fn += len(gt_ids) - len(rows)
        fp += len(tr_ids) - len(rows)
    mota = (tp - fp - idsw) / max(1.0, tp + fn)
    return {"MOTA": mota, "TP": tp, "FN": fn, "FP": fp, "IDSW": idsw}


def identity(data, num_gt_ids, num_tr_ids, threshold=0.5):
    """IDF1 (TrackEval's identity.py): each known track is paired with at most one identity, and each
    identity with at most one known track, so as to cover the most boxes (IoU >= threshold); IDTP is
    the number of boxes covered by the pairs."""
    potential = np.zeros((num_gt_ids, num_tr_ids))
    num_gt = num_tr = 0
    for gt_ids, tr_ids, sim in data:
        rows, cols = np.nonzero(sim >= threshold)
        np.add.at(potential, (gt_ids[rows], tr_ids[cols]), 1)
        num_gt += len(gt_ids)
        num_tr += len(tr_ids)
    idtp = 0
    if potential.size:
        rows, cols = linear_sum_assignment(-potential)
        idtp = int(potential[rows, cols].sum())
    idfn, idfp = num_gt - idtp, num_tr - idtp
    idf1 = idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)
    return {"IDF1": idf1, "IDTP": idtp, "IDFN": idfn, "IDFP": idfp}


def hota(data, num_gt_ids, num_tr_ids):
    """HOTA, DetA and AssA (TrackEval's hota.py), averaged over the IoU thresholds 0.05 to 0.95."""
    num_gt = sum(len(g) for g, _, _ in data)
    num_tr = sum(len(t) for _, t, _ in data)
    if num_tr == 0 or num_gt == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}
    potential = np.zeros((num_gt_ids, num_tr_ids))
    gt_count = np.zeros((num_gt_ids, 1))
    tr_count = np.zeros((1, num_tr_ids))
    for gt_ids, tr_ids, sim in data:
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 0 + kEps
        sim_iou[mask] = sim[mask] / denom[mask]
        potential[gt_ids[:, None], tr_ids[None, :]] += sim_iou
        gt_count[gt_ids] += 1
        tr_count[0, tr_ids] += 1
    global_alignment = potential / (gt_count + tr_count - potential)
    n = len(kHotaAlphas)
    tp, fn, fp = np.zeros(n), np.zeros(n), np.zeros(n)
    matches = [np.zeros_like(potential) for _ in range(n)]
    for gt_ids, tr_ids, sim in data:
        if len(gt_ids) == 0:
            fp += len(tr_ids)
            continue
        if len(tr_ids) == 0:
            fn += len(gt_ids)
            continue
        score = global_alignment[gt_ids[:, None], tr_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(kHotaAlphas):
            ok = sim[rows, cols] >= alpha - kEps
            r, c = rows[ok], cols[ok]
            tp[a] += len(r)
            fn[a] += len(gt_ids) - len(r)
            fp[a] += len(tr_ids) - len(r)
            if len(r):
                matches[a][gt_ids[r], tr_ids[c]] += 1
    det_a = tp / np.maximum(1, tp + fn + fp)
    ass_a = np.array([
        np.sum(m * (m / np.maximum(1, gt_count + tr_count - m))) / max(1.0, tp[a]) for a, m in enumerate(matches)
    ])
    hota_a = np.sqrt(det_a * ass_a)
    return {"HOTA": float(hota_a.mean()), "DetA": float(det_a.mean()), "AssA": float(ass_a.mean())}


def score_tracks(gt_frames, tracker_frames, scored_frames):
    """MOTA, IDF1 and HOTA, with their parts, of the tracker's boxes against the known ones, over the
    scored frames. Both are {frame: rows} as read_mot_boxes() gives them."""
    data, num_gt_ids, num_tr_ids = prepare(gt_frames, tracker_frames, scored_frames)
    out = {"frames": len(scored_frames), "known_ids": num_gt_ids, "tracker_ids": num_tr_ids,
           "known_boxes": int(sum(len(g) for g, _, _ in data)), "tracker_boxes": int(sum(len(t) for _, t, _ in data))}
    out.update(clear_mot(data, num_gt_ids))
    out.update(identity(data, num_gt_ids, num_tr_ids))
    out.update(hota(data, num_gt_ids, num_tr_ids))
    return out
