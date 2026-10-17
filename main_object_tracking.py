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

# The objects of every frame of a sequence, detected and tracked (with --tum, every colour image of
# rgb.txt, unless --associations names another list): every object the tracker keeps gets
# an identity, a number which should stay the same while the object is in view. The detector is
# YOLO (or YOLOE with a text prompt), the trackers ByteTrack and BoT-SORT, through Ultralytics' track
# mode (model.track(persist=True, tracker=...)), with their settings in settings/trackers/.
#
# --predict own replaces the trackers' prediction step with predict_box() of
# pyslam/semantics/track_utils.py. --evaluate DIR scores the tracker against known tracks (MOTA, IDF1
# and HOTA, pyslam/semantics/tracking_scores.py). --save DIR writes the tracked boxes (and masks) of
# every frame for main_slam.py --tracks DIR, which lifts them into the map.
#
#   pixi run -e semantics object-tracking --list
#   pixi run -e semantics object-tracking --tracker BYTETRACK --tum $S --settings settings/TUM3.yaml
#   pixi run -e semantics object-tracking --tracker BOTSORT --video data/videos/kitti06/video_color.mp4
#   pixi run -e semantics object-tracking --tracker BYTETRACK --evaluate data/tracking_sets/NAME
#   pixi run -e semantics object-tracking --tracker BOTSORT --tum $S --settings settings/TUM3.yaml --save tracks/NAME

import pyslam  # first: sets the OpenMP thread settings before numpy/torch are imported
import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)

# The trackers (--tracker): their settings file in settings/trackers/ (copies of Ultralytics' own) and what each does
kTrackers = {
    "BYTETRACK": ("bytetrack.yaml", "ByteTrack: a Kalman filter on each box, matched by box overlap in two passes, the "
                  "confident boxes first, then the weak ones. The camera's motion is ignored"),
    "BOTSORT": ("botsort.yaml", "BoT-SORT: as ByteTrack, after compensating the camera's motion, estimated from the "
                "image (sparse optical flow); appearance vectors can be added (with_reid, off by default)"),
}
# The detectors (--method) which feed them: their weights, in data/models/
kMethods = {
    "YOLO": ("yolo26s-seg.pt", "YOLO26-S with masks: 80 object classes (COCO), the detector of Lab 04"),
    "YOLOE": ("yoloe-26s-seg.pt", "YOLOE-26-S with masks: the classes named by --prompt (open vocabulary; the text "
              "encoder, MobileCLIP2, is downloaded on the first run)"),
}
kWeightsUrl = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}"
kModelsFolder = os.path.join(kScriptFolder, "data", "models")
kTrackerSettingsFolder = os.path.join(kScriptFolder, "settings", "trackers")
kPerson = "person"
kShortTrack = 10  # frames: a track seen in fewer frames is counted as short in the summary
kTrailLength = 30  # frames of each box's trail in the Tracks window
kTimelineWidth = 900  # pixels of the Timeline window given to the frames
kTimelineHeight = 560  # pixels of the Timeline window given to the rows


def main():
    from pyslam.io.sequence_options import add_sequence_options, has_sequence_options

    parser = argparse.ArgumentParser(
        description="The objects of every frame of a sequence, detected and tracked: each tracked object gets an "
        "identity which should stay the same while it is in view",
        epilog="examples: pixi run -e semantics object-tracking --tracker BYTETRACK --tum $S --settings "
        "settings/TUM3.yaml; pixi run -e semantics object-tracking --tracker BOTSORT --evaluate "
        "data/tracking_sets/NAME; pixi run -e semantics object-tracking --tracker BOTSORT --tum $S --settings "
        "settings/TUM3.yaml --save tracks/NAME",
    )
    parser.add_argument("--tracker", default="BYTETRACK", metavar="NAME", help="the tracker (default BYTETRACK; see --list)")
    parser.add_argument("--method", default="YOLO", metavar="NAME", help="the detector which feeds it (default YOLO; see --list)")
    parser.add_argument("--prompt", metavar="TEXT", help='YOLOE: the classes to detect, separated by commas, e.g. "monitor, keyboard, person"')
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="where the detector runs (default auto: an NVIDIA GPU, else Apple's GPU, else the CPU)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="the detector's lowest score: weaker boxes are not given to the tracker (default 0.1, as "
                        "Ultralytics' track mode: the second pass needs the weak boxes)")
    parser.add_argument("--second-pass", choices=("on", "off"), default="on",
                        help="off: no second pass, so the weak boxes (between track_low_thresh and track_high_thresh) "
                        "are never matched to a track (default on)")
    parser.add_argument("--predict", choices=("tracker", "own"), default="tracker",
                        help="own: predict each box with predict_box() of pyslam/semantics/track_utils.py instead of the "
                        "tracker's own Kalman prediction (default tracker)")
    add_sequence_options(parser, need_settings=False)
    sequence = parser.add_argument_group("with a sequence")
    sequence.add_argument("--evaluate", metavar="DIR",
                          help="track the frames of DIR (in the MOTChallenge layout: img1/, gt/gt.txt) and score the "
                          "tracks of the people against its known tracks: MOTA, IDF1, HOTA")
    sequence.add_argument("--save", metavar="DIR",
                          help="write the tracked boxes of every frame to this folder (tracks.csv, the MOTChallenge file "
                          "tracks.txt, masks/, info.json), for main_slam.py --tracks DIR")
    sequence.add_argument("--step", type=int, default=1, metavar="N", help="track every Nth frame (default 1: every frame)")
    sequence.add_argument("--max-frames", type=int, metavar="N", help="stop after N tracked frames")
    sequence.add_argument("--headless", action="store_true", help="no windows")
    parser.add_argument("--list", action="store_true", help="list the trackers and the detectors")
    args = parser.parse_args()

    if args.list:
        print("Trackers (--tracker), and what each does:")
        for name, (settings, what) in kTrackers.items():
            print(f"    {name:10s} {what}. Settings: settings/trackers/{settings}")
        print("Detectors (--method) which feed them:")
        for name, (weights, what) in kMethods.items():
            print(f"    {name:10s} {what}")
        print("Both need the semantics level: pixi run -e semantics object-tracking ... (the weights are downloaded "
              "to data/models/ on the first run).")
        sys.exit(0)
    args.tracker, args.method = args.tracker.upper(), args.method.upper()
    if args.tracker not in kTrackers:
        print(f"Unknown tracker '{args.tracker}'. Valid names (see --list): {' '.join(kTrackers)}", file=sys.stderr)
        sys.exit(2)
    if args.method not in kMethods:
        print(f"Unknown method '{args.method}'. Valid names (see --list): {' '.join(kMethods)}", file=sys.stderr)
        sys.exit(2)
    if args.method == "YOLOE" and not args.prompt:
        parser.error('--method YOLOE needs --prompt, e.g. --prompt "monitor, keyboard, person"')
    if args.prompt and args.method != "YOLOE":
        parser.error("--prompt is for --method YOLOE: YOLO has its 80 classes")
    if args.evaluate and has_sequence_options(args):
        parser.error("--evaluate DIR tracks the frames of DIR: give it without --video, --images or --tum")
    if not args.evaluate and not has_sequence_options(args):
        parser.error("give a sequence (--tum, --video or --images) or --evaluate DIR")
    if args.step < 1:
        parser.error("--step must be 1 or more")
    if importlib.util.find_spec("ultralytics") is None or importlib.util.find_spec("lap") is None:
        sys.exit("object-tracking needs the semantics level: run this command with `pixi run -e semantics object-tracking ...`.")

    from pyslam.utilities.logging import Printer

    if args.evaluate:
        frames = MotFrames(args.evaluate)
    else:
        frames = SequenceFrames(args, parser)
    tracker = Tracker(args, Printer)
    if args.evaluate and kPerson not in tracker.names.values():
        sys.exit(f"--evaluate scores the tracks of the people, and {tracker.describe_detector()} has no class '{kPerson}'"
                 + (": add it to --prompt" if args.method == "YOLOE" else ""))
    run(args, frames, tracker, Printer)


# ---------------------------------------------------------------------------------------------------
# The frames: a sequence of pySLAM's readers, or a folder in the MOTChallenge layout (--evaluate)
# ---------------------------------------------------------------------------------------------------


class SequenceFrames:
    """The frames of --tum, --video or --images, read by pySLAM's dataset readers (as main_slam.py reads
    them, so that the frame numbers and timestamps are the same)."""

    def __init__(self, args, parser):
        from pyslam.config import Config
        from pyslam.io.dataset_factory import dataset_factory
        from pyslam.io.sequence_options import sequence_config_path

        if args.tum and not args.associations:
            # every colour image, from rgb.txt: the tracker needs no depth, and associations.txt leaves out the
            # frames without a depth image (main_slam.py --tracks finds the boxes of its frames by timestamp)
            folder = os.path.abspath(os.path.expanduser(args.tum)).rstrip("/")
            if os.path.isfile(os.path.join(folder, "rgb.txt")):
                args.associations = "rgb.txt"
                args.sensor = args.sensor or "mono"
        config_path = sequence_config_path(args, parser, default_settings=os.path.join(kScriptFolder, "settings", "KITTI04-12.yaml"))
        config = Config(config_path)
        os.remove(config_path)  # the temporary file of sequence_config_path(): it has been read
        self.dataset = dataset_factory(config)
        self.source = args.tum or args.video or args.images
        self.name = os.path.basename(os.path.normpath(self.source))
        if args.video:  # e.g. kitti06/video_color.mp4
            self.name = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(self.source))), self.name)
        self.option = "--tum" if args.tum else "--video" if args.video else "--images"
        self.num_total = getattr(self.dataset, "num_frames", None)
        self.list_name = None
        if args.tum:
            folder = os.path.abspath(os.path.expanduser(args.tum)).rstrip("/")
            self.list_name = args.associations or ("associations.txt" if os.path.isfile(os.path.join(folder, "associations.txt")) else "rgb.txt")
            print(f"--tum: the frames of {self.list_name}" + (" (every colour image)" if self.list_name == "rgb.txt" else ""))

    def __iter__(self):
        """(frame number, timestamp, image), in order, until the sequence ends."""
        frame_id = 0
        while self.dataset.is_ok:
            img = self.dataset.getImageColor(frame_id)
            if img is None:
                break
            yield frame_id, self.dataset.getTimestamp(), img
            frame_id += 1


class MotFrames:
    """The frames of a folder in the MOTChallenge layout, with its known tracks (--evaluate)."""

    def __init__(self, folder):
        from pyslam.semantics.tracking_scores import read_mot_folder

        if not os.path.isdir(folder):
            sys.exit(f"--evaluate: no folder {folder}")
        try:
            self.images, self.gt, self.scored, self.name, self.fps = read_mot_folder(folder)
        except FileNotFoundError as e:
            sys.exit(f"--evaluate {folder}: {e} (the layout of the MOTChallenge benchmark: img1/000001.jpg ..., gt/gt.txt)")
        self.source = folder
        self.option = None
        self.num_total = len(self.images)
        self.list_name = None

    def __iter__(self):
        for i, path in enumerate(self.images):
            img = cv2.imread(path)
            if img is None:
                sys.exit(f"cannot read {path}")
            yield i, (i / self.fps if self.fps else float(i)), img


# ---------------------------------------------------------------------------------------------------
# The detector and the tracker
# ---------------------------------------------------------------------------------------------------


def read_yaml(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def box_from_state(mean, box_state):
    """x1, y1, x2, y2 of a Kalman state (centre x, centre y, aspect ratio or width, height, ...)."""
    cx, cy, a, h = (float(v) for v in mean[:4])
    w = a * h if box_state == "xyah" else a
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class OwnPrediction:
    """--predict own: the prediction step of the trackers' Kalman filters (Ultralytics' KalmanFilterXYAH
    for ByteTrack, KalmanFilterXYWH for BoT-SORT) replaced by predict_box(). If it returns None, a
    warning is printed once and the filter's own prediction is used for the rest of the run. Each own
    prediction is compared with the filter's own, for the summary."""

    def __init__(self, Printer):
        self.Printer = Printer
        self.returned_none = False
        self.failed = False
        self.used = 0  # tracks predicted by predict_box()
        self.largest_difference = 0.0  # pixels: the largest difference from the filter's own predicted box

    def install(self):
        from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH

        for kf_class, box_state in ((KalmanFilterXYAH, "xyah"), (KalmanFilterXYWH, "xywh")):
            own_predict = kf_class.__dict__["predict"]
            own_multi_predict = kf_class.__dict__["multi_predict"]

            def predict(kf, mean, covariance, _own=own_predict, _state=box_state):
                theirs = _own(kf, mean, covariance)
                ours = self._predict_one(mean, covariance, _state, theirs)
                return ours if ours is not None else theirs

            def multi_predict(kf, mean, covariance, _own=own_multi_predict, _state=box_state):
                theirs = _own(kf, mean, covariance)
                if self.returned_none or self.failed:
                    return theirs
                ours = [self._predict_one(m, P, _state, (tm, tP)) for m, P, tm, tP in zip(mean, covariance, *theirs)]
                if any(o is None for o in ours):
                    return theirs
                return np.stack([o[0] for o in ours]), np.stack([o[1] for o in ours])

            kf_class.predict = predict
            kf_class.multi_predict = multi_predict

    def _predict_one(self, mean, covariance, box_state, theirs):
        from pyslam.semantics import track_utils

        if self.returned_none or self.failed:
            return None
        track_utils.kBoxState = box_state
        out = track_utils.predict_box(np.array(mean, np.float64), np.array(covariance, np.float64))
        if out is None:
            self.returned_none = True
            self.Printer.orange(
                "predict_box() in pyslam/semantics/track_utils.py returned None: it is not written yet (COMP0227 Lab 05, "
                "Activity 4), so the tracker uses its own prediction"
            )
            return None
        try:
            m, P = out
            m = np.asarray(m, np.float64).reshape(8)
            P = np.asarray(P, np.float64).reshape(8, 8)
            if not (np.all(np.isfinite(m)) and np.all(np.isfinite(P))):
                raise ValueError("it is not finite")
        except Exception as e:
            self.failed = True
            self.Printer.red(
                f"predict_box() must return (the predicted state, shape (8,), the predicted covariance, shape (8, 8)): {e}. "
                "The tracker uses its own prediction."
            )
            return None
        self.used += 1
        mine, own = box_from_state(m, box_state), box_from_state(theirs[0], box_state)
        self.largest_difference = max(self.largest_difference, max(abs(a - b) for a, b in zip(mine, own)))
        return m, P


class Tracker:
    """A detector (YOLO or YOLOE) and a tracker (ByteTrack or BoT-SORT), through Ultralytics' track mode."""

    def __init__(self, args, Printer):
        import torch
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT

        self.name, self.method, self.Printer = args.tracker, args.method, Printer
        self.conf = args.conf
        self.second_pass = args.second_pass == "on"

        # the tracker's settings: settings/trackers/NAME.yaml (or Ultralytics' own, if the copy is missing)
        settings_name = kTrackers[self.name][0]
        self.settings_path = os.path.join(kTrackerSettingsFolder, settings_name)
        if not os.path.isfile(self.settings_path):
            from ultralytics.utils.checks import check_yaml

            Printer.orange(f"no settings/trackers/{settings_name}: Ultralytics' own settings are used")
            self.settings_path = str(check_yaml(settings_name))
        self.settings = read_yaml(self.settings_path)
        self.tracker_yaml = self.settings_path
        if not self.second_pass:
            # no box lies between the two thresholds, so the second pass has nothing to match
            settings = dict(self.settings, track_low_thresh=self.settings["track_high_thresh"])
            with tempfile.NamedTemporaryFile("w", prefix="pyslam_tracker_", suffix=".yaml", delete=False) as f:
                import yaml

                yaml.safe_dump(settings, f)
            self.tracker_yaml = f.name
        keys = ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh")
        print(f"{self.name} settings: {os.path.relpath(self.settings_path, kScriptFolder)} ("
              + ", ".join(f"{k} {self.settings.get(k)}" for k in keys) + ")"
              + ("" if self.second_pass else f"; --second-pass off: track_low_thresh is set to track_high_thresh "
                 f"({self.settings['track_high_thresh']}), so no box is left for the second pass"))

        # the device: CUDA, then Apple's MPS, then the CPU (or the one of --device)
        if args.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        elif args.device == "cuda" and not torch.cuda.is_available():
            sys.exit("--device cuda: there is no NVIDIA GPU that torch can use here")
        elif args.device == "mps" and not torch.backends.mps.is_available():
            sys.exit("--device mps: Apple's GPU is not available here")
        else:
            self.device = "cuda:0" if args.device == "cuda" else args.device
        self.device_name = self.device.split(":")[0]

        # the detector
        weights = os.path.join(kModelsFolder, kMethods[self.method][0])
        if not os.path.isfile(weights):
            os.makedirs(kModelsFolder, exist_ok=True)
            url = kWeightsUrl.format(os.path.basename(weights))
            Printer.green(f"Downloading {os.path.basename(weights)} to data/models/ ({url})")
            urllib.request.urlretrieve(url, weights + ".part")
            os.replace(weights + ".part", weights)
        start = time.time()
        if self.method == "YOLOE":
            from ultralytics import YOLOE

            self.model = YOLOE(weights)
            self.classes = [c.strip() for c in args.prompt.split(",") if c.strip()]
            cwd = os.getcwd()
            os.chdir(kModelsFolder)  # the text encoder (mobileclip2_b.ts) is downloaded to the working folder
            try:
                self.model.set_classes(self.classes)
            finally:
                os.chdir(cwd)
        else:
            from ultralytics import YOLO

            self.model = YOLO(weights)
        self.load_time = time.time() - start
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))

        # the predicted boxes of every track (for the dashed boxes of --predict own) and the time of the
        # tracker alone, from the trackers' multi_predict() and update()
        self.predicted = {}  # track id -> x1, y1, x2, y2 of its predicted box in the current frame
        self.tracker_seconds = []
        for tracker_class, box_state in ((BYTETracker, "xyah"), (BOTSORT, "xywh")):
            original_multi_predict = tracker_class.__dict__["multi_predict"]

            def multi_predict(tracker, tracks, _original=original_multi_predict, _state=box_state):
                _original(tracker, tracks)
                for t in tracks:
                    self.predicted[t.track_id] = box_from_state(t.mean, _state)

            tracker_class.multi_predict = multi_predict
        original_update = BYTETracker.update  # BoT-SORT uses ByteTrack's update()

        def update(tracker, *a, **kw):
            start = time.time()
            out = original_update(tracker, *a, **kw)
            self.tracker_seconds.append(time.time() - start)
            return out

        BYTETracker.update = update
        self.own_prediction = None
        if args.predict == "own":
            self.own_prediction = OwnPrediction(Printer)
            self.own_prediction.install()

    def describe_detector(self):
        return f"YOLOE ({', '.join(self.classes)})" if self.method == "YOLOE" else self.method

    def track(self, img):
        """The tracked boxes of the next frame: a list of dicts (id, class, score, box x1 y1 x2 y2, mask or
        None, second: matched in the second pass), and the time of the detector and the tracker."""
        self.predicted = {}
        start = time.time()
        result = self.model.track(
            img, persist=True, tracker=self.tracker_yaml, conf=self.conf, device=self.device, verbose=False, retina_masks=True
        )[0]
        seconds = time.time() - start
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return [], seconds
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()
        masks = result.masks.data.cpu().numpy() > 0.5 if result.masks is not None else None
        high = float(self.settings["track_high_thresh"])
        tracks = []
        for k, track_id in enumerate(ids):
            mask = masks[k] if masks is not None else None
            if mask is not None and mask.shape != img.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            tracks.append(dict(id=int(track_id), cls=str(self.names.get(classes[k], classes[k])), score=float(scores[k]),
                               box=[float(v) for v in xyxy[k]], mask=mask, second=bool(scores[k] < high)))
        return tracks, seconds


# ---------------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------------


def id_colour(track_id):
    """A colour (BGR) for each identity: hues spread by the golden ratio, so neighbours differ."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 180)
    hsv = np.uint8([[[hue, 200, 235]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def dashed_rectangle(img, p1, p2, colour, dash=7):
    x1, y1, x2, y2 = (int(round(v)) for v in (*p1, *p2))
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for s in range(0, length, 2 * dash):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            q0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            q1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, q0, q1, colour, 1, cv2.LINE_AA)


def label(img, text, x, y, colour):
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    (w, h), _ = cv2.getTextSize(text, font, size, 1)
    y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y), colour, -1)
    cv2.putText(img, text, (x + 2, y - 3), font, size, (0, 0, 0), 1, cv2.LINE_AA)


class Windows:
    """Tracks: the frame with every tracked box, its identity and the trail of its centre (and, with
    --predict own, each track's predicted box, dashed). Timeline: one row per identity, filled in the
    frames where it was seen; the mouse over a row names it in the title."""

    def __init__(self, num_frames, show_predictions):
        self.num_frames = max(int(num_frames or 0), 1)
        self.show_predictions = show_predictions
        self.trails = {}  # id -> [(frame, centre)]
        self.rows = []  # the identities of the Timeline's rows, top to bottom
        self.row_height = 1
        self.history = None
        self.previous_ids = set()
        cv2.namedWindow("Tracks", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Timeline", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Timeline", self._on_mouse)

    def show(self, img, frame_id, tracks, predicted, history, title):
        self.history = history
        shown = img.copy()
        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            centre = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.trails.setdefault(t["id"], []).append((frame_id, centre))
        for track_id, trail in self.trails.items():
            recent = [c for f, c in trail if frame_id - f < kTrailLength]
            if len(recent) > 1 and trail[-1][0] == frame_id:
                cv2.polylines(shown, [np.int32(np.round(recent))], False, id_colour(track_id), 2, cv2.LINE_AA)
        if self.show_predictions:  # the tracks of this frame, and those lost since the previous one
            shown_ids = self.previous_ids | {t["id"] for t in tracks}
            for track_id, box in predicted.items():
                if track_id in shown_ids:
                    dashed_rectangle(shown, box[:2], box[2:], id_colour(track_id))
        self.previous_ids = {t["id"] for t in tracks}
        for t in tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in t["box"])
            colour = id_colour(t["id"])
            cv2.rectangle(shown, (x1, y1), (x2, y2), colour, 1 if t["second"] else 2, cv2.LINE_AA)
            label(shown, f"{t['id']} {t['cls']}" + (" (2nd pass)" if t["second"] else ""), x1, y1, colour)
        cv2.imshow("Tracks", shown)
        cv2.setWindowTitle("Tracks", f"Tracks: {title}")
        cv2.imshow("Timeline", self._timeline(frame_id, history))

    def _timeline(self, frame_id, history):
        """history: id -> {"cls": class, "frames": [frame numbers]}. The rows are grouped by class (people
        first, then the classes with the most identities), in the order the identities appeared."""
        margin, top = 110, 24
        by_class = {}
        for track_id, h in history.items():
            by_class.setdefault(h["cls"], []).append(track_id)
        order = sorted(by_class, key=lambda c: (c != kPerson, -len(by_class[c]), c))
        rows = [i for c in order for i in sorted(by_class[c], key=lambda i: history[i]["frames"][0])]
        self.row_height = int(np.clip(kTimelineHeight // max(len(rows), 1), 2, 14))
        shown_rows = rows[: kTimelineHeight // self.row_height]
        self.rows = shown_rows
        height = top + len(shown_rows) * self.row_height + 8
        img = np.full((max(height, 80), margin + kTimelineWidth + 10, 3), 255, np.uint8)
        scale = kTimelineWidth / self.num_frames
        last_class = None
        for r, track_id in enumerate(shown_rows):
            y0 = top + r * self.row_height
            h = history[track_id]
            colour = id_colour(track_id)
            for f in h["frames"]:
                x0 = margin + int(f * scale)
                cv2.rectangle(img, (x0, y0), (max(x0, margin + int((f + 1) * scale) - 1), y0 + self.row_height - 2), colour, -1)
            if h["cls"] != last_class:  # the first row of a class: its name and number of identities
                last_class = h["cls"]
                cv2.line(img, (0, y0 - 1), (img.shape[1], y0 - 1), (200, 200, 200), 1)
                cv2.putText(img, f"{last_class} ({len(by_class[last_class])})", (4, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0), 1, cv2.LINE_AA)
        x = margin + int(frame_id * scale)
        cv2.line(img, (x, top - 4), (x, img.shape[0] - 4), (0, 0, 255), 1)
        more = len(rows) - len(shown_rows)
        text = f"frame {frame_id}: {len(rows)} identities" + (f" ({more} more not shown)" if more else "") + \
               "; the mouse over a row names it"
        cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE or not self.rows or self.history is None:
            return
        r = (y - 24) // max(self.row_height, 1)
        if 0 <= r < len(self.rows):
            track_id = self.rows[r]
            h = self.history[track_id]
            frames = h["frames"]
            cv2.setWindowTitle("Timeline", f"Timeline: identity {track_id} ({h['cls']}), seen in {len(frames)} frames "
                               f"from {frames[0]} to {frames[-1]}")


def key_action(wait_ms):
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (ord("q"), 27):
        return "stop"
    if key == ord("p"):
        return "pause"
    return None


# ---------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------


class Saver:
    """--save DIR: tracks.csv (every tracked box, with the frame number and timestamp of the sequence),
    tracks.txt (the same boxes in the MOTChallenge format: frame from 1, id, left, top, width, height,
    score, -1, -1, -1), masks/NNNNNN.png (16 bits: the identity of each pixel, 0 in no tracked object;
    where two masks overlap, the higher score wins) and info.json."""

    def __init__(self, folder):
        self.folder = folder
        if os.path.isdir(folder) and os.listdir(folder):
            print(f"--save {folder}: the folder exists: its files are replaced")
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        self.rows = []
        self.frames = []

    def add(self, frame_id, timestamp, tracks, image_shape):
        self.frames.append((frame_id, timestamp))
        has_mask = any(t["mask"] is not None for t in tracks)
        ids = np.zeros(image_shape[:2], np.uint16) if has_mask else None
        for t in sorted(tracks, key=lambda t: t["score"]):
            row = dict(frame=frame_id, timestamp=timestamp, id=t["id"], cls=t["cls"], score=t["score"], box=t["box"],
                       second=int(t["second"]), mask_pixels=0)
            if t["mask"] is not None:
                ids[t["mask"]] = t["id"]
                row["mask_pixels"] = int(np.count_nonzero(t["mask"]))
            self.rows.append(row)
        if ids is not None:
            cv2.imwrite(os.path.join(self.folder, "masks", f"{frame_id:06d}.png"), ids)

    def close(self, info):
        with open(os.path.join(self.folder, "tracks.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp", "id", "class", "score", "x1", "y1", "x2", "y2", "second_pass", "mask_pixels"])
            for r in sorted(self.rows, key=lambda r: (r["frame"], r["id"])):
                writer.writerow([r["frame"], repr(float(r["timestamp"])), r["id"], r["cls"], f"{r['score']:.4f}",
                                 *(f"{v:.1f}" for v in r["box"]), r["second"], r["mask_pixels"]])
        with open(os.path.join(self.folder, "tracks.txt"), "w") as f:
            for r in sorted(self.rows, key=lambda r: (r["frame"], r["id"])):
                x1, y1, x2, y2 = r["box"]
                f.write(f"{r['frame'] + 1},{r['id']},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{r['score']:.4f},-1,-1,-1\n")
        info["frames_tracked"] = len(self.frames)
        info["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.folder, "info.json"), "w") as f:
            json.dump(info, f, indent=1)


def run(args, frames, tracker, Printer):
    history = {}  # id -> {"cls": class (the most frequent), "frames": [...], "counts": {class: n}}
    seconds, num_boxes, num_second = [], 0, 0
    mot_rows = {}  # MOT frame (from 1) -> rows of the people, for --evaluate
    saver = Saver(args.save) if args.save else None
    windows = None if args.headless else Windows(frames.num_total, args.predict == "own")
    every = "every frame" if args.step == 1 else f"every {args.step}th frame"
    print(f"{tracker.name} on {tracker.describe_detector()}, {frames.name}: {every}"
          + ("" if args.headless else "; in a window, q or Esc stops and p pauses"))
    paused = stopped = False
    num_tracked = 0
    last_frame = None
    for frame_id, timestamp, img in frames:
        if frame_id % args.step != 0:
            continue
        if args.max_frames is not None and num_tracked >= args.max_frames:
            break
        tracks, secs = tracker.track(img)
        seconds.append(secs)
        num_tracked += 1
        last_frame = (frame_id, img)
        for t in tracks:
            h = history.setdefault(t["id"], {"frames": [], "counts": {}})
            h["frames"].append(frame_id)
            h["counts"][t["cls"]] = h["counts"].get(t["cls"], 0) + 1
            h["cls"] = max(h["counts"], key=h["counts"].get)
            num_boxes += 1
            num_second += t["second"]
            if t["cls"] == kPerson:
                x1, y1, x2, y2 = t["box"]
                mot_rows.setdefault(frame_id + 1, []).append((t["id"], x1, y1, x2 - x1, y2 - y1, 1, 1))
        if saver is not None:
            saver.add(frame_id, timestamp, tracks, img.shape)
        if num_tracked == 1 or num_tracked % 100 == 0:
            print(f"frame {frame_id}" + (f"/{frames.num_total}" if frames.num_total else "")
                  + f": {secs:.3f} s, {len(tracks)} tracked boxes, {len(history)} identities so far")
        if windows is not None:
            title = f"{frames.name}, frame {frame_id}"
            windows.show(img, frame_id, tracks, tracker.predicted, history, title)
            while True:
                action = key_action(50 if paused else 1)
                if action == "stop":
                    stopped = True
                    break
                if action == "pause":
                    paused = not paused
                    Printer.yellow("Paused: p resumes." if paused else "Resumed.")
                if not paused:
                    break
        if stopped:
            break

    if not seconds:
        sys.exit(f"no frames were read from {frames.source}")
    summary = print_summary(args, frames, tracker, history, seconds, num_boxes, num_second, mot_rows, Printer)
    if saver is not None:
        saver.close(dict(
            tracker=tracker.name, method=tracker.method, prompt=args.prompt, settings=tracker.settings,
            settings_file=os.path.relpath(tracker.settings_path, kScriptFolder), second_pass=args.second_pass,
            predict=args.predict, conf=args.conf, device=tracker.device_name, sequence=os.path.abspath(frames.source),
            frames_list=frames.list_name, step=args.step, summary=summary,
        ))
        print(f"Written to {args.save}: tracks.csv, tracks.txt (MOTChallenge format), masks/, info.json."
              + (f" The tracks in the map: pixi run slam {frames.option} {frames.source} --settings ... --tracks {args.save}"
                 if frames.option else ""))
    if windows is not None:
        if not stopped and last_frame is not None:
            Printer.green("End of the sequence: the windows stay open; press q or Esc in one of them to close them.")
            while key_action(50) != "stop":
                pass
        cv2.destroyAllWindows()


def print_summary(args, frames, tracker, history, seconds, num_boxes, num_second, mot_rows, Printer):
    n = len(seconds)
    per_frame = float(np.mean(seconds[1:])) if n > 1 else float(seconds[0])  # the first frame loads the model
    tracker_part = float(np.mean(tracker.tracker_seconds[1:])) if len(tracker.tracker_seconds) > 1 else None
    timing = f"{per_frame:.3f} s per frame on {tracker.device_name}" + (
        f" (the tracker alone {1000 * tracker_part:.1f} ms)" if tracker_part is not None else "")
    head = f"{tracker.name} on {tracker.describe_detector()}, {frames.name}"
    parts = []
    if args.evaluate:
        from pyslam.semantics.tracking_scores import score_tracks

        scored = [f for f in frames.scored if (f - 1) % args.step == 0 and f <= frames.num_total]
        s = score_tracks(frames.gt, mot_rows, scored)
        labelled = "" if len(scored) == n else f", {len(scored)} of them labelled"
        head += f" ({n} frames{labelled}; {s['known_ids']} people, {s['known_boxes']} true boxes)"
        parts.append(f"MOTA {s['MOTA']:.2f}, IDF1 {s['IDF1']:.2f}, HOTA {s['HOTA']:.2f} (DetA {s['DetA']:.2f}, AssA {s['AssA']:.2f})")
        parts.append(f"{s['IDSW']} identity switches, {s['FN']} misses, {s['FP']} false boxes; {s['tracker_ids']} identities "
                     f"given to the people")
        parts.append(timing)
    else:
        frames_text = f"{n} frames" if not frames.num_total or n == frames.num_total else f"{n} of {frames.num_total} frames"
        if args.step > 1:
            frames_text += f", every {args.step}"
        head += f" ({frames_text})"
        parts.append(timing)
        people = sum(1 for h in history.values() if h["cls"] == kPerson)
        parts.append(f"{len(history)} identities" + (f", {people} of them people" if kPerson in tracker.names.values() else ""))
        if history:
            longest_id = max(history, key=lambda i: len(history[i]["frames"]))
            parts.append(f"longest track {len(history[longest_id]['frames'])} frames ({history[longest_id]['cls']}, identity {longest_id})")
            parts.append(f"{sum(1 for h in history.values() if len(h['frames']) < kShortTrack)} tracks shorter than {kShortTrack} frames")
    if args.second_pass == "off":
        parts.append("second pass off")
    else:
        parts.append(f"{num_second} of {num_boxes} boxes matched in the second pass")
    own = tracker.own_prediction
    if own is not None:
        if own.returned_none or own.failed or own.used == 0:
            parts.append("prediction: the tracker's own (predict_box() returned None)" if own.returned_none else
                         "prediction: the tracker's own (predict_box() failed)")
        else:
            parts.append(f"prediction: predict_box(), {own.largest_difference:.2f} px at most from the tracker's own")
    if args.save:
        parts.append(f"saved to {args.save}")
    summary = f"{head}: {'; '.join(parts)}"
    Printer.green(f"\nSummary: {summary}\n")
    return summary


if __name__ == "__main__":
    main()
