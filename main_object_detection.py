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

# The objects in one image, or in every frame of a sequence (COMP0227 Lab 04): a box, a class, a score
# and a mask for every object a detector finds. The detectors are YOLO (YOLO26-S with masks, the 80
# classes of COCO) and YOLOE (YOLOE-26-S with masks, the classes named by a text prompt); SAM
# (MobileSAM) gives the masks of what a point or a box points at, without any class. All three come
# from Ultralytics, in the semantics level.
#
# The candidates of a detector are all its boxes above --conf, before non-maximum suppression (NMS)
# removed the overlapping ones. --nms own uses nms() of pyslam/semantics/detection_utils.py in place of
# the detector's own NMS. --evaluate DIR scores the detector against the true boxes of a labelled set
# (AP, with pycocotools). --save DIR writes the detections of every frame of a sequence, for
# main_slam.py --detections DIR, which lifts them into the map.
#
#   pixi run -e semantics object-detection --list
#   pixi run -e semantics object-detection --method YOLO --test desk
#   pixi run -e semantics object-detection --method YOLOE --test desk --prompt "monitor, keyboard, mug"
#   pixi run -e semantics object-detection --method SAM --test desk --point 523,651
#   pixi run -e semantics object-detection --method YOLO --test desk --nms own
#   pixi run -e semantics object-detection --method YOLO --evaluate data/detection_sets/room0
#   pixi run -e semantics object-detection --method YOLO --tum $S --settings settings/TUM1.yaml --save detections/NAME --step 5

import pyslam  # first: sets the OpenMP thread settings (and, on macOS, the MPS fallback) before torch is imported
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time
import urllib.request
import zlib
from datetime import datetime

import cv2
import numpy as np

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)

# The methods (--method): their weights in data/models/, and what each gives
kMethods = {
    "YOLO": ("yolo26s-seg.pt", "YOLO26-S with masks: a box, a class, a score and a mask for every object it finds, "
             "among the 80 classes of COCO"),
    "YOLOE": ("yoloe-26s-seg.pt", "YOLOE-26-S with masks: the same, for the classes named by --prompt (open "
              "vocabulary: a text encoder, MobileCLIP2-B, turns each name into a row of the classifier)"),
    "SAM": ("mobile_sam.pt", "MobileSAM, a small version of the Segment Anything Model: the masks of what a point "
            "(--point) or a box (--box) points at, with no class"),
}
kTextEncoder = "mobileclip2_b.ts"  # YOLOE's text encoder (MobileCLIP2-B), in data/models/
kWeightsUrl = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}"
kModelsFolder = os.path.join(kScriptFolder, "data", "models")

# The images of --test (the same as main_semantic_image_segmentation.py's)
kTests = {
    "kitti": "test/data/kitti06-12-color.png",  # frame 12 of the KITTI 06 video
    "kitti_next": "test/data/kitti06-17-color.png",  # half a second later
    "desk": "data/images/desk_detic.jpeg",
    "churchill": "test/data/churchill/1.ppm",
    "box": "test/data/box_in_scene.png",
    "mars": "test/data/mars1.png",
}

# The 80 classes of COCO, in the order of YOLO's class numbers
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

kCandidatesMax = 30000  # the candidates: every box above --conf (Ultralytics' NMS takes at most 30000)
kMaxDet = 300  # boxes kept per image (Ultralytics' max_det)
kEvaluateConf = 0.001  # --evaluate: AP is computed from every box above this score
kShowScore = 0.25  # --evaluate: the Detections window draws the boxes with at least this score
kMaxWindowWidth = 1800  # pixels: wider windows are reduced


def main():
    from pyslam.io.sequence_options import add_sequence_options, has_sequence_options

    parser = argparse.ArgumentParser(
        description="The objects in one image, or in every frame of a sequence: a box, a class, a score and a mask "
        "for every object a detector finds; SAM gives the masks of what a point or a box points at",
        epilog="examples: pixi run -e semantics object-detection --method YOLO --test desk; pixi run -e semantics "
        "object-detection --method SAM --test desk --point 523,651; pixi run -e semantics object-detection --method "
        "YOLO --evaluate data/detection_sets/room0; pixi run -e semantics object-detection --method YOLO --tum $S "
        "--settings settings/TUM1.yaml --save detections/NAME --step 5",
    )
    parser.add_argument("--method", default="YOLO", metavar="NAME", help="YOLO, YOLOE or SAM (default YOLO; see --list)")
    parser.add_argument("--test", default="desk", choices=sorted(kTests), help="the image (default desk)")
    parser.add_argument("--image", metavar="FILE", help="an image file of your own, instead of --test")
    parser.add_argument("--prompt", metavar="TEXT",
                        help='YOLOE: the classes to detect, separated by commas, e.g. "monitor, keyboard, mug". With '
                        '--evaluate, WORD=CLASS counts the detections of WORD as the class CLASS of the set, e.g. "pouf=chair"')
    parser.add_argument("--conf", type=float, help=f"the lowest score of a box (default 0.25; {kEvaluateConf} with --evaluate)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS: of two boxes of one class whose IoU is above this, the one with the lower score is "
                        "removed (default 0.7)")
    parser.add_argument("--one-to-one", action="store_true",
                        help="YOLO and YOLOE: the head trained with one cell per object, which needs no NMS")
    parser.add_argument("--nms", choices=("detector", "own", "none"), default="detector",
                        help="own: nms() of pyslam/semantics/detection_utils.py in place of the detector's own NMS; "
                        "none: no NMS, every candidate is kept (default detector)")
    parser.add_argument("--all-classes", action="store_true",
                        help="with --nms own: give nms() the candidates of all the classes at once, instead of one "
                        "class at a time")
    parser.add_argument("--sam", action="store_true",
                        help="YOLO and YOLOE: give the detector's boxes to SAM as box prompts, and use SAM's masks")
    parser.add_argument("--point", metavar="U,V", help="SAM: a point on the object, in pixels (u from the left, v from the top)")
    parser.add_argument("--box", metavar="X1,Y1,X2,Y2", help="SAM: a box around the object, in pixels")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="where the networks run (default auto: an NVIDIA GPU, else Apple's GPU, else the CPU)")
    parser.add_argument("--runs", type=int, default=3,
                        help="with one image: how many times to repeat the detection for the timing (default 3)")
    add_sequence_options(parser, need_settings=False)
    sequence = parser.add_argument_group("with a sequence or a labelled set")
    sequence.add_argument("--evaluate", metavar="DIR",
                          help="detect the images of a labelled set (DIR/boxes.json and DIR/images/) and score the "
                          "detector against their true boxes: AP50 per class, mAP50 and mAP50:95")
    sequence.add_argument("--save", metavar="DIR",
                          help="write the detections of every frame to this folder (detections.csv, masks/, counts.csv, "
                          "info.json), for main_slam.py --detections DIR")
    sequence.add_argument("--step", type=int, default=1, metavar="N", help="detect every Nth frame (default 1: every frame)")
    sequence.add_argument("--max-frames", type=int, metavar="N", help="stop after N detected frames")
    sequence.add_argument("--headless", action="store_true", help="no windows")
    parser.add_argument("--list", action="store_true", help="list the methods, and the classes of COCO")
    args = parser.parse_args()

    if args.list:
        list_methods()
        sys.exit(0)
    args.method = args.method.upper()
    if args.method not in kMethods:
        print(f"Unknown method '{args.method}'. Valid names (see --list): {' '.join(kMethods)}", file=sys.stderr)
        sys.exit(2)
    on_sequence = has_sequence_options(args)
    if args.method == "SAM":
        if bool(args.point) == bool(args.box):
            parser.error("--method SAM needs --point U,V or --box X1,Y1,X2,Y2 (one of them)")
        if on_sequence or args.evaluate or args.save:
            parser.error("--method SAM works on one image (--test or --image)")
        for option in ("prompt", "sam", "one_to_one", "all_classes"):
            if getattr(args, option):
                parser.error(f"--{option.replace('_', '-')} is not for --method SAM")
        if args.nms != "detector":
            parser.error("--nms is not for --method SAM: SAM gives masks, not candidate boxes")
    elif args.point or args.box:
        parser.error("--point and --box are for --method SAM")
    if args.method == "YOLOE" and not args.prompt:
        parser.error('--method YOLOE needs --prompt, e.g. --prompt "monitor, keyboard, mug"')
    if args.prompt and args.method != "YOLOE":
        parser.error("--prompt is for --method YOLOE: YOLO has its 80 classes")
    if args.one_to_one and args.nms != "detector":
        parser.error("--one-to-one uses no NMS, so --nms does not go with it")
    if args.all_classes and args.nms != "own":
        parser.error("--all-classes goes with --nms own")
    if args.evaluate and on_sequence:
        parser.error("--evaluate DIR detects the images of DIR: give it without --video, --images or --tum")
    if args.save and not on_sequence:
        parser.error("--save goes with a sequence (--tum, --video or --images)")
    if args.evaluate and args.sam:
        parser.error("--evaluate scores boxes, so --sam (masks) does not change it")
    if args.step < 1:
        parser.error("--step must be 1 or more")
    if args.conf is None:
        args.conf = kEvaluateConf if args.evaluate else 0.25
    if importlib.util.find_spec("ultralytics") is None:
        if os.name == "nt":
            sys.exit("object-detection needs the semantics level, which does not exist on native Windows: use WSL2.")
        sys.exit("object-detection needs the semantics level: run this command with `pixi run -e semantics object-detection ...`.")

    from pyslam.utilities.logging import Printer

    if args.method == "SAM":
        run_sam(args, Printer)
    elif args.evaluate:
        run_evaluate(args, Printer)
    elif on_sequence:
        run_sequence(args, parser, Printer)
    else:
        run_image(args, Printer)


def list_methods():
    print("Methods (--method), and what each gives:")
    for name, (weights, what) in kMethods.items():
        print(f"    {name:6s} {what}. Weights: data/models/{weights}")
    print("All three need the semantics level: pixi run -e semantics object-detection ...; their weights are "
          "downloaded by pixi run -e semantics models-detection.")
    print("\nThe 80 classes of COCO (YOLO):")
    names = coco_names()
    for i in range(0, len(names), 8):
        print("    " + ", ".join(names[i:i + 8]))


def coco_names():
    return list(kCocoClasses)


# ---------------------------------------------------------------------------------------------------
# The networks
# ---------------------------------------------------------------------------------------------------


def choose_device(name):
    import torch

    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda: there is no NVIDIA GPU that torch can use here")
    if name == "mps" and not torch.backends.mps.is_available():
        sys.exit("--device mps: Apple's GPU is not available here")
    return "cuda:0" if name == "cuda" else name


def weights_path(filename, Printer):
    """data/models/FILENAME, downloaded from Ultralytics' releases if it is not there yet."""
    path = os.path.join(kModelsFolder, filename)
    if not os.path.isfile(path):
        os.makedirs(kModelsFolder, exist_ok=True)
        url = kWeightsUrl.format(filename)
        Printer.green(f"Downloading {filename} to data/models/ ({url}); pixi run -e semantics models-detection downloads them all")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def parse_prompt(text):
    """The words of a prompt, and the class each word counts as ("pouf=chair": the word pouf, the class chair)."""
    words, classes = [], {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        word, _, cls = item.partition("=")
        word, cls = word.strip(), cls.strip()
        if word in words:
            continue
        words.append(word)
        classes[word] = cls or word
    if not words:
        sys.exit("--prompt: no class name in it")
    return words, classes


class Candidates:
    """While active, the segmentation predictor of Ultralytics returns the boxes without their masks,
    and keeps the mask coefficients and the prototypes, so that the masks of only the kept boxes are
    computed (at a low --conf there are thousands of candidates, and their masks would fill the memory)."""

    def __init__(self):
        self.captured = None

    @contextlib.contextmanager
    def active(self):
        from ultralytics.engine.results import Results
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor
        from ultralytics.utils import ops

        original = SegmentationPredictor.construct_result
        candidates = self

        def construct_result(predictor, pred, img, orig_img, img_path, proto):
            boxes_input = pred[:, :4].clone()
            if pred.shape[0]:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            candidates.captured = dict(pred=pred.clone(), boxes_input=boxes_input, proto=proto, input_shape=img.shape[2:],
                                       orig_shape=orig_img.shape[:2])
            return Results(orig_img, path=img_path, names=predictor.model.names, boxes=pred[:, :6])

        SegmentationPredictor.construct_result = construct_result
        try:
            yield self
        finally:
            SegmentationPredictor.construct_result = original

    def masks(self, idx, retina):
        """The masks (boolean, K x H x W, in the original image's size if retina) of the candidates idx,
        as Ultralytics computes them for the boxes it keeps."""
        import torch
        from ultralytics.utils import ops

        c = self.captured
        idx = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=c["pred"].device)
        if idx.numel() == 0:
            return np.zeros((0,) + tuple(c["orig_shape"]), bool)
        out = []
        for chunk in torch.split(idx, 64):
            coeffs = c["pred"][chunk, 6:]
            if retina:
                m = ops.process_mask_native(c["proto"], coeffs, c["pred"][chunk, :4], c["orig_shape"])
            else:
                m = ops.process_mask(c["proto"], coeffs, c["boxes_input"][chunk], c["input_shape"], upsample=True)
            out.append(m.bool().cpu().numpy())
        return np.concatenate(out)


class Detector:
    """YOLO, or YOLOE with a prompt. One model object for each way it is called (Ultralytics keeps the
    arguments of predict() between the calls on one object)."""

    def __init__(self, args, Printer):
        self.method = args.method
        self.device = choose_device(args.device)
        self.device_name = self.device.split(":")[0]
        self.weights = weights_path(kMethods[self.method][0], Printer)
        self.words, self.prompt_classes = parse_prompt(args.prompt) if args.prompt else (None, None)
        self.text_seconds = 0.0
        start = time.time()
        self.model = self._load()
        self.load_time = time.time() - start - self.text_seconds
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self._models = {}
        self._candidates = Candidates()

    def _load(self):
        if self.method == "YOLOE":
            from ultralytics import YOLOE

            model = YOLOE(self.weights)
            if not os.path.isfile(os.path.join(kModelsFolder, kTextEncoder)):
                print(f"Downloading YOLOE's text encoder ({kTextEncoder}, 0.25 GB) to data/models/")
            cwd = os.getcwd()
            os.chdir(kModelsFolder)  # Ultralytics looks for the text encoder in the working folder
            try:
                start = time.time()
                model.set_classes(self.words, model.get_text_pe(self.words))
                if not self.text_seconds:
                    self.text_seconds = time.time() - start
            finally:
                os.chdir(cwd)
            return model
        from ultralytics import YOLO

        return YOLO(self.weights)

    def _model(self, key):
        if key == "default":
            return self.model
        if key not in self._models:
            self._models[key] = self._load()
        return self._models[key]

    def cls_name(self, k):
        name = self.names[int(k)]
        return self.prompt_classes.get(name, name) if self.prompt_classes else name

    def describe_classes(self):
        if self.method == "YOLOE":
            return f"{len(self.words)} prompt{'s' if len(self.words) != 1 else ''}"
        return f"{len(self.names)} classes (COCO)"

    def detect(self, img, conf, iou=0.7, one_to_one=False, nms="detector", all_classes=False, masks=True,
               candidates=False, own_nms=None):
        """The detections of one image: a dict with boxes (K x 4), scores, classes (names), masks (K x H x W
        booleans, or None), and, if asked for or needed, the candidates (all the boxes above conf, before
        NMS) and how the own NMS compared with the detector's."""
        out = {}
        start = time.time()
        if one_to_one:
            r = self._model("one_to_one").predict(img, conf=conf, nms=False, max_det=kMaxDet, device=self.device,
                                                  verbose=False, retina_masks=masks)[0]
            self._fill(out, r, masks)
        else:
            if nms == "detector" or candidates or nms == "own":
                r = self.model.predict(img, conf=conf, iou=iou, max_det=kMaxDet, device=self.device, verbose=False,
                                       retina_masks=masks)[0]
                if nms == "detector":
                    self._fill(out, r, masks)
                detector = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
            if candidates or nms in ("own", "none"):
                with self._candidates.active():
                    self._model("candidates").predict(img, conf=conf, iou=1.0, max_det=kCandidatesMax, device=self.device,
                                                      verbose=False, retina_masks=masks)
                c = self._candidates.captured["pred"]
                cb, cs, cc = c[:, :4].cpu().numpy(), c[:, 4].cpu().numpy(), c[:, 5].cpu().numpy().astype(int)
                out["candidates"] = (cb, cs, cc)
                if nms in ("own", "none"):
                    nms_start = time.time()
                    if nms == "none":
                        keep = np.argsort(-cs, kind="stable")[:kMaxDet]
                    else:
                        keep = own_nms(cb, cs, cc, iou, all_classes)
                    out["nms_seconds"] = time.time() - nms_start
                    if keep is None:  # the stub: the detector's own NMS
                        self._fill(out, r, masks)
                        out["own_missing"] = True
                    else:
                        keep = np.asarray(keep, dtype=np.int64)[:kMaxDet]
                        m = self._candidates.masks(keep, retina=masks)
                        if nms == "own":  # as Ultralytics: a box whose mask is empty is dropped
                            nonempty = m.reshape(len(m), -1).any(axis=1) if len(m) else np.zeros(0, bool)
                            keep, m = keep[nonempty], m[nonempty]
                        out["boxes"], out["scores"] = cb[keep], cs[keep]
                        out["classes"] = [self.cls_name(k) for k in cc[keep]]
                        out["masks"] = m if masks else None
                        if nms == "own":
                            out["same_as_detector"] = same_boxes(out["boxes"], out["scores"], detector[0], detector[1])
                            out["detector_kept"] = len(detector[1])
        out["seconds"] = time.time() - start
        return out

    def _fill(self, out, r, masks):
        out["boxes"] = r.boxes.xyxy.cpu().numpy()
        out["scores"] = r.boxes.conf.cpu().numpy()
        out["classes"] = [self.cls_name(k) for k in r.boxes.cls.cpu().numpy().astype(int)]
        out["masks"] = r.masks.data.bool().cpu().numpy() if (masks and r.masks is not None) else None


def same_boxes(b1, s1, b2, s2, tol_px=0.5, tol_score=1e-4):
    """Whether two sets of boxes are the same, whatever their order."""
    if len(s1) != len(s2):
        return False
    o1, o2 = np.lexsort((b1[:, 0], -np.round(s1, 4))), np.lexsort((b2[:, 0], -np.round(s2, 4)))
    return bool(np.all(np.abs(b1[o1] - b2[o2]) <= tol_px) and np.all(np.abs(s1[o1] - s2[o2]) <= tol_score))


class OwnNMS:
    """--nms own: nms() of pyslam/semantics/detection_utils.py, called once per class (or once for all
    the classes with --all-classes). If it returns None, a warning is printed once and the detector's own
    NMS is used."""

    def __init__(self, Printer):
        from pyslam.semantics import detection_utils

        self.nms = detection_utils.nms
        self.Printer = Printer
        self.missing = False

    def __call__(self, boxes, scores, classes, iou, all_classes):
        if self.missing:
            return None
        groups = [np.arange(len(scores))] if all_classes else [np.where(classes == c)[0] for c in np.unique(classes)]
        keep = []
        for idx in groups:
            if len(idx) == 0:
                continue
            k = self.nms(boxes[idx], scores[idx], iou)
            if k is None:
                self.missing = True
                self.Printer.orange("nms() in pyslam/semantics/detection_utils.py returned None: it is not written yet "
                                    "(Lab 04, Activity 5). The detector's own NMS is used instead.")
                return None
            keep.extend(idx[np.asarray(k, dtype=np.int64).ravel()])
        keep = np.asarray(keep, dtype=np.int64)
        return keep[np.argsort(-scores[keep], kind="stable")]


class Sam:
    """MobileSAM through Ultralytics' SAM predictor: three masks for a point (from small to large, each
    with a score), one for a box."""

    def __init__(self, device, Printer):
        from ultralytics.models.sam import Predictor as SAMPredictor

        self.device = device
        self.device_name = device.split(":")[0]
        weights = weights_path(kMethods["SAM"][0], Printer)
        start = time.time()
        self.predictor = SAMPredictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=weights,
                                                     device=device, verbose=False, save=False))
        self.predictor.setup_model(None)
        self.load_time = time.time() - start
        self.encode_seconds = 0.0

    def set_image(self, img):
        start = time.time()
        self.predictor.set_image(img)
        self.encode_seconds = time.time() - start

    def point(self, u, v):
        start = time.time()
        r = self.predictor(points=[[u, v]], labels=[1], multimask_output=True)[0]
        seconds = time.time() - start
        masks = r.masks.data.bool().cpu().numpy()
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(len(masks))
        order = np.argsort([m.sum() for m in masks])  # from the smallest mask to the largest
        return masks[order], scores[order], seconds

    def boxes(self, boxes):
        start = time.time()
        r = self.predictor(bboxes=[[float(v) for v in b] for b in boxes])[0]
        seconds = time.time() - start
        masks = r.masks.data.bool().cpu().numpy() if r.masks is not None else np.zeros((0,) + r.orig_shape, bool)
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(len(masks))
        return masks, scores, seconds

    def reset(self):
        self.predictor.reset_image()


# ---------------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------------


def class_colour(name):
    """The colour of a class, the same in every run (BGR); main_slam.py --detections draws the objects of
    the Map Viewer in the same colours (pyslam/semantics/object_lifting.py, class_colour_bgr)."""
    name = str(name)
    i = (7 * kCocoClasses.index(name)) % 20 if name in kCocoClasses else zlib.crc32(name.encode()) % 20
    h = kClassPalette[i]
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def label(img, text, x, y, colour):
    x, y = int(x), int(max(14, y))
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def shade(img, mask, colour, alpha=0.45):
    if mask is None or mask.shape[:2] != img.shape[:2]:
        return
    img[mask] = ((1 - alpha) * img[mask] + alpha * np.array(colour, np.float32)).astype(np.uint8)


def draw_detections(img, det, min_score=0.0, true_boxes=None):
    shown = img.copy()
    masks = det.get("masks")
    order = np.argsort(det["scores"]) if len(det["scores"]) else []
    for i in order:  # the best last, on top
        if det["scores"][i] < min_score:
            continue
        colour = class_colour(det["classes"][i])
        if masks is not None and i < len(masks):
            shade(shown, masks[i], colour)
    for i in order:
        if det["scores"][i] < min_score:
            continue
        x1, y1, x2, y2 = det["boxes"][i]
        colour = class_colour(det["classes"][i])
        cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        label(shown, f"{det['classes'][i]} {det['scores'][i]:.2f}", x1, y1, colour)
    if true_boxes:
        for x1, y1, x2, y2, name in true_boxes:
            dashed_rectangle(shown, (x1, y1), (x2, y2), (255, 255, 255))
    return shown


def draw_candidates(img, candidates, names):
    shown = img.copy()
    boxes, scores, classes = candidates
    for i in np.argsort(scores):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), class_colour(names(classes[i])), 1)
    return shown


def dashed_rectangle(img, p1, p2, colour, dash=8):
    (x1, y1), (x2, y2) = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    for x in range(x1, x2, 2 * dash):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, 1)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, 1)
    for y in range(y1, y2, 2 * dash):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, 1)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, 1)


def fit_width(img, width=kMaxWindowWidth):
    if img.shape[1] <= width:
        return img
    f = width / img.shape[1]
    return cv2.resize(img, (width, int(round(img.shape[0] * f))), interpolation=cv2.INTER_AREA)


class ImageWindow:
    """The Image window, whose title gives the pixel under the mouse."""

    def __init__(self, img, name):
        self.name = name
        self.scale = 1.0
        shown = fit_width(img)
        self.scale = img.shape[1] / shown.shape[1]
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Image", shown)
        cv2.setWindowTitle("Image", f"Image: {name} (the mouse shows the pixel)")
        cv2.setMouseCallback("Image", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            cv2.setWindowTitle("Image", f"Image: {self.name}; the pixel ({int(x * self.scale)}, {int(y * self.scale)})")


def wait_for_key():
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            return
        try:
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                return
        except cv2.error:
            return


def key_action(wait_ms):
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (ord("q"), 27):
        return "stop"
    if key == ord("p"):
        return "pause"
    return None


def count_text(classes):
    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    return ", ".join(f"{c} {n}" for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "nothing"


def timing_text(per_image, times, first, load, extra=""):
    if len(times) > 1:
        runs = f"mean of {len(times)} runs after the first, which took {first:.2f} s"
    elif times:
        runs = f"one run after the first, which took {first:.2f} s"
    else:
        runs = "the first run"
    return f"{per_image:.2f} s per image ({runs}; {load:.1f} s to load the model{extra})"


def read_image(args):
    if args.image:
        path, name = args.image, os.path.basename(args.image)
    else:
        path, name = os.path.join(kScriptFolder, kTests[args.test]), args.test
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, as Ultralytics expects a NumPy image
    if img is None:
        sys.exit(f"cannot read the image {path}")
    return img, name


# ---------------------------------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------------------------------


def run_image(args, Printer):
    img, name = read_image(args)
    h, w = img.shape[:2]
    detector = Detector(args, Printer)
    own = OwnNMS(Printer) if args.nms == "own" else None
    kw = dict(conf=args.conf, iou=args.iou, one_to_one=args.one_to_one, nms=args.nms, all_classes=args.all_classes,
              candidates=not args.one_to_one, own_nms=own)
    det = detector.detect(img, **kw)
    first = det["seconds"]
    times = []
    for _ in range(max(0, args.runs)):
        times.append(detector.detect(img, **{**kw, "candidates": False})["seconds"])
    per_image = float(np.mean(times)) if times else first

    sam_text = ""
    if args.sam and len(det["scores"]):
        sam = Sam(detector.device, Printer)
        sam.set_image(img)
        masks, _, seconds = sam.boxes(det["boxes"])
        det["detector_masks"] = det.get("masks")
        det["masks"] = masks
        sam_text = f"; SAM drew their {len(masks)} masks in {sam.encode_seconds + seconds:.2f} s (MobileSAM, box prompts)"
        sam.reset()

    for i in np.argsort(-det["scores"]):
        x1, y1, x2, y2 = det["boxes"][i]
        mask = det.get("masks")
        px = f", mask {int(mask[i].sum())} px" if mask is not None and i < len(mask) else ""
        print(f"    {det['classes'][i]:16s} {det['scores'][i]:.3f}  box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}){px}")

    extra = f", {detector.text_seconds:.2f} s for the text encoder, once for the list" if detector.method == "YOLOE" else ""
    summary = (f"{detector.method} on {name} ({w}x{h} pixels): {detector.describe_classes()} on {detector.device_name}; "
               + timing_text(per_image, times, first, detector.load_time, extra) + "; ")
    num = len(det["scores"])
    objects = f"{num} object{'s' if num != 1 else ''}"
    if args.one_to_one:
        summary += f"{objects} (conf {args.conf:g}, the one-to-one head, no NMS)"
    else:
        num_candidates = len(det["candidates"][1])
        if args.nms == "none":
            summary += f"{objects}: every candidate kept (conf {args.conf:g}, no NMS)"
        elif args.nms == "own" and not det.get("own_missing"):
            calls = "one call for all the classes" if args.all_classes else "one call per class"
            summary += f"{objects} kept of {num_candidates} candidates by your nms() (conf {args.conf:g}, iou {args.iou:g}, {calls})"
        else:
            summary += f"{objects} kept of {num_candidates} candidates (conf {args.conf:g}, iou {args.iou:g})"
    summary += f": {count_text(det['classes'])}"
    if args.nms == "own" and not det.get("own_missing"):
        if det["same_as_detector"]:
            summary += f"; the detector's own NMS keeps the same {det['detector_kept']} boxes"
        else:
            summary += f"; the detector's own NMS keeps {det['detector_kept']} boxes, and not all of them are yours"
    summary += sam_text
    Printer.green(f"\nSummary: {summary}\n")

    if not args.headless:
        ImageWindow(img, name)
        cv2.imshow("Detections", fit_width(draw_detections(img, det)))
        what = "SAM's masks" if args.sam else "the masks"
        cv2.setWindowTitle("Detections", f"Detections: {name}, {num} kept ({what} shaded)")
        if "candidates" in det:
            cv2.imshow("Candidates", fit_width(draw_candidates(img, det["candidates"], detector.cls_name)))
            cv2.setWindowTitle("Candidates", f"Candidates: {name}, the {len(det['candidates'][1])} boxes above conf {args.conf:g}, before NMS")
        print("Press q or Esc in a window to stop.")
        wait_for_key()
        cv2.destroyAllWindows()


def run_sam(args, Printer):
    img, name = read_image(args)
    h, w = img.shape[:2]
    sam = Sam(choose_device(args.device), Printer)
    sam.set_image(img)
    if args.point:
        try:
            u, v = (float(x) for x in args.point.split(","))
        except ValueError:
            sys.exit("--point U,V: two numbers, e.g. --point 523,651")
        if not (0 <= u < w and 0 <= v < h):
            sys.exit(f"--point {args.point}: outside the image ({w}x{h} pixels)")
        masks, scores, seconds = sam.point(u, v)
        listed = ", ".join(f"{int(m.sum())} px ({'score ' if i == 0 else ''}{s:.2f})" for i, (m, s) in enumerate(zip(masks, scores)))
        what = f"point ({u:.0f}, {v:.0f}): {len(masks)} masks: {listed}"
        prompt_text = "the point"
    else:
        try:
            box = [float(x) for x in args.box.split(",")]
            assert len(box) == 4 and box[2] > box[0] and box[3] > box[1]
        except (ValueError, AssertionError):
            sys.exit("--box X1,Y1,X2,Y2: four numbers, the left, top, right and bottom edges, e.g. --box 324,551,724,751")
        masks, scores, seconds = sam.boxes([box])
        box_px = (box[2] - box[0]) * (box[3] - box[1])
        what = (f"box ({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}): {len(masks)} mask of {int(masks[0].sum())} px "
                f"(score {scores[0]:.2f}), {100 * masks[0].sum() / box_px:.0f}% of the box")
        prompt_text = "the box"
    summary = (f"SAM on {name} ({w}x{h} pixels) on {sam.device_name}, {what}; {sam.encode_seconds:.2f} s to encode the image, "
               f"{seconds:.2f} s for {prompt_text} ({sam.load_time:.1f} s to load the model)")
    Printer.green(f"\nSummary: {summary}\n")
    if not args.headless:
        ImageWindow(img, name)
        panels = []
        for i, (m, s) in enumerate(zip(masks, scores)):
            panel = img.copy()
            shade(panel, m, (203, 102, 255), 0.55)
            if args.point:
                cv2.circle(panel, (int(u), int(v)), 7, (0, 0, 0), -1)
                cv2.circle(panel, (int(u), int(v)), 5, (255, 255, 255), -1)
            else:
                cv2.rectangle(panel, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
            text = f"mask {i + 1}: {int(m.sum())} px, score {s:.2f}"
            cv2.putText(panel, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(panel, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        cv2.imshow("Detections", fit_width(np.hstack(panels)))
        cv2.setWindowTitle("Detections", f"Detections: SAM's masks for {prompt_text}, from small to large")
        print("Press q or Esc in a window to stop.")
        wait_for_key()
        cv2.destroyAllWindows()
    sam.reset()


# ---------------------------------------------------------------------------------------------------
# A labelled set: AP (--evaluate)
# ---------------------------------------------------------------------------------------------------


class LabelledSet:
    """DIR/boxes.json (the true boxes, in the COCO format) and DIR/images/."""

    def __init__(self, folder):
        path = os.path.join(folder, "boxes.json")
        if not os.path.isfile(path):
            sys.exit(f"--evaluate {folder}: no boxes.json there. The labelled set of Lab 04 is downloaded by "
                     "`pixi run -e semantics models-detection` into data/detection_sets/room0.")
        with open(path) as f:
            self.gt = json.load(f)
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        cats = self.gt["categories"]
        with_boxes = {a["category_id"] for a in self.gt["annotations"]}
        self.categories = [c for c in cats if c["id"] in with_boxes]  # the classes with true boxes, in the file's order
        self.cat_ids = {c["name"]: c["id"] for c in self.categories}
        self.images = self.gt["images"]
        self.boxes_of = {}
        for a in self.gt["annotations"]:
            x, y, bw, bh = a["bbox"]
            name = next(c["name"] for c in cats if c["id"] == a["category_id"])
            self.boxes_of.setdefault(a["image_id"], []).append((x, y, x + bw, y + bh, name))

    def image_path(self, im):
        return os.path.join(self.folder, "images", im["file_name"])


def coco_scores(gt, dets, cat_ids):
    """AP50 and AP50:95 of each class, and their means, with pycocotools (101 recall points, at most 100
    detections per image); also the precision-recall curve of each class at IoU 0.5."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = json.loads(json.dumps(gt))
        coco_gt.createIndex()
        if not dets:
            return {c: (0.0, 0.0) for c in cat_ids}, 0.0, 0.0, {}
        coco_dt = coco_gt.loadRes(dets)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.catIds = list(cat_ids)
        e.evaluate()
        e.accumulate()
    precision = e.eval["precision"]  # T x R x K x A x M
    per, curves = {}, {}
    for k, cid in enumerate(e.params.catIds):
        p50 = precision[0, :, k, 0, -1]
        pall = precision[:, :, k, 0, -1]
        ap50 = float(np.mean(p50[p50 > -1])) if np.any(p50 > -1) else 0.0
        ap = float(np.mean(pall[pall > -1])) if np.any(pall > -1) else 0.0
        per[cid] = (ap50, ap)
        curves[cid] = p50
    return per, float(np.mean([v[0] for v in per.values()])), float(np.mean([v[1] for v in per.values()])), curves


def draw_pr_curves(curves, names, title, size=(720, 560)):
    w, h = size
    m = 60
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (m, 20), (w - 20, h - m), (0, 0, 0), 1)
    for t in np.linspace(0, 1, 6):
        x = int(m + t * (w - 20 - m))
        y = int(h - m - t * (h - m - 20))
        cv2.putText(img, f"{t:.1f}", (x - 12, h - m + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{t:.1f}", (m - 40, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "recall", (w // 2 - 20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "precision", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    recall = np.linspace(0, 1, 101)
    for row, (cid, p) in enumerate(curves.items()):
        colour = class_colour(names[cid])
        valid = p > -1
        pts = [(int(m + r * (w - 20 - m)), int(h - m - max(q, 0) * (h - m - 20))) for r, q, ok in zip(recall, p, valid) if ok]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, colour, 2, cv2.LINE_AA)
        cv2.putText(img, names[cid], (w - 190, 45 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    cv2.putText(img, title, (m + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_evaluate(args, Printer):
    data = LabelledSet(args.evaluate)
    detector = Detector(args, Printer)
    own = OwnNMS(Printer) if args.nms == "own" else None
    dets, seconds = [], []
    names = {c["id"]: c["name"] for c in data.categories}
    unknown = set()
    paused = False
    for n, im in enumerate(data.images):
        img = cv2.imread(data.image_path(im), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read {data.image_path(im)}")
        det = detector.detect(img, args.conf, args.iou, args.one_to_one, args.nms, args.all_classes, masks=False, own_nms=own)
        seconds.append(det["seconds"])
        for b, s, c in zip(det["boxes"], det["scores"], det["classes"]):
            if c not in data.cat_ids:
                unknown.add(c)
                continue
            dets.append(dict(image_id=im["id"], category_id=data.cat_ids[c],
                             bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], score=float(s)))
        if not args.headless:
            shown = draw_detections(img, det, min_score=kShowScore, true_boxes=data.boxes_of.get(im["id"]))
            cv2.imshow("Detections", fit_width(shown))
            cv2.setWindowTitle("Detections", f"Detections: {im['file_name']} ({n + 1} of {len(data.images)}; boxes with a "
                               f"score of at least {kShowScore}; the true boxes dashed)")
            action = key_action(1)
            if action == "pause":
                paused = True
            while paused:
                action = key_action(50)
                if action in ("pause", "stop"):
                    paused = False
            if action == "stop":
                break
    per, map50, map5095, curves = coco_scores(data.gt, dets, [c["id"] for c in data.categories])
    per_image = float(np.mean(seconds[1:])) if len(seconds) > 1 else float(np.mean(seconds))
    if args.one_to_one:
        how = f"conf {args.conf:g}, the one-to-one head, no NMS"
    elif args.nms == "none":
        how = f"conf {args.conf:g}, no NMS"
    else:
        how = f"conf {args.conf:g}, iou {args.iou:g}"
        if args.nms == "own" and not (own and own.missing):
            how += ", your nms(), " + ("one call for all the classes" if args.all_classes else "one call per class")
    what = detector.describe_classes() if detector.method == "YOLOE" else ""
    summary = (f"{detector.method} on {data.name} ({len(data.images)} images, {len(data.gt['annotations'])} true boxes, "
               f"{len(data.categories)} classes{', ' + what if what else ''}) on {detector.device_name}, {how}: AP50 "
               + ", ".join(f"{names[cid]} {per[cid][0]:.2f}" for cid in names)
               + f"; mAP50 {map50:.2f}, mAP50:95 {map5095:.2f}; {per_image:.2f} s per image")
    if detector.method == "YOLOE" and unknown:
        print(f"Not classes of the set, so not scored: {', '.join(sorted(unknown))} (WORD=CLASS counts a word as a class)")
    Printer.green(f"\nSummary: {summary}\n")
    if not args.headless:
        cv2.imshow("Precision-recall", draw_pr_curves(curves, names, f"{detector.method}, IoU 0.5: mAP50 {map50:.2f}"))
        cv2.setWindowTitle("Precision-recall", f"Precision-recall: each class at IoU 0.5 ({data.name})")
        print("Press q or Esc in a window to stop.")
        while key_action(50) != "stop":
            pass
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------
# A sequence
# ---------------------------------------------------------------------------------------------------


class SequenceFrames:
    """The frames of --tum, --video or --images, read by pySLAM's dataset readers, with the timestamps
    main_slam.py gives them (main_slam.py --detections finds the detections of each of its frames by
    timestamp). With --tum, the frames of associations.txt if there is one (the frames SLAM reads)."""

    def __init__(self, args, parser):
        from pyslam.config import Config
        from pyslam.io.dataset_factory import dataset_factory
        from pyslam.io.sequence_options import sequence_config_path

        config_path = sequence_config_path(args, parser, default_settings=os.path.join(kScriptFolder, "settings", "KITTI04-12.yaml"))
        config = Config(config_path)
        os.remove(config_path)  # the temporary file of sequence_config_path(): it has been read
        self.dataset = dataset_factory(config)
        self.source = args.tum or args.video or args.images
        self.name = os.path.basename(os.path.normpath(self.source))
        if args.video:  # e.g. kitti06/video_color.mp4
            self.name = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(self.source))), self.name)
        self.num_total = getattr(self.dataset, "num_frames", None)
        self.list_name = None
        if args.tum:
            folder = os.path.abspath(os.path.expanduser(args.tum)).rstrip("/")
            self.list_name = args.associations or ("associations.txt" if os.path.isfile(os.path.join(folder, "associations.txt")) else "rgb.txt")
            print(f"--tum: the frames of {self.list_name}")

    def __iter__(self):
        frame_id = 0
        while self.dataset.is_ok:
            img = self.dataset.getImageColor(frame_id)
            if img is None:
                break
            yield frame_id, self.dataset.getTimestamp(), img
            frame_id += 1


class Saver:
    """--save DIR: detections.csv (every detection, with the frame number and timestamp of the sequence
    and its index in the frame), masks/NNNNNN.png (16 bits: the index of the detection which each pixel
    belongs to, 0 in none; where two masks overlap, the higher score wins), counts.csv (one line per
    frame: the time it took and the number of objects of each class) and info.json."""

    def __init__(self, folder):
        self.folder = folder
        if os.path.isdir(folder) and os.listdir(folder):
            print(f"--save {folder}: the folder exists: its files are replaced")
            for old in ("detections.csv", "counts.csv", "info.json"):
                if os.path.isfile(os.path.join(folder, old)):
                    os.remove(os.path.join(folder, old))
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        self.rows, self.frames = [], []

    def add(self, frame_id, timestamp, det, seconds, image_shape):
        counts = {}
        index = np.zeros(image_shape[:2], np.uint16)
        masks = det.get("masks")
        order = np.argsort(det["scores"])  # the best last, so that it wins where masks overlap
        for i in order:
            mask_pixels = 0
            if masks is not None and i < len(masks):
                index[masks[i]] = i + 1
        for i in range(len(det["scores"])):
            cls = det["classes"][i]
            counts[cls] = counts.get(cls, 0) + 1
            mask_pixels = int(np.count_nonzero(index == i + 1)) if masks is not None else 0
            self.rows.append(dict(frame=frame_id, timestamp=timestamp, index=i + 1, cls=cls, score=float(det["scores"][i]),
                                  box=[float(v) for v in det["boxes"][i]], mask_pixels=mask_pixels))
        cv2.imwrite(os.path.join(self.folder, "masks", f"{frame_id:06d}.png"), index)
        self.frames.append(dict(frame=frame_id, timestamp=timestamp, seconds=seconds, counts=counts))

    def close(self, info):
        with open(os.path.join(self.folder, "detections.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp", "index", "class", "score", "x1", "y1", "x2", "y2", "mask_pixels"])
            for r in self.rows:
                writer.writerow([r["frame"], repr(float(r["timestamp"])), r["index"], r["cls"], f"{r['score']:.4f}",
                                 *(f"{v:.1f}" for v in r["box"]), r["mask_pixels"]])
        classes = sorted({c for fr in self.frames for c in fr["counts"]})
        with open(os.path.join(self.folder, "counts.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp", "seconds"] + classes)
            for fr in self.frames:
                writer.writerow([fr["frame"], repr(float(fr["timestamp"])), f"{fr['seconds']:.4f}"]
                                + [fr["counts"].get(c, 0) for c in classes])
        info["frames_detected"] = len(self.frames)
        info["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.folder, "info.json"), "w") as f:
            json.dump(info, f, indent=1)


def run_sequence(args, parser, Printer):
    frames = SequenceFrames(args, parser)
    detector = Detector(args, Printer)
    own = OwnNMS(Printer) if args.nms == "own" else None
    sam = Sam(detector.device, Printer) if args.sam else None
    saver = Saver(args.save) if args.save else None
    seconds, num_detected, num_boxes, totals = [], 0, 0, {}
    image_size = None
    paused = False
    start_all = time.time()
    for frame_id, timestamp, img in frames:
        if frame_id % args.step:
            continue
        image_size = (img.shape[1], img.shape[0])
        det = detector.detect(img, args.conf, args.iou, args.one_to_one, args.nms, args.all_classes, masks=True, own_nms=own)
        if sam is not None and len(det["scores"]):
            sam.set_image(img)
            det["masks"], _, _ = sam.boxes(det["boxes"])
            sam.reset()
        seconds.append(det["seconds"])
        num_detected += 1
        num_boxes += len(det["scores"])
        for c in det["classes"]:
            totals[c] = totals.get(c, 0) + 1
        if saver:
            saver.add(frame_id, timestamp, det, det["seconds"], img.shape)
        if not args.headless:
            cv2.imshow("Detections", fit_width(draw_detections(img, det)))
            cv2.setWindowTitle("Detections", f"Detections: frame {frame_id}" + (f" of {frames.num_total}" if frames.num_total else "")
                               + f" ({len(det['scores'])} objects; p pauses, q stops)")
            action = key_action(1)
            if action == "pause":
                paused = True
            while paused:
                action = key_action(50)
                if action in ("pause", "stop"):
                    paused = False
            if action == "stop":
                break
        if args.max_frames and num_detected >= args.max_frames:
            break
    if not args.headless:
        cv2.destroyAllWindows()
    if num_detected == 0:
        sys.exit("no frame was detected")
    per_frame = float(np.median(seconds[1:])) if len(seconds) > 1 else seconds[0]
    of = f" of {frames.num_total}" if frames.num_total else ""
    every = f", every {args.step}" if args.step > 1 else ""
    listed = ", ".join(f"{c} {n / num_detected:.1f}" for c, n in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    summary = (f"{detector.method} on {frames.name} ({num_detected}{of} frames{every}, {image_size[0]}x{image_size[1]}) on "
               f"{detector.device_name}: {per_frame:.2f} s per frame; {num_boxes} detections (conf {args.conf:g}); per frame: {listed or 'nothing'}")
    if saver:
        saver.close(dict(method=detector.method, prompt=args.prompt, conf=args.conf, iou=args.iou, one_to_one=args.one_to_one,
                         nms=args.nms, sam=args.sam, step=args.step, sequence=os.path.abspath(frames.source),
                         frames_list=frames.list_name, device=detector.device_name,
                         seconds=round(time.time() - start_all, 1)))
        summary += f"; {num_detected} frames saved to {args.save}"
    Printer.green(f"\nSummary: {summary}\n")


if __name__ == "__main__":
    main()
