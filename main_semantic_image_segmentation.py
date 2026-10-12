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

# Semantic segmentation of one image, or of every frame of a sequence, with one of pySLAM's
# segmentation methods: the class of every pixel, in the method's colours, and a one-line summary
# (the classes found, their shares of the image, and the time per image). With --save, the label
# maps of a sequence are written for main_slam.py --semantic-mask.
#
#   pixi run -e semantics semantic-segmentation --list
#   pixi run -e semantics semantic-segmentation --method SEGFORMER --test kitti
#   pixi run -e semantics semantic-segmentation --method DEEPLABV3 --image photo.jpg
#   pixi run -e semantics semantic-segmentation --method SEGFORMER --video data/videos/kitti06/video_color.mp4 --save labels/kitti06

import pyslam  # first: sets the OpenMP thread settings before numpy/torch are imported
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)

# The methods (SemanticSegmentationType entries): what each gives
kMethods = {
    "SEGFORMER": "a class for every pixel: 19 street classes (Cityscapes). SegFormer-B0, a transformer",
    "DEEPLABV3": "a class for every pixel: 21 object classes (Pascal VOC). DeepLabV3 with a ResNet-50",
    "CLIP": "a class for every pixel, from text: the best matching of the 19 Cityscapes class names. CLIP ViT-L/14",
    "YOLO": "the pixels of each object it detects, nothing elsewhere: 80 object classes (COCO). YOLO26-S",
    "RFDETR": "the pixels of each object it detects, nothing elsewhere: 80 object classes (COCO). RF-DETR Seg-S",
    "DETIC": "the pixels of each object it detects, nothing elsewhere: 1203 object classes (LVIS). Detic, Swin-B",
    "EOV_SEG": "a class for every pixel, open vocabulary. EOV-Seg (NVIDIA GPU only)",
    "ODISE": "a class for every pixel, open vocabulary. ODISE (NVIDIA GPU only)",
}
# These give label 0 to every pixel which is in no object, and label 0 is also their first class
# (YOLO: person; Detic: aerosol can). Their instance map tells the two apart: it is 0 outside the objects.
kObjectsOnly = ("YOLO", "RFDETR", "DETIC")
# The set of classes each method really uses (the detectors keep their own, whatever they are asked for)
kClassSets = {"SEGFORMER": "Cityscapes", "DEEPLABV3": "Pascal VOC", "CLIP": "Cityscapes", "YOLO": "COCO",
              "RFDETR": "COCO", "DETIC": "LVIS", "EOV_SEG": "open vocabulary", "ODISE": "open vocabulary"}
kCudaOnly = ("EOV_SEG", "ODISE")
kNoObject = "no object"

# The images of --test
kTests = {
    "kitti": "test/data/kitti06-12-color.png",  # frame 12 of the KITTI 06 video
    "kitti_next": "test/data/kitti06-17-color.png",  # half a second later
    "desk": "data/images/desk_detic.jpeg",
    "churchill": "test/data/churchill/1.ppm",
    "box": "test/data/box_in_scene.png",
    "mars": "test/data/mars1.png",
}


def main():
    from pyslam.io.sequence_options import add_sequence_options, has_sequence_options, sequence_config_path

    parser = argparse.ArgumentParser(
        description="Semantic segmentation of one image, or of every frame of a sequence, with one "
        "of pySLAM's segmentation methods",
        epilog="examples: pixi run -e semantics semantic-segmentation --method SEGFORMER --test kitti; "
        "pixi run -e semantics semantic-segmentation --method DEEPLABV3 --image photo.jpg; "
        "pixi run -e semantics semantic-segmentation --method SEGFORMER "
        "--video data/videos/kitti06/video_color.mp4 --save labels/kitti06",
    )
    parser.add_argument(
        "--method", default="SEGFORMER", metavar="NAME", help="the segmentation method (default SEGFORMER; see --list)"
    )
    parser.add_argument("--test", default="kitti", choices=sorted(kTests), help="the image (default kitti)")
    parser.add_argument("--image", metavar="FILE", help="an image file of your own, instead of --test")
    parser.add_argument(
        "--grey", action="store_true", help="turn the image (or the frames) to greyscale before segmenting"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="with one image: how many times to repeat it for the timing (default 3)"
    )
    parser.add_argument(
        "--image-size", type=int, default=512, metavar="N", help="the image is resized to NxN for the network (default 512)"
    )
    add_sequence_options(parser, need_settings=False)
    sequence = parser.add_argument_group("with a sequence")
    sequence.add_argument("--save", metavar="DIR", help="write the label map of every frame to this folder, for main_slam.py --semantic-mask")
    sequence.add_argument("--step", type=int, default=1, metavar="N", help="segment every Nth frame (default 1: every frame)")
    sequence.add_argument("--max-frames", type=int, metavar="N", help="stop after N segmented frames")
    sequence.add_argument("--headless", action="store_true", help="no windows")
    parser.add_argument("--list", action="store_true", help="list the methods")
    args = parser.parse_args()

    if args.list:
        print("Methods (--method), and what each gives:")
        for name, what in kMethods.items():
            print(f"    {name:10s} {what}")
        print("Their models are installed by: pixi run -e semantics models-semantics")
        print(f"{' and '.join(kCudaOnly)} need an NVIDIA GPU (the full level: pixi run -e full ...).")
        sys.exit(0)
    if args.method not in kMethods:
        print(f"Unknown method '{args.method}'. Valid names (see --list):", file=sys.stderr)
        print("    " + " ".join(kMethods), file=sys.stderr)
        sys.exit(2)
    if args.image and has_sequence_options(args):
        parser.error("give --image or a sequence, not both")

    # the configuration (it sets the library paths; its dataset is only used for a sequence)
    from pyslam.config import Config

    config_path = sequence_config_path(args, parser, default_settings=os.path.join(kScriptFolder, "settings", "KITTI04-12.yaml"))
    config = Config(config_path) if config_path else Config()
    if config_path:
        os.remove(config_path)  # the temporary file of sequence_config_path(): it has been read

    from pyslam.utilities.logging import Printer

    segmenter = Segmenter(args.method, args.image_size)
    if has_sequence_options(args):
        from pyslam.io.dataset_factory import dataset_factory

        dataset = dataset_factory(config)
        run_sequence(args, segmenter, dataset, Printer)
    else:
        run_image(args, segmenter, Printer)


class Segmenter:
    """One of pySLAM's segmentation methods, with the names and colours of its classes."""

    def __init__(self, method, image_size):
        from pyslam.semantics.semantic_segmentation_factory import semantic_segmentation_factory
        from pyslam.semantics.semantic_segmentation_types import SemanticSegmentationType
        from pyslam.semantics.semantic_types import SemanticDatasetType, SemanticFeatureType

        self.method = method
        # pySLAM's DeepLabV3 has Pascal VOC weights only; the others are asked for the Cityscapes classes
        self.dataset = SemanticDatasetType.VOC if method == "DEEPLABV3" else SemanticDatasetType.CITYSCAPES
        self.class_set = kClassSets[method]
        start = time.time()
        self.model = semantic_segmentation_factory(
            semantic_segmentation_type=SemanticSegmentationType[method],
            semantic_feature_type=SemanticFeatureType.LABEL,
            semantic_dataset_type=self.dataset,
            image_size=(image_size, image_size),
            device=None,  # autodetect
        )
        self.load_time = time.time() - start
        self.device = str(getattr(self.model, "device", ""))
        self.num_classes = int(self.model.num_classes())
        self.objects_only = method in kObjectsOnly
        # the label of the pixels in no object, when the method only labels the objects it finds
        self.no_object_label = self.num_classes if self.objects_only else None
        self.class_names = self._class_names()
        if self.no_object_label is not None:
            self.class_names[self.no_object_label] = kNoObject
        self.colours = {}  # label -> BGR colour, as seen

    def _class_names(self):
        """The names of the classes, by label, as far as the method gives them."""
        model = getattr(self.model, "model", None)
        for source in (
            getattr(model, "names", None),
            getattr(model, "class_names", None),
            getattr(getattr(model, "model", None), "names", None),
        ):
            if isinstance(source, dict) and source:
                return {int(k): str(v) for k, v in source.items()}
            if isinstance(source, (list, tuple)) and source:
                return dict(enumerate(str(v) for v in source))
        metadata = getattr(model, "metadata", None)
        listed = getattr(metadata, "thing_classes", None) if self.method == "DETIC" else (
            getattr(metadata, "stuff_classes", None) or getattr(metadata, "thing_classes", None)
        )
        if listed:
            return dict(enumerate(str(v) for v in listed))
        try:
            from pyslam.semantics.semantic_labels_factory import semantic_labels_factory

            return dict(enumerate(str(v) for v in semantic_labels_factory(self.dataset)))
        except Exception:
            return {}

    def name(self, label):
        return self.class_names.get(int(label), str(int(label)))

    def segment(self, img):
        """The label map of img (with no_object_label on the pixels in no object), its colours, and
        the colours over the image."""
        result = self.model.infer(img)
        labels = np.asarray(result.semantics)
        instances = result.instances
        colours = np.ascontiguousarray(self.model.sem_img_to_rgb(labels, bgr=True))
        labelled = np.ones(labels.shape, bool)
        if self.objects_only and instances is not None:
            labelled = np.asarray(instances) > 0
        labels = labels.astype(np.int32)
        if self.no_object_label is not None:
            labels[~labelled] = self.no_object_label
            colours[~labelled] = 0
        for label in np.unique(labels[labelled]):
            if int(label) not in self.colours:
                self.colours[int(label)] = [int(v) for v in colours[labelled & (labels == label)][0]]
        over_image = cv2.addWeighted(img, 0.45, np.zeros_like(img), 0.55, 0)  # in no object: the image, darkened
        over_image[labelled] = cv2.addWeighted(colours, 0.6, img, 0.4, 0)[labelled]
        return labels, colours, over_image

    def shares(self, labels):
        """[(label, share of the pixels)], largest first."""
        ids, counts = np.unique(labels, return_counts=True)
        return sorted(((int(i), float(c) / labels.size) for i, c in zip(ids, counts)), key=lambda s: -s[1])

    def shares_text(self, shares, most=10):
        return ", ".join(f"{self.name(i)} {100 * share:.0f}%" for i, share in shares[:most])

    def legend(self, shares, width, most=8):
        """A strip with the colours and names of the largest classes."""
        strip = np.full((30, width, 3), 255, np.uint8)
        x = 8
        for label, share in shares[:most]:
            text = f"{self.name(label)} {100 * share:.0f}%"
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            if x + 24 + text_width > width - 8:
                break
            colour = self.colours.get(label, [40, 40, 40])
            cv2.rectangle(strip, (x, 7), (x + 16, 23), colour, -1)
            cv2.rectangle(strip, (x, 7), (x + 16, 23), (60, 60, 60), 1)
            cv2.putText(strip, text, (x + 22, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            x += 22 + text_width + 16
        return strip

    def label_dtype(self):
        return np.uint8 if self.num_classes < 255 else np.uint16


def to_grey(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def show(segmenter, img, labels, colours, over_image, title):
    shares = segmenter.shares(labels)
    cv2.imshow("Image", img)
    cv2.imshow(f"Classes - {segmenter.method}", np.vstack([colours, segmenter.legend(shares, colours.shape[1])]))
    cv2.imshow(f"Classes over the image - {segmenter.method}", over_image)
    cv2.setWindowTitle("Image", title)


def run_image(args, segmenter, Printer):
    if args.image:
        path, name = args.image, os.path.basename(args.image)
    else:
        path, name = os.path.join(kScriptFolder, kTests[args.test]), args.test
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read the image {path}")
    if args.grey:
        img = to_grey(img)
    h, w = img.shape[:2]

    start = time.time()
    labels, colours, over_image = segmenter.segment(img)
    first = time.time() - start
    times = []
    for _ in range(max(0, args.runs)):
        start = time.time()
        segmenter.segment(img)
        times.append(time.time() - start)
    per_image = np.mean(times) if times else first
    shares = segmenter.shares(labels)

    summary = (
        f"{segmenter.method} on {name} ({w}x{h} pixels{', in greyscale' if args.grey else ''}): "
        f"{segmenter.num_classes} classes ({segmenter.class_set}) on {segmenter.device}; "
        f"{per_image:.2f} s per image"
        + (
            f" (mean of {len(times)} runs after the first, which took {first:.2f} s"
            if len(times) > 1
            else f" (one run after the first, which took {first:.2f} s" if times else " (the first run"
        )
        + f"; {segmenter.load_time:.1f} s to load the model); {segmenter.shares_text(shares)}"
    )
    Printer.green(f"\nSummary: {summary}\n")
    if not args.headless:
        show(segmenter, img, labels, colours, over_image, f"Image: {name}")
        print("Press a key in one of the windows to close them.")
        cv2.waitKey()
        cv2.destroyAllWindows()


def run_sequence(args, segmenter, dataset, Printer):
    source = args.video or args.images or args.tum
    name = os.path.basename(os.path.normpath(source))
    num_total = getattr(dataset, "num_frames", None)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    info = {
        "method": segmenter.method,
        "class_set": segmenter.class_set,
        "num_classes": segmenter.num_classes,
        "class_names": {str(k): v for k, v in sorted(segmenter.class_names.items())},
        "no_object_label": segmenter.no_object_label,
        "sequence": os.path.abspath(source),
        "step": args.step,
        "greyscale": bool(args.grey),
        "device": segmenter.device,
        "frames": [],
    }
    rows = []  # one per segmented frame: id, time, changed fraction, {label: share}
    totals = {}  # label -> sum of the shares
    prev_labels = None
    img_id = 0
    paused = False
    stopped = False
    print(f"{segmenter.method} on {name}: every {args.step} frame(s); q or Esc in a window stops, p pauses")
    while dataset.is_ok and not stopped:
        img = dataset.getImageColor(img_id)
        if img is None:
            break
        if img_id % args.step == 0:
            if args.max_frames is not None and len(rows) >= args.max_frames:
                break
            if args.grey:
                img = to_grey(img)
            start = time.time()
            labels, colours, over_image = segmenter.segment(img)
            seconds = time.time() - start
            shares = segmenter.shares(labels)
            changed = None
            if prev_labels is not None and prev_labels.shape == labels.shape:
                changed = float(np.count_nonzero(prev_labels != labels)) / labels.size
            prev_labels = labels
            rows.append((img_id, seconds, changed, dict(shares)))
            for label, share in shares:
                totals[label] = totals.get(label, 0.0) + share
            if args.save:
                cv2.imwrite(os.path.join(args.save, f"labels_{img_id:06d}.png"), labels.astype(segmenter.label_dtype()))
                info["frames"].append(img_id)
            if len(rows) == 1 or len(rows) % 50 == 0:
                print(
                    f"frame {img_id}" + (f"/{num_total}" if num_total else "") + f": {seconds:.2f} s; "
                    + segmenter.shares_text(shares, most=5)
                )
            if not args.headless:
                show(segmenter, img, labels, colours, over_image, f"Image: {name}, frame {img_id}")
                while True:
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key in (ord("q"), 27):
                        stopped = True
                        break
                    if key == ord("p"):
                        paused = not paused
                        Printer.yellow("Paused: p resumes." if paused else "Resumed.")
                    if not paused:
                        break
        img_id += 1

    if not rows:
        sys.exit(f"no frames were read from {source}")
    seconds = float(np.mean([r[1] for r in rows]))
    changes = [r[2] for r in rows if r[2] is not None]
    mean_shares = sorted(((label, total / len(rows)) for label, total in totals.items()), key=lambda s: -s[1])
    summary = (
        f"{segmenter.method} on {name}: {len(rows)} frames" + (f" of {num_total}" if num_total else "")
        + (f" (every {args.step})" if args.step > 1 else "") + (", in greyscale" if args.grey else "")
        + f"; {segmenter.num_classes} classes ({segmenter.class_set}) on {segmenter.device}; "
        f"{seconds:.2f} s per frame; mean shares: {segmenter.shares_text(mean_shares)}"
    )
    if changes:
        summary += f"; the label changed on {100 * np.mean(changes):.1f}% of the pixels from one segmented frame to the next (mean)"
    if args.save:
        info["seconds_per_frame"] = seconds
        with open(os.path.join(args.save, "info.json"), "w") as f:
            json.dump(info, f, indent=1)
        labels_seen = sorted(totals)
        with open(os.path.join(args.save, "shares.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "seconds", "changed_fraction"] + [segmenter.name(l) for l in labels_seen])
            for frame_id, secs, changed, shares in rows:
                writer.writerow([frame_id, f"{secs:.4f}", "" if changed is None else f"{changed:.4f}"] + [f"{shares.get(l, 0.0):.4f}" for l in labels_seen])
        summary += f"; saved to {args.save} (labels_NNNNNN.png, info.json, shares.csv)"
    Printer.green(f"\nSummary: {summary}\n")
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
