#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Download what main_object_detection.py needs, and check it (pixi task models-detection, in the semantics
level):

- the weights of the three methods, from Ultralytics' releases, into data/models/: yolo26s-seg.pt (YOLO),
  yoloe-26s-seg.pt and its text encoder mobileclip2_b.ts (YOLOE), mobile_sam.pt (SAM): 0.35 GB.

Then each method runs once on the test image desk, and one line per component says OK or what failed.
With --set DIR, one more line checks a labelled set (DIR/boxes.json, the true boxes in the COCO format,
as main_object_detection.py --evaluate DIR reads them) and, with --images PATH, that its images are in
PATH or in PATH/results (the layout of the Replica package). If boxes.json's "info" has "images_sha256",
the SHA-256 of the lines "<file name> <SHA-256 of the file>\n" of its images in the order of their names,
the line also says whether the images are the files the boxes were made on. What is already downloaded
is not downloaded again. Exit code: 0 if every component is OK, 1 otherwise.

usage: pixi run -e semantics models-detection [--set DIR [--images PATH]]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

# torchvision's NMS has no kernel for Apple's GPU: set the CPU fallback before torch is imported, as
# `import pyslam` does (pyslam/__init__.py)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "data", "models")
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}"
WEIGHTS = ("yolo26s-seg.pt", "yoloe-26s-seg.pt", "mobileclip2_b.ts", "mobile_sam.pt")

TEST_IMAGE = os.path.join(ROOT_DIR, "data", "images", "desk_detic.jpeg")


def green(text):
    print(f"\033[92m{text}\033[0m")


def red(text):
    print(f"\033[91m{text}\033[0m")


def download(url, path, retries=2):
    """Download url to path (through path.part), trying again after a pause if it fails."""
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
            return
        except Exception as e:
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
            if attempt == retries:
                raise RuntimeError(f"cannot download {url}: {e}") from e
            time.sleep(5 * (attempt + 1))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install_weights():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in WEIGHTS:
        path = os.path.join(MODELS_DIR, name)
        if os.path.isfile(path):
            continue
        print(f"Downloading {name} to data/models/")
        download(WEIGHTS_URL.format(name), path)


def check_set(folder, images):
    """A labelled set: DIR/boxes.json, and with images (the --images PATH) its images. Returns (ok, text)."""
    path = os.path.join(os.path.expanduser(folder), "boxes.json")
    if not os.path.isfile(path):
        return False, f"no boxes.json in {folder}"
    with open(path) as f:
        gt = json.load(f)
    text = f"{len(gt['annotations'])} true boxes of {len(gt['images'])} images in {folder}"
    if not images:
        return True, text + "; to check that you have its images, run this command again with --images PATH"
    names = sorted(im["file_name"] for im in gt["images"])
    places = [os.path.expanduser(images), os.path.join(os.path.expanduser(images), "results")]
    found = [sum(os.path.isfile(os.path.join(p, n)) for n in names) for p in places]
    folder_found = places[found.index(max(found))]
    if max(found) < len(names):
        missing = next(n for n in names if not os.path.isfile(os.path.join(folder_found, n)))
        return False, (f"{len(names) - max(found)} of the {len(names)} images (e.g. {missing}) are neither in "
                       f"{images} nor in its results/ folder")
    text += f", and its {len(names)} images in {folder_found}"
    expected = gt.get("info", {}).get("images_sha256")
    if expected:
        h = hashlib.sha256()
        for n in names:
            h.update(f"{n} {sha256(os.path.join(folder_found, n))}\n".encode())
        if h.hexdigest() != expected:
            text += " (but not the same files as those the boxes were made on, so the scores may differ)"
    return True, text


def check_methods():
    """Each method once on desk: [(name, ok, text)]."""
    import cv2
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    img = cv2.imread(TEST_IMAGE)
    results = []
    try:
        from ultralytics import YOLO

        start = time.time()
        r = YOLO(os.path.join(MODELS_DIR, "yolo26s-seg.pt")).predict(img, device=device, verbose=False)[0]
        results.append(("YOLO", True, f"{len(r.boxes)} objects on desk ({time.time() - start:.1f} s, {device.split(':')[0]})"))
    except Exception as e:
        results.append(("YOLO", False, str(e)))
    try:
        from ultralytics import YOLOE

        start = time.time()
        model = YOLOE(os.path.join(MODELS_DIR, "yoloe-26s-seg.pt"))
        cwd = os.getcwd()
        os.chdir(MODELS_DIR)  # Ultralytics looks for the text encoder in the working folder
        try:
            model.set_classes(["keyboard", "mug"], model.get_text_pe(["keyboard", "mug"]))
        finally:
            os.chdir(cwd)
        r = model.predict(img, device=device, verbose=False)[0]
        results.append(("YOLOE", True, f"{len(r.boxes)} objects on desk for \"keyboard, mug\" ({time.time() - start:.1f} s)"))
    except Exception as e:
        results.append(("YOLOE", False, str(e)))
    try:
        from ultralytics.models.sam import Predictor as SAMPredictor

        start = time.time()
        p = SAMPredictor(overrides=dict(task="segment", mode="predict", imgsz=1024, model=os.path.join(MODELS_DIR, "mobile_sam.pt"),
                                        device=device, verbose=False, save=False))
        p.set_image(img)
        r = p(points=[[523, 651]], labels=[1], multimask_output=True)[0]
        p.reset_image()
        results.append(("SAM", True, f"{len(r.masks.data)} masks for a point on desk ({time.time() - start:.1f} s)"))
    except Exception as e:
        results.append(("SAM", False, str(e)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Download and check what object-detection needs")
    parser.add_argument("--set", metavar="DIR",
                        help="also check a labelled set: DIR/boxes.json, as object-detection --evaluate DIR reads it")
    parser.add_argument("--images", metavar="PATH",
                        help="with --set: check that the set's images are in PATH or in PATH/results (for frames of "
                        "Replica, the scene's folder of the Replica package)")
    args = parser.parse_args()
    if args.images and not args.set:
        parser.error("--images goes with --set DIR")
    ok = 0
    total = 4 if args.set else 3
    try:
        install_weights()
    except Exception as e:
        red(f"FAILED  the weights: {e}")
        red("Check the network connection and run the same command again: what is already downloaded is kept.")
        return 1
    for name, good, text in check_methods():
        (green if good else red)(f"{'OK     ' if good else 'FAILED '} {name}: {text}")
        ok += bool(good)
    if args.set:
        try:
            good, text = check_set(args.set, args.images)
            (green if good else red)(f"{'OK     ' if good else 'FAILED '} the labelled set: {text}")
            ok += bool(good)
        except Exception as e:
            red(f"FAILED  the labelled set: {e}")
    (green if ok == total else red)(f"{ok}/{total} components of 'detection' OK")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
