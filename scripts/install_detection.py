#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
COMP0227 Lab 04: download what main_object_detection.py needs, and check it (pixi task models-detection,
in the semantics level):

- the weights of the three methods, from Ultralytics' releases, into data/models/: yolo26s-seg.pt (YOLO),
  yoloe-26s-seg.pt and its text encoder mobileclip2_b.ts (YOLOE), mobile_sam.pt (SAM): 0.35 GB;
- the labelled set of Lab 04 (59 images of a room with the true boxes of their objects) into
  data/detection_sets/room0/. Its images are not part of pySLAM: they come from a staff address, and
  the download is checked against its SHA-256. PYSLAM_DETECTION_SET_URL overrides the address.

Then each method runs once on the test image desk, and one line per component says OK or what failed.
What is already there is not downloaded again. Exit code: 0 if every component is OK, 1 otherwise.

usage: pixi run -e semantics models-detection
"""
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile

# torchvision's NMS has no kernel for Apple's GPU: set the CPU fallback before torch is imported, as
# `import pyslam` does (pyslam/__init__.py)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "data", "models")
SETS_DIR = os.path.join(ROOT_DIR, "data", "detection_sets")
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}"
WEIGHTS = ("yolo26s-seg.pt", "yoloe-26s-seg.pt", "mobileclip2_b.ts", "mobile_sam.pt")

# The labelled set: a zip of room0/ (boxes.json, images/, README.txt), checked by its SHA-256
DETECTION_SET = "room0"
DETECTION_SET_URL = ""  # TODO (COMP0227 staff): the address of room0.zip, once it is decided where it is kept
DETECTION_SET_SHA256 = "ab9b4676b7a5ed52317a93a48ad4b51b9f7dd2b026272e3847f2c9b220ad46b1"
DETECTION_SET_IMAGES, DETECTION_SET_BOXES = 59, 351

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


def install_detection_set():
    """data/detection_sets/room0/ from the staff address, checked against its SHA-256. Returns a reason
    if the set is not there and cannot be downloaded, else None."""
    folder = os.path.join(SETS_DIR, DETECTION_SET)
    if os.path.isfile(os.path.join(folder, "boxes.json")):
        return None
    url = os.environ.get("PYSLAM_DETECTION_SET_URL") or DETECTION_SET_URL
    if not url:
        return ("the address of the labelled set is not set yet (DETECTION_SET_URL in scripts/install_detection.py): "
                "ask the module team, or set PYSLAM_DETECTION_SET_URL")
    os.makedirs(SETS_DIR, exist_ok=True)
    zip_path = os.path.join(SETS_DIR, DETECTION_SET + ".zip")
    print(f"Downloading the labelled set {DETECTION_SET} to data/detection_sets/")
    download(url, zip_path)
    digest = sha256(zip_path)
    if digest != DETECTION_SET_SHA256:
        os.remove(zip_path)
        return f"the download is not the expected file (SHA-256 {digest[:12]}..., expected {DETECTION_SET_SHA256[:12]}...)"
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
        if any(os.path.isabs(n) or ".." in n.split("/") for n in names) or not all(n.startswith(DETECTION_SET + "/") for n in names):
            os.remove(zip_path)
            return "the zip file has an unexpected layout"
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        z.extractall(SETS_DIR, members=names)
    os.remove(zip_path)
    return None


def check_set():
    folder = os.path.join(SETS_DIR, DETECTION_SET)
    with open(os.path.join(folder, "boxes.json")) as f:
        gt = json.load(f)
    missing = [im["file_name"] for im in gt["images"] if not os.path.isfile(os.path.join(folder, "images", im["file_name"]))]
    if missing:
        raise RuntimeError(f"{len(missing)} images of boxes.json are missing from images/")
    return f"{len(gt['images'])} images, {len(gt['annotations'])} true boxes in data/detection_sets/{DETECTION_SET}"


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
    ok = 0
    total = 4
    try:
        install_weights()
    except Exception as e:
        red(f"FAILED  the weights: {e}")
        red("Check the network connection and run the same command again: what is already downloaded is kept.")
        return 1
    for name, good, text in check_methods():
        (green if good else red)(f"{'OK     ' if good else 'FAILED '} {name}: {text}")
        ok += bool(good)
    try:
        reason = install_detection_set()
        if reason:
            red(f"FAILED  the labelled set: {reason}")
        else:
            green(f"OK      the labelled set: {check_set()}")
            ok += 1
    except Exception as e:
        red(f"FAILED  the labelled set: {e}")
    (green if ok == total else red)(f"{ok}/{total} components of 'detection' OK")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
