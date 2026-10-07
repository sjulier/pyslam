#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
Check the components of an optional pySLAM extra (see scripts/install_extra.sh) and download their
model weights ahead of time, so that first use in a lab session does not stall on a download.

Each component runs in its own process on the bundled KITTI 06 test images:
- local features / matchers: create the FeatureTrackerConfigs entry and match kitti06-12 vs kitti06-17;
- VPR loop detectors: create the LoopDetectorConfigs entry and describe kitti06-12, -17 (same place)
  and -435 (different place);
- depth and stereo estimators: create the DepthEstimatorType entry and infer a depth map from the
  stereo pair kitti06-12 (the monocular models use the left image only);
- semantic segmentation models: create the SemanticSegmentationType entry and segment kitti06-12;
- scene-from-views models: create the SceneFromViewsType entry and reconstruct kitti06-12, -13, -14;
  Gaussian splatting: run its three CUDA extensions (both skipped, with the reason, on a machine
  without an NVIDIA GPU);
- TensorFlow-based features and place recognition: as the local features and the VPR detectors; in an
  environment without TensorFlow they run in the TensorFlow worker (pyslam/workers), device "tfw".

usage: python scripts/extras_check.py <extra> [COMPONENT ...]   (run from anywhere)
       <extra>: features, features-core, vpr, vpr-core, depth, semantics, scene3d, tf (see install_extra.sh)

A component whose model weights could not be downloaded (no network, a server that does not answer,
a download quota) is reported as UNTRIED, not as failed: run the check again later.
Exit code: 0 if every component is OK or skipped, 1 if one failed, 3 if some are only untried.
"""
import json
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXTRA_COMPONENTS = {
    "features": [
        "SUPERPOINT", "XFEAT", "XFEAT_XFEAT", "XFEAT_LIGHTGLUE", "LIGHTGLUE", "LIGHTGLUE_DISK",
        "LIGHTGLUE_ALIKED", "LIGHTGLUESIFT", "DISK", "ALIKED", "D2NET", "R2D2", "KEYNET",
        "KEYNETAFFNETHARDNET", "BRISK_TFEAT", "ORB2_HARDNET", "ORB2_SOSNET", "ORB2_L2NET", "LOFTR",
    ],
    "vpr": ["ALEXNET", "NETVLAD", "COSPLACE", "EIGENPLACES", "MEGALOC"],
    # the recommended first steps (pixi tasks models-features and models-vpr)
    "features-core": ["SUPERPOINT", "LIGHTGLUE"],
    "vpr-core": ["COSPLACE"],
    "depth": [
        "DEPTH_ANYTHING_V2", "DEPTH_PRO", "DEPTH_RAFT_STEREO", "DEPTH_CRESTEREO_PYTORCH",
        "DEPTH_ANYTHING_V3",
    ],
    "semantics": ["DEEPLABV3", "SEGFORMER", "YOLO", "RFDETR", "CLIP", "DETIC", "EOV_SEG", "ODISE"],
    "scene3d": ["MAST3R", "DUST3R", "MVDUST3R", "VGGT", "VGGT_ROBUST", "FAST3R", "GAUSSIAN_SPLATTING"],
    "tf": ["DELF", "LFNET", "CONTEXTDESC", "GEODESC", "HDC_DELF"],
}

# The TensorFlow-based components are local features, except these place recognition detectors
TF_VPR_COMPONENTS = ("HDC_DELF",)

# Code run in a child process for one component. It prints one JSON line prefixed by RESULT.
_CHILD = r'''
import json, os, socket, sys, time, traceback
kind, name = sys.argv[1], sys.argv[2]
out = {"status": "FAIL"}
# a model download whose connection goes silent (e.g. after a change of network) fails after this
# many seconds without data, instead of blocking the check for good
socket.setdefaulttimeout(120)
try:
    # import pyslam before torch: on macOS pyslam sets PYTORCH_ENABLE_MPS_FALLBACK, which torch only
    # reads when it is first imported (otherwise e.g. ALIKED fails on MPS with NotImplementedError)
    from pyslam.config import Config  # noqa: F401  (also sets up the thirdparty paths)
    import cv2, numpy as np, torch
    data = os.path.join(os.getcwd(), "test", "data")
    read = lambda f: cv2.imread(os.path.join(data, f))
    t0 = time.time()
    uses_tf = kind.startswith("tf-")
    if uses_tf:
        kind = kind[3:]
        try:
            import tensorflow as tf
        except ImportError:
            tf = None  # TensorFlow is not installed here: the model runs in the TensorFlow worker
    if kind == "features":
        from pyslam.local_features.feature_tracker import feature_tracker_factory
        from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
        if name == "GEODESC":  # a descriptor only, without a ready-made configuration: SIFT keypoints + GeoDesc
            from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes
            config = dict(FeatureTrackerConfigs.ORB2)
            config["detector_type"] = FeatureDetectorTypes.SIFT
            config["descriptor_type"] = FeatureDescriptorTypes.GEODESC
        else:
            config = dict(getattr(FeatureTrackerConfigs, name))
        config["num_features"] = 2000
        tracker = feature_tracker_factory(**config)
        img1, img2 = read("kitti06-12-color.png"), read("kitti06-17-color.png")
        kps1, des1 = tracker.detectAndCompute(img1)
        kps2, des2 = tracker.detectAndCompute(img2)
        res = tracker.matcher.match(img1, img2, des1, des2, kps1, kps2)
        out["result"] = f"{len(res.idxs1)} matches"
    elif kind == "depth":
        from pyslam.depth_estimation.depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
        from pyslam.io.dataset_types import DatasetEnvironmentType
        from pyslam.slam import PinholeCamera
        cam = PinholeCamera(Config())
        if not cam.bf:  # the stereo models need the baseline: KITTI's, as the test images are from KITTI 06
            cam.bf = 379.8145
            cam.b = cam.bf / cam.fx
        est = depth_estimator_factory(
            depth_estimator_type=DepthEstimatorType[name], camera=cam, max_depth=50,
            dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        )
        # a rectified stereo pair; the monocular models ignore the right image
        depth, _ = est.infer(read("kitti06-12-color.png"), read("kitti06-12-R-color.png"))
        depth = np.asarray(depth, dtype=np.float64)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if depth.ndim != 2 or valid.size < 0.5 * depth.size:
            raise RuntimeError(f"depth map {depth.shape} has only {valid.size} valid values")
        out["result"] = f"depth map {depth.shape[1]}x{depth.shape[0]}, median {np.median(valid):.1f} m"
    elif kind == "semantics":
        if name == "EOV_SEG" and not torch.cuda.is_available():
            # its backbone uses detectron2's deformable convolution, which has no CPU implementation
            out["status"] = "SKIP"
            out["result"] = "needs an NVIDIA GPU with CUDA (deformable convolution)"
            raise SystemExit
        if name == "ODISE" and not torch.cuda.is_available():
            # it runs on the CPU, but its diffusion backbone took over an hour for one image on a Mac
            out["status"] = "SKIP"
            out["result"] = "needs an NVIDIA GPU with CUDA to run in reasonable time (diffusion backbone)"
            raise SystemExit
        from pyslam.semantics.semantic_segmentation_factory import semantic_segmentation_factory
        from pyslam.semantics.semantic_segmentation_types import SemanticSegmentationType
        from pyslam.semantics.semantic_types import SemanticFeatureType, SemanticDatasetType
        seg = semantic_segmentation_factory(
            semantic_segmentation_type=SemanticSegmentationType[name],
            semantic_feature_type=SemanticFeatureType.LABEL,
            semantic_dataset_type=SemanticDatasetType.CITYSCAPES, image_size=(512, 512),
        )
        img = read("kitti06-12-color.png")
        res = seg.infer(img)
        labels = np.asarray(res.semantics)
        if labels.shape[:2] != img.shape[:2]:
            raise RuntimeError(f"label map {labels.shape} does not match the image {img.shape[:2]}")
        n_inst = 0 if res.instances is None else len(np.unique(res.instances)) - 1
        out["result"] = f"{len(np.unique(labels))} classes" + (f", {n_inst} instances" if res.instances is not None else "")
    elif kind == "scene3d":
        if not torch.cuda.is_available():
            out["status"] = "SKIP"
            out["result"] = "needs an NVIDIA GPU with CUDA"
            raise SystemExit
        if name == "GAUSSIAN_SPLATTING":
            # the three CUDA extensions used by the Gaussian splatting integrator (MonoGS)
            import glob
            import pyslam.config as config
            config.cfg.set_lib("gaussian_splatting")
            built = [glob.glob(os.path.join(os.getcwd(), "thirdparty", p)) for p in (
                "monogs/submodules/simple-knn/simple_knn/_C*.so",
                "monogs/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/_C*.so",
                "lietorch/install/lietorch_backends.so")]
            if not all(built):
                try:  # they may also be installed in the environment (the older install scripts did that)
                    import simple_knn._C, diff_gaussian_rasterization  # noqa: F401
                    from lietorch import SE3  # noqa: F401
                except ImportError:
                    out["status"] = "SKIP"
                    out["result"] = "its CUDA extensions are not built (they need the CUDA compiler nvcc)"
                    raise SystemExit
            from simple_knn._C import distCUDA2
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: F401
            import lietorch
            pts = torch.rand(2000, 3, device="cuda")
            d2 = distCUDA2(pts)
            T = lietorch.SE3.exp(0.1 * torch.randn(8, 6, device="cuda"))
            err = float((T * T.inv()).log().abs().max())
            if not (d2.shape[0] == 2000 and bool(torch.isfinite(d2).all()) and err < 1e-4):
                raise RuntimeError(f"wrong results from the CUDA extensions (lietorch error {err:.2e})")
            from monogs.gaussian_splatting_manager import GaussianSplattingManager  # noqa: F401
            out["result"] = "simple_knn, diff_gaussian_rasterization and lietorch run on the GPU"
            out["seconds"] = round(time.time() - t0, 1)
            out["device"] = "cuda"
            out["status"] = "OK"
            raise SystemExit
        from pyslam.scene_from_views import SceneFromViewsType, scene_from_views_factory
        rec = scene_from_views_factory(scene_from_views_type=SceneFromViewsType[name])
        imgs = [read(f) for f in ("kitti06-12-color.png", "kitti06-13-color.png", "kitti06-14-color.png")]
        res = rec.reconstruct(images=imgs, as_pointcloud=True)
        n_pts = 0 if res.global_point_cloud is None else len(res.global_point_cloud.vertices)
        n_poses = 0 if res.camera_poses is None else len(res.camera_poses)
        if n_pts < 1000:
            raise RuntimeError(f"the reconstruction has only {n_pts} points")
        out["result"] = f"{n_pts} points, {n_poses} camera poses from {len(imgs)} views"
    else:
        from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs, loop_detector_factory
        det = loop_detector_factory(**getattr(LoopDetectorConfigs, name))
        det.init()
        if getattr(det, "global_feature_extractor", "n/a") is None:
            raise RuntimeError("the global feature extractor could not be initialised")
        d = [np.asarray(det.compute_global_des(None, read(f)), dtype=np.float64).ravel()
             for f in ("kitti06-12-color.png", "kitti06-17-color.png", "kitti06-435.png")]
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        same, diff = cos(d[0], d[1]), cos(d[0], d[2])
        out["result"] = f"similarity same place {same:.2f} > different place {diff:.2f}"
        if not same > diff:
            raise RuntimeError(out["result"] + " does not hold")
    out["seconds"] = round(time.time() - t0, 1)
    tf_on_gpu = False
    if uses_tf and tf is None:
        out["device"] = "tfw"  # in the TensorFlow worker (pyslam/workers/tf_worker.py)
        out["status"] = "OK"
        raise SystemExit
    if uses_tf and tf.config.list_physical_devices("GPU"):
        tf_on_gpu = tf.config.experimental.get_memory_info("GPU:0")["peak"] > 0
    if tf_on_gpu or (torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0):
        out["device"] = "cuda"
    elif torch.backends.mps.is_available() and torch.mps.current_allocated_memory() > 0:
        out["device"] = "mps"
    else:
        out["device"] = "cpu"
    out["status"] = "OK"
except SystemExit:
    pass  # the status and the result are already set
except BaseException as e:  # noqa: BLE001
    out["error"] = f"{type(e).__name__}: {e}".splitlines()[0][:200]
    # a download that failed (network, server, quota) leaves the component untried, not failed
    network_errors = {
        "URLError", "timeout", "TimeoutError", "ConnectionError", "ConnectionResetError",
        "ConnectionRefusedError", "ConnectTimeout", "ReadTimeout", "IncompleteRead",
        "RemoteDisconnected", "ChunkedEncodingError", "ContentTooShortError",
        "LocalEntryNotFoundError", "FileURLRetrievalError", "gaierror",
    }
    chain, x = [], e
    while x is not None and len(chain) < 8:
        chain.append(x)
        x = x.__cause__ or x.__context__
    for x in chain:
        names = {c.__name__ for c in type(x).__mro__}
        code = getattr(x, "code", None) or getattr(getattr(x, "response", None), "status_code", None)
        if names & network_errors or ("HTTPError" in names and code in (429, 500, 502, 503, 504)):
            out["status"] = "UNTRIED"
            break
print("RESULT " + json.dumps(out), flush=True)
'''


def check(kind, name, timeout_s=3600):
    t0 = time.time()
    try:
        proc = subprocess.run(  # features-core is checked like features, vpr-core like vpr
            [sys.executable, "-c", _CHILD, kind.removesuffix("-core"), name], cwd=ROOT_DIR, capture_output=True,
            text=True, errors="replace", timeout=timeout_s,
        )
        lines = [l for l in proc.stdout.splitlines() if "RESULT {" in l]
        if lines:
            return json.loads(lines[-1][lines[-1].index("{"):])
        tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return {"status": "FAIL", "error": f"exit code {proc.returncode}: {tail[:200]}"}
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "error": f"timed out after {time.time() - t0:.0f} s"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in EXTRA_COMPONENTS:
        print(__doc__)
        sys.exit(2)
    kind = sys.argv[1]
    names = sys.argv[2:] or EXTRA_COMPONENTS[kind]
    failed, skipped, untried = [], [], []
    tty = sys.stdout.isatty()
    for name in names:
        if tty:  # show what is running (a first run may be downloading model weights)
            print(f"  {name:22s} ...", end="\r", flush=True)
        child_kind = kind
        if kind == "tf":
            child_kind = "tf-vpr" if name in TF_VPR_COMPONENTS else "tf-features"
        r = check(child_kind, name)
        if r["status"] == "OK":
            print(f"  {name:22s} OK    {r['device']:4s} {r['seconds']:6.1f} s  {r['result']}", flush=True)
        elif r["status"] == "SKIP":
            skipped.append(name)
            print(f"  {name:22s} SKIP  {r.get('result', '')}", flush=True)
        elif r["status"] == "UNTRIED":
            untried.append(name)
            print(f"  {name:22s} UNTRIED  download failed: {r.get('error', '')}", flush=True)
        else:
            failed.append(name)
            print(f"  {name:22s} FAIL  {r.get('error', '')}", flush=True)
    print(f"{len(names) - len(failed) - len(skipped) - len(untried)}/{len(names)} components of '{kind}' OK"
          + (f"; skipped: {', '.join(skipped)}" if skipped else "")
          + (f"; untried (download failed, run again to retry): {', '.join(untried)}" if untried else "")
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else (3 if untried else 0))


if __name__ == "__main__":
    main()
