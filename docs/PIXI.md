# pySLAM with pixi

[pixi](https://pixi.sh) installs pySLAM's Python environment from a lock file (`pixi.lock`), so
everybody gets exactly the same package versions. The environment lives inside the repository
folder (`.pixi/`): pixi does not touch conda, your `base` environment or `~/.condarc`.

<!-- TOC -->

- [pySLAM with pixi](#pyslam-with-pixi)
  - [Levels](#levels)
  - [Supported systems](#supported-systems)
  - [Install](#install)
    - [1. Install pixi](#1-install-pixi)
    - [2. Get the code](#2-get-the-code)
    - [3. Build the default level](#3-build-the-default-level)
    - [4. Move up a level](#4-move-up-a-level)
  - [Run](#run)
  - [Tasks](#tasks)
  - [Good to know](#good-to-know)
  - [For maintainers](#for-maintainers)

<!-- /TOC -->

## Levels

The environment comes in levels. Each level runs more of the `main_*.py` scripts and **includes the
levels below it**.

| level | adds these scripts | adds these components |
|---|---|---|
| `default` | `main_vo.py`, `main_feature_matching.py`, `main_slam.py`, `main_map_viewer.py`, `main_slam_evaluation.py`, `main_map_dense_reconstruction.py` (TSDF / voxel grid) | classic visual odometry and SLAM, loop closing, the viewers; learned features and matchers (SuperPoint, LightGlue, XFeat, DISK, ALIKED, ...) and learned place recognition (NetVLAD, CosPlace, EigenPlaces, MegaLoc) |
| `depth` | `main_depth_prediction.py`; depth prediction inside SLAM and dense reconstruction | Depth Anything V2 and V3, Depth Pro, RAFT-Stereo, CREStereo |
| `semantics` | `main_semantic_image_segmentation.py`; semantic mapping | DeepLabV3, SegFormer, YOLO, RF-DETR, CLIP, Detic, EOV-Seg, ODISE |
| `full` | `main_scene_from_views.py`; Gaussian splatting | MASt3R, DUSt3R, MV-DUSt3R, VGGT, Robust VGGT, Fast3R; MonoGS |
| `tf` | no new script: more features for `main_slam.py` and `main_feature_matching.py` | TensorFlow, for the TensorFlow-based features (DELF, LF-Net, ContextDesc, GeoDesc) and the HDC-DELF place recognition |

Start with `default` and move up when you need a script of a higher level. Moving up downloads only
the packages that level adds; the native modules do **not** need to be rebuilt.

On **Linux without an NVIDIA GPU**, use the CPU levels instead: `default-cpu`, `depth-cpu`,
`semantics-cpu`, `full-cpu`, `tf-cpu`.

## Supported systems

| system | GPU | levels |
|---|---|---|
| Linux x86-64 with an NVIDIA GPU and a recent driver | CUDA 12.9, from Pascal (GTX 10xx, Titan Xp) to the RTX 50 series | `default`, `depth`, `semantics`, `full`: tested (RTX 5090, driver 575) |
| Linux x86-64 without an NVIDIA GPU | none | `default-cpu`, `depth-cpu`, `semantics-cpu`, `full-cpu`: tested. EOV-Seg and everything `full` adds need an NVIDIA GPU and are skipped |
| macOS 14 or later, Apple silicon | the Apple GPU (MPS) | `default`, `depth`, `semantics`, `full` (the 3R models and Gaussian splatting of `full` need an NVIDIA GPU): not tested yet |
| Windows | via WSL2 (Ubuntu inside Windows): follow the Linux instructions | as Linux: not tested yet |

You do not need to install the CUDA toolkit or a compiler: both are part of the environment.

Disk space: about 20 GB for all four levels together (they share their files), plus the model
weights you download, which add up to tens of GB if you install every level.

## Install

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new terminal afterwards, and check with `pixi --version` (0.81 or later).

### 2. Get the code

```bash
git clone https://github.com/sjulier/pyslam.git
cd pyslam
```

Run all the commands below from this folder.

### 3. Build the default level

```bash
pixi run build      # downloads the environment, builds pySLAM's native modules (about 15 min on a fast
                    # machine, up to an hour on a laptop) and fetches the ORB vocabulary (about 100 MB)
pixi run check      # the native modules load and the optimiser tests pass
pixi run models     # learned features and place recognition: code, model weights, and a check of each
```

`pixi run build` is safe to re-run: it skips what is already built. `pixi run models` prints one line
per component, `OK` with the device it ran on (`cuda`, `mps` or `cpu`), and ends with a summary such
as `19/19 components of 'features' OK`.

On Linux without an NVIDIA GPU, add `-e default-cpu` to each command (`pixi run -e default-cpu build`).

### 4. Move up a level

Install a level's models when you first need it. Each command also checks every component it
installed.

```bash
pixi run -e depth models-depth            # depth and stereo models
pixi run -e semantics models-semantics    # segmentation and detection models
pixi run -e full models-scene3d           # 3R models and Gaussian splatting (NVIDIA GPU; builds CUDA extensions)
pixi run -e tf models-tf                  # TensorFlow-based features
```

## Run

Every main script has a task. Arguments after the task name are passed to the script.

```bash
pixi run slam                         # main_slam.py on the bundled KITTI 06 video
pixi run slam --headless              # without windows; prints the trajectory error at the end
pixi run vo
pixi run feature-matching
pixi run -e depth depth-prediction
pixi run -e semantics semantic-segmentation
pixi run -e full scene-from-views --method MAST3R
```

A level also runs the tasks of the levels below it (`pixi run -e full slam`). To work inside an
environment instead of prefixing every command, open a shell in it:

```bash
pixi shell -e semantics
python main_semantic_image_segmentation.py
```

## Tasks

| task | level | what it does |
|---|---|---|
| `build` | default | build the native modules (GTSAM, g2o, Pangolin, DBoW2/3, iBoW, ORB-SLAM2 features, C++ utilities, C++ core), then check that they share one pybind11 ABI |
| `vocabulary` | default | download the ORB vocabulary of the default loop detector (part of `build`); an interrupted or stalled download is resumed |
| `check` | default | the ABI check and the GTSAM and g2o optimiser tests |
| `models` | default | learned features and place recognition: `scripts/install_extra.sh features vpr` |
| `models-depth` | depth | `scripts/install_extra.sh depth` |
| `models-semantics` | semantics | `scripts/install_extra.sh semantics` |
| `models-scene3d` | full | `scripts/install_extra.sh scene3d` |
| `models-tf` | tf | `scripts/install_extra.sh tf` |
| `slam`, `vo`, `feature-matching`, `map-viewer`, `slam-evaluation`, `dense-reconstruction` | default | the main scripts |
| `depth-prediction` | depth | `main_depth_prediction.py` |
| `semantic-segmentation` | semantics | `main_semantic_image_segmentation.py` |
| `scene-from-views` | full | `main_scene_from_views.py` |

`pixi task list` shows them all. The single build steps are tasks too (`build-gtsam`, `build-g2o`,
`build-cpp-core`, ...), to re-run one of them.

## Good to know

- **The native modules belong to the level they were built in.** They are built once, in the source
  tree, and every level of a ladder uses them. If you delete the environment you built them in
  (`pixi clean`, or removing `.pixi/envs/default`), run `pixi run build` again after removing the
  build folders (`./clean.sh`). The GPU levels and the CPU levels are two separate ladders: build
  again when you change between them.
- **Model code is not installed into the environment.** `scripts/install_extra.sh` clones each model
  at a fixed version into `thirdparty/`, applies pySLAM's patches and downloads the weights. It can be
  re-run at any time, and it skips what is already there.
- **`main_slam.py` feeds frames at the camera's rate**, with or without `--headless`. `--speed 2`
  plays twice as fast, and `--speed 0` as fast as possible. Faster than the camera, the mapping thread
  gets less time per frame, and on KITTI 06 tracking is then sometimes lost at the turns. Use the
  default speed when you compare results.
- **If a download fails**, run the same task again: what is already installed is skipped, and an
  interrupted download continues where it stopped. A component that could not be installed does not
  stop the others; the task lists what is missing at the end (`Not installed: ...`). Model weights
  are fetched by the `models*` tasks and the ORB vocabulary by `build`, not during the first SLAM run.
- **A component is skipped, with the reason, when the machine cannot run it** (for example the 3R
  models without an NVIDIA GPU). A skipped component is not a failure.
- **Gaussian splatting on another GPU.** Its CUDA extensions are built for the GPUs in the machine.
  To build them for other GPUs, set `PYSLAM_CUDA_ARCHS` and `LIETORCH_CUDA_ARCHS` (for example `61`
  for Pascal, `86` for the RTX 30 series) before `pixi run -e full models-scene3d`.
- **Not available with pixi**: SURF (non-free); CREStereo's original MegEngine version (the PyTorch
  port is installed); pytorch3d.
- **Do not `pip install` into the environment.** A package that is missing belongs in `pixi.toml`.
- **`pip check`** reports nothing in the `default` and `depth` levels. In `semantics` and `full` it
  reports that detectron2 requires `black`: detectron2's metadata pins that code formatter, which is
  not used at run time and is not installed.
- **Remove everything pixi installed** with `pixi clean` (it deletes `.pixi/`).

## For maintainers

- `pixi.toml` defines small *features* (`build`, `slam`, `torch`, `cuda`, `cpu`, `depth`,
  `semantics`, `recon3d`, `tf`, `dev`) and combines them into the level environments. The levels of
  a ladder share a *solve group*, so they have identical versions of every shared package: that is
  why one build of the native modules serves them all.
- Packages come from conda-forge. PyPI is used only for packages that conda-forge does not have, or
  has in a version that would force older versions of the main packages.
- After changing `pixi.toml`, run `pixi lock` and **read the lock's diff**: one package with old
  requirements can pull OpenCV, PyTorch or Open3D back to older versions in every level. Then run
  `pixi run check` and the `models*` tasks of the levels you changed.
- Linux is pinned to CUDA 12.9 (`cuda-version`), whose builds still support Pascal and Volta GPUs.
- TensorFlow shares the ladder's solve group, so the whole ladder uses the versions that conda-forge's
  TensorFlow build allows (at the moment PyTorch 2.12.0 instead of 2.12.1, because of protobuf).
  If that ever blocks an update, the alternative is a separate environment for TensorFlow with a
  worker process.
