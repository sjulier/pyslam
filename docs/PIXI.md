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

Start with `default` and move up when you need a script of a higher level. Moving up downloads only
the packages that level adds; the native modules do **not** need to be rebuilt.

On **Linux without an NVIDIA GPU**, use the CPU levels instead: `default-cpu`, `depth-cpu`,
`semantics-cpu`, `full-cpu`.

## Supported systems

| system | GPU | levels |
|---|---|---|
| Linux x86-64 with an NVIDIA GPU and a recent driver | CUDA 12.9, from Pascal (GTX 10xx, Titan Xp) to the RTX 50 series | `default`, `depth`, `semantics`, `full`: tested (RTX 5090, driver 575) |
| Linux x86-64 without an NVIDIA GPU | none | `default-cpu`, `depth-cpu`, `semantics-cpu`, `full-cpu`: tested. EOV-Seg and everything `full` adds need an NVIDIA GPU and are skipped |
| macOS 14 or later, Apple silicon | the Apple GPU (MPS) | `default`, `depth`, `semantics`: tested (MacBook Air M1). EOV-Seg, ODISE and everything `full` adds need an NVIDIA GPU and are skipped |
| Windows, via WSL2 (Ubuntu inside Windows) | as Linux: follow the Linux instructions | `default`: tested (Windows 11). The other levels: not tested yet |
| Windows, native (experimental) | none: the CPU build of PyTorch | `default-win` only, the default level without the TensorFlow features: see [Windows (native, experimental)](../README.md#windows-native-experimental) |

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
git clone --depth 1 https://github.com/sjulier/pyslam.git
cd pyslam
```

`--depth 1` downloads only the current version, without the history (about 0.6 GB instead of
1.1 GB). Run all the commands below from this folder.

### 3. Build the default level

```bash
pixi run build      # downloads the environment, builds pySLAM's native modules (about 15 min on a fast
                    # machine, up to an hour on a laptop) and fetches the ORB vocabulary (a 31 MB download)
pixi run check      # the native modules load and the optimiser tests pass
pixi run models     # the recommended learned models: SuperPoint, LightGlue, CosPlace (about 0.3 GB)
```

`pixi run build` is safe to re-run: it skips what is already built. `pixi run models` prints one line
per component, `OK` with the device it ran on (`cuda`, `mps` or `cpu`), and ends with summaries such
as `2/2 components of 'features-core' OK`. A component marked `UNTRIED` could not be downloaded: run
the command again later. `pixi run models-all-features` and `pixi run models-all-vpr` install all 19
learned features and all 5 place recognition models (about 3.4 GB).

On Linux without an NVIDIA GPU, add `-e default-cpu` to each command (`pixi run -e default-cpu build`).

### 4. Move up a level

Install a level's models when you first need it. Each command also checks every component it
installed.

```bash
pixi run -e depth models-depth            # depth and stereo models
pixi run -e semantics models-semantics    # segmentation and detection models
pixi run -e full models-scene3d           # 3R models and Gaussian splatting (NVIDIA GPU; builds CUDA extensions)
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
| `build` | default | install or build the native modules in two parts, then check that they share one pybind11 ABI: the prerequisites (GTSAM, g2o, Pangolin, DBoW2/3, iBoW) and pySLAM's own C++ code (ORB-SLAM2 features, C++ utilities, C++ core). Each part comes prebuilt when there is a bundle for the checkout and machine (`scripts/native_bundle.py`); after a change to pySLAM's C++ code only that part is rebuilt |
| `build-prerequisites`, `build-pyslam` | default | build one part from source |
| `vocabulary` | default | download the ORB vocabulary of the default loop detector (part of `build`); an interrupted or stalled download is resumed |
| `check` | default | the ABI check and the GTSAM and g2o optimiser tests |
| `models` | default | the recommended learned models: `models-features` + `models-vpr` |
| `models-features` | default | SuperPoint, and SuperPoint with LightGlue: `scripts/install_extra.sh features-core` |
| `models-vpr` | default | CosPlace place recognition: `scripts/install_extra.sh vpr-core` |
| `models-all-features` | default | all 19 learned features and matchers: `scripts/install_extra.sh features` |
| `models-all-vpr` | default | all 5 place recognition models: `scripts/install_extra.sh vpr` |
| `models-tf` | default | the TensorFlow-based features (DELF, LF-Net, ContextDesc, GeoDesc) and the HDC-DELF place recognition, in their own environment: `scripts/install_extra.sh tf` |
| `models-depth` | depth | `scripts/install_extra.sh depth` |
| `models-semantics` | semantics | `scripts/install_extra.sh semantics` |
| `models-scene3d` | full | `scripts/install_extra.sh scene3d` |
| `doctor` | default | check the machine, the environment, every native module and the data, one line per check; `doctor --run` also runs SLAM on KITTI 06 without windows |
| `slam`, `vo`, `feature-matching`, `map-viewer`, `slam-evaluation`, `dense-reconstruction` | default | the main scripts |
| `depth-prediction` | depth | `main_depth_prediction.py` |
| `semantic-segmentation` | semantics | `main_semantic_image_segmentation.py` |
| `scene-from-views` | full | `main_scene_from_views.py` |

`pixi task list` shows them all. The single build steps are tasks too (`build-gtsam`, `build-g2o`,
`build-cpp-core`, ...), to re-run one of them. `build-cpp-core` and the other steps of the pySLAM part
expect the prerequisites to be there (built, or installed by `build`).

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
  default speed when you compare results. With the windows open, the playback slows down by itself
  whenever tracking gets weak (it prints `Playback speed: ...`; `--no-throttle` turns this off). With
  `--headless` it is off: if tracking is lost at the same places in every run, the machine may be too
  slow for the camera's frame rate, and `--throttle` turns it on.
- **If a download fails**, run the same task again: what is already installed is skipped, and an
  interrupted download continues where it stopped. A component that could not be installed does not
  stop the others; the task lists what is missing at the end (`Not installed: ...`). Model weights
  are fetched by the `models*` tasks and the ORB vocabulary by `build`, not during the first SLAM run.
- **A component is skipped, with the reason, when the machine cannot run it** (for example the 3R
  models without an NVIDIA GPU). A skipped component is not a failure.
- **Gaussian splatting on another GPU.** Its CUDA extensions are built for the GPUs in the machine.
  To build them for other GPUs, set `PYSLAM_CUDA_ARCHS` and `LIETORCH_CUDA_ARCHS` (for example `61`
  for Pascal, `86` for the RTX 30 series) before `pixi run -e full models-scene3d`.
- **The TensorFlow-based features** (DELF, LF-Net, ContextDesc, GeoDesc, and the HDC-DELF place
  recognition) have their own environment, because TensorFlow's requirements do not fit the levels:
  `pixi run models-tf` installs it (`pixi run -e default-cpu models-tf` on Linux without an NVIDIA
  GPU; about 6 GB) and checks the five components. pySLAM then runs them in a worker process that it
  starts on first use: `pixi run slam --features CONTEXTDESC`. GeoDesc is a descriptor only, without a
  SLAM configuration, and HDC-DELF is a loop detector, chosen in the settings file. Not on native
  Windows.
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
- The TensorFlow environments (`tf`, and `tf-cpu` for the CPU ladder) are not levels: the features
  that need TensorFlow run in them through a worker process (`pyslam/workers/`).
