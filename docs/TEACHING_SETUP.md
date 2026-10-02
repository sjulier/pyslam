# pySLAM setup for the course (draft)

> **Draft.** This is the short installation path for the course. It installs the **core** of pySLAM
> (visual odometry, full SLAM with classical features, loop closing, g2o/GTSAM optimisation and the
> viewers) and then **optional components on request**, week by week. A pixi-based setup that
> replaces steps 3–6 with a few commands is described in [PIXI.md](./PIXI.md); it has been tested
> on Linux with an NVIDIA GPU so far.
>
> Do **not** run `./install_all.sh` for the course: it installs every optional component at once
> (over 1.5 hours and many GB of downloads).

## Supported systems

| system | status |
|---|---|
| **Linux** (Ubuntu 22.04 / 24.04, x86-64) | tested. NVIDIA GPUs are optional; older GPUs (e.g. Pascal, such as the Titan Xp) are supported |
| **macOS** (Apple silicon, macOS 14 or later) | tested. Learned features use the Apple GPU (MPS) |
| **Windows** | via **WSL2** only (Ubuntu inside Windows); being tested |

You need **about 20 GB of free disk space** (the Python environment alone is about 13 GB) and an
internet connection. The core installation takes **about 1 hour** on a recent machine, mostly
downloading and compiling; you only do it once.

## 1. Prerequisites

- **conda**, version 23.10 or later (for its fast "libmamba" solver). We recommend
  [Miniforge](https://github.com/conda-forge/miniforge). Check with `conda --version`.
  The setup script never changes your conda configuration or your `base` environment: it only creates
  the `pyslam` environment.
- **git**.
- **Linux with an NVIDIA GPU**: an up-to-date NVIDIA driver (`nvidia-smi` should work). You do *not*
  need to install the CUDA toolkit.
- **macOS**: the Xcode command-line tools (`xcode-select --install`).

## 2. Get the code

```bash
git clone https://github.com/sjulier/pyslam.git
cd pyslam
```

(The git submodules are only needed for some optional components; `scripts/install_extra.sh`
fetches the ones it needs.)

## 3. Create the Python environment

```bash
bash scripts/pyenv-conda-create.sh pyslam
conda activate pyslam
```

This creates the `pyslam` environment with OpenCV, PyTorch (matched to your GPU on Linux; with Apple
GPU support on macOS), Qt and the other dependencies. It checks at the end that the main packages
import, and stops with an error if anything failed.

From now on, always run `conda activate pyslam` before using pySLAM.

## 4. Build the core

Run these from the `pyslam` folder, in this order, with the environment activated:

```bash
bash scripts/install_gtsam.sh                       # GTSAM (10-15 min)
bash scripts/install_json_nlohmann.sh
bash scripts/install_qhull.sh
(cd thirdparty/orbslam2_features && ./build.sh)     # ORB features
(cd thirdparty/pangolin && ./build.sh)              # 3D viewer
(cd thirdparty/g2opy && ./build.sh)                 # g2o optimiser (~5 min)
(cd thirdparty/pydbow3 && ./build.sh)               # default loop-closure vocabulary (DBoW3)
(cd cpp && ./build.sh)                              # C++ utilities (~10 min)
./build_cpp_core.sh                                 # pySLAM C++ core (~3 min)
```

Optional loop-closure back-ends, only needed if you select them:

```bash
(cd thirdparty/pydbow2 && ./build.sh)               # DBoW2
(cd thirdparty/pyibow && ./build.sh)                # iBoW / OBIndex2
```

Each step stops with an error message if it fails. Fix the cause and re-run that step; re-running a
step that already succeeded is safe.

## 5. Check the installation

```bash
python scripts/check_pybind11_abi.py                # all native modules must report one ABI: "OK"
python -m pytest -q -p no:warnings test/gtsam/test_gtsam_factors_jacobians.py test/gtsam/test_optimize_pose.py test/gtsam/test_optimize_sim3.py
python -m pytest -q -p no:warnings test/g2o/test_optimize_pose.py test/g2o/test_optimize_sim3.py   # optimiser tests
python main_slam.py                                 # SLAM on the bundled KITTI 06 video
```

`main_slam.py` opens an image window, a 3D viewer and plots. Press `q` in a window to quit. The
**first run downloads the ORB vocabulary** (about 105 MB) before tracking starts. With
`python main_slam.py --headless` it runs without windows and prints the trajectory error at the end.
Frames are fed at the camera's rate, with or without windows; `--speed 2` plays twice as fast and
`--speed 0` as fast as possible (see [here](./TROUBLESHOOTING.md#non-determinism-and-run-to-run-variability)).

## 6. Optional components (extras)

Install the components you need for a given week on top of the core:

```bash
bash scripts/install_extra.sh --list
bash scripts/install_extra.sh features     # learned local features and matchers (SuperPoint, LightGlue, DISK, ALIKED, XFeat, LoFTR, ...)
bash scripts/install_extra.sh vpr          # learned place recognition for loop closing (NetVLAD, CosPlace, EigenPlaces, MegaLoc)
```

Each extra fetches only what it needs, **downloads the model weights at install time** (some take
several minutes, e.g. NetVLAD), and checks every component on the bundled test images. Install
extras before the lab session, not during it.

More groups of components (depth and stereo estimation, semantic segmentation and object detection,
3D representations such as Gaussian splatting and the DUSt3R/MASt3R family) will be added as extras
for the weeks that use them. Some of them need an NVIDIA GPU on Linux.

## Good to know

- **Results vary from run to run.** SLAM runs several threads, so two runs on the same video differ:
  on KITTI 06 the trajectory error (ATE) of monocular SLAM ranged from about 12 m to 30 m in our
  tests, mostly from scale drift before the loop closes. Compare methods over several runs. See
  [non-determinism](./TROUBLESHOOTING.md#non-determinism-and-run-to-run-variability).
- **Memory for the build.** Compiling GTSAM needs about 12 GB of free memory: most files need under
  4 GB, but a few files of its Python wrapper need up to 12 GB each. The build scripts run as many
  compiler jobs as the free memory allows (about 4 GB each); set `PYSLAM_BUILD_JOBS=1` to build one
  file at a time. On **Windows**, WSL2 gets only half of the computer's memory by default: if the
  build stops with `Killed signal terminated program cc1plus`, close other programs, or give WSL
  more memory (`memory=` and `swap=` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`).
- **Don't run `./clean.sh` casually.** It deletes the build folders, including the GTSAM build
  (`thirdparty/gtsam_local`), which then takes about 10 minutes to rebuild.
- **Error messages tell you what to do.** If a component is not installed or not built, pySLAM says
  which one and how to install or build it (for example
  `The native module 'pydbow2' is not built: run thirdparty/pydbow2/build.sh`).
- **Not available**: SURF (non-free); the TensorFlow-based features (DELF, LF-Net, ContextDesc,
  GeoDesc) are not part of the core.
- **Slow environment creation** ("Solving environment" for many minutes) usually means an old conda
  without the libmamba solver: update conda (`conda update -n base conda`) yourself, then re-run the
  script.

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md). When asking for help, include the full error message
and the output of `conda list -n pyslam | grep -E "^(python|numpy|pytorch|torch|libopencv|py-opencv) "`.
