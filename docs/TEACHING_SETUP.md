# pySLAM setup for the course

This page installs pySLAM's **default level** for the course with [pixi](https://pixi.sh): visual
odometry, full SLAM with classical and learned features, loop closing (also with learned place
recognition), g2o/GTSAM optimisation, dense reconstruction and the viewers. Later weeks add more
levels (depth prediction, semantic segmentation, 3D reconstruction); they are described in
[PIXI.md](./PIXI.md).

pixi installs everything pySLAM needs into the repository folder (`.pixi/`), from a lock file, so
everybody gets the same versions. It does not touch conda or the rest of your system, and you do not
need to install CUDA or a compiler.

> **Do the installation before the lab.** It downloads about 9 GB with an NVIDIA GPU (the environment
> and the model weights; about 4 GB without one) and compiles pySLAM's C++ modules, which takes from
> 15 minutes to over an hour.

## Supported systems

| system | status |
|---|---|
| **Linux** (x86-64), with an NVIDIA GPU | tested, from older GPUs (Pascal, e.g. Titan Xp) to the RTX 50 series |
| **Linux** (x86-64), without an NVIDIA GPU | tested: add `-e default-cpu` to every `pixi run` command below |
| **macOS** (Apple silicon, macOS 14 or later) | tested. Learned features use the Apple GPU |
| **Windows** | via **WSL2** only (Ubuntu inside Windows), then as Linux: being tested |

You need about **30 GB of free disk space** (less without an NVIDIA GPU) and an internet connection.
If pySLAM's C++ modules have to be compiled on your machine (see step 3), that also needs about
**12 GB of free memory** (see [Good to know](#good-to-know)).

## 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new terminal, and check with `pixi --version` (0.81 or later). You also need `git`.

## 2. Get the code

```bash
git clone --depth 1 https://github.com/sjulier/pyslam.git
cd pyslam
```

`--depth 1` downloads only the current version, without the project's history: about 0.6 GB
instead of 1.1 GB. `git pull` still updates it later.

Run all the commands below from this folder. If you already have an older pySLAM checkout (for
example one set up with conda), make a fresh clone instead of updating it.

## 3. Build

```bash
pixi run build      # the environment, pySLAM's C++ modules and the ORB vocabulary
pixi run check      # the C++ modules load and the optimiser tests pass
pixi run models     # learned features and place recognition: code and model weights, each one checked
```

- The first `build` downloads the environment (about 6 GB with an NVIDIA GPU): this takes 20 minutes
  or more, mostly **without any output**, which is normal.
- `build` then installs pySLAM's C++ modules ready-made when there is a prebuilt copy for your system
  (`[native bundle] installed the prebuilt native modules`, a 27 MB download), and otherwise compiles
  them, which takes from 15 minutes to over an hour.
- `build` ends by checking that all native modules share one pybind11 ABI (`OK: ... module(s) share ...`).
- `check` ends with the GTSAM and g2o tests passing.
- `models` prints one line per component, `OK` with the device it ran on (`cuda`, `mps` or `cpu`),
  and summaries such as `19/19 components of 'features' OK` and `5/5 components of 'vpr' OK`.

If a command stops (for example a download breaks), **run the same command again**: what is already
done is skipped, and interrupted downloads continue where they stopped.

## 4. Run SLAM

```bash
pixi run slam              # main_slam.py on the bundled KITTI 06 video
pixi run slam --headless   # without windows; prints the trajectory error (ATE) at the end
```

`pixi run slam` opens an image window, a 3D viewer, plots and loop closing's debug windows (the
keyframe similarity matrix and the loop candidates). Press `q` or `Esc` in the image window to quit.

The other main scripts have tasks too (`pixi run vo`, `pixi run feature-matching`,
`pixi run map-viewer`, ...); `pixi task list` shows them all, and [PIXI.md](./PIXI.md#run) explains
how to work in a pixi shell instead.

## Good to know

- **Results vary from run to run.** SLAM runs several threads, so two runs on the same video differ.
  On KITTI 06 the trajectory error (ATE) of monocular SLAM was about 13-18 m in our tests. Compare
  methods over several runs. See
  [non-determinism](./TROUBLESHOOTING.md#non-determinism-and-run-to-run-variability).
- **SLAM runs at the camera's frame rate**, also with `--headless`. `--speed 2` plays twice as fast and
  `--speed 0` as fast as possible; faster than the camera, tracking is sometimes lost at the turns.
  Use the default speed when you compare results. If tracking is lost at the same places in every
  run, your machine may be too slow for the camera's frame rate: add `--throttle`, which slows the
  playback down when tracking gets weak (it prints `Playback speed: ...`).
- **Memory for the build.** Compiling GTSAM needs about 12 GB of free memory: a few files of its Python
  wrapper need up to 12 GB each. The build runs as many compiler jobs as the free memory allows; set
  `PYSLAM_BUILD_JOBS=1` to build one file at a time. On **Windows**, WSL2 gets only half of the
  computer's memory by default: if the build stops with `Killed signal terminated program cc1plus`,
  close other programs, or give WSL more memory (`memory=` and `swap=` in
  `%UserProfile%\.wslconfig`, then `wsl --shutdown`) and run `pixi run build` again. Single files of
  GTSAM and of pySLAM's C++ code need up to 12 GB, so on a laptop with 16 GB of memory (WSL2 then gets about 8 GB) give
  WSL more memory or swap **before** a build from source. None of this applies when `build`
  installs the prebuilt modules.
- **Compiler warnings** such as `-Wmaybe-uninitialized` from Eigen during a build from source are
  harmless (they appear on CPUs with AVX-512).
- **Do not `pip install` into the environment** and do not run `./clean.sh` casually: it deletes the
  build folders, and rebuilding takes up to an hour.
- **If you change pySLAM's C++ code, rebuild with `pixi run build`**, never with a module's own
  `build.sh` outside pixi (the scripts stop with an error there). After a change to the C++ code,
  `build` compiles **all** the C++ modules from source, GTSAM included: that takes from 30 minutes to
  over an hour and needs the memory described above. If a build fails or is interrupted, run
  `./clean.sh` and then `pixi run build`.
- **Error messages tell you what to do.** If a component is not installed or not built, pySLAM says
  which one and which command installs or builds it.
- **Not available**: SURF (non-free). The TensorFlow-based features (DELF, LF-Net, ContextDesc,
  GeoDesc) are not in the default level.

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) and the
[good-to-know section of PIXI.md](./PIXI.md#good-to-know). When asking for help, include the full
error message, the command you ran, and the output of `pixi info`.
