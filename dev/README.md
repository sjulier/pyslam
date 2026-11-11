# dev/: tools for developing pySLAM

Everything in `dev/` is for pySLAM's own development. None of it is for the course, so it is kept
out of the course branch `comp0227-2026` (below).

## Keeping dev/ out of the course branch

`comp0227-2026` gets main's fixes by merging main into it. When you do, take `dev/` out in the same
merge:

```bash
git switch comp0227-2026
git merge --no-commit main                 # stops before committing (or on a conflict)
git rm -r -q --ignore-unmatch dev          # never in the course branch
git commit                                 # e.g. "Merge main <sha> into comp0227-2026 for <tag>: ..."
```

This works in every case:
- **The first merge after dev/ appears:** the merge brings `dev/` in, and `git rm` takes it out again.
- **Later merges, main hasn't touched dev/:** the earlier deletion stands, and `git rm` finds nothing.
- **Later merges, main changed files in dev/:** git stops with a modify/delete conflict on those
  files only, and `git rm` resolves it.

Check before pushing: `git ls-tree -d comp0227-2026 dev` prints nothing.

## benchmark/: Python vs C++ SLAM pipeline

These compare `PYSLAM_USE_CPP_PIPELINE=0` (the Python pipeline) with `=1` (the C++ pipeline) on the
same build. They report:
- per-frame tracking time;
- run time;
- peak memory;
- accuracy;
- frames tracked;
- hangs.

```bash
# TUM fr1_xyz mono and RGB-D (RGB-D needs associations_rgbd.txt in the sequence folder:
# python scripts/associate.py rgb.txt depth.txt > associations_rgbd.txt), 3 runs per cell
pixi run python dev/benchmark/pipeline_ab.py runs/tum --runs 3 \
    --cells 0:0:tum-mono 1:0:tum-mono 0:0:tum-rgbd 1:0:tum-rgbd \
    --tum ~/datasets/tum/rgbd_dataset_freiburg1_xyz

# the dataset of config.yaml (the KITTI 06 video), flat out with the playback throttle
pixi run python dev/benchmark/pipeline_ab.py runs/kitti --runs 5 --cells 0:0:config 1:0:config -- --throttle

pixi run python dev/benchmark/summarize.py runs/tum
```

- `pipeline_ab.py` runs `main_slam.py --headless` with `--timing-report`, and with `--no-throttle`
  unless `--throttle` is given after `--`. The cells are interleaved round by round.
- `run_monitored.py` runs one command and records:
  - its wall time and exit code;
  - peak RSS, and on macOS peak phys_footprint, summed over the process tree once a second;
  - stalls: if the log is silent for `--stall` seconds (default 120), it records every thread's
    stack (macOS `sample` for native stacks; SIGUSR1 for main_slam's Python stacks) before
    killing the run.
- `summarize.py` prints, for each cell, the mean ± sd over its runs, then every run's % tracked
  and ATE, and the "Playback:" line of throttled runs.
- The tracking time is `slam.track()` alone: main_slam's timing report has a "tracking" stage. It
  skips the first 30 frames.

### Measuring on a Mac: lessons from Nov 2026
- **Turn Low Power Mode off** (`pmset -g | grep lowpower`): it slows the CPU, and the SLAM
  timings with it. pySLAM's `doctor` warns about it.
- **Sleep looks like a hang.** Idle sleep or hibernation suspends a run, and a timeout then calls
  it a freeze. pipeline_ab.py wraps each run in `caffeinate -i -s`, and run_monitored.py reports
  sleep as `slept_s`. Before calling a timeout a hang, check `pmset -g log | grep -E "Sleep|Wake"`.
- **Compare memory by footprint, not RSS.** macOS RSS swung ±0.3 GB between sessions on the same
  build, as memory pressure compressed pages. phys_footprint stayed within ±0.05 GB.
- **Run nothing else meanwhile:** builds and other benchmarks double the sd.
