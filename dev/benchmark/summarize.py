"""The table of a pipeline_ab.py output folder: mean +- sd over the runs of each cell, then every run.

    pixi run python dev/benchmark/summarize.py OUT_DIR

Columns: ATE (cm), % of frames tracked, slam.track() time per frame (mean and p95, ms, after the
first 30 frames), whole frame time (ms), wall time (s), peak RSS, and macOS footprint or Linux PSS (GB), exit
codes, and the runs that timed out or stalled. A sd of '-' means one run.
"""

import collections
import glob
import json
import os
import re
import statistics
import sys


def mean_sd(values, scale=1.0, digits=2):
    values = [v * scale for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f}±{statistics.stdev(values):.{digits}f}"


def read_run(run_json):
    run = json.load(open(run_json))
    log = open(run_json[: -len(".run.json")] + ".log", errors="replace").read()
    found = re.findall(r"ATE rmse ([0-9.]+) over the tracked frames: (\d+) of (\d+) frames tracked", log)
    ate, tracked = (float(found[-1][0]), int(found[-1][1]) / int(found[-1][2])) if found else (None, None)
    playback = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in re.findall(r"Playback:[^\n]*", log)]
    row = dict(ate=ate, tracked=tracked, wall=run["wall_s"], rss=run["peak_rss_gb"],
               footprint=run.get("peak_footprint_gb") or run.get("peak_pss_gb"), rc=run["returncode"],
               bad=bool(run["timed_out"] or run.get("stalled_at_s")), playback=playback[-1] if playback else None)
    timing = run_json[: -len(".run.json")] + ".timing.json"
    if os.path.exists(timing):
        stages = json.load(open(timing))["after_warmup"].get("stages_ms", {})
        row.update(track=stages.get("tracking", {}).get("mean"), track_p95=stages.get("tracking", {}).get("p95"),
                   frame=stages.get("total", {}).get("mean"))
    return row


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cells = collections.defaultdict(list)
    for run_json in sorted(glob.glob(os.path.join(sys.argv[1], "*.run.json")),
                           key=lambda p: (p.rsplit("_run", 1)[0], int(p.rsplit("_run", 1)[1].split(".")[0]))):
        cells[os.path.basename(run_json).rsplit("_run", 1)[0]].append(read_run(run_json))
    print(f"{'cell':28} n  {'ATE cm':12} {'tracked %':11} {'track ms':12} {'p95 ms':12} {'frame ms':12} "
          f"{'wall s':12} {'RSS GB':10} {'footprint/PSS':14} exit codes / timed out or stalled")
    for cell, rows in sorted(cells.items()):
        get = lambda key: [r.get(key) for r in rows]  # noqa: E731
        print(f"{cell:28} {len(rows)}  {mean_sd(get('ate'), 100):12} {mean_sd(get('tracked'), 100, 1):11} "
              f"{mean_sd(get('track')):12} {mean_sd(get('track_p95')):12} {mean_sd(get('frame')):12} "
              f"{mean_sd(get('wall'), 1, 1):12} {mean_sd(get('rss')):10} {mean_sd(get('footprint')):14} "
              f"{get('rc')} / {sum(get('bad'))}")
    print("\nevery run: % tracked (ATE m)")
    for cell, rows in sorted(cells.items()):
        print(f"  {cell:28} " + "  ".join(
            "no result" if r["tracked"] is None else f"{r['tracked'] * 100:.1f} ({r['ate']:.2f})" for r in rows))
    playback = [(cell, r["playback"]) for cell, rows in sorted(cells.items()) for r in rows if r["playback"]]
    if playback:
        print("\nplayback (the slow-downs are those of --throttle; without it there are none):")
        for cell, line in playback:
            print(f"  {cell:28} {line}")


if __name__ == "__main__":
    main()
