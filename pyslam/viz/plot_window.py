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

# Sliding window of the 2D plots (Mplot2d, Qplot2d): the plots show the last `x_window` units of
# their x axis (frames, for the plots of main_slam.py) instead of the whole run. All the data is
# kept, so the window can be made wider again. 0 means the whole run.

import numpy as np

kXWindowLadder = (25, 50, 100, 250, 500, 1000, 2000, 5000)  # the windows the keys step through
kXWindowKeysHelp = "+ / - / 0"  # shown in the title of the plots: wider, narrower, the whole run


def sliding_window_limits(curves, x_window):
    """The axis limits that show the last `x_window` x units of `curves`, a list of (x, y) arrays or
    lists: (xmin, xmax, ymin, ymax), with y over the visible samples only, or None if there is
    nothing to show."""
    curves = [(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for x, y in curves]
    curves = [(x, y) for x, y in curves if x.size > 0 and x.size == y.size]
    if not curves:
        return None
    x_last = max(np.max(x) for x, _ in curves)
    x_first = max(min(np.min(x) for x, _ in curves), x_last - x_window)
    visible = [y[(x >= x_first) & np.isfinite(y)] for x, y in curves]
    visible = [y for y in visible if y.size > 0]
    if not visible:
        return None
    y_min = min(np.min(y) for y in visible)
    y_max = max(np.max(y) for y in visible)
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.5
    return x_first, max(x_last, x_first + 1), y_min - margin, y_max + margin


def x_window_from_key(x_window, key, x_span):
    """The new window after `key` is pressed in a plot, or None if the key is not one of these:
    '-' the next value of kXWindowLadder below the window, '+' or '=' the next one above, '0' the
    whole run (0). The keys step through fixed, round values, so the same keys give the same
    windows in every run. `x_span` is the extent of the data on the x axis: from the whole run '-'
    goes to the largest value below it, and '+' goes back to the whole run once the next value
    would show all the data anyway."""
    if key == "-":
        below = [w for w in kXWindowLadder if w < (x_window if x_window > 0 else x_span)]
        return below[-1] if below else kXWindowLadder[0]
    if key in ("+", "="):
        if x_window <= 0:
            return 0
        above = [w for w in kXWindowLadder if w > x_window]
        if not above or (x_span > 0 and above[0] >= x_span):
            return 0  # as wide as the data, or wider: the whole run
        return above[0]
    if key == "0":
        return 0
    return None


def x_window_str(x_window, unit=""):
    return f"the last {x_window:g} {unit}".rstrip() if x_window > 0 else "the whole run"


def x_window_title(title, x_window, unit):
    """The title of a plot with a sliding window: what it shows and the keys that change it, e.g.
    "# matches · last 1000 frames · keys + / - / 0". `unit` names the x axis (e.g. "frames"); a plot
    without a unit has no sliding window to show, and keeps its title."""
    if not unit:
        return title
    span = f"last {x_window:g} {unit}" if x_window > 0 else "whole run"
    return f"{title} \u00b7 {span} \u00b7 keys {kXWindowKeysHelp}"
