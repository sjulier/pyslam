"""The sliding window of the plots (pyslam/viz/plot_window.py): the keys step through a fixed ladder
of round values. Run: python -m pytest -q test/viz/test_plot_window.py"""

import importlib.util
import os

import numpy as np

# loaded from its file: the module only needs numpy (no pySLAM native modules)
_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyslam", "viz", "plot_window.py")
_spec = importlib.util.spec_from_file_location("plot_window", _path)
plot_window = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_window)
x_window_from_key = plot_window.x_window_from_key


def press(x_window, keys, x_span):
    windows = []
    for key in keys:
        x_window = x_window_from_key(x_window, key, x_span)
        windows.append(x_window)
    return windows


def test_minus_from_the_whole_run_gives_round_windows():
    # from the whole run: the largest value of the ladder below the frames shown so far, whatever
    # their exact number (it gave 221.5, 227, 271.5 when it halved the extent of the data)
    for x_span in (443, 454, 499.9):
        assert press(0, "---", x_span) == [250, 100, 50]
    assert press(0, "---", 543.7) == [500, 250, 100]


def test_minus_then_plus_from_the_default():
    assert press(1000, "-+", x_span=1100) == [500, 1000]
    assert press(1000, "----", x_span=1100) == [500, 250, 100, 50]


def test_plus_goes_back_to_the_whole_run_when_the_window_covers_the_data():
    assert press(500, "+", x_span=1100) == [1000]
    assert press(1000, "+", x_span=1100) == [0]  # 2000 would show all the data anyway
    assert press(500, "+", x_span=800) == [0]
    assert press(5000, "+", x_span=20000) == [0]  # above the top of the ladder
    assert press(0, "+", x_span=1100) == [0]


def test_the_ends_of_the_ladder():
    assert press(25, "-", x_span=1100) == [25]
    assert press(0, "-", x_span=10) == [25]  # hardly any data yet
    assert press(0, "-", x_span=20000) == [5000]


def test_a_window_off_the_ladder_snaps_to_it_at_the_first_key():
    assert press(300, "-", x_span=1100) == [250]
    assert press(300, "+", x_span=1100) == [500]
    assert press(62.5, "-", x_span=1100) == [50]


def test_zero_and_other_keys():
    assert press(250, "0", x_span=1100) == [0]
    assert x_window_from_key(250, "=", 1100) == 500
    assert x_window_from_key(250, "q", 1100) is None
    assert x_window_from_key(250, None, 1100) is None


def test_windows_are_integers_in_the_title_and_the_message():
    for window in press(0, "---", 543.7) + press(1000, "-+", 1100):
        assert window == int(window)
    assert plot_window.x_window_title("# matches", 500, "frames") == "# matches · last 500 frames · keys + / - / 0"
    assert plot_window.x_window_title("# matches", 0, "frames") == "# matches · whole run · keys + / - / 0"
    assert plot_window.x_window_str(500, "frames") == "the last 500 frames"
    assert plot_window.x_window_str(0, "frames") == "the whole run"


def test_sliding_window_limits():
    x = np.arange(2000)
    y = np.where(x < 1500, 100.0, x - 1500.0)
    x_min, x_max, y_min, y_max = plot_window.sliding_window_limits([(x, y)], 250)
    assert (x_min, x_max) == (1749, 1999)
    assert y_min < 249 and y_max > 499 and y_max < 600  # scaled to the visible samples
    assert plot_window.sliding_window_limits([([], [])], 250) is None
