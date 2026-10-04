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

import pyqtgraph as pg
import pyqtgraph.opengl as gl

try:
    # Prefer the compatibility shim so any installed Qt binding works.
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except Exception:
    # Fallback (only used if the shim isn't available)
    from PyQt5 import QtCore, QtGui, QtWidgets

# Unified names
QLabel = QtWidgets.QLabel
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

Qt = QtCore.Qt
QFont = QtGui.QFont
QColor = QtGui.QColor
QVector3D = QtGui.QVector3D


import numpy as np
import time
import math
import re
import os
import logging

import sys

sys.path.append("../")
from pyslam.config import Config

# import multiprocessing as mp
import torch.multiprocessing as mp
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.logging import Logging
from pyslam.utilities.system import locally_configure_qt_environment

from pyslam.config_parameters import Parameters
from pyslam.viz.plot_window import sliding_window_limits, x_window_from_key, x_window_str

kVerbose = False
kDebugAndPrintToFile = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
# kLogsFolder = kRootFolder + '/logs'


if kVerbose and kDebugAndPrintToFile:
    # redirect the prints of local mapping to the file logs/local_mapping.log
    # you can watch the output in separate shell by running:
    # $ tail -f logs/mplot_thread.log
    logging_file = Parameters.kLogsFolder + "/qplot_thread.log"
    local_logger = Logging.setup_file_logger(
        "qplot_thread_logger", logging_file, formatter=Logging.simple_log_formatter
    )

    def print(*args, **kwargs):
        message = " ".join(
            str(arg) for arg in args
        )  # Convert all arguments to strings and join with spaces
        return local_logger.info(message, **kwargs)


class FigureNum:
    figure_num = 0

    @staticmethod
    def getFigureNum():
        FigureNum.figure_num += 1
        return FigureNum.figure_num


# empty a queue before exiting from the consumer thread/process for safety
def empty_queue(queue):
    while not queue.empty():
        try:
            queue.get(timeout=0.001)
        except:
            pass


# global lock for drawing with matplotlib
class SharedSingletonLock:
    _instance = None  # Placeholder for singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedSingletonLock, cls).__new__(cls)
            cls._instance.lock = mp.Lock()  # Create the lock only once
        return cls._instance

    @property
    def get_lock(self):
        return self.lock


# Qplot2d class for dynamic 2D plotting using pyqtgraph
# NOTE: This is a good tutorial https://www.pythonguis.com/tutorials/plotting-pyqtgraph/
# pyqtgraph's single-letter colours that have poor contrast on the white plot background ('c', 'g'
# and 'y' are (0,255,255), (0,255,0) and (255,255,0)): drawn with darker shades instead.
_COLORS_ON_WHITE = {"c": (0, 150, 190), "g": (0, 160, 0), "y": (200, 150, 0)}


# Labels are written in matplotlib's mathtext (e.g. "# $KF_{ref}$ tracked pts", rendered by Mplot2d);
# pyqtgraph shows them as HTML, so the $...$ parts are converted: X_{ab} -> X<sub>ab</sub>,
# X^{ab} -> X<sup>ab</sup>, and the usual Greek letters.
_MATHTEXT_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "theta": "θ",
    "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
    "chi": "χ", "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Phi": "Φ",
}


def mathtext_to_html(text):
    def convert(expr):
        expr = re.sub(r"\\([A-Za-z]+)", lambda m: _MATHTEXT_SYMBOLS.get(m.group(1), m.group(1)), expr)
        expr = re.sub(r"_\{([^}]*)\}|_(\w)", lambda m: f"<sub>{m.group(1) or m.group(2)}</sub>", expr)
        expr = re.sub(r"\^\{([^}]*)\}|\^(\w)", lambda m: f"<sup>{m.group(1) or m.group(2)}</sup>", expr)
        return expr

    if not isinstance(text, str) or "$" not in text:
        return text
    return re.sub(r"\$([^$]*)\$", lambda m: convert(m.group(1)), text)


class Qplot2d:
    def __init__(self, xlabel: str = "", ylabel: str = "", title: str = "", x_window: float = 0):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        # show the last x_window units of the x axis (0: the whole run); see plot_window.py
        self.x_window = x_window

        self.data = None
        self.got_data = False
        self.handle_map = {}
        self.handle_data_map = {}

        self.axis_computed = False
        self.xlim = [float("inf"), float("-inf")]
        self.ylim = [float("inf"), float("-inf")]
        self.xmin = float("inf")
        self.xmax = float("-inf")
        self.ymin = float("inf")
        self.ymax = float("-inf")

        self.screen_width = None
        self.screen_height = None

        self.key = mp.Value("i", 0)
        self.is_running = mp.Value("i", 1)

        # Use manager to handle queues and avoid pickling problems
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        self.figure_num = mp.Value("i", int(FigureNum.getFigureNum()))
        # print(f'Qplot2d: starting the process on figure: {self.figure_num.value}')

        self.lock = SharedSingletonLock().get_lock

        self.win = None
        self.initialized = False  # New flag to track figure initialization

        args = (
            self.figure_num,
            self.queue,
            self.lock,
            self.key,
            self.is_running,
            self.key_queue,
        )
        self.process = mp.Process(target=self.run, args=args)
        self.process.start()

    def quit(self):
        print(f'Qplot2d "{self.title}" closing...')
        self.is_running.value = 0
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            print(
                f'Warning: Qplot2d "{self.title}" process did not terminate in time, forced kill.'
            )
            self.process.terminate()
        print(f'Qplot2d "{self.title}" closed')

    def get_screen_dimensions(self):
        def_width = 900
        def_height = 500
        try:
            # Try the modern Qt6 approach first
            screen = self.app.primaryScreen()
            if screen is not None:
                screen_rect = screen.geometry()
                self.screen_width = screen_rect.width()
                self.screen_height = screen_rect.height()
            else:
                # Fallback to default values
                self.screen_width = def_width
                self.screen_height = def_height
        except AttributeError:
            # Fallback for older Qt versions or if primaryScreen doesn't exist
            try:
                desktop = self.app.desktop()
                screen_rect = desktop.screenGeometry()
                self.screen_width = screen_rect.width()
                self.screen_height = screen_rect.height()
            except AttributeError:
                # Final fallback to default values
                self.screen_width = def_width
                self.screen_height = def_height
        return self.screen_width, self.screen_height

    def init(self, figure_num, lock):
        lock.acquire()
        locally_configure_qt_environment()
        self.app = pg.mkQApp()  # Create a PyQtGraph application

        # Fetch desktop dimensions
        self.screen_width, self.screen_height = self.get_screen_dimensions()

        self.win = pg.PlotWidget(title=self.title)  # Create a plot widget
        # Line width in logical pixels, scaled by the screen's device-pixel ratio: pyqtgraph's default
        # 1-px pens look hair-thin on high-DPI (e.g. macOS Retina, 2x) screens. 1 on standard screens.
        self.line_width = 1.0
        try:
            screen = self.win.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                self.line_width = max(1.0, float(screen.devicePixelRatio()))
        except Exception:
            pass
        # Legend in a light, semi-transparent box with dark text, anchored in the top-left corner of the
        # plot area and drawn above the curves (which show through it, muted). Curves get their legend
        # entry from their name (see drawer_refresh). The legend can be dragged.
        self.legend = self.win.addLegend(
            offset=(10, 10),
            brush=pg.mkBrush(240, 240, 240, 200),
            pen=pg.mkPen(160, 160, 160),
            labelTextColor=(20, 20, 20),
        )
        self.legend.setZValue(1000)
        self.win.setLabel("left", mathtext_to_html(self.ylabel))  # Set the y-axis label
        self.win.setLabel("bottom", mathtext_to_html(self.xlabel))  # Set the x-axis label

        self.win.showGrid(x=True, y=True, alpha=0.5)  # Show grid

        # # Define maximum size for the window (e.g., half the screen size)
        # max_width = int(self.screen_width * 3.0/4)
        # max_height = int(self.screen_height * 3.0/4)

        # # Set maximum size constraints
        # self.win.setMaximumSize(max_width, max_height)

        offset = (figure_num - 1) * 100
        current_position = self.win.frameGeometry().topLeft()
        self.win.setGeometry(
            current_position.x() + offset,
            current_position.y() + offset,
            self.win.width(),
            self.win.height(),
        )  # Set a default size

        # Define the grid color and line width
        # grid_color = pg.mkColor(0, 0, 255)  # Red grid lines
        # grid_pen = pg.mkPen(color=grid_color, width=0.3)  # Grid pen with width 2

        # Set the grid with the custom color and style
        self.win.showGrid(x=True, y=True, alpha=0.6)  # Show grid and adjust alpha transparency

        # self.win.getAxis('left').setPen(grid_pen)  # Set the grid color for the Y-axis
        # self.win.getAxis('bottom').setPen(grid_pen)  # Set the grid color for the X-axis
        # self.win.getAxis('left').setGrid(100)    # Increase resolution of Y grid
        # self.win.getAxis('bottom').setGrid(100)  # Increase resolution of X grid

        self.win.setYRange(-2, 2)  # Set Y-axis range (can be dynamic)
        if self.title != "":
            self.win.setWindowTitle(self.title)
        self.win.setBackground("w")  # 'w' stands for white

        self.win.keyPressEvent = self.on_key_press
        self.win.keyReleaseEvent = self.on_key_release
        self.win.closeEvent = self.on_close

        self.win.setInteractive(True)
        # self.win.setAspectLocked(True)

        # Display the window
        self.win.show()
        lock.release()
        self.initialized = True  # Set the flag to True after initialization

    def run(self, figure_num, queue, lock, key, is_running, key_queue):
        if kVerbose:
            print(f'Qplot2d "{self.title}": starting run on figure ', figure_num.value)
        self.key = key
        self.key_queue_thread = key_queue
        # self.init(figure_num.value, lock)
        while is_running.value == 1:
            self.drawer_refresh(queue, lock, figure_num)
            if True:  # Adjust condition if needed
                time.sleep(0.02)  # Slow the loop down for real-time effect

        empty_queue(queue)  # Empty the queue before exiting
        print(f'{mp.current_process().name} - Qplot2d "{self.title}": closing plot')
        if self.win:
            self.win.close()  # Close the PyQtGraph window
        print(f"Qplot2d: run: closed")

    def drawer_refresh(self, queue, lock, figure_num):
        while not queue.empty():
            self.got_data = True
            self.data = queue.get()
            xy_signal, name, color, marker, linestyle, append = self.data
            if isinstance(color, str):
                color = _COLORS_ON_WHITE.get(color, color)

            # Initialize figure upon receiving the first data
            if not self.initialized:
                self.init(figure_num.value, lock)

            if name in self.handle_map:
                handle = self.handle_map[name]
                if append:
                    x_data, y_data = self.handle_data_map[name]
                    x_data.append(xy_signal[0])
                    y_data.append(xy_signal[1])
                    self.updateMinMax(xy_signal[0], xy_signal[1])
                    self.handle_data_map[name] = (x_data, y_data)
                    handle.setData(x=x_data, y=y_data)
                else:
                    handle_data = (xy_signal[0], xy_signal[1])
                    self.updateMinMax(xy_signal[0], xy_signal[1])
                    self.handle_data_map[name] = handle_data
                    handle.setData(x=xy_signal[0], y=xy_signal[1])
            else:
                if append:
                    handle_data = ([xy_signal[0]], [xy_signal[1]])  # append the first sample
                    kwargs = {"x": [xy_signal[0]], "y": [xy_signal[1]], "pen": pg.mkPen(color, width=self.line_width), "name": mathtext_to_html(name)}
                else:
                    handle_data = (xy_signal[0], xy_signal[1])
                    kwargs = {"x": xy_signal[0], "y": xy_signal[1], "pen": pg.mkPen(color, width=self.line_width), "name": mathtext_to_html(name)}
                    self.updateMinMax(xy_signal[0], xy_signal[1])
                if linestyle != "":
                    kwargs["style"] = linestyle
                if marker != "":
                    kwargs["symbol"] = marker
                handle = self.win.plot(**kwargs)  # also adds the legend entry (from kwargs["name"])
                self.handle_map[name] = handle
                self.handle_data_map[name] = handle_data

        if self.got_data is True:
            self.plot_refresh(lock)

    def on_key_press(self, event):
        key = event.key()
        self.update_x_window(event.text())
        try:
            self.key.value = ord(event.text())  # Convert to int
            self.key_queue_thread.put(self.key.value)
        except Exception as e:
            print(f"Qplot2d: on_key_press: encountered exception: {e}")
        # if key == Qt.Key_Up:
        #     print('Up arrow key pressed')
        # elif key == Qt.Key_Down:
        #     print('Down arrow key pressed')
        # elif key == Qt.Key_Left:
        #     print('Left arrow key pressed')
        # elif key == Qt.Key_Right:
        #     print('Right arrow key pressed')
        # else:
        #     print(f'Key pressed: {chr(key)}')  # Print the character of the key

    def on_key_release(self, event):
        # self.key.value = 0  # Reset to no key symbol
        pass

    def on_close(self, event):
        self.is_running.value = 0
        print(f'{mp.current_process().name} - Qplot2d "{self.title}" closed figure')

    def get_key(self):
        if not self.key_queue.empty():
            return chr(self.key_queue.get())
        else:
            return ""

    def update_x_window(self, key):
        """Keys in the plot window: '+' and '-' change the sliding window, '0' shows the whole run."""
        x_span = self.xlim[1] - self.xlim[0] if self.axis_computed else 0
        x_window = x_window_from_key(self.x_window, key, x_span)
        if x_window is None:
            return
        self.x_window = x_window
        print(f'Qplot2d "{self.title}": showing {x_window_str(x_window, self.xlabel)}')
        self.got_data = True  # redraw

    def setGridAxis(self, xlim=None, ylim=None):
        xlim = self.xlim if xlim is None else xlim
        ylim = self.ylim if ylim is None else ylim
        deltax = xlim[1] - xlim[0]
        deltay = ylim[1] - ylim[0]
        # self.win.getAxis('left').setTickSpacing(10, 100)  # Y-axis: major and minor ticks

        # def get_values(delta):
        #     # xminor = (delta/10)
        #     # xmajor = max(delta,10)
        #     # if delta > 10:
        #     log = int(math.log10(delta))
        #     xminor = 10**(log)/5
        #     xmajor = 10**(log)
        #     return xminor, xmajor

        def get_values(delta):
            if delta <= 0:
                return 1, 5  # Default spacing if delta is invalid
            # Calculate major tick spacing to ensure at least 5 ticks
            major = delta / 5
            # Round to a nice value for readability
            magnitude = 10 ** int(math.floor(math.log10(major)))
            normalized = major / magnitude
            if normalized < 1.5:
                major = 1 * magnitude
            elif normalized < 3:
                major = 2 * magnitude
            elif normalized < 7:
                major = 5 * magnitude
            else:
                major = 10 * magnitude
            # Minor ticks are 1/5th of major ticks
            minor = major / 5
            return minor, major

        if deltax > 0 and deltay > 0:
            xminor, xmajor = get_values(deltax)
            yminor, ymajor = get_values(deltay)

            self.win.getAxis("left").setTickSpacing(ymajor, yminor)  # Y-axis: major and minor ticks
            self.win.getAxis("bottom").setTickSpacing(
                xmajor, xminor
            )  # X-axis: major and minor ticks

    def setAxis(self):
        if self.axis_computed and self.x_window > 0:
            limits = sliding_window_limits(list(self.handle_data_map.values()), self.x_window)
            if limits is not None:
                self.win.setXRange(limits[0], limits[1])
                self.win.setYRange(limits[2], limits[3])
                self.setGridAxis(limits[0:2], limits[2:4])
                return
        if self.axis_computed:
            if self.xlim != [float("inf"), float("-inf")]:
                self.win.setXRange(self.xlim[0], self.xlim[1])
            if self.ylim != [float("inf"), float("-inf")]:
                self.win.setYRange(self.ylim[0], self.ylim[1])

            self.setGridAxis()

    def updateMinMax(self, x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            self.xmin = np.amin(x)
            self.xmax = np.amax(x)
            self.ymin = np.amin(y)
            self.ymax = np.amax(y)

            self.xlim = [self.xmin, self.xmax]
            self.ylim = [self.ymin, self.ymax]
        else:
            # incremental update
            if self.xmin == float("inf"):
                self.xmin = x
            if self.xmax == float("-inf"):
                self.xmax = x
            if self.ymin == float("inf"):
                self.ymin = y
            if self.ymax == float("-inf"):
                self.ymax = y

            self.xmin = min(self.xmin, x)
            self.xmax = max(self.xmax, x)
            self.ymin = min(self.ymin, y)
            self.ymax = max(self.ymax, y)

            # update mins and maxs
            if True:
                if self.xmax > self.xlim[1]:
                    self.xlim[1] = self.xmax
                if self.xmin < self.xlim[0]:
                    self.xlim[0] = self.xmin
                # cx = 0.5*(self.xlim[1]+self.xlim[0])

                if self.ymax > self.ylim[1]:
                    self.ylim[1] = self.ymax
                if self.ymin < self.ylim[0]:
                    self.ylim[0] = self.ymin
                # cy = 0.5*(self.ylim[1]+self.ylim[0])

                # deltay = 0.5*(self.ylim[1]-self.ylim[0])
                # self.ylim = [cy-deltay,cy+deltay]

            # make axis actually squared
            if False:
                cx = 0.5 * (xmax + xmin)
                cy = 0.5 * (ymax + ymin)
                smin = min(xmin, ymin)
                smax = max(xmax, ymax)
                delta = 0.9 * (smax - smin)
                self.xlim = [cx - delta, cx + delta]
                self.ylim = [cy - delta, cy + delta]

        self.axis_computed = True

    # NOTE: Qt's line styles: Qt.SolidLine, Qt.DotLine, Qt.DashDotLine, and Qt.DashDotDotLine
    def draw(self, xy_signal, name, color="r", marker="", linestyle="", append=True):
        if self.queue is None:
            return
        self.queue.put((xy_signal, name, color, marker, linestyle, append))

    def plot_refresh(self, lock):
        lock.acquire()
        if self.is_running.value == 1:
            self.setAxis()
        self.app.processEvents()  # Process events to update the plot
        lock.release()


# ===================================================


# Custom GLViewWidget to enable mouse-based translation
class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.pan_speed = 0.05  # Adjust to control panning sensitivity
        self.last_mouse_pos = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.RightButton:  # Left mouse button for panning
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() == Qt.RightButton and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()

            dx = delta.x() * self.pan_speed
            dy = -delta.y() * self.pan_speed
            dz = 0

            # Translate the view by adjusting the center
            center = self.opts["center"]
            self.opts["center"] = QVector3D(center[0] + dx, center[1] + dy, center[2] + dz)
            self.update()


class LegendOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.legend_items = []
        self.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setFont(QFont("Arial", 10))
        self.setTextFormat(Qt.RichText)

        self.setText("")

        # Position the legend box at the top-left corner
        # self.move(10, 10)  # 10px margin from top-left corner

        self.update_legend([])

    def update_legend(self, items):
        new_row_height = 30
        """Update the legend with a list of (label, color)."""
        self.legend_items = items
        legend_html = "<b>Legend:</b><br>"
        for label, color in items:
            # Handle QColor and tuples consistently
            if isinstance(color, QColor):
                hex_color = color.name()  # QColor to hex
            else:
                # Assuming color is an RGBA tuple, convert to QColor first
                q_color = QColor(*[int(c * 255) for c in color[:3]])
                hex_color = q_color.name()
            legend_html += f'<span style="color:{hex_color}">■</span> {label}<br>'
        self.setText(legend_html)
        self.setMaximumHeight(self.height() + new_row_height)


class Qplot3d:
    def __init__(
        self,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        auto_view: bool = True,
    ):
        self.xlabel = xlabel if xlabel != "" else "X Axis"
        self.ylabel = ylabel if ylabel != "" else "Y Axis"
        self.zlabel = zlabel if zlabel != "" else "Z Axis"
        self.title = title
        self.auto_view = auto_view

        self.data = None
        self.got_data = False
        self.handle_map = {}
        self.handle_data_map = {}
        self.legend_items = []  # Store legend items for the 2D overlay

        self.axis_computed = False
        self.limits = {
            "x": [float("inf"), float("-inf")],
            "y": [float("inf"), float("-inf")],
            "z": [float("inf"), float("-inf")],
        }

        self.key = mp.Value("i", 0)
        self.is_running = mp.Value("i", 1)

        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        self.figure_num = mp.Value("i", int(FigureNum.getFigureNum()))
        print(f"Qplot3d: starting process on figure: {self.figure_num.value}")

        self.lock = mp.Lock()

        args = (self.figure_num, self.queue, self.lock, self.key, self.is_running, self.key_queue)
        self.process = mp.Process(target=self.run, args=args)
        self.process.start()

    def quit(self):
        print(f'Qplot3d "{self.title}" closing...')
        self.is_running.value = 0
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            print(
                f'Warning: Qplot3d "{self.title}" process did not terminate in time, forced kill.'
            )
            self.process.terminate()
        print(f'Qplot3d "{self.title}" closed')

    def init(self, figure_num, lock):
        lock.acquire()
        locally_configure_qt_environment()
        self.app = pg.mkQApp()  # Create a PyQtGraph application
        # self.win = gl.GLViewWidget()
        self.win = CustomGLViewWidget()
        self.win.setWindowTitle(self.title)
        self.win.setBackgroundColor("w")

        self.main_widget = QWidget()  # Create a main widget
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.addWidget(self.win)

        self.overlay = LegendOverlay()  # Add legend overlay
        self.main_layout.addWidget(self.overlay)

        default_width = 800
        default_height = 600
        self.main_widget.resize(default_width, default_height)
        self.overlay.setMaximumHeight(int(default_height / 50))

        self.create_axis()

        self.main_widget.keyPressEvent = self.on_key_press
        self.main_widget.keyReleaseEvent = self.on_key_release
        self.main_widget.closeEvent = self.on_close

        self.win.keyPressEvent = self.on_key_press
        self.win.keyReleaseEvent = self.on_key_release
        self.win.closeEvent = self.on_close

        self.win.show()
        self.main_widget.show()

        lock.release()

    def addItem(self, item):
        item.setGLOptions(
            "translucent"
        )  # this is neccessary to visualize items when the background is white!
        self.win.addItem(item)

    def add_legend_item(self, label, color):
        """Add an item to the legend."""
        self.legend_items.append((label, color))
        self.overlay.update_legend(self.legend_items)

    def create_axis(self):
        axis_length = 10
        arrow_size = 0.1  # Size of the arrowhead
        text_scale = 1.0

        # NOTE: the very basic approach is to add
        # axis = gl.GLAxisItem()
        # axis.setSize(10, 10, 10)
        # self.win.addItem(axis)

        # X axis (Red)
        x_axis = gl.GLLinePlotItem(
            pos=[[0, 0, 0], [axis_length, 0, 0]], color=(1, 0, 0, 1), width=2
        )
        self.addItem(x_axis)

        # X axis arrow
        x_arrow = gl.MeshData.cylinder(
            rows=10, cols=20, radius=[0, arrow_size], length=arrow_size * 2
        )
        x_arrow_item = gl.GLMeshItem(
            meshdata=x_arrow, smooth=True, color=(1, 0, 0, 1), shader="shaded"
        )
        x_arrow_item.rotate(-90, 0, 1, 0)  # Rotate to point along the X axis
        x_arrow_item.translate(axis_length, 0, 0)
        self.addItem(x_arrow_item)

        # X axis label
        x_label = gl.GLTextItem(pos=(axis_length * 1.2, 0, 0), text="X", color=(1, 0, 0, 1))
        x_label.scale(text_scale, text_scale, text_scale)  # Scale the text size
        self.addItem(x_label)

        # Y axis (Green)
        y_axis = gl.GLLinePlotItem(
            pos=[[0, 0, 0], [0, axis_length, 0]], color=(0, 1, 0, 1), width=2
        )
        self.addItem(y_axis)

        # Y axis arrow
        y_arrow = gl.MeshData.cylinder(
            rows=10, cols=20, radius=[0, arrow_size], length=arrow_size * 2
        )
        y_arrow_item = gl.GLMeshItem(
            meshdata=y_arrow, smooth=True, color=(0, 1, 0, 1), shader="shaded"
        )
        y_arrow_item.rotate(90, 1, 0, 0)  # Rotate to point along the Y axis
        y_arrow_item.translate(0, axis_length, 0)
        self.addItem(y_arrow_item)

        # Y axis label
        y_label = gl.GLTextItem(pos=(0, axis_length * 1.2, 0), text="Y", color=(0, 1, 0, 1))
        y_label.scale(text_scale, text_scale, text_scale)  # Scale the text size
        self.addItem(y_label)

        # Z axis (Blue)
        z_axis = gl.GLLinePlotItem(
            pos=[[0, 0, 0], [0, 0, axis_length]], color=(0, 0, 1, 1), width=2
        )
        self.addItem(z_axis)

        # Z axis arrow
        z_arrow = gl.MeshData.cylinder(
            rows=10, cols=20, radius=[0, arrow_size], length=arrow_size * 2
        )
        z_arrow_item = gl.GLMeshItem(
            meshdata=z_arrow, smooth=True, color=(0, 0, 1, 1), shader="shaded"
        )
        z_arrow_item.rotate(180, 1, 0, 0)  # Rotate to point along the Y axis
        z_arrow_item.translate(0, 0, axis_length)
        self.addItem(z_arrow_item)

        # Z axis label
        z_label = gl.GLTextItem(pos=(0, 0, axis_length * 1.2), text="Z", color=(0, 0, 1, 1))
        z_label.scale(text_scale, text_scale, text_scale)  # Scale the text size
        self.addItem(z_label)

    def run(self, figure_num, queue, lock, key, is_running, key_queue):
        self.key = key
        self.key_queue_thread = key_queue
        self.legend_items = []
        self.init(figure_num.value, lock)

        while is_running.value == 1:
            self.drawer_refresh(queue, lock)
            time.sleep(0.02)

        self.win.close()
        print(f"Qplot3d: run: closed")

    def drawer_refresh(self, queue, lock):
        while not queue.empty():
            self.got_data = True
            self.data = queue.get()
            xyz_signal, name, color, marker, size = self.data

            xyz_data = np.array(xyz_signal)
            # print(f'Qplot3d: xyz_data shape: {xyz_data.shape}')

            if self.auto_view:
                self.update_limits(xyz_data[-1, :])

            handle = None
            if name in self.handle_map:
                # print(f'Qplot3d: updating plot {name}')
                handle = self.handle_map[name]
                handle.setData(pos=xyz_data)  # Update existing plot
            else:
                if False:
                    handle = gl.GLScatterPlotItem(pos=xyz_data, color=color, size=size)
                else:
                    # Create a GLLinePlotItem for continuous line (instead of GLScatterPlotItem)
                    handle = gl.GLLinePlotItem(pos=xyz_data, color=color, width=size)

                self.addItem(handle)
                self.add_legend_item(name, color)  # Add to legend
                self.handle_map[name] = handle

        if self.got_data:
            self.plot_refresh(lock)

    def on_key_press(self, event):
        key = event.key()
        try:
            self.key.value = ord(event.text())  # Convert to int
            self.key_queue_thread.put(self.key.value)
        except Exception as e:
            print(f"Qplot3d: on_key_press: encountered exception: {e}")
        # if key == Qt.Key_Up:
        #     print('Up arrow key pressed')
        # elif key == Qt.Key_Down:
        #     print('Down arrow key pressed')
        # elif key == Qt.Key_Left:
        #     print('Left arrow key pressed')
        # elif key == Qt.Key_Right:
        #     print('Right arrow key pressed')
        # else:
        #     print(f'Key pressed: {chr(key)}')  # Print the character of the key

    def on_key_release(self, event):
        # self.key.value = 0  # Reset to no key symbol
        pass

    def get_key(self):
        if not self.key_queue.empty():
            return chr(self.key_queue.get())
        else:
            return ""

    def on_close(self, event):
        self.is_running.value = 0
        print(f'Qplot3d "{self.title}" closing figure.')

    def update_limits(self, point):
        x, y, z = point[0], point[1], point[2]
        self.limits["x"] = [min(self.limits["x"][0], x), max(self.limits["x"][1], x)]
        self.limits["y"] = [min(self.limits["y"][0], y), max(self.limits["y"][1], y)]
        self.limits["z"] = [min(self.limits["z"][0], z), max(self.limits["z"][1], z)]
        self.axis_computed = True

    def set_axis(self):
        scale_distance = 1.8
        if self.axis_computed:
            xlim, ylim, zlim = self.limits["x"], self.limits["y"], self.limits["z"]
            self.win.opts["center"] = QVector3D(
                (xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2, (zlim[1] + zlim[0]) / 2
            )
            self.win.opts["distance"] = (
                max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) * scale_distance
            )

    def plot_refresh(self, lock):
        lock.acquire()
        if self.is_running.value == 1:
            self.set_axis()
        self.win.update()  # Process events to update the plot
        self.app.processEvents()  # Process events to update the plot
        lock.release()

    def draw(self, xyz_signal, name, color=(1, 0, 0, 1), marker="", size=5):
        if isinstance(color, str):
            color = pg.mkColor(color)
        if self.queue is None:
            return
        self.queue.put((xyz_signal, name, color, marker, size))
