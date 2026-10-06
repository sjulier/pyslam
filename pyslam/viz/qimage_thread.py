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

# (not torch.multiprocessing: see pyslam/utilities/logging.py)
import multiprocessing as mp
from pyslam.utilities.multi_processing import MultiprocessingManager

import numpy as np
import cv2
import time
import traceback

from pyslam.utilities.logging import Logging
from pyslam.utilities.system import locally_configure_qt_environment
from pyslam.utilities.data_management import empty_queue

try:
    # Prefer the compatibility shim so any installed Qt binding works.
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except Exception:
    # Fallback (only used if the shim isn't available)
    from PyQt5 import QtCore, QtGui, QtWidgets

# Unified names
QApplication = QtWidgets.QApplication
QLabel = QtWidgets.QLabel
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

Qt = QtCore.Qt
QFont = QtGui.QFont
QColor = QtGui.QColor
QVector3D = QtGui.QVector3D
QPixmap = QtGui.QPixmap
QImage = QtGui.QImage
QTimer = QtCore.QTimer
QMainWindow = QtWidgets.QMainWindow

kVerbose = False


class ImageWidget(QWidget):
    """Shows an image scaled to the widget, keeping its aspect ratio (centred on a dark background),
    so that the window can be resized freely."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(80, 60)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self.pixmap is not None and not self.pixmap.isNull():
            scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(
                (self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2, scaled
            )
        painter.end()


def fit_size(width, height, max_width, max_height):
    """(width, height) scaled down, with the same aspect ratio, to fit in max_width x max_height."""
    scale = min(1.0, max_width / float(width), max_height / float(height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def check_image_format(image):
    """Check if image is grayscale or color and determine appropriate Qt format"""
    if len(image.shape) == 2:
        # Grayscale image (height, width)
        # print(f"Grayscale image: {image.shape}")
        return QImage.Format_Grayscale8
    elif len(image.shape) == 3:
        height, width, channels = image.shape
        # print(f"Color image: {image.shape}, channels: {channels}")
        if channels == 1:
            return QImage.Format_Grayscale8
        elif channels == 3:
            # print("Color image")
            return QImage.Format_BGR888
        elif channels == 4:
            return QImage.Format_RGBA8888
    return QImage.Format_BGR888  # Default fallback


class QimageViewer:
    _instance = None

    def __init__(self):

        self.is_running = mp.Value("i", 1)  # Flag to manage process lifecycle
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        self.key = mp.Value("i", 0)

        self.screen_width = None
        self.screen_height = None

        # Start the GUI process
        self.process = mp.Process(
            target=self.run, args=(self.queue, self.key_queue, self.key, self.is_running)
        )
        self.process.start()

    # destructor
    def __del__(self):
        self.quit()

    @classmethod
    def is_running(cls):
        return (cls._instance is not None) and (cls._instance.is_running.value != 0)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def quit(self):
        """Stop the viewer process."""
        print(f"QimageViewer closing...")
        if self.is_running.value == 0:
            print(f"QimageViewer already closed")
            return
        self.is_running.value = 0
        # Give the process time to finish its current iteration
        time.sleep(0.1)
        # Join the process with a timeout
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            print(f"Warning: QimageViewer process did not terminate in time, forcing kill.")
            self.process.terminate()
        print(f"QimageViewer closed")

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

    def init(self):
        locally_configure_qt_environment()
        self.app = QApplication([])  # Initialize the application
        self.image_map = {}  # name -> [image widget, window, aspect ratio of its image]
        self.key
        self.offset = 0
        self.screen_width, self.screen_height = self.get_screen_dimensions()

    def close(self):
        for name, (label, window, aspect_ratio) in self.image_map.items():
            window.close()

    def init_window(self, name, width, height):
        window = QMainWindow()
        window.setWindowTitle(name)
        window.setGeometry(10 + self.offset, 10 + self.offset, 200, 200)  # Set a default size
        self.offset += 30

        # The largest size the viewer gives a window itself (the user can make it larger)
        self.max_width = int(self.screen_width * 4.0 / 5)
        self.max_height = int(self.screen_height * 4.0 / 5)

        # The first image at its own size, if it fits
        width, height = fit_size(width, height, self.max_width, self.max_height)
        current_position = window.frameGeometry().topLeft()
        window.setGeometry(
            current_position.x() + self.offset, current_position.y() + self.offset, width, height
        )

        label = ImageWidget(window)  # scales the image to the window, keeping its aspect ratio
        window.setCentralWidget(label)

        window.keyPressEvent = self.on_key_press
        window.keyReleaseEvent = self.on_key_release
        window.closeEvent = self.on_close

        window.show()
        return label, window

    def update_image(self, image, name, format=QImage.Format_BGR888):
        height, width, channels = image.shape
        new_aspect_ratio = float(width) / height
        if name not in self.image_map:
            label, window = self.init_window(name, width, height)
            self.image_map[name] = [label, window, new_aspect_ratio]
        label, window, aspect_ratio = self.image_map[name]
        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, format)
        label.setPixmap(QPixmap.fromImage(q_image.copy()))  # copy: the pixmap outlives the numpy array
        if abs(aspect_ratio - new_aspect_ratio) > 1e-3:
            # The shape of the image has changed (e.g. another number of loop candidates): keep the
            # width of the window, which the user may have chosen, and adapt its height. With the same
            # shape the window is left as it is, so the user's resizing is kept.
            if kVerbose:
                print(
                    f"QimageViewer: window {name}:  image size: {width}x{height}, aspect ratio changed from {aspect_ratio} to {new_aspect_ratio}"
                )
            self.image_map[name][2] = new_aspect_ratio
            new_width = window.width()
            new_height = int(round(new_width / new_aspect_ratio))
            if new_height > self.max_height:
                new_height = self.max_height
                new_width = int(round(new_height * new_aspect_ratio))
            window.resize(new_width, new_height)

    def run(self, queue, key_queue, key, is_running):
        self.key = key
        self.key_queue_thread = key_queue
        try:
            self.init()
        except Exception as e:
            print(f"QimageViewer: init: encountered exception: {e}")
            traceback.print_exc()
            return

        while is_running.value == 1:
            try:
                if not queue.empty():
                    image, name, format = queue.get()
                    if image is not None:
                        try:
                            self.update_image(image, name, format)
                        except Exception as e:
                            print(f"QimageViewer: update_image: encountered exception: {e}")
                            traceback.print_exc()
            except Exception as e:
                print(f"QimageViewer: Unexpected error: {e}")
                # traceback.print_exc()
                break

            self.app.processEvents()  # Process PyQt events
            time.sleep(0.01)  # Prevent high CPU usage

        try:
            empty_queue(queue)
        except Exception as e:
            print(f"QimageViewer: Error emptying queue: {e}")

        self.close()
        print(f"QimageViewer: run: closed")

    def draw(self, image, name, format=QImage.Format_BGR888):
        if self.is_running.value == 0:
            return  # closed: an image left in the queue could hold up the exit of the process
        format = check_image_format(image)
        if image is not None:
            self.queue.put((image, name, format))

    def on_key_press(self, event):
        key = event.key()
        try:
            self.key.value = ord(event.text())  # Convert to int
            self.key_queue_thread.put(self.key.value)
        except Exception as e:
            print(f"QimageViewer: on_key_press: encountered exception: {e}")
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
        print(f"{mp.current_process().name} - QimageViewer closed figure")

    def get_key(self):
        if not self.key_queue.empty():
            return chr(self.key_queue.get())
        else:
            return ""
