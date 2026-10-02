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

# From torchvision.uitls
import smtplib, socket, hashlib, errno
import __main__ as main
from email.mime.text import MIMEText
import os
import gdown
import glob
import cv2
from pyslam.utilities.logging import Printer


def check_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def create_folder(file):
    if not file:
        return
    path = os.path.dirname(file)
    if path and not os.path.exists(path):
        os.makedirs(path)


def mkdir_for(f):
    """Create directory for a file path if it doesn't exist.

    Args:
        f: File path

    Returns:
        The file path (for chaining)
    """
    if not f:
        return f
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f


def list_dir(root, prefix=False):
    ### List all directories at a given root
    # root (str): Path to directory whose folders need to be listed
    # prefix (bool, optional): If true, prepends the path to each result, otherwise only returns the name of the directories found
    root = os.path.expanduser(root)
    directories = list(filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root)))

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    ### List all files ending with a suffix at a given root
    # root (str): Path to directory whose folders need to be listed
    # suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png'). It uses the Python "str.endswith" method and is passed directly
    # prefix (bool, optional): If true, prepends the path to each result, otherwise only returns the name of the files found
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix), os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def make_dirs(dir, exist_ok=True):
    # helper function, adds exist_ok to python2
    if exist_ok:
        try:
            os.makedirs(dir)
        except:
            pass
    else:
        os.makedirs(dir)


# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# with open(textfile) as fp:
# Create a text/plain message
# msg = MIMEText(fp.read())
def send_email(recipient, ignore_host="", login_gmail="", login_password=""):
    msg = MIMEText("")

    if socket.gethostname() == ignore_host:
        return
    msg["Subject"] = (
        socket.gethostname() + " just finished running a job " + os.path.basename(main.__file__)
    )
    msg["From"] = "pyslam@gmail.com"
    msg["To"] = recipient

    s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    s.ehlo()
    s.login(login_gmail, login_password)
    s.sendmail(login_gmail, recipient, msg.as_string())
    s.quit()


# download file from google drive
# need to pass as input the 'url' and output 'path'
def gdrive_download_lambda(*args, **kwargs):
    url = kwargs["url"]
    output = kwargs["path"]
    # check if outfolder exists or create it
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output):
        print(f"downloading {url} to {output}")
        gdrive_download_with_retry(url, output)
    else:
        print(f"file already exists: {output}")


def gdrive_download_with_retry(url, output, stall_timeout=60, max_attempts=5):
    """
    Download a file from Google Drive into `output`, without leaving a hung or truncated download.

    gdown has no timeout: a stalled connection blocks forever, and an interrupted download leaves a
    partial file under the final name, which later looks like a complete one. So gdown runs in a child
    process and writes to `output + ".part"`; if the file stops growing for `stall_timeout` seconds
    the child is stopped and the download resumed, up to `max_attempts` times. The file gets its
    final name only when gdown has finished successfully.
    """
    import subprocess
    import sys
    import time

    part = output + ".part"
    for attempt in range(1, max_attempts + 1):
        cmd = [sys.executable, "-m", "gdown", url, "-O", part, "--continue"]
        if "/file/d/" in url:
            cmd.append("--fuzzy")
        proc = subprocess.Popen(cmd)
        last_size, last_change = -1, time.time()
        stalled = False
        while proc.poll() is None:
            time.sleep(2)
            # gdown writes to a temporary file next to the output while downloading
            folder = os.path.dirname(part) or "."
            prefix = os.path.basename(part)
            size = sum(
                os.path.getsize(os.path.join(folder, name))
                for name in os.listdir(folder)
                if name.startswith(prefix)
            )
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > stall_timeout:
                stalled = True
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        if not stalled and proc.returncode == 0 and os.path.exists(part):
            os.replace(part, output)
            return
        reason = f"no data for {stall_timeout} s" if stalled else f"gdown exited with code {proc.returncode}"
        print(f"download of {os.path.basename(output)} interrupted ({reason}): attempt {attempt}/{max_attempts}")
    raise RuntimeError(
        f"Could not download {url} after {max_attempts} attempts. "
        f"Download it by hand (in a browser) and save it as {output}"
    )


# Select n_frame images from images_path starting from start_frame_name with delta_frame between each frame
def select_image_files(images_path, start_frame_name, n_frame, delta_frame):
    # List and sort all image paths in the directory
    img_paths = os.listdir(images_path)
    img_paths.sort()

    # Ensure the start_frame_name exists in img_paths
    if start_frame_name not in img_paths:
        raise ValueError(f"'{start_frame_name}' not found in the directory '{images_path}'.")

    # Get the index of the starting frame
    start_idx = img_paths.index(start_frame_name)
    num_files = len(img_paths)

    # Validate that the selection does not go out of bounds
    if start_idx + (n_frame - 1) * delta_frame >= num_files:
        raise ValueError(
            f"Not enough images in the directory to select {n_frame} frames with delta {delta_frame}."
        )

    # Select the images
    selected_img_paths = [img_paths[start_idx + i * delta_frame] for i in range(n_frame)]
    return selected_img_paths


def load_images_from_directory(image_dir, pattern="*.png", max_num_images=None):
    """
    Load images from a directory.

    Args:
        image_dir: Path to directory containing images
        pattern: Glob pattern for image files (default: "*.png")
        max_num_images: Maximum number of images to load (None for all)

    Returns:
        List of images (BGR format)
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if max_num_images is not None:
        image_paths = image_paths[:max_num_images]

    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            print(f"Loaded: {img_path} ({img.shape[1]}x{img.shape[0]})")
        else:
            Printer.red(f"Failed to load image: {img_path}")

    return images


def hash_md5(s):
    """Compute MD5 hash of a string.

    Args:
        s: Input string

    Returns:
        MD5 hash as hexadecimal string
    """
    return hashlib.md5(s.encode("utf-8")).hexdigest()
