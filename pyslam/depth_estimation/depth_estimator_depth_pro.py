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

import cv2
import numpy as np
import os
import sys

import pyslam.config as config

config.cfg.set_lib("depth_pro")

import torch
import depth_pro

from pyslam.slam import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.depth import img_from_depth
from pyslam.utilities.logging import Printer
from pyslam.utilities.system import set_rlimit

from .depth_estimator_base import DepthEstimator


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# Monocular depth estimator using the DepthPro model.
class DepthEstimatorDepthPro(DepthEstimator):
    def __init__(
        self,
        device=None,
        camera: Camera = None,
        min_depth=0,
        max_depth=50,
        dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        precision=torch.float16,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda" and torch.backends.mps.is_available():
                # Apple GPU: about 2 s per image on an M4 Max against about 35 s on its CPU, and the
                # same depth (fp16 and fp32 agree)
                device = torch.device("mps")
        if device.type == "cuda":
            print("DepthEstimatorDepthPro: Using CUDA")
        elif device.type == "mps":
            print("DepthEstimatorDepthPro: Using MPS")
        else:
            print("DepthEstimatorDepthPro: Using CPU")
        # Load model and preprocessing transform
        # NOTE: precision=torch.float16 makes the inference much faster
        if device.type == "cpu":
            print(
                f"DepthEstimatorDepthPro: Forcing precision {precision} on CPU since float16 may not be supported"
            )
            precision = torch.float32
        model, transform = depth_pro.create_model_and_transforms(device=device, precision=precision)
        model = model.to(device).eval()
        super().__init__(
            model,
            transform,
            device,
            camera=camera,
            min_depth=min_depth,
            max_depth=max_depth,
            dataset_env_type=dataset_env_type,
            precision=precision,
        )

    # Return the predicted depth map and the point cloud (if any)
    def infer(self, image, image_right=None):
        # Depth Pro reads its images as RGB (depth_pro.load_rgb); pySLAM's images are BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        # With a camera, its focal length; without one, Depth Pro estimates the focal length itself
        f_px = torch.tensor(float(self.camera.fx), dtype=torch.float32) if self.camera is not None else None
        prediction = self.model.infer(image, f_px=f_px)
        # Extract depth and focal length.
        depth_prediction = prediction["depth"]  # Depth in [m].
        self.focal_length_px = float(prediction["focallength_px"])  # the one given, or its estimate
        depth_prediction = depth_prediction.squeeze().float().cpu().numpy().astype(np.float32)
        self.depth_map = depth_prediction
        return depth_prediction, None
