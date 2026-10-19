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
import platform

import pyslam.config as config

config.cfg.set_lib("depth_anything_v2")


from pyslam.slam import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.depth import img_from_depth
from pyslam.utilities.logging import Printer
from pyslam.utilities.system import set_rlimit

from .depth_estimator_base import DepthEstimator


import torch
from depth_anything_v2.dpt import DepthAnythingV2


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# Monocular depth estimator using the DepthAnythingV2 model.
class DepthEstimatorDepthAnythingV2(DepthEstimator):
    depth_anything_v2_path = kRootFolder + "/thirdparty/depth_anything_v2/metric_depth/checkpoints"
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    # the max_depth each metric checkpoint was fine-tuned with (indoor: Hypersim, outdoor: Virtual KITTI)
    checkpoint_max_depth = {"hypersim": 20.0, "vkitti": 80.0}

    def __init__(
        self,
        device=None,
        camera: Camera = None,
        min_depth=0,
        max_depth=50,
        dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        encoder_name="vitl",
        precision=None,
    ):  # or 'vits', 'vitb'   we use the largest by default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == "cuda":
            print("DepthEstimatorDepthAnythingV2: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():  # Should return True for MPS availability
                raise Exception("DepthEstimatorDepthAnythingV2: MPS is not available")
            print("DepthEstimatorDepthAnythingV2: Using MPS")
        else:
            print("DepthEstimatorDepthAnythingV2: Using CPU")
        transform = None
        dataset_name = (
            "vkitti" if dataset_env_type == DatasetEnvironmentType.OUTDOOR else "hypersim"
        )
        # The metric head is a sigmoid times the model's max_depth, so the model is built with the
        # max_depth of its checkpoint (Depth-Anything-V2/metric_depth/README.md: 20 for Hypersim, 80
        # for Virtual KITTI): any other value scales every depth by value / checkpoint value. The
        # max_depth of the caller is only the range for display and clipping.
        self.model_max_depth = DepthEstimatorDepthAnythingV2.checkpoint_max_depth[dataset_name]
        model = DepthAnythingV2(
            **{
                **DepthEstimatorDepthAnythingV2.model_configs[encoder_name],
                "max_depth": self.model_max_depth,
            }
        )
        model_path = f"{DepthEstimatorDepthAnythingV2.depth_anything_v2_path}/depth_anything_v2_metric_{dataset_name}_{encoder_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device).eval()
        super().__init__(
            model,
            transform,
            device,
            camera=camera,
            min_depth=min_depth,
            max_depth=max_depth,
            dataset_env_type=dataset_env_type,
            precision=None,
        )

    # Return the predicted depth map and the point cloud (if any)
    def infer(self, image, image_right=None):
        depth_prediction = self.model.infer_image(image)
        self.depth_map = depth_prediction
        return depth_prediction, None
