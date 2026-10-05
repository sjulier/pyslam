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

import os
import time
import math
import numpy as np
import cv2

from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.logging import Printer

from pyslam.config_parameters import Parameters

from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_manager_configs import FeatureManagerConfigs
from pyslam.local_features.feature_types import FeatureDescriptorTypes
from pyslam.slam.feature_tracker_shared import SlamFeatureManagerInfo

from .loop_detector_base import LoopDetectorBase

# The loop detector classes are imported in loop_detector_factory(), only for the selected type: their
# modules need optional native or deep-learning packages (pydbow2, pydbow3, pyibow/pyobindex2, torch
# models, tensorflow), and importing them all here made every detector's package a requirement of
# main_slam.py, even with the default DBOW3.
from .loop_detector_vocabulary import (
    DBow3OrbVocabularyData,
    DBow2OrbVocabularyData,
    VladOrbVocabularyData,
    dbow2_orb_vocabulary_factory,
    dbow3_orb_vocabulary_factory,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.slam import Slam  # Only imported when type checking, not at runtime
    from pyslam.local_features.feature_manager import FeatureManager


kVerbose = True
kPrintTrackebackDetails = True

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


@register_class
class GlobalDescriptorType(SerializableEnum):
    # These types take the name from the adopted aggregation type and underlying local descriptor type.
    DBOW2 = 0  # Bags of Words (BoW). This implementation only works with ORB local features.
    # It needs an ORB vocabulary (available).
    # Reference: "Bags of Binary Words for Fast Place Recognition in Image Sequences"
    DBOW3 = 1  # Bags of Words (BoW).
    # It needs a vocabulary (available for ORB).
    OBINDEX2 = 2  # Hierarchical indexing scheme. Incremental Bags of binary Words.
    # Incrementally builds a vocabulary. If needed, it transforms the input non-binary local descriptors into binary descriptors.
    # Reference: "iBoW-LCD: An Appearance-based Loop Closure Detection Approach using Incremental Bags of Binary Words"
    IBOW = 3  # Incremental Bags of binary Words (iBoW). Built on the top of OBINDEX2.
    # It incrementally builds a vocabulary. If needed, it transforms input non-binary local descriptors into binary descriptors.
    # Reference: "iBoW-LCD: An Appearance-based Loop Closure Detection Approach using Incremental Bags of Binary Words"
    HDC_DELF = 4  # Local DELF descriptor + Hyperdimensional Computing (HDC).
    # Reference: "Large-Scale Image Retrieval with Attentive Deep Local Features", "Hyperdimensional Computing as a Framework for Systematic Aggregation of Image Descriptors"
    SAD = 5  # Sum of Absolute Differences (SAD)
    # Reference: "SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights".
    ALEXNET = 6  # AlexNetConv3Extractor model.
    # Reference: "On the performance of ConvNet features for place recognition".
    NETVLAD = 7  # PatchNetVLADFeatureExtractor model.
    # Reference: "Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition".
    COSPLACE = 8  # CosPlaceFeatureExtractor model.
    # Reference: "Rethinking Visual Geo-localization for Large-Scale Applications"
    EIGENPLACES = 9  # EigenPlacesFeatureExtractor model.
    # Reference: "EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition"
    VLAD = 10  # VLAD, Vector of Locally Aggregated Descriptors. It needs a vocabulary (available for ORB).
    # Reference: "All about VLAD".
    MEGALOC = (
        11  # Global feature descriptor based on MegaLoc model. https://github.com/gmberton/MegaLoc
    )
    # Reference: "MegaLoc: One Retrieval to Place Them All".


# Additional information about used local descriptors aggregation method
@register_class
class LocalDescriptorAggregationType(SerializableEnum):
    NONE = 0
    DBOW2 = 1  # Bags of Words (BoW). This version only works with ORB2 local features (DBOW3 is easier to use).
    # Reference: "Bags of Binary Words for Fast Place Recognition in Image Sequences"
    DBOW3 = 2  # Bags of Words (BoW).
    OBINDEX2 = 3  # Hierarchical indexing scheme. Incremental Bags of binary Words.
    # Reference: "iBoW-LCD: An Appearance-based Loop Closure Detection Approach using Incremental Bags of Binary Words"
    IBOW = 4  # Incremental Bags of binary Words (iBoW). Built on the top of OBINDEX2.
    # Reference: "iBoW-LCD: An Appearance-based Loop Closure Detection Approach using Incremental Bags of Binary Words"
    HDC = 5  # Hyperdimensional Computing (HDC).
    # Reference: "Hyperdimensional Computing as a Framework for Systematic Aggregation of Image Descriptors"
    NETVLAD = 6  # Patch-level features from NetVLAD residuals
    # Reference: "Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition".
    VLAD = 7  # VLAD, Vector of Locally Aggregated Descriptors.
    # Reference: "All about VLAD"


"""
LoopDetectorConfigs contains a collection of ready-to-used loop detection configurations. These configurations are used by the class LoopDetectingProcess.
You can test any of these configurations separataly (without SLAM) by using the script: test/loopclosing/test_loop_detecting_process.py or test/loopclosing/test_loop_detector.py.
See the README for further details.

Template configuration: 

    LoopDetectorConfigs.XXX = dict(
        global_descriptor_type = GlobalDescriptorType.XXX,
        local_feature_manager_config = FeatureManagerConfigs.XXX,                  # If None the frontend local descriptors will be re-used (must be compatible with the used descriptor aggregator and loaded vocabulary). 
                                                                                   # Otherwise, an independent local feature manager is created and used.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.XXX,
        vocabulary_data = XXX)                                                     # Must be a vocabulary built with the frontend local descriptor type (see the file loop_detector_vocabulary.py)
"""


def is_cosplace_installed():
    """Whether the CosPlace loop detector can run without downloading anything: its code (thirdparty/vpr)
    and its model (torch.hub) are there. `pixi run models` (or models-vpr) installs both."""
    code = os.path.join(kRootFolder, "thirdparty", "vpr", "feature_extraction", "feature_extractor_cosplace.py")
    if not os.path.isfile(code):
        return False
    try:
        import torch

        hub = torch.hub.get_dir()
    except Exception:  # noqa: BLE001
        return False
    return os.path.isdir(os.path.join(hub, "gmberton_cosplace_main")) and os.path.isfile(
        os.path.join(hub, "checkpoints", "ResNet50_2048_cosplace.pth")
    )


def loop_detector_name_for_features(descriptor_type):
    """The loop detector for features with these descriptors when nothing names one (main_slam.py and
    main_slam_evaluation.py --features): None (the default DBOW3) for ORB; otherwise CosPlace when it is
    installed, else DBOW3_INDEPENDENT (DBoW3 on ORB descriptors of its own), since the vocabularies are
    made of ORB descriptors."""
    if descriptor_type in (FeatureDescriptorTypes.ORB, FeatureDescriptorTypes.ORB2):
        return None
    return "COSPLACE" if is_cosplace_installed() else "DBOW3_INDEPENDENT"


class LoopDetectorConfigs:

    @staticmethod
    def get_config_from_name(config_name):
        config_dict = getattr(LoopDetectorConfigs, config_name, None)
        if config_dict is not None:
            Printer.cyan("LoopDetectorConfigs: Configuration loaded:", config_dict)
        else:
            Printer.red(f"LoopDetectorConfigs: No configuration found for '{config_name}'")
        return config_dict

    # NOTE: Under mac, loading the DBOW2 vocabulary may be very slow (both from text and from boost archive).
    DBOW2 = dict(
        global_descriptor_type=GlobalDescriptorType.DBOW2,
        local_feature_manager_config=None,  # If None the frontend local descriptors will be re-used (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.DBOW2,
        vocabulary_data=dbow2_orb_vocabulary_factory(),
    )  # Must be a vocabulary built with the frontend local descriptor type

    DBOW2_INDEPENDENT = dict(
        global_descriptor_type=GlobalDescriptorType.DBOW2,
        local_feature_manager_config=FeatureManagerConfigs.ORB2,  # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.DBOW2,
        vocabulary_data=dbow2_orb_vocabulary_factory(),
    )  # Must be a vocabulary built with the frontend local descriptor type

    # NOTE: Under mac, loading the DBOW2 vocabulary may be very slow (both from text and from boost archive).
    DBOW3 = dict(
        global_descriptor_type=GlobalDescriptorType.DBOW3,
        local_feature_manager_config=None,  # If None the frontend local descriptors will be re-used (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.DBOW3,
        vocabulary_data=dbow3_orb_vocabulary_factory(),
    )  # Must be a vocabulary built with the frontend local descriptor type

    DBOW3_INDEPENDENT = dict(
        global_descriptor_type=GlobalDescriptorType.DBOW3,
        local_feature_manager_config=FeatureManagerConfigs.ORB2,  # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.DBOW3,
        vocabulary_data=dbow3_orb_vocabulary_factory(),
    )  # Must be a vocabulary built with the adopted local descriptor type

    VLAD = dict(
        global_descriptor_type=GlobalDescriptorType.VLAD,
        local_feature_manager_config=None,  # If None the frontend local descriptors will be re-used (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.VLAD,
        vocabulary_data=VladOrbVocabularyData(),
    )  # Must be a vocabulary built with the adopted local descriptor type

    VLAD_INDEPENDENT = dict(
        global_descriptor_type=GlobalDescriptorType.VLAD,
        local_feature_manager_config=FeatureManagerConfigs.ORB2,  # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.VLAD,
        vocabulary_data=VladOrbVocabularyData(),
    )  # Must be a vocabulary built with the adopted local descriptor type

    OBINDEX2 = dict(
        global_descriptor_type=GlobalDescriptorType.OBINDEX2,
        local_feature_manager_config=None,  # If None the frontend local descriptors will be re-used. If they are non-binary, they will be converted to binary.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.OBINDEX2,
        vocabulary_data=None,
    )  # OBIndex2 does not need a vocabulary. It incrementally builds it.

    IBOW = dict(
        global_descriptor_type=GlobalDescriptorType.IBOW,
        local_feature_manager_config=None,  # If None the frontend local descriptors will be re-used. If they are non-binary, they will be converted to binary.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.IBOW,
        vocabulary_data=None,
    )  # IBow does not need a vocabulary. It incrementally builds it.

    IBOW_INDEPENDENT = dict(
        global_descriptor_type=GlobalDescriptorType.IBOW,
        local_feature_manager_config=FeatureManagerConfigs.ORB2,  # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and loaded vocabulary)
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.IBOW,
        vocabulary_data=None,
    )  # IBow does not need a vocabulary. It incrementally builds it.

    # NOTE: HDC_DELF seems very slow and does not work well with SLAM. In fact,
    #       the online computation of global descriptors for keyframes struggles to keep up with the real-time demands of SLAM processing.
    #       You can test HDC_DELF separataly (without SLAM) by using the script: test/loopclosing/test_loop_detecting_process.py.
    HDC_DELF = dict(
        global_descriptor_type=GlobalDescriptorType.HDC_DELF,
        local_feature_manager_config=None,  # It does use its own local feature manager: Delf.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.HDC,
        vocabulary_data=None,
    )  # It does not need a vocabulary

    SAD = dict(
        global_descriptor_type=GlobalDescriptorType.SAD,
        local_feature_manager_config=None,  # Not needed.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.NONE,
        vocabulary_data=None,
    )  # It does not need a vocabulary

    # NOTE: ALEXNET seems very slow and does not work well with SLAM. In fact,
    #       the online computation of global descriptors for keyframes struggles to keep up with the real-time demands of SLAM processing.
    #       You can test ALEXNET separataly (without SLAM) by using the script: test/loopclosing/test_loop_detecting_process.py.
    ALEXNET = dict(
        global_descriptor_type=GlobalDescriptorType.ALEXNET,
        local_feature_manager_config=None,  # Not needed.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.NONE,
        vocabulary_data=None,
    )  # It does not need a vocabulary

    NETVLAD = dict(
        global_descriptor_type=GlobalDescriptorType.NETVLAD,
        local_feature_manager_config=None,  # Not needed.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.NETVLAD,
        vocabulary_data=None,
    )  # It does not need a vocabulary

    COSPLACE = dict(
        global_descriptor_type=GlobalDescriptorType.COSPLACE,
        local_feature_manager_config=None,  # Not needed.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.NONE,
        vocabulary_data=None,
    )  # It does not need a vocabulary

    EIGENPLACES = dict(
        global_descriptor_type=GlobalDescriptorType.EIGENPLACES,
        local_feature_manager_config=None,  # Not needed.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.NONE,
        vocabulary_data=None,
    )  # It does not need a vocabulary

    MEGALOC = dict(
        global_descriptor_type=GlobalDescriptorType.MEGALOC,
        local_feature_manager_config=None,  # Not needed.
        local_descriptor_aggregation_type=LocalDescriptorAggregationType.NONE,
        vocabulary_data=None,
    )  # It does not need a vocabulary


def loop_detector_factory(
    global_descriptor_type=GlobalDescriptorType.DBOW3,
    local_feature_manager_config=None,  # If None the frontend local descriptors will be re-used (depending on the used descriptor aggregator and vocabulary)
    local_descriptor_aggregation_type=LocalDescriptorAggregationType.DBOW3,
    vocabulary_data=DBow3OrbVocabularyData(),
    slam_info=SlamFeatureManagerInfo(),
):

    if vocabulary_data is not None:
        vocabulary_data.check_download()  # check if the vocabulary exists or we need to download it

    # create an independent local feature manager if requested for the loop detection
    if local_feature_manager_config is None:
        local_feature_manager = None  # re-use the frontend local descriptors
    else:
        local_feature_manager = feature_manager_factory(
            **local_feature_manager_config
        )  # use an independent local feature manager in loop closing

    loop_detector = None
    if global_descriptor_type == GlobalDescriptorType.DBOW2:
        if local_feature_manager is not None:
            if (
                local_feature_manager.descriptor_type != FeatureDescriptorTypes.ORB2
                and local_feature_manager.descriptor_type != FeatureDescriptorTypes.ORB
            ):
                raise ValueError(
                    "loop_detector_factory: local_feature_manager.descriptor_type must be ORB2 or ORB"
                )
        from .loop_detector_dbow2 import LoopDetectorDBoW2

        loop_detector = LoopDetectorDBoW2(
            vocabulary_data=vocabulary_data, local_feature_manager=local_feature_manager
        )
    elif global_descriptor_type == GlobalDescriptorType.DBOW3:
        from .loop_detector_dbow3 import LoopDetectorDBoW3

        loop_detector = LoopDetectorDBoW3(
            vocabulary_data=vocabulary_data, local_feature_manager=local_feature_manager
        )
    elif global_descriptor_type == GlobalDescriptorType.VLAD:
        from .loop_detector_vlad import LoopDetectorVlad

        loop_detector = LoopDetectorVlad(
            vocabulary_data=vocabulary_data, local_feature_manager=local_feature_manager
        )
    elif global_descriptor_type == GlobalDescriptorType.OBINDEX2:
        from .loop_detector_obindex2 import LoopDetectorOBIndex2

        loop_detector = LoopDetectorOBIndex2(
            local_feature_manager=local_feature_manager, slam_info=slam_info
        )
    elif global_descriptor_type == GlobalDescriptorType.IBOW:
        from .loop_detector_ibow import LoopDetectorIBow

        loop_detector = LoopDetectorIBow(
            local_feature_manager=local_feature_manager, slam_info=slam_info
        )
    elif global_descriptor_type == GlobalDescriptorType.HDC_DELF:
        from .loop_detector_vpr import LoopDetectorHdcDelf

        loop_detector = LoopDetectorHdcDelf(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.SAD:
        from .loop_detector_vpr import LoopDetectorSad

        loop_detector = LoopDetectorSad(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.ALEXNET:
        from .loop_detector_vpr import LoopDetectorAlexNet

        loop_detector = LoopDetectorAlexNet(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.NETVLAD:
        from .loop_detector_vpr import LoopDetectorNetVLAD

        loop_detector = LoopDetectorNetVLAD(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.COSPLACE:
        from .loop_detector_vpr import LoopDetectorCosPlace

        loop_detector = LoopDetectorCosPlace(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.EIGENPLACES:
        from .loop_detector_vpr import LoopDetectorEigenPlaces

        loop_detector = LoopDetectorEigenPlaces(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.MEGALOC:
        from .loop_detector_vpr import LoopDetectorMegaloc

        loop_detector = LoopDetectorMegaloc(local_feature_manager=local_feature_manager)
    else:
        raise ValueError("loop_detector_factory: unknown global_descriptor_type")

    if loop_detector is not None:
        loop_detector.global_descriptor_type = global_descriptor_type
        loop_detector.local_descriptor_aggregation_type = local_descriptor_aggregation_type
        loop_detector.local_feature_manager = local_feature_manager
        loop_detector.vocabulary_data = vocabulary_data

    return loop_detector


def loop_detector_config_check(
    loop_detector: LoopDetectorBase, slam_feature_descriptor_type: FeatureDescriptorTypes
):
    descriptor_type_to_check = None
    if loop_detector.local_feature_manager is None:
        descriptor_type_to_check = slam_feature_descriptor_type
    else:
        descriptor_type_to_check = loop_detector.local_feature_manager.descriptor_type

    if loop_detector.global_descriptor_type == GlobalDescriptorType.DBOW2:
        if (
            descriptor_type_to_check != FeatureDescriptorTypes.ORB2
            and descriptor_type_to_check != FeatureDescriptorTypes.ORB
        ):
            message = f"loop_detector_config_check: ERROR: incompatible descriptor_type! With DBOW2, the only available voculary that is loaded needs a local_feature_manager with ORB2 or ORB."
            message += f"\n\t As a quick solution, you can set the loop detector to work with an independent ORB2 local_feature_manager: Use the configuration DBOW2_INDEPENDENT."
            message += f"\n\t Alternatively, you may want to use DBOW3 instead and then create a vocabulary with your favorite descriptors (see the pyslam README file) and then load it."
            Printer.red(message)
            raise ValueError(message)

    if loop_detector.vocabulary_data is not None:
        # Let's encode the fact that ORB and ORB2 are compatible
        if (
            loop_detector.vocabulary_data.descriptor_type == FeatureDescriptorTypes.ORB2
            or loop_detector.vocabulary_data.descriptor_type == FeatureDescriptorTypes.ORB
        ) and (
            descriptor_type_to_check == FeatureDescriptorTypes.ORB2
            or descriptor_type_to_check == FeatureDescriptorTypes.ORB
        ):
            return
        if descriptor_type_to_check != loop_detector.vocabulary_data.descriptor_type:
            message = f"loop_detector_config_check: ERROR: incompatible vocabulary type!"
            message += f"\n\t The loaded vocabulary_data.descriptor_type is {loop_detector.vocabulary_data.descriptor_type.name}."
            message += f"\n\t On the other end, the loop detector is configured to work with a {descriptor_type_to_check.name} local_feature_manager."
            message += f"\n\t If you dont have a vocabulary with {descriptor_type_to_check.name} then you must create it (see the pyslam README file) and then load it."
            message += f"\n\t Otherwise, you can set the loop detector to work with an independent {loop_detector.vocabulary_data.descriptor_type.name} local_feature_manager."
            message += f"\n\t See the file loop_detector_configs.py for further details."
            Printer.red(message)
            raise ValueError(message)
