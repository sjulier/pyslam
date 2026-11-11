/*
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
 */
#include "config_parameters.h"

#include <algorithm>
#include <cmath>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace pyslam {

std::atomic<float> Parameters::kMaxDescriptorDistance{
    0.0f}; // It is initialized by the first created instance of feature_manager.py at runtime

float Parameters::kFeatureMatchDefaultRatioTest =
    0.7; // This is the default ratio test used by all feature matchers. It can be configured
         // per descriptor in feature_tracker_configs.py

// Constants (config_parameters.h): defaults, replaced by the Python values at Slam init
float Parameters::kChi2Mono = 5.991;
float Parameters::kChi2Stereo = 7.815;
double Parameters::kThHuberMono = 2.4476519360399265;
double Parameters::kThHuberStereo = 2.7955321496988725;
int Parameters::kSparseImageColorPatchDelta = 1;
std::size_t Parameters::kMinWeightForDrawingCovisibilityEdge = 100;
std::size_t Parameters::kMaxSparseMapPointsToVisualize = 1000000;
int Parameters::kMinNumOfCovisiblePointsForCreatingConnection = 15;
int Parameters::kLargeBAWindowSize = 20;
std::size_t Parameters::kMaxLenFrameDeque = 20;
int Parameters::kMaxNumOfKeyframesInLocalMap = 80;
int Parameters::kNumBestCovisibilityKeyFrames = 10;
double Parameters::kMaxOutliersRatioInPoseOptimization = 0.9;
float Parameters::kViewingCosLimitForPoint = 0.5;
double Parameters::kScaleConsistencyFactor = 1.5;
double Parameters::kMaxDistanceToleranceFactor = 1.2;
double Parameters::kMinDistanceToleranceFactor = 0.8;
double Parameters::kRansacThresholdNormalized = 0.0004;
double Parameters::kRansacProb = 0.999;
int Parameters::kNumMinInliersEssentialMat = 8;
int Parameters::kNumMinObsForKeyFrameTrackedPoints = 3;
double Parameters::kRansacReprojThreshold = 5;
int Parameters::kRansacMinNumInliers = 15;
int Parameters::kMaxNumVisualOdometryPoints = 100;
int Parameters::kMaxNumStereoPointsOnNewKeyframe = 100;
float Parameters::kMaxReprojectionDistanceFrame = 7.0f;
float Parameters::kMaxReprojectionDistanceFrameNonStereo = 15.0f;
float Parameters::kMaxReprojectionDistanceMap = 3.0f;
float Parameters::kMaxReprojectionDistanceMapRgbd = 3.0f;
float Parameters::kMaxReprojectionDistanceMapReloc = 5.0f;
float Parameters::kMaxReprojectionDistanceFuse = 3.0f;
float Parameters::kMaxReprojectionDistanceSim3 = 7.5f;
float Parameters::kMatchRatioTestFrameByProjection = 0.9f;
float Parameters::kMatchRatioTestMap = 0.8f;
float Parameters::kMinDistanceFromEpipole = 10.0f;
bool Parameters::kCheckFeaturesOrientation = true;
int Parameters::kLocalBAWindowSize = 20;
double Parameters::kCosMaxParallax = 0.9998;
double Parameters::kMinRatioBaselineDepth = 0.01;
float Parameters::kLoopClosingMaxReprojectionDistanceFuse = 4.0f;
float Parameters::kKeyframeCullingRedundantObsRatio = 0.9f;
double Parameters::kKeyframeMaxTimeDistanceInSecForCulling = 0.5;
int Parameters::kKeyframeCullingMinNumPoints = 0;
int Parameters::kLocalMappingParallelFusePointsNumWorkers = 2;
int Parameters::kLocalMappingNumNeighborKeyFramesStereo = 10;
int Parameters::kLocalMappingNumNeighborKeyFramesMonocular = 20;

bool Parameters::kLocalMappingOnSeparateThread = true;
bool Parameters::kTrackingWaitForLocalMappingToGetIdle = false;
double Parameters::kWaitForLocalMappingTimeout = 0.05;
double Parameters::kParallelLBAWaitIdleTimeout = 0.03;
int Parameters::kPlaybackThrottleWindow = 30;
double Parameters::kPlaybackThrottleWeakTrackingRatio = 0.5;
double Parameters::kLocalMappingTimeoutPopKeyframe = 0.5;
bool Parameters::kUseDynamicDesDistanceTh = true;
bool Parameters::kUseDescriptorSigmaMadv2 = false;
double Parameters::kCosMaxParallaxInitializer = 0.99998;
double Parameters::kInitializerDesiredMedianDepth = 1.0;
double Parameters::kInitializerMinRatioDepthBaseline = 100.0;
int Parameters::kInitializerNumMinFeatures = 100;
int Parameters::kInitializerNumMinFeaturesStereo = 500;
int Parameters::kInitializerNumMinTriangulatedPoints = 150;
int Parameters::kInitializerNumMinTriangulatedPointsStereo = 100;
float Parameters::kInitializerFeatureMatchRatioTest = 0.9f;
int Parameters::kInitializerNumMinNumPointsForPnPWithDepth = 15;
bool Parameters::kInitializerUseCellCoverageCheck = true;
bool Parameters::kInitializerUseMinFrameDistanceCheck = true;
bool Parameters::kUseMotionModel = true;
bool Parameters::kUseSearchFrameByProjection = true;
int Parameters::kMinNumMatchedFeaturesSearchFrameByProjection = 20;
bool Parameters::kUseEssentialMatrixFitting = false;
int Parameters::kMinNumMatchedFeaturesSearchReferenceFrame = 15;
bool Parameters::kUseVisualOdometryPoints = true;
bool Parameters::kUseInterruptLocalMapping = false;
bool Parameters::kUseMotionBlurDection = true;
double Parameters::kMotionBlurDetectionLalacianVarianceThreshold = 100.0;
int Parameters::kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography = 30;
int Parameters::kNumMinPointsForNewKf = 15;
int Parameters::kNumMinTrackedClosePointsForNewKfNonMonocular = 100;
int Parameters::kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70;
double Parameters::kThNewKfRefRatioMonocular = 0.9;
double Parameters::kThNewKfRefRatioStereo = 0.75;
double Parameters::kThNewKfRefRatioNonMonocular = 0.25;
bool Parameters::kUseFeatureCoverageControlForNewKf = false;
bool Parameters::kLocalMappingParallelKpsMatching = true;
int Parameters::kLocalMappingParallelKpsMatchingNumWorkers = 2;
bool Parameters::kUseLargeWindowBA = false;
int Parameters::kEveryNumFramesLargeWindowBA = 10;
bool Parameters::kUseParallelProcessLBA = false;
bool Parameters::kOptimizationFrontEndUseGtsam = false;
bool Parameters::kUseLoopClosing = true;
bool Parameters::kDoVolumetricIntegration = false;
bool Parameters::kDoSparseSemanticMappingAndSegmentation = false;
bool Parameters::kSemanticMappingOnSeparateThread = true;


#ifdef USE_PYTHON
namespace {

template <typename T> void read_attr(const py::object &params, const char *name, T &dst) {
    if (py::hasattr(params, name)) {
        dst = py::cast<T>(params.attr(name));
    }
}

// a number that Python may hold as an int or a float (kMaxSparseMapPointsToVisualize = 1e6)
template <typename T> void read_number(const py::object &params, const char *name, T &dst) {
    if (py::hasattr(params, name)) {
        dst = static_cast<T>(py::cast<double>(params.attr(name)));
    }
}

} // namespace

void Parameters::sync_runtime_from_python() {
    py::object params = py::module_::import("pyslam.config_parameters").attr("Parameters");
    read_attr(params, "kLocalMappingOnSeparateThread", kLocalMappingOnSeparateThread);
    read_attr(params, "kTrackingWaitForLocalMappingToGetIdle",
              kTrackingWaitForLocalMappingToGetIdle);
    read_attr(params, "kWaitForLocalMappingTimeout", kWaitForLocalMappingTimeout);
    read_attr(params, "kParallelLBAWaitIdleTimeout", kParallelLBAWaitIdleTimeout);
    read_attr(params, "kPlaybackThrottleWindow", kPlaybackThrottleWindow);
    read_attr(params, "kPlaybackThrottleWeakTrackingRatio", kPlaybackThrottleWeakTrackingRatio);
    read_attr(params, "kLocalMappingTimeoutPopKeyframe", kLocalMappingTimeoutPopKeyframe);
    read_attr(params, "kUseDynamicDesDistanceTh", kUseDynamicDesDistanceTh);
    read_attr(params, "kUseDescriptorSigmaMadv2", kUseDescriptorSigmaMadv2);
    read_attr(params, "kCosMaxParallaxInitializer", kCosMaxParallaxInitializer);
    read_attr(params, "kInitializerDesiredMedianDepth", kInitializerDesiredMedianDepth);
    read_attr(params, "kInitializerMinRatioDepthBaseline", kInitializerMinRatioDepthBaseline);
    read_attr(params, "kInitializerNumMinFeatures", kInitializerNumMinFeatures);
    read_attr(params, "kInitializerNumMinFeaturesStereo", kInitializerNumMinFeaturesStereo);
    read_attr(params, "kInitializerNumMinTriangulatedPoints", kInitializerNumMinTriangulatedPoints);
    read_attr(params, "kInitializerNumMinTriangulatedPointsStereo",
              kInitializerNumMinTriangulatedPointsStereo);
    read_attr(params, "kInitializerFeatureMatchRatioTest", kInitializerFeatureMatchRatioTest);
    read_attr(params, "kInitializerNumMinNumPointsForPnPWithDepth",
              kInitializerNumMinNumPointsForPnPWithDepth);
    read_attr(params, "kInitializerUseCellCoverageCheck", kInitializerUseCellCoverageCheck);
    read_attr(params, "kInitializerUseMinFrameDistanceCheck", kInitializerUseMinFrameDistanceCheck);
    read_attr(params, "kUseMotionModel", kUseMotionModel);
    read_attr(params, "kUseSearchFrameByProjection", kUseSearchFrameByProjection);
    read_attr(params, "kMinNumMatchedFeaturesSearchFrameByProjection",
              kMinNumMatchedFeaturesSearchFrameByProjection);
    read_attr(params, "kUseEssentialMatrixFitting", kUseEssentialMatrixFitting);
    read_attr(params, "kMinNumMatchedFeaturesSearchReferenceFrame",
              kMinNumMatchedFeaturesSearchReferenceFrame);
    read_attr(params, "kUseVisualOdometryPoints", kUseVisualOdometryPoints);
    read_attr(params, "kUseInterruptLocalMapping", kUseInterruptLocalMapping);
    read_attr(params, "kUseMotionBlurDection", kUseMotionBlurDection);
    read_attr(params, "kMotionBlurDetectionLalacianVarianceThreshold",
              kMotionBlurDetectionLalacianVarianceThreshold);
    read_attr(params, "kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography",
              kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography);
    read_attr(params, "kNumMinPointsForNewKf", kNumMinPointsForNewKf);
    read_attr(params, "kNumMinTrackedClosePointsForNewKfNonMonocular",
              kNumMinTrackedClosePointsForNewKfNonMonocular);
    read_attr(params, "kNumMaxNonTrackedClosePointsForNewKfNonMonocular",
              kNumMaxNonTrackedClosePointsForNewKfNonMonocular);
    read_attr(params, "kThNewKfRefRatioMonocular", kThNewKfRefRatioMonocular);
    read_attr(params, "kThNewKfRefRatioStereo", kThNewKfRefRatioStereo);
    read_attr(params, "kThNewKfRefRatioNonMonocular", kThNewKfRefRatioNonMonocular);
    read_attr(params, "kUseFeatureCoverageControlForNewKf", kUseFeatureCoverageControlForNewKf);
    read_attr(params, "kLocalMappingParallelKpsMatching", kLocalMappingParallelKpsMatching);
    read_attr(params, "kLocalMappingParallelKpsMatchingNumWorkers",
              kLocalMappingParallelKpsMatchingNumWorkers);
    read_attr(params, "kUseLargeWindowBA", kUseLargeWindowBA);
    read_attr(params, "kEveryNumFramesLargeWindowBA", kEveryNumFramesLargeWindowBA);
    read_attr(params, "kUseParallelProcessLBA", kUseParallelProcessLBA);
    read_attr(params, "kOptimizationFrontEndUseGtsam", kOptimizationFrontEndUseGtsam);
    read_attr(params, "kUseLoopClosing", kUseLoopClosing);
    read_attr(params, "kDoVolumetricIntegration", kDoVolumetricIntegration);
    read_attr(params, "kDoSparseSemanticMappingAndSegmentation",
              kDoSparseSemanticMappingAndSegmentation);
    read_attr(params, "kSemanticMappingOnSeparateThread", kSemanticMappingOnSeparateThread);
    if (py::hasattr(params, "kMaxDescriptorDistance")) {
        kMaxDescriptorDistance = py::cast<float>(params.attr("kMaxDescriptorDistance"));
    }
    // the constants (config_parameters.h)
    read_number(params, "kChi2Mono", kChi2Mono);
    read_number(params, "kChi2Stereo", kChi2Stereo);
    read_number(params, "kSparseImageColorPatchDelta", kSparseImageColorPatchDelta);
    read_number(params, "kMinWeightForDrawingCovisibilityEdge", kMinWeightForDrawingCovisibilityEdge);
    read_number(params, "kMaxSparseMapPointsToVisualize", kMaxSparseMapPointsToVisualize);
    read_number(params, "kMinNumOfCovisiblePointsForCreatingConnection", kMinNumOfCovisiblePointsForCreatingConnection);
    read_number(params, "kLargeBAWindowSize", kLargeBAWindowSize);
    read_number(params, "kMaxNumOfKeyframesInLocalMap", kMaxNumOfKeyframesInLocalMap);
    read_number(params, "kNumBestCovisibilityKeyFrames", kNumBestCovisibilityKeyFrames);
    read_number(params, "kMaxOutliersRatioInPoseOptimization", kMaxOutliersRatioInPoseOptimization);
    read_number(params, "kViewingCosLimitForPoint", kViewingCosLimitForPoint);
    read_number(params, "kScaleConsistencyFactor", kScaleConsistencyFactor);
    read_number(params, "kMaxDistanceToleranceFactor", kMaxDistanceToleranceFactor);
    read_number(params, "kMinDistanceToleranceFactor", kMinDistanceToleranceFactor);
    read_number(params, "kMaxNumVisualOdometryPoints", kMaxNumVisualOdometryPoints);
    read_number(params, "kMaxNumStereoPointsOnNewKeyframe", kMaxNumStereoPointsOnNewKeyframe);
    read_number(params, "kMaxReprojectionDistanceFrame", kMaxReprojectionDistanceFrame);
    read_number(params, "kMaxReprojectionDistanceFrameNonStereo", kMaxReprojectionDistanceFrameNonStereo);
    read_number(params, "kMaxReprojectionDistanceMap", kMaxReprojectionDistanceMap);
    read_number(params, "kMaxReprojectionDistanceMapRgbd", kMaxReprojectionDistanceMapRgbd);
    read_number(params, "kMaxReprojectionDistanceMapReloc", kMaxReprojectionDistanceMapReloc);
    read_number(params, "kMaxReprojectionDistanceFuse", kMaxReprojectionDistanceFuse);
    read_number(params, "kMaxReprojectionDistanceSim3", kMaxReprojectionDistanceSim3);
    read_number(params, "kMatchRatioTestFrameByProjection", kMatchRatioTestFrameByProjection);
    read_number(params, "kMatchRatioTestMap", kMatchRatioTestMap);
    read_number(params, "kMinDistanceFromEpipole", kMinDistanceFromEpipole);
    read_attr(params, "kCheckFeaturesOrientation", kCheckFeaturesOrientation);
    read_number(params, "kLocalBAWindowSize", kLocalBAWindowSize);
    read_number(params, "kCosMaxParallax", kCosMaxParallax);
    read_number(params, "kMinRatioBaselineDepth", kMinRatioBaselineDepth);
    read_number(params, "kLoopClosingMaxReprojectionDistanceFuse", kLoopClosingMaxReprojectionDistanceFuse);
    read_number(params, "kKeyframeCullingRedundantObsRatio", kKeyframeCullingRedundantObsRatio);
    read_number(params, "kKeyframeMaxTimeDistanceInSecForCulling", kKeyframeMaxTimeDistanceInSecForCulling);
    read_number(params, "kKeyframeCullingMinNumPoints", kKeyframeCullingMinNumPoints);
    read_number(params, "kLocalMappingParallelFusePointsNumWorkers", kLocalMappingParallelFusePointsNumWorkers);
    read_number(params, "kLocalMappingNumNeighborKeyFramesStereo", kLocalMappingNumNeighborKeyFramesStereo);
    read_number(params, "kLocalMappingNumNeighborKeyFramesMonocular", kLocalMappingNumNeighborKeyFramesMonocular);
    // optimizer_g2o.py: thHuberMono = sqrt(5.991), thHuberStereo = sqrt(7.815)
    kThHuberMono = std::sqrt(static_cast<double>(kChi2Mono));
    kThHuberStereo = std::sqrt(static_cast<double>(kChi2Stereo));
    // module-level constants of the Python pipeline
    {
        const py::object map = py::module_::import("pyslam.slam.map");
        read_number(map, "kMaxLenFrameDeque", kMaxLenFrameDeque);
    }
    {
        const py::object tracking_core = py::module_::import("pyslam.slam.tracking_core");
        read_number(tracking_core, "kRansacThresholdNormalized", kRansacThresholdNormalized);
        read_number(tracking_core, "kRansacProb", kRansacProb);
        read_number(tracking_core, "kNumMinInliersEssentialMat", kNumMinInliersEssentialMat);
        read_number(tracking_core, "kRansacReprojThreshold", kRansacReprojThreshold);
        read_number(tracking_core, "kRansacMinNumInliers", kRansacMinNumInliers);
    }
    {
        const py::object local_mapping_core = py::module_::import("pyslam.slam.local_mapping_core");
        read_number(local_mapping_core, "kNumMinObsForKeyFrameTrackedPoints", kNumMinObsForKeyFrameTrackedPoints);
    }
    if (py::hasattr(params, "kMinDepth") &&
        std::abs(py::cast<double>(params.attr("kMinDepth")) - kMinDepth) > 1e-12) {
        py::module_::import("warnings")
            .attr("warn")("C++ pipeline: Parameters.kMinDepth is fixed when cpp_core and camera_models "
                          "are built, and keeps its built-in value instead of the Python one");
    }
}
#endif

} // namespace pyslam
