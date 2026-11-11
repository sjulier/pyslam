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

#pragma once
#include <atomic>

#include <array>
#include <cstddef>
#include <string_view>

namespace pyslam {

class Parameters {
  public:
    // ----------------------------------------------------------------------------
    // Constants
    // Slam init copies every field from the Python Parameters (pyslam/config_parameters.py, or
    // the module-level constant the Python pipeline uses), so Python holds the values; those here
    // are the defaults for C++ without Python. Two stay compile-time constants: kMinDepth (the
    // camera_models module uses it too) and kStereoMatchingShowMatchedPoints (debug windows).
    // ----------------------------------------------------------------------------

    static float kChi2Mono; // chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
    static float kChi2Stereo; // chi-square 3 DOFs, used for reprojection error  (Hartley Zisserman pg 119)

    static double kThHuberMono; // sqrt(kChi2Mono), as in optimizer_g2o.py
    static double kThHuberStereo; // sqrt(kChi2Stereo), as in optimizer_g2o.py

    static int kSparseImageColorPatchDelta; // centre +- delta
    static std::size_t kMinWeightForDrawingCovisibilityEdge;
    static std::size_t kMaxSparseMapPointsToVisualize; // Sparse pointcloud downsampling for very
                                                       // large clouds to reduce queue bandwidth and GL load

    static constexpr double kMinDepth = 1e-2; // [meters] Minimum depth for a map point

    // Threshold constant (equivalent to Parameters.kMinNumOfCovisiblePointsForCreatingConnection =
    // 15)
    static int kMinNumOfCovisiblePointsForCreatingConnection;

    static int kLargeBAWindowSize;
    static std::size_t kMaxLenFrameDeque;
    static int kMaxNumOfKeyframesInLocalMap;
    static int kNumBestCovisibilityKeyFrames;

    static double kMaxOutliersRatioInPoseOptimization;

    // Tracking: Point visibility
    static float kViewingCosLimitForPoint; // Must be viewing cos < kViewingCosLimitForPoint (viewing angle must be less than 60
    // deg)
    static double kScaleConsistencyFactor;
    static double kMaxDistanceToleranceFactor;
    static double kMinDistanceToleranceFactor;

    // Tracking: Constants
    static double kRansacThresholdNormalized; // 0.0003 // metric threshold used for normalized image coordinates
    static double kRansacProb;
    static int kNumMinInliersEssentialMat;
    // static constexpr int kNumMinInliersPoseOptimizationTrackFrame = 10;
    // static constexpr int kNumMinInliersPoseOptimizationTrackLocalMap = 20;
    // static constexpr int kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = 50;
    // static constexpr int kNumMinObsForKeyFrameDefault = 3;
    static int kNumMinObsForKeyFrameTrackedPoints;

    static double kRansacReprojThreshold;
    static int kRansacMinNumInliers;

    // Stereo Matching: Show matched points
    static constexpr bool kStereoMatchingShowMatchedPoints = false; // debug windows: true needs OpenCV highgui, which cpp_core does not link

    // Tracking: Visual Odometry
    static int kMaxNumVisualOdometryPoints;
    static int kMaxNumStereoPointsOnNewKeyframe;

    // Search matches by projection
    static float kMaxReprojectionDistanceFrame; // [pixels]    o:7
    static float kMaxReprojectionDistanceFrameNonStereo; // [pixels]    o:15
    static float kMaxReprojectionDistanceMap; // [pixels]    o:1
    static float kMaxReprojectionDistanceMapRgbd; // [pixels]    o:3
    static float kMaxReprojectionDistanceMapReloc; // [pixels]    o:5
    static float kMaxReprojectionDistanceFuse; // [pixels]    o:3
    static float kMaxReprojectionDistanceSim3; // [pixels]    o:7.5
    //
    static float kMatchRatioTestFrameByProjection;
    static float kMatchRatioTestMap;

    // Search matches for triangulation by using epipolar lines
    static float kMinDistanceFromEpipole; // [pixels] Used with search by epipolar lines

    static bool kCheckFeaturesOrientation;

    static int kLocalBAWindowSize; // as in config_parameters.py

    // Point triangulation
    static double kCosMaxParallax;
    static double kMinRatioBaselineDepth;

    // Loop closing
    static float kLoopClosingMaxReprojectionDistanceFuse; // [pixels]    o:4

    // Local mapping
    static float kKeyframeCullingRedundantObsRatio;
    static double kKeyframeMaxTimeDistanceInSecForCulling;
    static int kKeyframeCullingMinNumPoints;

    static int kLocalMappingParallelFusePointsNumWorkers;

    static int kLocalMappingNumNeighborKeyFramesStereo; //  [# frames]   for generating new points and fusing them under stereo or RGBD

    static int kLocalMappingNumNeighborKeyFramesMonocular; //  [# frames]   for generating new points and fusing them under monocular

    // ----------------------------------------------------------------------------
    // Modifiable parameters
    // ----------------------------------------------------------------------------

    static float kFeatureMatchDefaultRatioTest; // This is the default ratio test used by all
                                                // feature matchers. It can be configured per
                                                // descriptor in feature_tracker_configs.py

    static std::atomic<float> kMaxDescriptorDistance; // It is initialized by the first created instance of
                                         // feature_manager.py at runtime

    // Runtime copies of the Python Parameters flags the C++ tracker and local mapper read.
    // Slam construction copies the current Python values into these fields.
    static bool kLocalMappingOnSeparateThread;
    static bool kTrackingWaitForLocalMappingToGetIdle;
    static double kWaitForLocalMappingTimeout;
    static double kParallelLBAWaitIdleTimeout;
    static int kPlaybackThrottleWindow;               // KeyframeDemand, for main_slam.py's throttle
    static double kPlaybackThrottleWeakTrackingRatio;
    static double kLocalMappingTimeoutPopKeyframe;
    static bool kUseDynamicDesDistanceTh;
    static bool kUseDescriptorSigmaMadv2;
    static double kCosMaxParallaxInitializer;
    static double kInitializerDesiredMedianDepth;
    static double kInitializerMinRatioDepthBaseline;
    static int kInitializerNumMinFeatures;
    static int kInitializerNumMinFeaturesStereo;
    static int kInitializerNumMinTriangulatedPoints;
    static int kInitializerNumMinTriangulatedPointsStereo;
    static float kInitializerFeatureMatchRatioTest;
    static int kInitializerNumMinNumPointsForPnPWithDepth;
    static bool kInitializerUseCellCoverageCheck;
    static bool kInitializerUseMinFrameDistanceCheck;
    static bool kUseMotionModel;
    static bool kUseSearchFrameByProjection;
    static int kMinNumMatchedFeaturesSearchFrameByProjection;
    static bool kUseEssentialMatrixFitting;
    static int kMinNumMatchedFeaturesSearchReferenceFrame;
    static bool kUseVisualOdometryPoints;
    static bool kUseInterruptLocalMapping;
    static bool kUseMotionBlurDection;
    static double kMotionBlurDetectionLalacianVarianceThreshold;
    static int kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography;
    static int kNumMinPointsForNewKf;
    static int kNumMinTrackedClosePointsForNewKfNonMonocular;
    static int kNumMaxNonTrackedClosePointsForNewKfNonMonocular;
    static double kThNewKfRefRatioMonocular;
    static double kThNewKfRefRatioStereo;
    static double kThNewKfRefRatioNonMonocular;
    static bool kUseFeatureCoverageControlForNewKf;
    static bool kLocalMappingParallelKpsMatching;
    static int kLocalMappingParallelKpsMatchingNumWorkers;
    static bool kUseLargeWindowBA;
    static int kEveryNumFramesLargeWindowBA;
    static bool kUseParallelProcessLBA;
    static bool kOptimizationFrontEndUseGtsam;
    static bool kUseLoopClosing;
    static bool kDoVolumetricIntegration;
    static bool kDoSparseSemanticMappingAndSegmentation;
    static bool kSemanticMappingOnSeparateThread;

#ifdef USE_PYTHON
    // Copy the live Python Parameters values so a C++ Slam follows the same config.
    // It warns if the Python kMinDepth differs from the compile-time one.
    static void sync_runtime_from_python();
#endif
};

} // namespace pyslam
