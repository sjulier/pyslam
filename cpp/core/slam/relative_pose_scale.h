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

// The scale of the frames' relative poses in the tracking history: the C++ version of
// pyslam/slam/relative_pose_scale.py (#93), which explains it. In short: monocular SLAM changes the
// map's scale after a frame is tracked (loop corrections, BA); the keyframes follow, the stored
// relative translations do not. Tracking therefore also stores the positions of a few keyframes
// around the reference keyframe ("anchors"), and the final trajectory scales the relative
// translation by the anchors' scale change since then. Monocular only.

#include "keyframe.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <vector>

namespace pyslam {

class RelativePoseAnchors {
  public:
    static constexpr int kNumAnchorCovisibleKeyFrames = 8; // + the reference keyframe and its parent
    static constexpr double kMinAnchorSpread = 1e-9;       // [map units^2] below it, no scale
    static constexpr size_t kMinAnchorsForScale = 3;        // with two, the ratio is too noisy

    RelativePoseAnchors(const KeyFramePtr &kf_ref, const std::shared_ptr<const RelativePoseAnchors> &previous) {
        std::vector<KeyFramePtr> candidates;
        if (KeyFramePtr parent = kf_ref->get_parent()) {
            candidates.push_back(parent);
        }
        for (const auto &kf : kf_ref->get_best_covisible_keyframes(kNumAnchorCovisibleKeyFrames)) {
            candidates.push_back(kf);
        }
        if (candidates.size() < 2 && previous) {
            // a keyframe that local mapping has not connected yet: the previous frame's anchors are
            // around the same place
            candidates.insert(candidates.end(), previous->keyframes.begin(), previous->keyframes.end());
        }
        keyframes.push_back(kf_ref);
        std::unordered_set<int> kids{kf_ref->kid};
        for (const auto &kf : candidates) {
            if (kf && !kf->is_bad() && kids.insert(kf->kid).second) {
                keyframes.push_back(kf);
            }
        }
        positions.reserve(keyframes.size());
        for (const auto &kf : keyframes) {
            positions.push_back(kf->Ow());
        }
    }

    // the local scale change since the frame was tracked (1.0 if the anchors cannot tell)
    double scale() const {
        std::vector<Eigen::Vector3d> then, now;
        for (size_t i = 0; i < keyframes.size(); ++i) {
            if (!keyframes[i]->is_bad()) {
                then.push_back(positions[i]);
                now.push_back(keyframes[i]->Ow());
            }
        }
        return scale_change(then, now);
    }

    // sqrt(spread now / spread then) of matched point sets (1.0 if they cannot tell)
    static double scale_change(const std::vector<Eigen::Vector3d> &then,
                               const std::vector<Eigen::Vector3d> &now) {
        if (then.size() < kMinAnchorsForScale || then.size() != now.size()) {
            return 1.0;
        }
        const auto spread = [](const std::vector<Eigen::Vector3d> &points) {
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            for (const auto &p : points) {
                mean += p;
            }
            mean /= static_cast<double>(points.size());
            double sum = 0.0;
            for (const auto &p : points) {
                sum += (p - mean).squaredNorm();
            }
            return sum;
        };
        const double spread_then = spread(then);
        if (spread_then < kMinAnchorSpread) {
            return 1.0;
        }
        return std::sqrt(spread(now) / spread_then);
    }

    std::vector<KeyFramePtr> keyframes;
    std::vector<Eigen::Vector3d> positions; // the keyframes' positions when the frame was tracked
};

using RelativePoseAnchorsPtr = std::shared_ptr<const RelativePoseAnchors>;

} // namespace pyslam
