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

#include <deque>
#include <mutex>

namespace pyslam {

// What tracking records at each keyframe decision, for main_slam.py's playback throttle.
// The C++ version of pyslam/slam/playback_throttle.py's KeyframeDemand, with the same Python
// interface, so the throttle reads either pipeline the same way.
// - Weak tracking: the frame matched fewer than weak_ratio of the map points its reference keyframe
//   tracks. The fraction of weak frames among the recent ones is the throttle's signal.
// - Suppressed keyframe requests: a keyframe would have been requested if local mapping were idle.
//   Only reported at the end of a run.
// Tracking writes it and the main thread reads it, so every access takes the lock.
class KeyframeDemand {
  public:
    KeyframeDemand(int window, double weak_ratio)
        : weak_ratio(weak_ratio), window_(window > 0 ? static_cast<size_t>(window) : 1) {}

    // num_ref_tracked_points <= 0: no reference to compare with, the frame is not sampled
    void record(bool wanted_if_idle, bool is_local_mapping_idle, int num_tracked_points,
                int num_ref_tracked_points) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (wanted_if_idle) {
            ++num_wanted_;
            num_suppressed_ += is_local_mapping_idle ? 0 : 1;
        }
        if (num_ref_tracked_points > 0) {
            const bool weak = num_tracked_points < weak_ratio * num_ref_tracked_points;
            weak_.push_back(weak);
            if (weak_.size() > window_) {
                weak_.pop_front();
            }
            ++num_frames_;
            num_weak_ += weak ? 1 : 0;
        }
    }

    int num_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(weak_.size());
    }

    // forget the recent samples (the totals are kept): they were taken at another playback speed
    void reset_window() {
        std::lock_guard<std::mutex> lock(mutex_);
        weak_.clear();
    }

    // fraction of the frames in the window in which tracking was weak (0 with no samples)
    double weak_fraction() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (weak_.empty()) {
            return 0.0;
        }
        size_t n = 0;
        for (bool w : weak_) {
            n += w ? 1 : 0;
        }
        return static_cast<double>(n) / static_cast<double>(weak_.size());
    }

    double total_weak_fraction() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_frames_ > 0 ? static_cast<double>(num_weak_) / num_frames_ : 0.0;
    }

    double total_suppressed_fraction() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_wanted_ > 0 ? static_cast<double>(num_suppressed_) / num_wanted_ : 0.0;
    }

    int num_frames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_frames_;
    }
    int num_weak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_weak_;
    }
    int num_wanted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_wanted_;
    }
    int num_suppressed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_suppressed_;
    }

    const double weak_ratio;

  private:
    mutable std::mutex mutex_;
    size_t window_;
    std::deque<bool> weak_; // for each recent frame: true if tracking was weak
    int num_frames_ = 0;
    int num_weak_ = 0;
    int num_wanted_ = 0;
    int num_suppressed_ = 0;
};

} // namespace pyslam
