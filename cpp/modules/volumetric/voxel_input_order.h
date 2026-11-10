/**
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

#include <algorithm>

#ifdef TBB_FOUND
#include <tbb/parallel_for_each.h>
#endif

namespace volumetric {

// Sorts the points of each voxel (or block) group by their input index.
// The parallel grouping appends points in thread-scheduling order; integrating them in input
// order makes the result deterministic (semantic voting depends on it) and matches the serial
// path. GroupsMap maps a key to a std::vector of structs with a size_t `index` member.
// No concurrent insertion into `groups` may happen while this runs.
template <typename GroupsMap> void sort_voxel_groups_by_input_order(GroupsMap &groups) {
    auto sort_group = [](auto &pair) {
        auto &points = pair.second;
        if (points.size() < 2) {
            return;
        }
        const auto by_index = [](const auto &a, const auto &b) { return a.index < b.index; };
        if (!std::is_sorted(points.begin(), points.end(), by_index)) {
            std::sort(points.begin(), points.end(), by_index);
        }
    };
#ifdef TBB_FOUND
    tbb::parallel_for_each(groups.begin(), groups.end(), sort_group);
#else
    for (auto &pair : groups) {
        sort_group(pair);
    }
#endif
}

} // namespace volumetric
