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

// Containers of frames, keyframes and map points ordered or hashed by the object's id instead of
// its address. A pointer-keyed container iterates in an order set by where the allocator put the
// objects, which changes from run to run; everything downstream of that iteration (keyframe
// culling, fusion, the order of optimizer edges, ...) then changes too. Keyed by id, the same input
// gives the same output (on a given platform and standard library).
//
// Equality stays pointer equality, and objects with equal ids fall back to their address, so two
// distinct objects are never merged even if their ids collide (e.g. objects built in tests).

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "smart_pointers.h"

namespace pyslam {

// Strict weak order by id; null first. Works with shared_ptr and raw pointers to any type with
// an `id` member (the type only needs to be complete where the comparison is instantiated).
struct IdLess {
    template <typename P> bool operator()(const P &a, const P &b) const noexcept {
        if (!a || !b) {
            return !a && b;
        }
        if (a->id != b->id) {
            return a->id < b->id;
        }
        return std::less<const void *>{}(&*a, &*b);
    }
};

// Hash by id; null hashes to 0.
struct IdHash {
    template <typename P> std::size_t operator()(const P &p) const noexcept {
        return p ? std::hash<decltype(p->id)>{}(p->id) : 0;
    }
};

template <typename T> using IdSet = std::set<std::shared_ptr<T>, IdLess>;
template <typename K, typename V> using IdMap = std::map<std::shared_ptr<K>, V, IdLess>;
template <typename T> using IdHashSet = std::unordered_set<std::shared_ptr<T>, IdHash>;
template <typename K, typename V>
using IdHashMap = std::unordered_map<std::shared_ptr<K>, V, IdHash>;

} // namespace pyslam
