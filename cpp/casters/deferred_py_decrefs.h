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

#include <Python.h>
#include <pybind11/pybind11.h>

#include <mutex>
#include <vector>

namespace pybind11 {
namespace detail {

// Python objects owned by C++ (the NumPy owners of released Mats, a keyframe's global descriptor)
// that a thread without the GIL could not DECREF itself.
// That thread must not wait for the GIL: it may hold a C++ lock that the thread holding the GIL
// is waiting for. For example, local BA (GIL released) drops a culled keyframe, and with it its
// descriptors, under the local map's lock, while tracking (GIL held) waits for that lock in
// local_map.is_empty(): a deadlock with every thread idle. Instead the owners are queued here,
// and the interpreter DECREFs them in its main thread, holding the GIL.
class deferred_py_decrefs {
  public:
    static deferred_py_decrefs &instance() {
        static auto *d = new deferred_py_decrefs(); // never destroyed: threads may still push at exit
        return *d;
    }

    // Needs neither the GIL nor a Python thread state.
    void push(PyObject *owner) {
        bool schedule = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            owners_.push_back(owner);
            if (!scheduled_) {
                scheduled_ = true;
                schedule = true;
            }
        }
        // Py_AddPendingCall needs neither the GIL nor a thread state either. It fails only when
        // the interpreter's queue is full: then the next push tries again.
        if (schedule && Py_AddPendingCall(&deferred_py_decrefs::drain_pending, nullptr) != 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            scheduled_ = false;
        }
    }

  private:
    // Runs in the main thread with the GIL held.
    static int drain_pending(void *) {
        auto &self = instance();
        std::vector<PyObject *> owners;
        {
            std::lock_guard<std::mutex> lock(self.mutex_);
            owners.swap(self.owners_);
            self.scheduled_ = false;
        }
        for (PyObject *owner : owners)
            Py_DECREF(owner);
        return 0;
    }

    std::mutex mutex_;
    std::vector<PyObject *> owners_;
    bool scheduled_ = false;
};

// Drop the reference that obj holds, from any thread: with the GIL now, otherwise later in the
// main thread (see deferred_py_decrefs). For Python objects owned by C++ objects that a C++
// thread without the GIL may free (KeyFrame::g_des, even when it is None: before Python 3.12 the
// count of references to None changes too).
inline void release_py_object_without_waiting_for_gil(object &obj) {
    if (!obj.ptr())
        return;
    if (!Py_IsInitialized()) {
        obj.release(); // the interpreter is gone: nothing to DECREF
        return;
    }
    if (PyGILState_Check()) {
        obj = object();
        return;
    }
    deferred_py_decrefs::instance().push(obj.release().ptr());
}

} // namespace detail
} // namespace pybind11
