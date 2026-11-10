/*
 * This file is part of PYSLAM
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

// gtsam::Values::at<Point3> in a module other than the one that inserted the value (macOS).
//
// Python's gtsam stores a numpy point as GenericValue<Eigen::VectorXd>, and Values::at looks it
// up with dynamic_cast. GTSAM and the pybind11 modules are built with hidden visibility, so
// GenericValue<Eigen::...> is a hidden type, and every module has its own copy of its RTTI.
// On Linux libstdc++ compares RTTI by name, so the cast works across modules; on macOS arm64
// libc++ compares hidden RTTI by address, so it fails, and a factor of this module could not
// read a point that Python had inserted ("type stored in Values is
// gtsam::GenericValue<Eigen::Matrix<double, -1, 1, ...>> but requested type was ...").
//
// This specialisation of gtsam's own lookup for Point3 (what the factors read) compares the
// names when the cast fails, as libstdc++ does. Include this header before anything that reads
// a Point3 from gtsam::Values.

#include <gtsam/nonlinear/Values.h>

#if defined(__APPLE__)

#include <cstring>
#include <typeinfo>

namespace gtsam {
namespace internal {

namespace cross_module {
template <typename T> const GenericValue<T> *generic_value_cast(const Value *value) {
    if (auto cast = dynamic_cast<const GenericValue<T> *>(value)) {
        return cast;
    }
    // the same type, with the RTTI of another module
    if (std::strcmp(typeid(*value).name(), typeid(GenericValue<T>).name()) == 0) {
        return static_cast<const GenericValue<T> *>(value);
    }
    return nullptr;
}
} // namespace cross_module

// Point3: Values::at<Point3> ends here. Like gtsam's handle_matrix for fixed sizes, it accepts
// a stored Vector3, MatrixXd or VectorXd of the right size. (handle_matrix<VectorXd> itself can't
// be specialised here: Values-inl.h already instantiates it.)
template <> struct handle<Eigen::Vector3d> {
    inline Eigen::Vector3d operator()(Key j, const Value *const pointer) {
        if (auto value = cross_module::generic_value_cast<Eigen::Vector3d>(pointer)) {
            return value->value();
        }
        Matrix A;
        if (auto value = cross_module::generic_value_cast<Eigen::VectorXd>(pointer)) {
            A = value->value();
        } else if (auto value = cross_module::generic_value_cast<Eigen::MatrixXd>(pointer)) {
            A = value->value();
        } else {
            throw ValuesIncorrectType(j, typeid(*pointer), typeid(Eigen::Vector3d));
        }
        if (A.rows() != 3 || A.cols() != 1) {
            throw NoMatchFoundForFixed(3, 1, A.rows(), A.cols());
        }
        return A;
    }
};

} // namespace internal
} // namespace gtsam

#endif // __APPLE__
