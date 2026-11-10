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

// #define GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR  // before including gtsam


#include "values_cross_module.h"  // first: gtsam::Values lookups across modules on macOS

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <memory>

#include <algorithm>
#include <iostream>

using namespace gtsam;
using symbol_shorthand::X;

#ifndef SIM3_FACTOR_REVERSE_ERROR_DIRECTION
#define SIM3_FACTOR_REVERSE_ERROR_DIRECTION 0
#endif

namespace gtsam_factors {

// Right Jacobian of Sim(3) at xi (tangent order [w, u, lambda], as in gtsam::Similarity3):
//   Exp(xi + d) ~= Exp(xi) * Exp(J_r(xi) d),   J_r(xi) = sum_k (-ad_xi)^k / (k+1)!
// The series is the top-right block of expm([[-ad_xi, I], [0, 0]]), evaluated exactly by Eigen's
// Pade matrix exponential. Uniformly scaling translations is an automorphism of Sim(3): with
// D = diag(I3, k I3, 1), J_r(D xi) = D J_r(xi) D^-1. J_r is therefore evaluated at unit translation
// and scaled back, which keeps expm well conditioned for any translation magnitude (arbitrary
// monocular scale). No finite-difference step is involved.
inline gtsam::Matrix7 Sim3RightJacobian(const gtsam::Vector7 &xi) {
    const double k = 1.0 / std::max(1.0, xi.segment<3>(3).norm());
    gtsam::Vector7 xi_scaled = xi;
    xi_scaled.segment<3>(3) *= k;

    Eigen::Matrix<double, 14, 14> B = Eigen::Matrix<double, 14, 14>::Zero();
    B.topLeftCorner<7, 7>() = -gtsam::Similarity3::adjointMap(xi_scaled);
    B.topRightCorner<7, 7>().setIdentity();
    gtsam::Matrix7 J = B.exp().topRightCorner<7, 7>(); // = D J_r(xi) D^-1

    J.block<3, 7>(3, 0) /= k; // D^-1 (.)
    J.block<7, 3>(0, 3) *= k; // (.) D
    return J;
}

// Inverse right Jacobian of Sim(3): d Log(g Exp(d)) / d d at d = 0, for xi = Log(g).
inline gtsam::Matrix7 Sim3LogmapDerivative(const gtsam::Vector7 &xi) {
    return Sim3RightJacobian(xi).inverse();
}

// Similarity3 prior factor. Goal is to penalize all terms.
// Error e(x) = Log(prior^-1 * x). With GTSAM's right retraction x * Exp(d), the Jacobian is exactly
// J_r^-1(e(x)) (Sim3LogmapDerivative), computed analytically.
// NOTE: GTSAM (4.3) also provides PriorFactor<Similarity3> (gtsam.PriorFactorSimilarity3) with the
// same error, but Similarity3's chart has no Local() Jacobian, so its prior uses the identity as the
// Jacobian (exact only at x == prior).
class PriorFactorSimilarity3 : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const gtsam::Similarity3 prior_inverse_;

    PriorFactorSimilarity3(gtsam::Key key, const gtsam::Similarity3 &prior,
                           const gtsam::SharedNoiseModel &noise)
        : Base(noise, key), prior_inverse_(prior.inverse()) {}

    // Define error function: Logmap of transformation error
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim,
                                gtsam::OptionalMatrixType H = OptionalNone) const override {
        gtsam::Vector7 error = gtsam::Similarity3::Logmap(prior_inverse_ * sim);

        if (H) {
            *H = Sim3LogmapDerivative(error);
        }
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<PriorFactorSimilarity3>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef std::shared_ptr<PriorFactorSimilarity3> shared_ptr;
};

// Similarity3 prior factor with only scale. Goal is to only penalize scale difference.
class PriorFactorSimilarity3ScaleOnly : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const double prior_scale_; // Prior scale value

    PriorFactorSimilarity3ScaleOnly(gtsam::Key key, double prior_scale, double prior_sigma)
        : Base(gtsam::noiseModel::Isotropic::Sigma(1, prior_sigma), key),
          prior_scale_(prior_scale) {}

    // Define error function: Only penalize scale difference
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim,
                                gtsam::OptionalMatrixType H = OptionalNone) const override {
#define USE_LOG_SCALE 1
#if USE_LOG_SCALE
        double scale_error = std::log(sim.scale()) - std::log(prior_scale_); // Log-scale difference
#else
        double scale_error = sim.scale() - prior_scale_;
#endif

        if (H) {
            // Compute Jacobian: d(log(s)) / ds = 1 / s
            Eigen::Matrix<double, 1, 7> J = Eigen::Matrix<double, 1, 7>::Zero();
#if USE_LOG_SCALE
            J(6) = 1.0 / sim.scale(); // Derivative w.r.t. scale
#else
            J(6) = 1.0; // Derivative w.r.t. scale
#endif
            *H = J;
        }
        return gtsam::Vector1(scale_error); // Return 1D error
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<PriorFactorSimilarity3ScaleOnly>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef std::shared_ptr<PriorFactorSimilarity3ScaleOnly> shared_ptr;
};

// =====================================================================================================================

// Custom version of BetweenFactor<Similarity3> with analytic Jacobians
// Assuming 
// sim3_1 = Swc1 
// sim3_2 = Swc2
// measured = Sc1c2
class BetweenFactorSimilarity3
    : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3(gtsam::Key key1, gtsam::Key key2, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key1, key2), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured) 
#else
        measured_inverse_(measured.inverse())
#endif
        {}

    // Compute error (7D residual). Jacobians are analytic (see Sim3LogmapDerivative); with right
    // perturbations S * Exp(d):  g * Exp(d) -> J,  and  Exp(d) inside g -> J * Ad(.).
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_1, const gtsam::Similarity3 &sim3_2,
                                gtsam::OptionalMatrixType H1 = OptionalNone,
                                gtsam::OptionalMatrixType H2 = OptionalNone) const override {
        const gtsam::Similarity3 predicted = sim3_1.inverse() * sim3_2; // Swc1^-1 * Swc2 = Sc1c2
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        const gtsam::Vector7 error = Similarity3::Logmap(measured_ * predicted.inverse());
        if (H1 || H2) {
            const gtsam::Matrix7 J = Sim3LogmapDerivative(error);
            if (H1) *H1 = J;
            if (H2) *H2 = -J * predicted.AdjointMap();
        }
#else
        const gtsam::Vector7 error = Similarity3::Logmap(measured_inverse_ * predicted);
        if (H1 || H2) {
            const gtsam::Matrix7 J = Sim3LogmapDerivative(error);
            if (H1) *H1 = -J * predicted.inverse().AdjointMap();
            if (H2) *H2 = J;
        }
#endif
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<BetweenFactorSimilarity3>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef std::shared_ptr<BetweenFactorSimilarity3> shared_ptr;
};


// =====================================================================================================================

// Custom version of BetweenFactor<Similarity3> with analytic Jacobians for inverse error
// Assuming:
// sim3_1 = Sc1w
// sim3_2 = Sc2w
// measured = Sc1c2
class BetweenFactorSimilarity3Inverse
    : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3Inverse(gtsam::Key key1, gtsam::Key key2, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key1, key2), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured) 
#else
        measured_inverse_(measured.inverse())
#endif
        {}

    // Compute error (7D residual), with analytic Jacobians (see BetweenFactorSimilarity3)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_1, const gtsam::Similarity3 &sim3_2,
                                gtsam::OptionalMatrixType H1 = OptionalNone,
                                gtsam::OptionalMatrixType H2 = OptionalNone) const override {
        const gtsam::Similarity3 predicted = sim3_1 * sim3_2.inverse(); // Sc1w * Sc2w^-1 = Sc1c2
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        // use predicted.inverse() * measured_ to match backward error semantics
        const gtsam::Vector7 error = Similarity3::Logmap(predicted.inverse() * measured_);
        if (H1 || H2) {
            const gtsam::Matrix7 JAd = Sim3LogmapDerivative(error) * (measured_.inverse() * sim3_1).AdjointMap();
            if (H1) *H1 = -JAd;
            if (H2) *H2 = JAd;
        }
#else
        const gtsam::Vector7 error = Similarity3::Logmap(measured_inverse_ * predicted);
        if (H1 || H2) {
            const gtsam::Matrix7 JAd = Sim3LogmapDerivative(error) * sim3_2.AdjointMap();
            if (H1) *H1 = JAd;
            if (H2) *H2 = -JAd;
        }
#endif
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<BetweenFactorSimilarity3Inverse>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef std::shared_ptr<BetweenFactorSimilarity3Inverse> shared_ptr;
};



// Custom version of BetweenFactor<Similarity3> with analytic Jacobians for inverse error. Only s1 is optimized
// Assuming:
// sim3_1 = Sc1w
// sim3_2 = Sc2w  FIXED
// measured = Sc1c2
class BetweenFactorSimilarity3InverseOnlyS1
    : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const gtsam::Similarity3 sim3_2_inverse_;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3InverseOnlyS1(gtsam::Key key_sim3_1, const gtsam::Similarity3 &sim3_2, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key_sim3_1), sim3_2_inverse_(sim3_2.inverse()), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured)
#else
        measured_inverse_(measured.inverse())
#endif
    {}

    // Compute error (7D residual), with analytic Jacobian (see BetweenFactorSimilarity3)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_1,
                                gtsam::OptionalMatrixType H = OptionalNone) const override {
        const gtsam::Similarity3 predicted = sim3_1 * sim3_2_inverse_; // Sc1w * Sc2w^-1 = Sc1c2
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        const gtsam::Vector7 error = Similarity3::Logmap(predicted.inverse() * measured_);
        if (H) *H = -Sim3LogmapDerivative(error) * (measured_.inverse() * sim3_1).AdjointMap();
#else
        const gtsam::Vector7 error = Similarity3::Logmap(measured_inverse_ * predicted);
        if (H) *H = Sim3LogmapDerivative(error) * sim3_2_inverse_.inverse().AdjointMap();
#endif
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<BetweenFactorSimilarity3InverseOnlyS1>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef std::shared_ptr<BetweenFactorSimilarity3InverseOnlyS1> shared_ptr;
};




// Custom version of BetweenFactor<Similarity3> with analytic Jacobians for inverse error. Only s2 is optimized
// Assuming:
// sim3_1 = Sc1w FIXED
// sim3_2 = Sc2w  
// measured = Sc1c2
class BetweenFactorSimilarity3InverseOnlyS2
    : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const gtsam::Similarity3 sim3_1_;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3InverseOnlyS2(gtsam::Key key_sim3_2, const gtsam::Similarity3 &sim3_1, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key_sim3_2), sim3_1_(sim3_1), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured)
#else
        measured_inverse_(measured.inverse())
#endif
    {}

    // Compute error (7D residual), with analytic Jacobian (see BetweenFactorSimilarity3)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_2,
                                gtsam::OptionalMatrixType H = OptionalNone) const override {
        const gtsam::Similarity3 predicted = sim3_1_ * sim3_2.inverse(); // Sc1w * Sc2w^-1 = Sc1c2
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        const gtsam::Vector7 error = Similarity3::Logmap(predicted.inverse() * measured_);
        if (H) *H = Sim3LogmapDerivative(error) * (measured_.inverse() * sim3_1_).AdjointMap();
#else
        const gtsam::Vector7 error = Similarity3::Logmap(measured_inverse_ * predicted);
        if (H) *H = -Sim3LogmapDerivative(error) * sim3_2.AdjointMap();
#endif
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::make_shared<BetweenFactorSimilarity3InverseOnlyS2>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef std::shared_ptr<BetweenFactorSimilarity3InverseOnlyS2> shared_ptr;
};

} // namespace gtsam_factors
