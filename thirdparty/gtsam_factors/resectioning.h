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



#include "values_cross_module.h"  // first: gtsam::Values lookups across modules on macOS

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Similarity3.h>

#include <memory>

using namespace gtsam;
using namespace gtsam::noiseModel;
using symbol_shorthand::X;



namespace gtsam_factors {

/**
 * Monocular Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactor : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    const gtsam::Cal3_S2& K_;
    gtsam::Point3 P_;  // world point
    gtsam::Point2 p_;  // pixel point 
    double weight_ = 1.0;

    ResectioningFactor(const gtsam::SharedNoiseModel& model, 
                        const gtsam::Key& key,
                        const gtsam::Cal3_S2& calib,
                        const gtsam::Point2& measured_p,
                        const gtsam::Point3& world_P)
        : Base(model, key), K_(calib), P_(world_P), p_(measured_p) {}

    Vector evaluateError(const Pose3& pose, gtsam::OptionalMatrixType H = OptionalNone) const override {
        gtsam::PinholeCamera<gtsam::Cal3_S2> camera(pose, K_);        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(2,6);
                return gtsam::Vector::Zero(2);
            } else {
                const gtsam::Point2 delta = camera.project(P_, H, {}, {}) - p_;
                const gtsam::Vector error(delta);
                if (H) *H *= weight_;
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactor]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector::Zero(2);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 


/**
 * Monocular Resectioning Factor.
 * The pose of the camera is assumed to be Tcw.
 */
class ResectioningFactorTcw : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    double fx_, fy_, cx_, cy_;            
    gtsam::Point3 P_;  // world point
    gtsam::Point2 p_;  // pixel point 
    double weight_ = 1.0;    

    ResectioningFactorTcw(const gtsam::SharedNoiseModel& model, 
                        const gtsam::Key& key,
                        const gtsam::Cal3_S2& calib,
                        const gtsam::Point2& measured_p,
                        const gtsam::Point3& world_P)
        : Base(model, key), fx_(calib.fx()), fy_(calib.fy()), cx_(calib.px()), cy_(calib.py()), 
            P_(world_P), p_(measured_p) {}

    Vector evaluateError(const gtsam::Pose3& Tcw, gtsam::OptionalMatrixType H = OptionalNone) const override {
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(2,6);
                return gtsam::Vector::Zero(2);
            }

            const gtsam::Rot3& Rcw = Tcw.rotation();
            const gtsam::Point3& tcw = Tcw.translation();
            const gtsam::Matrix3 R = Rcw.matrix();
            const gtsam::Vector3 Pc = R * P_ + tcw;

            const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
            const double Zinv = 1.0 / Z;
            const double Zinv2 = Zinv * Zinv;

            // Projection
            const double u = fx_ * X * Zinv + cx_;
            const double v = fy_ * Y * Zinv + cy_;
            gtsam::Vector2 error;
            error << u - p_.x(), v - p_.y();
            error *= weight_;

            if (H) {
                // d(projected)/dPc
                gtsam::Matrix23 J_proj;
                J_proj << fx_ * Zinv,         0.0, -fx_ * X * Zinv2,
                            0.0,    fy_ * Zinv, -fy_ * Y * Zinv2;

                // dPc/dTcw
                gtsam::Matrix36 J_pose;
                J_pose.leftCols<3>() = -R * gtsam::skewSymmetric(P_); // wrt rotation
                J_pose.rightCols<3>() = R;                                     // wrt translation

                *H = weight_ * (J_proj * J_pose);  // Chain rule
            }

            return error;

        } catch( std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(2,6);
            std::cerr << "\x1b[31m [ResectioningFactorTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector::Zero(2);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 


/**
 * Stereo Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactorStereo : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    gtsam::Cal3_S2Stereo::shared_ptr K_;    
    gtsam::Point3 P_;
    gtsam::StereoPoint2 p_stereo_;  // pixel point (uL, uR, vL)
    double weight_ = 1.0;    

    ResectioningFactorStereo(const SharedNoiseModel& model, 
                                const Key& key,
                                const Cal3_S2Stereo& calib,
                                const StereoPoint2& measured_p_stereo,
                                const Point3& world_P)
        : Base(model, key), /*K_(calib),*/ P_(world_P), p_stereo_(measured_p_stereo) {
            K_ = std::make_shared<Cal3_S2Stereo>(calib);
        }

    Vector evaluateError(const Pose3& pose, gtsam::OptionalMatrixType H = OptionalNone) const override {
        StereoCamera camera(pose, K_);
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(3,6);
                return Vector::Zero(3);
            } else {
                const StereoPoint2 delta = camera.project(P_, H, {}, {}) - p_stereo_;
                const Vector error = delta.vector();
                if (H) *H *= weight_;
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(3,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorStereo]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(3);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }    
};


/**
 * Stereo Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactorStereoTcw : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    double fx_, fy_, cx_, cy_, bf_;            
    gtsam::Point3 P_;               // world point
    gtsam::StereoPoint2 p_stereo_;  // pixel point (uL, uR, vL)
    double weight_ = 1.0;    

    ResectioningFactorStereoTcw(const gtsam::SharedNoiseModel& model, 
                        const gtsam::Key& key,
                        const gtsam::Cal3_S2Stereo& calib,
                        const gtsam::StereoPoint2& measured_p_stereo,
                        const gtsam::Point3& world_P)
        : Base(model, key), fx_(calib.fx()), fy_(calib.fy()), cx_(calib.px()), cy_(calib.py()), 
          bf_(calib.baseline()*calib.fx()), P_(world_P), p_stereo_(measured_p_stereo) {}

    Vector evaluateError(const gtsam::Pose3& Tcw, gtsam::OptionalMatrixType H = OptionalNone) const override {
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(3,6);
                return gtsam::Vector3::Zero();
            }

            const gtsam::Rot3& Rcw = Tcw.rotation();
            const gtsam::Point3& tcw = Tcw.translation();
            const gtsam::Matrix3 R = Rcw.matrix();
            const gtsam::Vector3 Pc = R * P_ + tcw;

            const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
            const double invZ = 1.0 / Z;
            const double invZ2 = invZ * invZ;

            // Projection
            const double uL = fx_ * X * invZ + cx_;
            const double uR = uL - bf_ * invZ;
            const double vL = fy_ * Y * invZ + cy_;
            gtsam::Vector3 error;
            error << uL - p_stereo_.uL(), uR - p_stereo_.uR(), vL - p_stereo_.v();
            error *= weight_;

            if (H) {
                // d(cam projection) / d(Pc)
                gtsam::Matrix33 J_proj;
                J_proj <<
                    fx_ * invZ,       0.0, -fx_ * X * invZ2,
                    fx_ * invZ,       0.0, -fx_ * X * invZ2 + bf_ * invZ2,  // d(uL - bf/Z)/dZ
                        0.0, fy_ * invZ, -fy_ * Y * invZ2;

                // d(Pc) / d(Tcw)
                gtsam::Matrix36 J_pose;
                J_pose.leftCols<3>()  = -R * gtsam::skewSymmetric(P_);  // Rotation
                J_pose.rightCols<3>() = R;                              // Translation

                *H = weight_ * (J_proj * J_pose);  // Chain rule
            }

            return error;

        } catch (const std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(3,6);
            std::cerr << "\x1b[31m [ResectioningFactorStereoTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector3::Zero();
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 
        
    

// =====================================================================================================================


// Jacobian of the perspective division uv = p.head<2>() / p[2] w.r.t. p (homogeneous image point).
inline gtsam::Matrix23 projectionJacobian(const gtsam::Vector3 &p) {
    const double z_inv = 1.0 / p[2];
    gtsam::Matrix23 J;
    J << z_inv, 0.0, -p[0] * z_inv * z_inv,
         0.0, z_inv, -p[1] * z_inv * z_inv;
    return J;
}

// Used by optimizer_gtsam.optimize_sim3()
class SimResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
    private:
        const Cal3_S2 calib_;   // Camera intrinsics (stored by value)
        const Point2 uv_;       // Observed 2D point
        const Point3 P_;        // 3D point
        double weight_ = 1.0;   // Weight
    
    public:
        SimResectioningFactor(const gtsam::Key& sim_pose_key,
                                const gtsam::Cal3_S2& calib,
                                const gtsam::Point2& uv,
                                const gtsam::Point3& P,
                                const gtsam::SharedNoiseModel& noiseModel)
            : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
                calib_(calib), uv_(uv), P_(P) {}
    
        gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                    gtsam::OptionalMatrixType H = OptionalNone) const override {
            // Q = s * R * P + t, projected with K.
            // With the right perturbation sim * Exp([w, u, lambda]): dR = R [w]x, dt = R u - t lambda,
            // ds = s lambda, so dQ/d(w, u, lambda) = [-s R [P]x, R, s R P - t] (analytic Jacobian).
            const gtsam::Matrix3 R = sim3.rotation().matrix();
            const gtsam::Vector3 t = sim3.translation();
            const double s = sim3.scale();
            const gtsam::Vector3 sRP = s * (R * P_);
            const gtsam::Vector3 Q = sRP + t;
            const gtsam::Matrix3 K = calib_.K();
            const gtsam::Vector3 projected = K * Q;
            const gtsam::Point2 uv = projected.head<2>() / projected[2];
            const gtsam::Vector2 error = weight_ * (uv - uv_);

            if (H) {
                gtsam::Matrix37 dQ;
                dQ << -s * R * gtsam::skewSymmetric(P_), R, sRP - t;
                *H = weight_ * (projectionJacobian(projected) * K * dQ);
            }
            return error;
        }
    
        void setWeight(double weight) {
            weight_ = weight;
        }
    
        double getWeight() const {
            return weight_;
        }
    
        virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
            return std::make_shared<SimResectioningFactor>(*this);
        }    
    };
    
    
    // Used by optimizer_gtsam.optimize_sim3()
    class SimInvResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
    private:
        const Cal3_S2 calib_;   // Camera intrinsics (stored by value)
        const Point2 uv_;       // Observed 2D pixel point
        const Point3 P_;        // 3D camera point
        double weight_ = 1.0;   // Weight    
    
    public:
        SimInvResectioningFactor(const gtsam::Key& sim_pose_key,
                                 const gtsam::Cal3_S2& calib,
                                 const gtsam::Point2& p,
                                 const gtsam::Point3& P,
                                 const gtsam::SharedNoiseModel& noiseModel)
            : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
                calib_(calib), uv_(p), P_(P) {}
    
        gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                    gtsam::OptionalMatrixType H = OptionalNone) const override {
            // Q = s^-1 * R^T * (P - t) (inverse of the Sim3 action), projected with K.
            // With the right perturbation sim * Exp([w, u, lambda]) (dR = R [w]x, dt = R u - t lambda,
            // ds = s lambda): dQ/d(w, u, lambda) = [[Q]x, -I / s, R^T (2 t - P) / s] (analytic Jacobian).
            const gtsam::Matrix3 R = sim3.rotation().matrix();
            const gtsam::Vector3 t = sim3.translation();
            const double s_inv = 1.0 / sim3.scale();
            const gtsam::Vector3 Q = s_inv * (R.transpose() * (P_ - t));
            const gtsam::Matrix3 K = calib_.K();
            const gtsam::Vector3 projected = K * Q;
            const gtsam::Point2 uv = projected.head<2>() / projected[2];
            const gtsam::Vector2 error = weight_ * (uv - uv_);

            if (H) {
                gtsam::Matrix37 dQ;
                dQ << gtsam::skewSymmetric(Q), -s_inv * gtsam::I_3x3, s_inv * (R.transpose() * (2.0 * t - P_));
                *H = weight_ * (projectionJacobian(projected) * K * dQ);
            }
            return error;
        }
    
        void setWeight(double weight) {
            weight_ = weight;
        }
    
        double getWeight() const {
            return weight_;
        }
    
        virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
            return std::make_shared<SimInvResectioningFactor>(*this);
        }
    };
    
} // namespace gtsam_factors