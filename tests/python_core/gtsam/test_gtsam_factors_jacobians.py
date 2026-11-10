import sys

import pyslam.config as config

import numpy as np
import gtsam
import gtsam_factors

import unittest
from unittest import TestCase


# Checks the analytic Jacobians of the gtsam_factors factors against Richardson-extrapolated central
# differences of their error w.r.t. GTSAM's retraction (x.retract(d) = x * Exp(d)), including arbitrary
# (monocular) scales and very small/large translations.


def random_rot(rng, angle):
    w = rng.normal(size=3)
    return gtsam.Rot3.Expmap(w / np.linalg.norm(w) * angle)


def random_sim3(rng, rot, trans, log_scale):
    return gtsam.Similarity3(random_rot(rng, rot), rng.normal(size=3) * trans, float(np.exp(log_scale)))


def factor_error(factor, keys_values):
    # The factors use unit-sigma noise models, so the linearized right-hand side is -e(x).
    values = gtsam.Values()
    for key, value in keys_values:
        values.insert(key, value)
    _, b = factor.linearize(values).jacobian()
    return -b


def analytic_jacobian(factor, keys_values):
    values = gtsam.Values()
    for key, value in keys_values:
        values.insert(key, value)
    A, _ = factor.linearize(values).jacobian()
    return A


def reference_jacobian(factor, keys_values):
    """Richardson-extrapolated central differences over every variable, with a per-column step scaled so
    that the perturbation stays in the linear range (large columns would otherwise be dominated by
    truncation error)."""
    columns = []
    for k, (key, x) in enumerate(keys_values):
        dim = 7 if isinstance(x, gtsam.Similarity3) else 6

        def err(xk):
            kv = list(keys_values)
            kv[k] = (key, xk)
            return factor_error(factor, kv)

        for i in range(dim):

            def central(h):
                d = np.zeros(dim)
                d[i] = h
                return (err(x.retract(d)) - err(x.retract(-d))) / (2 * h)

            h = 1e-3 / max(1.0, np.linalg.norm(central(1e-6)) / 1e3)
            columns.append((4.0 * central(h / 2) - central(h)) / 3.0)
    return np.column_stack(columns)


def relative_error(A, R):
    return np.abs(A - R).max() / max(1.0, np.abs(R).max())


# (relative rotation angle, translation magnitude, log-scale)
SIM3_REGIMES = {
    "typical": (1.0, 1.0, 0.5),
    "large rotation": (3.0, 1.0, 0.5),
    "scale 1e-4": (1.0, 1.0, np.log(1e-4)),
    "scale 1e4": (1.0, 1.0, np.log(1e4)),
    "tiny translation": (1.0, 1e-9, 0.5),
    "huge translation": (1.0, 1e6, 0.5),
}


class TestSim3FactorJacobians(TestCase):
    def check(self, make_factor_and_values, tol=1e-8, trials=10):
        rng = np.random.default_rng(0)
        for name, regime in SIM3_REGIMES.items():
            with self.subTest(regime=name):
                for _ in range(trials):
                    factor, keys_values = make_factor_and_values(rng, *regime)
                    rel = relative_error(analytic_jacobian(factor, keys_values), reference_jacobian(factor, keys_values))
                    self.assertLess(rel, tol, f"{name}: relative Jacobian error {rel:.2e}")

    def test_prior(self):
        noise = gtsam.noiseModel.Isotropic.Sigma(7, 1.0)

        def make(rng, rot, trans, log_scale):
            prior = random_sim3(rng, 0.7, 1.0, 0.3 * rng.normal())
            x = prior.compose(random_sim3(rng, rot, trans, log_scale))
            return gtsam_factors.PriorFactorSimilarity3(1, prior, noise), [(1, x)]

        self.check(make)

    def test_between_inverse(self):
        noise = gtsam.noiseModel.Isotropic.Sigma(7, 1.0)

        def make(rng, rot, trans, log_scale):
            A = random_sim3(rng, 0.7, 1.0, 0.3 * rng.normal())
            S1, S2 = A.compose(random_sim3(rng, rot, trans, log_scale)), A.compose(random_sim3(rng, rot, trans, log_scale))
            M = S1.compose(S2.inverse()).compose(random_sim3(rng, 0.3, 0.1 * max(trans, 1e-9), 0.1))
            return gtsam_factors.BetweenFactorSimilarity3Inverse(1, 2, M, noise), [(1, S1), (2, S2)]

        self.check(make)

    def test_between_inverse_only_s1_and_s2(self):
        noise = gtsam.noiseModel.Isotropic.Sigma(7, 1.0)

        def make_s1(rng, rot, trans, log_scale):
            A = random_sim3(rng, 0.7, 1.0, 0.3 * rng.normal())
            S1, S2 = A.compose(random_sim3(rng, rot, trans, log_scale)), A.compose(random_sim3(rng, rot, trans, log_scale))
            M = S1.compose(S2.inverse()).compose(random_sim3(rng, 0.3, 0.1 * max(trans, 1e-9), 0.1))
            return gtsam_factors.BetweenFactorSimilarity3InverseOnlyS1(1, S2, M, noise), [(1, S1)]

        def make_s2(rng, rot, trans, log_scale):
            A = random_sim3(rng, 0.7, 1.0, 0.3 * rng.normal())
            S1, S2 = A.compose(random_sim3(rng, rot, trans, log_scale)), A.compose(random_sim3(rng, rot, trans, log_scale))
            M = S1.compose(S2.inverse()).compose(random_sim3(rng, 0.3, 0.1 * max(trans, 1e-9), 0.1))
            return gtsam_factors.BetweenFactorSimilarity3InverseOnlyS2(2, S1, M, noise), [(2, S2)]

        self.check(make_s1)
        self.check(make_s2)

    def test_between(self):
        noise = gtsam.noiseModel.Isotropic.Sigma(7, 1.0)

        def make(rng, rot, trans, log_scale):
            A = random_sim3(rng, 0.7, 1.0, 0.3 * rng.normal())
            S1, S2 = A.compose(random_sim3(rng, rot, trans, log_scale)), A.compose(random_sim3(rng, rot, trans, log_scale))
            M = S1.inverse().compose(S2).compose(random_sim3(rng, 0.3, 0.1 * max(trans, 1e-9), 0.1))
            return gtsam_factors.BetweenFactorSimilarity3(1, 2, M, noise), [(1, S1), (2, S2)]

        # S1^-1 * S2 amplifies translations by up to 1/s (~1e4 at scale 1e-4), which limits the accuracy of
        # the finite-difference reference itself (checked: the FD error vs step size is V-shaped around ~1e-9).
        self.check(make, tol=5e-8)

    def test_sim_resectioning(self):
        K = gtsam.Cal3_S2(520.0, 515.0, 0.0, 320.0, 240.0)
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        def make(rng, rot, trans, log_scale, inverse):
            sim = random_sim3(rng, rot, trans, log_scale)
            R, t, s = sim.rotation().matrix(), sim.translation(), sim.scale()
            Q = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)])  # camera point in front
            uv = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(), K).project(Q) + rng.uniform(-2, 2, size=2)
            if inverse:  # Q = R^T (P - t) / s
                return gtsam_factors.SimInvResectioningFactor(1, K, uv, s * (R @ Q) + t, noise), [(1, sim)]
            return gtsam_factors.SimResectioningFactor(1, K, uv, R.T @ (Q - t) / s, noise), [(1, sim)]  # Q = sRP + t

        # With |t| ~ 1e6 the Jacobian reaches ~1e8-1e9 and the finite-difference reference is only accurate to
        # ~1e-7 (V-shaped error vs step size); elsewhere agreement is ~1e-11.
        self.check(lambda rng, *r: make(rng, *r, inverse=False), tol=1e-6)
        self.check(lambda rng, *r: make(rng, *r, inverse=True), tol=1e-6)

    def test_error_is_log_of_relative_transform(self):
        rng = np.random.default_rng(1)
        prior, x = random_sim3(rng, 0.5, 1.0, 0.2), random_sim3(rng, 0.8, 2.0, -0.4)
        factor = gtsam_factors.PriorFactorSimilarity3(1, prior, gtsam.noiseModel.Isotropic.Sigma(7, 1.0))
        expected = gtsam.Similarity3.Logmap(prior.inverse().compose(x))
        np.testing.assert_allclose(factor_error(factor, [(1, x)]), expected, atol=1e-12)

    def test_prior_converges_from_far_away(self):
        rng = np.random.default_rng(2)
        prior = random_sim3(rng, 0.7, 1.0, np.log(20.0))
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam_factors.PriorFactorSimilarity3(1, prior, gtsam.noiseModel.Isotropic.Sigma(7, 0.1)))
        values = gtsam.Values()
        values.insert(1, prior.compose(random_sim3(rng, 2.5, 5.0, np.log(1e-3))))
        result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
        np.testing.assert_allclose(
            gtsam.Similarity3.Logmap(prior.inverse().compose(result.atSimilarity3(1))), np.zeros(7), atol=1e-8
        )


class TestResectioningFactorJacobians(TestCase):
    K = gtsam.Cal3_S2(520.0, 515.0, 0.0, 320.0, 240.0)
    Ks = gtsam.Cal3_S2Stereo(520.0, 515.0, 0.0, 320.0, 240.0, 0.1)

    def cases(self, rng, n=10):
        for trans in (1e-6, 1.0, 1e3):
            for _ in range(n):
                Twc = gtsam.Pose3(random_rot(rng, rng.uniform(0, 3)), rng.normal(size=3) * trans)
                Q = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)]) * max(1.0, trans)
                yield trans, Twc, Twc.transformFrom(Q)

    def check(self, factor, pose):
        kv = [(1, pose)]
        rel = relative_error(analytic_jacobian(factor, kv), reference_jacobian(factor, kv))
        self.assertLess(rel, 1e-9, f"relative Jacobian error {rel:.2e}")

    def test_mono(self):
        rng = np.random.default_rng(3)
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        for trans, Twc, Pw in self.cases(rng):
            with self.subTest(trans=trans):
                uv = gtsam.PinholeCameraCal3_S2(Twc, self.K).project(Pw) + rng.uniform(-1, 1, size=2)
                self.check(gtsam_factors.ResectioningFactor(noise, 1, self.K, uv, Pw), Twc)
                self.check(gtsam_factors.ResectioningFactorTcw(noise, 1, self.K, uv, Pw), Twc.inverse())

    def test_stereo(self):
        rng = np.random.default_rng(4)
        noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        for trans, Twc, Pw in self.cases(rng):
            with self.subTest(trans=trans):
                sp = gtsam.StereoCamera(Twc, self.Ks).project(Pw)
                sp = gtsam.StereoPoint2(sp.uL() + rng.uniform(-1, 1), sp.uR() + rng.uniform(-1, 1), sp.v() + rng.uniform(-1, 1))
                self.check(gtsam_factors.ResectioningFactorStereo(noise, 1, self.Ks, sp, Pw), Twc)
                self.check(gtsam_factors.ResectioningFactorStereoTcw(noise, 1, self.Ks, sp, Pw), Twc.inverse())


if __name__ == "__main__":
    unittest.main()
