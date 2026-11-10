"""
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
*
* gtsam_factors' factors read the points that Python inserted into gtsam.Values. On macOS the
* lookup crosses modules with hidden RTTI (see thirdparty/gtsam_factors/values_cross_module.h).
"""

import unittest

import pyslam.config as config  # noqa: F401  (puts gtsam_factors on the path)
import gtsam
import gtsam_factors
import numpy as np


class TestValuesCrossModule(unittest.TestCase):
    def setUp(self):
        self.noise2 = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.noise3 = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        self.K = gtsam.Cal3_S2(500.0, 500.0, 0.0, 320.0, 240.0)
        self.Ks = gtsam.Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 0.1)
        self.values = gtsam.Values()
        self.values.insert(0, gtsam.Pose3(gtsam.Rot3.Rz(0.05), np.array([0.1, -0.05, 0.02])))
        self.values.insert(1, gtsam.Point3(0.3, -0.2, 3.0))  # a numpy array: stored as a dynamic vector

    def test_projection_factor_reads_point_inserted_from_python(self):
        measured = gtsam.Point2(330.0, 250.0)
        factor = gtsam_factors.WeightedGenericProjectionFactorCal3_S2(measured, self.noise2, 0, 1, self.K)
        # gtsam's own factor lives in the same module as Values.insert, so its lookup always works
        reference = gtsam.GenericProjectionFactorCal3_S2(measured, self.noise2, 0, 1, self.K)
        self.assertAlmostEqual(factor.error(self.values), reference.error(self.values), places=9)
        factor.linearize(self.values)  # the Jacobians read the point too

    def test_stereo_projection_factor_reads_point_inserted_from_python(self):
        measured = gtsam.StereoPoint2(330.0, 320.0, 250.0)
        factor = gtsam_factors.WeightedGenericStereoProjectionFactor3D(measured, self.noise3, 0, 1, self.Ks)
        reference = gtsam.GenericStereoFactor3D(measured, self.noise3, 0, 1, self.Ks)
        self.assertAlmostEqual(factor.error(self.values), reference.error(self.values), places=9)

    def test_wrong_type_still_raises(self):
        self.values.insert(2, gtsam.Pose3())
        factor = gtsam_factors.WeightedGenericProjectionFactorCal3_S2(
            gtsam.Point2(330.0, 250.0), self.noise2, 0, 2, self.K)
        with self.assertRaises(RuntimeError):
            factor.error(self.values)


if __name__ == "__main__":
    unittest.main()
