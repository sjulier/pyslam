"""
The quality map of the ORB-SLAM2 extractor (feature quality): thirdparty/orbslam2_features/ORBextractor.h.

    python -m pytest -q test/cv/test_orb_quality.py
"""

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "thirdparty", "orbslam2_features", "lib"))

from orbslam2_features import ORBextractorDeterministic  # noqa: E402

IMAGE = os.path.join(ROOT, "test", "data", "kitti06-435.png")


@pytest.fixture(scope="module")
def image():
    img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    assert img is not None, IMAGE
    return img


def extract(image, quality=None):
    kps, des = ORBextractorDeterministic(2000, 1.2, 8).detectAndCompute(image, quality=quality)
    return np.array(kps, dtype=np.float64), des


def test_no_map_and_uniform_map_give_the_original_features(image):
    kps, des = extract(image)
    for quality in (np.ones(image.shape, np.float32), np.full((3, 5), 2.5, np.float32), np.full(image.shape, 255, np.uint8)):
        kps_q, des_q = extract(image, quality)
        assert np.array_equal(kps, kps_q)
        assert np.array_equal(des, des_q)


def test_zero_weight_excludes_keypoints(image):
    h, w = image.shape
    mask = np.full((h, w), 255, np.uint8)
    mask[:, : w // 2] = 0
    kps, _ = extract(image)
    kps_q, _ = extract(image, mask)
    assert np.all(kps_q[:, 0] >= w // 2 - 1)
    # the octree spends the features on the rest of the image
    assert len(kps_q) > 0.8 * len(kps)


def test_coarse_map_is_scaled_to_the_image(image):
    w = image.shape[1]
    quality = np.array([[0.0, 1.0]], np.float32)  # 1x2: left half excluded
    kps_q, _ = extract(image, quality)
    assert len(kps_q) > 0
    assert np.all(kps_q[:, 0] >= w // 2 - 1)


def test_weights_change_the_choice_but_not_the_responses(image):
    rng = np.random.default_rng(0)
    quality = rng.uniform(0.1, 1.0, (24, 80)).astype(np.float32)
    kps, _ = extract(image)
    kps_q, _ = extract(image, quality)
    assert not np.array_equal(kps, kps_q)
    # the returned responses are FAST's (integers), not the weighted ones
    response = kps_q[:, 4]
    assert np.array_equal(response, np.round(response))


def test_map_with_several_channels_is_refused(image):
    with pytest.raises(Exception, match="single channel"):
        extract(image, np.ones(image.shape + (3,), np.float32))
