# ABOUTME: Tests the crop adapter that puts AutoMorph's M0 geometry behind a reusable interface.
# ABOUTME: The load-bearing test is parity: the adapter must equal fundus_prep.process_without_gb.

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "M0_Preprocess"))

import fundus_prep as prep

from benchmark.preprocess import CANONICAL_SIZE, apply_crop, crop_transform, imread_rgb


def synthetic_fundus(side=400, radius=150, offset=(0, 0)):
    """A black frame with a bright, slightly off-centre retina."""
    image = np.zeros((side, side, 3), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:side, :side]
    centre_y, centre_x = side // 2 + offset[0], side // 2 + offset[1]
    disc = (grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2 <= radius**2
    image[disc] = (180, 120, 90)
    return image


def test_canonical_size_is_the_pipeline_grid():
    assert CANONICAL_SIZE == 912


def test_crop_matches_automorph_m0_exactly():
    """The whole point of the adapter: identical output to the function M0 itself calls."""
    image = synthetic_fundus(offset=(9, -7))

    expected, _, _, _, radii, centres_w, centres_h = prep.process_without_gb(
        image.copy(), image.copy(), [], [], []
    )
    transform = crop_transform(image.copy())
    actual = apply_crop(image.copy(), transform)

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)
    assert transform.radius == radii[0]
    assert int(transform.centre[0]) == centres_w[0]
    assert int(transform.centre[1]) == centres_h[0]


def test_paired_mask_crop_matches_automorph():
    """A mask carried alongside the image must land where M0's own label path puts it."""
    image = synthetic_fundus(offset=(-5, 11))
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[180:220, 120:280] = 255

    _, _, _, expected_label, _, _, _ = prep.process_without_gb(
        image.copy(), annotation.copy(), [], [], []
    )
    cropped = apply_crop(annotation.copy(), crop_transform(image.copy()))

    assert cropped.shape == expected_label.shape
    # M0 masks the image outside the field of view but not the label; the adapter masks both, so
    # compare inside the field of view, which is the region anything is ever scored on.
    inside = apply_crop(crop_transform(image.copy()).fov_mask.copy(), crop_transform(image.copy())) > 0
    assert np.array_equal(cropped[inside], expected_label[inside])


def test_crop_produces_a_square():
    transform = crop_transform(synthetic_fundus(offset=(20, -20)))
    square = apply_crop(synthetic_fundus(offset=(20, -20)), transform)
    assert square.shape[0] == square.shape[1]


def test_crop_side_is_about_twice_the_radius():
    image = synthetic_fundus(radius=150)
    transform = crop_transform(image)
    square = apply_crop(image, transform)
    assert abs(square.shape[0] - 2 * transform.radius) <= 2


def test_apply_crop_preserves_dtype():
    image = synthetic_fundus()
    transform = crop_transform(image)

    boolean = np.zeros(image.shape[:2], dtype=bool)
    boolean[190:210, 190:210] = True

    assert apply_crop(boolean, transform).dtype == bool
    assert apply_crop(image, transform).dtype == np.uint8


def test_apply_crop_does_not_mutate_its_input():
    image = synthetic_fundus()
    transform = crop_transform(image)
    original = image.copy()

    apply_crop(image, transform)

    assert np.array_equal(image, original)


def test_crop_transform_does_not_mutate_its_input():
    image = synthetic_fundus()
    original = image.copy()
    crop_transform(image)
    assert np.array_equal(image, original)


def test_imread_rgb_returns_rgb_not_bgr(tmp_path):
    path = tmp_path / "red.png"
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255  # red in BGR, which is what cv2.imwrite expects
    cv2.imwrite(str(path), bgr)

    image = imread_rgb(str(path))

    assert image.shape == (8, 8, 3)
    assert image[0, 0, 0] == 255  # red must land in channel 0
    assert image[0, 0, 2] == 0


def test_imread_rgb_reports_a_missing_file(tmp_path):
    with pytest.raises(OSError, match="cannot read"):
        imread_rgb(str(tmp_path / "absent.png"))
