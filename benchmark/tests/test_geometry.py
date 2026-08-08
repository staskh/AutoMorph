# ABOUTME: Tests moving ground truth into the pipeline's 912 frame, and both mask resize rules.
# ABOUTME: The two rules differ on thin structures, which is the whole reason the choice is explicit.

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.geometry import evaluation_mask, resize_mask, to_pipeline_frame, to_pipeline_image
from benchmark.preprocess import crop_transform


def synthetic_fundus(side=400, radius=150):
    image = np.zeros((side, side, 3), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:side, :side]
    disc = (grid_y - side // 2) ** 2 + (grid_x - side // 2) ** 2 <= radius**2
    image[disc] = (180, 120, 90)
    return image


def test_resize_is_a_no_op_at_the_target_size():
    mask = np.zeros((912, 912), dtype=bool)
    mask[100:200, 100:200] = True
    assert np.array_equal(resize_mask(mask, 912), mask)


def test_resize_returns_booleans_from_any_input_form():
    for mask in (
        np.array([[0, 255], [255, 0]], dtype=np.uint8),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[False, True], [True, False]]),
    ):
        assert resize_mask(mask, 4).dtype == bool


def test_area_downsampling_keeps_a_solid_block():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    resized = resize_mask(mask, 50, method="area")
    assert resized[15:35, 15:35].all()
    assert not resized[0:5, 0:5].any()


def test_area_downsampling_keeps_exactly_half_coverage():
    """The threshold is inclusive, so a structure covering exactly half an output pixel survives.

    An exclusive rule erodes, worst at an exact 2x reduction where many cells land on 0.5.
    """
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True  # at 2x reduction every output cell in row 0 is covered exactly half
    assert resize_mask(mask, 2, method="area")[0].all()


def test_area_downsampling_drops_a_structure_thinner_than_half_a_pixel():
    """A one-pixel line at 4x reduction covers a quarter of an output pixel, so it goes."""
    mask = np.zeros((80, 80), dtype=bool)
    mask[40, :] = True
    assert not resize_mask(mask, 20, method="area").any()


def test_nearest_downsampling_keeps_a_thin_line_the_area_rule_drops():
    """The two rules genuinely disagree; this is why the choice must be explicit, not implicit."""
    mask = np.zeros((80, 80), dtype=bool)
    mask[40, :] = True

    assert resize_mask(mask, 20, method="nearest").any()
    assert not resize_mask(mask, 20, method="area").any()


def test_upsampling_replicates_rather_than_inventing():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    resized = resize_mask(mask, 20)
    assert resized.sum() == 4  # one pixel becomes exactly a 2x2 block


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        resize_mask(np.zeros((10, 10), dtype=bool), 5, method="bicubic")


def test_to_pipeline_image_returns_an_rgb_square():
    image = synthetic_fundus()
    framed = to_pipeline_image(image, crop_transform(image), 912)
    assert framed.shape == (912, 912, 3)
    assert framed.dtype == np.uint8


def test_to_pipeline_frame_returns_a_boolean_square():
    image = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[180:220, 120:280] = 255

    framed = to_pipeline_frame(annotation, crop_transform(image), 912)

    assert framed.shape == (912, 912)
    assert framed.dtype == bool
    assert framed.any()


def test_evaluation_mask_is_the_field_of_view_in_the_pipeline_frame():
    image = synthetic_fundus()
    inside = evaluation_mask(crop_transform(image), 912)

    assert inside.shape == (912, 912)
    assert inside[456, 456]
    assert not inside[0, 0]  # the corner is outside the circular retina
    assert 0.5 < inside.mean() < 0.9  # a disc inscribed in its square
