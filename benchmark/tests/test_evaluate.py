# ABOUTME: Tests for scoring AutoMorph's vessel segmentation against the FIVES annotations.
# ABOUTME: Covers the confusion counts, the derived scores, FOV restriction, and crop alignment.

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from benchmark.evaluate import align_annotation, confusion, resize_mask, scores, summarise


def test_confusion_counts_are_exact():
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    truth = np.array([[1, 0], [1, 0]], dtype=bool)
    inside = np.ones((2, 2), dtype=bool)
    assert confusion(prediction, truth, inside) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_confusion_ignores_pixels_outside_the_field_of_view():
    prediction = np.array([[1, 1], [1, 1]], dtype=bool)
    truth = np.zeros((2, 2), dtype=bool)
    inside = np.array([[1, 0], [0, 0]], dtype=bool)
    assert confusion(prediction, truth, inside) == {"tp": 0, "fp": 1, "fn": 0, "tn": 0}


def test_confusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        confusion(np.zeros((2, 2), bool), np.zeros((3, 3), bool), np.zeros((2, 2), bool))


def test_perfect_prediction_scores_one():
    result = scores({"tp": 10, "fp": 0, "fn": 0, "tn": 90})
    assert result["dice"] == 1.0
    assert result["iou"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["accuracy"] == 1.0


def test_scores_match_hand_computed_values():
    result = scores({"tp": 6, "fp": 2, "fn": 4, "tn": 88})
    assert result["dice"] == pytest.approx(2 * 6 / (2 * 6 + 2 + 4))
    assert result["iou"] == pytest.approx(6 / (6 + 2 + 4))
    assert result["sensitivity"] == pytest.approx(6 / 10)
    assert result["specificity"] == pytest.approx(88 / 90)
    assert result["accuracy"] == pytest.approx(94 / 100)


def test_empty_prediction_and_truth_scores_one_not_nan():
    """An image with no annotated vessels inside the FOV must not poison the average."""
    result = scores({"tp": 0, "fp": 0, "fn": 0, "tn": 100})
    assert result["dice"] == 1.0
    assert result["iou"] == 1.0


def test_scores_are_zero_when_nothing_overlaps():
    result = scores({"tp": 0, "fp": 5, "fn": 5, "tn": 90})
    assert result["dice"] == 0.0
    assert result["iou"] == 0.0
    assert result["sensitivity"] == 0.0


def test_resize_mask_preserves_booleans_and_shape():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    resized = resize_mask(mask, 50)
    assert resized.shape == (50, 50)
    assert resized.dtype == bool
    assert resized[20:30, 20:30].all()
    assert not resized[0:10, 0:10].any()


def synthetic_fundus(side=400, radius=150):
    """A black frame with a bright circular retina, the shape M0's crop detector expects."""
    image = np.zeros((side, side, 3), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:side, :side]
    disc = (grid_y - side // 2) ** 2 + (grid_x - side // 2) ** 2 <= radius**2
    image[disc] = 180
    return image, disc


def test_align_annotation_matches_the_cropped_image_shape():
    image, disc = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[disc] = 255

    aligned, inside, radius = align_annotation(image, annotation)

    assert aligned.shape == inside.shape
    assert aligned.dtype == bool
    assert radius > 0
    assert aligned.shape[0] == aligned.shape[1]


def test_align_annotation_carries_the_annotation_through():
    image, disc = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[disc] = 255

    aligned, inside, _ = align_annotation(image, annotation)

    assert aligned.sum() > 0
    assert (aligned & inside).sum() > 0


def test_annotation_outside_the_field_of_view_is_not_counted_as_missed():
    """M0 crops the annotation to the retina's bounding box but does not circle-mask it, so a
    little annotation can land outside the field of view. Those pixels must not become false
    negatives."""
    image, disc = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[disc] = 255

    aligned, inside, _ = align_annotation(image, annotation)
    assert (aligned & ~inside).any(), "expected some annotation outside the detected field of view"

    counts = confusion(np.zeros_like(aligned), aligned, inside)
    assert counts["fn"] == int((aligned & inside).sum())


def scored_frame():
    """Two amd images scored, one dropped by the quality gate; one dr image scored."""
    return pd.DataFrame(
        [
            {"key": "a", "disease": "amd", "status": "ok", "dice": 0.8, "iou": 0.7,
             "sensitivity": 0.7, "specificity": 1.0, "accuracy": 1.0},
            {"key": "b", "disease": "amd", "status": "ok", "dice": 0.6, "iou": 0.5,
             "sensitivity": 0.5, "specificity": 1.0, "accuracy": 1.0},
            {"key": "c", "disease": "amd", "status": "no_segmentation"},
            {"key": "d", "disease": "dr", "status": "ok", "dice": 0.9, "iou": 0.8,
             "sensitivity": 0.9, "specificity": 1.0, "accuracy": 1.0},
        ]
    )


def test_summary_averages_only_the_scored_images():
    summary = summarise(scored_frame())
    assert summary.loc["amd", "dice"] == pytest.approx(0.7)
    assert summary.loc["all", "dice"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)


def test_summary_shows_how_many_images_were_selected_and_scored():
    summary = summarise(scored_frame())
    assert summary.loc["amd", "images_selected"] == 3
    assert summary.loc["amd", "images_scored"] == 2
    assert summary.loc["all", "images_selected"] == 4
    assert summary.loc["all", "images_scored"] == 3


def test_summary_counts_a_disease_that_produced_nothing():
    frame = pd.DataFrame([{"key": "x", "disease": "glaucoma", "status": "no_segmentation"}])
    summary = summarise(frame)
    assert summary.loc["glaucoma", "images_selected"] == 1
    assert summary.loc["glaucoma", "images_scored"] == 0


def test_align_annotation_drops_annotation_outside_the_retina():
    image, _ = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[0:5, 0:5] = 255  # a corner, well outside the bright disc

    aligned, _, _ = align_annotation(image, annotation)

    assert aligned.sum() == 0
