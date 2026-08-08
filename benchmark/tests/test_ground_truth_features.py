# ABOUTME: Tests for building 912 ground-truth vessel maps and measuring them with retipy.
# ABOUTME: Pins the two path conventions retipy relies on, which are easy to break by renaming.

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage.io import imread

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.ground_truth_features import (
    PROCESS_SUBDIR,
    SKELETON_SUBDIR,
    build_mask_pair,
    mask_directories,
    validate_paths,
    write_mask_pair,
)


def clean_root(tmp_path):
    """A run root whose own name contains neither token retipy splits on.

    pytest derives tmp_path from the test name, so a test named ..._skeleton_... would otherwise
    poison the very substring under test.
    """
    root = tmp_path / "run" / "Results"
    root.mkdir(parents=True)
    return root


def test_retipy_can_recover_the_binary_map_from_a_skeleton_path(tmp_path):
    """retipy does store_path + path.split('_skeleton')[1]; the token must appear exactly once."""
    skeleton_dir, process_dir = mask_directories(clean_root(tmp_path))
    skeleton_path = str(skeleton_dir / "test_11_A.png")

    assert skeleton_path.count("_skeleton") == 1
    assert str(process_dir) + skeleton_path.split("_skeleton")[1] == str(process_dir / "test_11_A.png")


def test_retipy_can_find_crop_info_from_the_binary_map_path(tmp_path):
    """retipy does store_path.split('M2')[0] + 'M0/crop_info.csv'."""
    results = clean_root(tmp_path)
    _, process_dir = mask_directories(results)

    assert str(process_dir).count("M2") == 1
    assert str(process_dir).split("M2")[0] + "M0/crop_info.csv" == str(results / "M0" / "crop_info.csv")


def test_mask_directories_are_named_consistently(tmp_path):
    skeleton_dir, process_dir = mask_directories(clean_root(tmp_path))
    assert skeleton_dir.name == SKELETON_SUBDIR
    assert process_dir.name == PROCESS_SUBDIR


def test_validate_paths_accepts_a_clean_root(tmp_path):
    validate_paths(clean_root(tmp_path))


def test_validate_paths_rejects_a_root_that_breaks_the_skeleton_split(tmp_path):
    root = tmp_path / "my_skeleton_dir" / "Results"
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="_skeleton"):
        validate_paths(root)


def test_validate_paths_rejects_a_root_that_breaks_the_crop_info_split(tmp_path):
    root = tmp_path / "M2_experiments" / "Results"
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="M2"):
        validate_paths(root)


def synthetic_fundus(side=400, radius=150):
    image = np.zeros((side, side, 3), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:side, :side]
    disc = (grid_y - side // 2) ** 2 + (grid_x - side // 2) ** 2 <= radius**2
    image[disc] = 180
    return image, disc


def test_build_mask_pair_returns_binary_912_masks():
    image, disc = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[disc] = 255

    processed, skeleton = build_mask_pair(image, annotation, size=912)

    assert processed.shape == (912, 912)
    assert skeleton.shape == (912, 912)
    assert processed.dtype == bool
    assert skeleton.dtype == bool
    assert processed.sum() > skeleton.sum() > 0


def test_build_mask_pair_removes_specks_the_pipeline_would_remove():
    """M2 applies remove_small_objects(30) before measuring; the truth must get the same."""
    image, disc = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    # A thick bar that survives, plus a 2x2 speck that must not.
    annotation[190:210, 130:270] = 255
    annotation[160:162, 160:162] = 255

    processed, _ = build_mask_pair(image, annotation, size=912)

    labelled_area = processed.sum()
    assert labelled_area > 0
    # The speck is far from the bar; nothing should survive up there.
    assert not processed[:150, :].any()


def test_build_mask_pair_restricts_to_the_field_of_view():
    image, _ = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[:, :] = 255  # annotate everything, including outside the retina

    processed, _ = build_mask_pair(image, annotation, size=912)

    # The retina is a circle, so the corners of the square must be excluded.
    assert not processed[0, 0]
    assert not processed[-1, -1]
    assert processed[456, 456]


def test_write_mask_pair_writes_readable_255_pngs(tmp_path):
    image, disc = synthetic_fundus()
    annotation = np.zeros(image.shape[:2], dtype=np.uint8)
    annotation[disc] = 255
    processed, skeleton = build_mask_pair(image, annotation, size=912)

    skeleton_dir, process_dir = mask_directories(tmp_path)
    write_mask_pair("test_11_A", processed, skeleton, skeleton_dir, process_dir)

    written = imread(str(process_dir / "test_11_A.png"))
    assert written.shape == (912, 912)
    assert set(np.unique(written)) <= {0, 255}
    assert (written == 255).sum() == processed.sum()
    assert (imread(str(skeleton_dir / "test_11_A.png")) == 255).sum() == skeleton.sum()
