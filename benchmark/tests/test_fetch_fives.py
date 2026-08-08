# ABOUTME: Tests the FIVES store writer, including that a failed PNG write is never swallowed.
# ABOUTME: cv2.imwrite returns False instead of raising, which silently loses images.

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.datasets import fives
from benchmark.fetch_fives import _imwrite, _store_subdirs, _write


def arrays(size=8):
    return {
        fives.IMAGES_SUBDIR: np.zeros((size, size, 3), dtype=np.uint8),
        fives.VESSEL_SUBDIR: np.zeros((size, size), dtype=bool),
        fives.FOV_SUBDIR: np.ones((size, size), dtype=bool),
    }


def test_store_subdirs_creates_all_three(tmp_path):
    _store_subdirs(tmp_path)
    for subdir in (fives.IMAGES_SUBDIR, fives.VESSEL_SUBDIR, fives.FOV_SUBDIR):
        assert (tmp_path / subdir).is_dir()


def test_store_subdirs_is_idempotent(tmp_path):
    _store_subdirs(tmp_path)
    _store_subdirs(tmp_path)


def test_write_produces_all_three_pngs(tmp_path):
    _store_subdirs(tmp_path)
    _write(arrays(), "test_1_A", tmp_path)

    for subdir in (fives.IMAGES_SUBDIR, fives.VESSEL_SUBDIR, fives.FOV_SUBDIR):
        assert (tmp_path / subdir / "test_1_A.png").is_file()


def test_write_raises_rather_than_losing_an_image(tmp_path):
    """Without the subdirectories, cv2.imwrite returns False and writes nothing."""
    with pytest.raises(OSError, match="could not write"):
        _write(arrays(), "test_1_A", tmp_path)


def test_imwrite_raises_on_an_unwritable_path(tmp_path):
    with pytest.raises(OSError, match="could not write"):
        _imwrite(tmp_path / "absent" / "x.png", np.zeros((4, 4), dtype=np.uint8))


def test_write_stores_masks_as_255_not_1(tmp_path):
    import cv2

    _store_subdirs(tmp_path)
    _write(arrays(), "test_1_A", tmp_path)

    fov = cv2.imread(str(tmp_path / fives.FOV_SUBDIR / "test_1_A.png"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(fov)) == {255}
