# ABOUTME: Tests for building the resolution_information.csv that M0 reads.
# ABOUTME: Covers the column contract, image discovery, ordering, and per-image overrides.

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.resolution import DEFAULT_RESOLUTION_MM, discover_images, resolution_table, write


def populate(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x")
    return directory


def test_columns_match_what_m0_reads(tmp_path):
    frame = resolution_table(["a.png", "b.png"])
    assert list(frame.columns) == ["fundus", "res"]


def test_every_image_gets_the_resolution():
    frame = resolution_table(["a.png", "b.png"], resolution=0.0075)
    assert list(frame["res"]) == [0.0075, 0.0075]


def test_default_resolution_is_applied():
    frame = resolution_table(["a.png"])
    assert frame["res"].iloc[0] == DEFAULT_RESOLUTION_MM


def test_rows_are_sorted_by_name():
    frame = resolution_table(["b.png", "a.png"])
    assert list(frame["fundus"]) == ["a.png", "b.png"]


def test_discover_skips_non_images(tmp_path):
    directory = populate(tmp_path / "images", ["a.png", ".DS_Store", "notes.txt", "b.jpg"])
    assert discover_images(directory) == ["a.png", "b.jpg"]


def test_discover_rejects_an_empty_directory(tmp_path):
    populate(tmp_path / "images", [])
    with pytest.raises(ValueError, match="no images"):
        discover_images(tmp_path / "images")


def test_write_round_trips(tmp_path):
    target = tmp_path / "nested" / "resolution_information.csv"
    write(target, resolution_table(["a.png"], resolution=0.008))
    assert pd.read_csv(target).to_dict("records") == [{"fundus": "a.png", "res": 0.008}]


def test_overrides_replace_the_default():
    frame = resolution_table(["a.png", "b.png"], resolution=0.008, overrides={"b.png": 0.006})
    assert dict(zip(frame["fundus"], frame["res"])) == {"a.png": 0.008, "b.png": 0.006}


def test_override_for_an_unknown_image_is_rejected():
    with pytest.raises(ValueError, match="c.png"):
        resolution_table(["a.png"], overrides={"c.png": 0.006})
