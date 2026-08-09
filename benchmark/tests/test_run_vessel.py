# ABOUTME: Tests the vessel-only benchmark: gate bypass, stage list, and isolation from other runs.
# ABOUTME: Isolation matters most — this run must not be able to overwrite the full run's results.

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.run import DEFAULT_OUTPUT as FULL_OUTPUT
from benchmark.run import DEFAULT_RUN_ROOT as FULL_RUN_ROOT
from benchmark.run_vessel import (
    DEFAULT_OUTPUT,
    DEFAULT_RUN_ROOT,
    STAGES,
    bypass_quality_gate,
    prediction_mask_directories,
)


def test_stages_are_preprocessing_and_vessel_only():
    assert [stage.name for stage in STAGES] == ["M0 preprocess", "M2 vessel segmentation"]


def test_no_quality_stage_is_run():
    """The gate is bypassed by populating its output, not by running and ignoring it."""
    assert not any("M1" in stage.name or "quality" in stage.name.lower() for stage in STAGES)


def test_no_artery_vein_or_disc_stage_is_run():
    joined = " ".join(stage.name for stage in STAGES)
    for absent in ("artery", "disc", "M3", "csv merge"):
        assert absent not in joined


def test_run_root_is_separate_from_the_full_run():
    assert DEFAULT_RUN_ROOT != FULL_RUN_ROOT


def test_output_is_separate_from_the_full_run():
    assert DEFAULT_OUTPUT != FULL_OUTPUT
    assert DEFAULT_OUTPUT.name == "M2_vessels"


def test_output_is_nested_under_the_full_run_output():
    """A subdirectory, so the two sets of results sit together without colliding."""
    assert DEFAULT_OUTPUT.parent == FULL_OUTPUT


def test_bypass_copies_every_preprocessed_image(tmp_path):
    source = tmp_path / "Results" / "M0" / "images"
    source.mkdir(parents=True)
    for index in range(5):
        (source / f"image_{index}.png").write_bytes(b"x")

    copied = bypass_quality_gate(tmp_path)

    good = tmp_path / "Results" / "M1" / "Good_quality"
    assert copied == 5
    assert sorted(path.name for path in good.iterdir()) == sorted(
        f"image_{index}.png" for index in range(5)
    )


def test_bypass_leaves_no_image_behind_in_bad_quality(tmp_path):
    source = tmp_path / "Results" / "M0" / "images"
    source.mkdir(parents=True)
    (source / "a.png").write_bytes(b"x")

    bypass_quality_gate(tmp_path)

    bad = tmp_path / "Results" / "M1" / "Bad_quality"
    assert not bad.exists() or not list(bad.iterdir())


def test_bypass_is_idempotent(tmp_path):
    source = tmp_path / "Results" / "M0" / "images"
    source.mkdir(parents=True)
    (source / "a.png").write_bytes(b"x")

    bypass_quality_gate(tmp_path)
    assert bypass_quality_gate(tmp_path) == 1


def test_bypass_reports_an_empty_m0(tmp_path):
    (tmp_path / "Results" / "M0" / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="no preprocessed images"):
        bypass_quality_gate(tmp_path)


def test_bypass_reports_a_missing_m0(tmp_path):
    with pytest.raises(FileNotFoundError, match="M0"):
        bypass_quality_gate(tmp_path)


def test_prediction_directories_satisfy_retipy_path_rules(tmp_path):
    """retipy splits on '_skeleton' and on 'M2'; M2's own output must satisfy both."""
    skeleton_dir, process_dir = prediction_mask_directories(tmp_path / "run" / "Results")

    assert str(skeleton_dir).count("_skeleton") == 1
    assert str(process_dir).count("M2") == 1
    assert skeleton_dir.name == "binary_skeleton"
    assert process_dir.name == "binary_process"


def test_prediction_directories_are_where_m2_writes(tmp_path):
    results = tmp_path / "Results"
    skeleton_dir, process_dir = prediction_mask_directories(results)
    assert skeleton_dir == results / "M2" / "binary_vessel" / "binary_skeleton"
    assert process_dir == results / "M2" / "binary_vessel" / "binary_process"
