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


def test_default_output_does_not_clobber_a_frozen_result_set():
    """M2_vessels holds the pre-fix baseline the comparison notebook reads; nothing may overwrite it."""
    assert DEFAULT_OUTPUT.name != "M2_vessels"
    assert (FULL_OUTPUT.parent / "M2_vessels").is_dir(), "the pre-fix baseline has gone missing"


def test_every_run_gets_its_own_directory_under_results():
    """One directory per run, so runs stay distinguishable and cannot overwrite each other."""
    assert DEFAULT_OUTPUT.parent == FULL_OUTPUT.parent == Path(FULL_OUTPUT).parents[0]
    assert DEFAULT_OUTPUT.parent.name == "results"
    assert DEFAULT_OUTPUT.name != FULL_OUTPUT.name


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


def make_source_run(tmp_path, images=3):
    """A minimal finished run: crop_info.csv plus prediction masks."""
    results = tmp_path / "source" / "Results"
    (results / "M0").mkdir(parents=True)
    (results / "M0" / "crop_info.csv").write_text("Name,radius,Scale,Scale_resolution\na.png,1,1,1\n")
    for subdir in ("binary_process", "binary_skeleton"):
        directory = results / "M2" / "binary_vessel" / subdir
        directory.mkdir(parents=True)
        for index in range(images):
            (directory / f"img_{index}.png").write_bytes(b"x")
    return tmp_path / "source"


def test_reuse_segmentation_carries_masks_and_crop_info(tmp_path):
    from benchmark.run_vessel import reuse_segmentation

    source = make_source_run(tmp_path)
    target = tmp_path / "target"

    copied = reuse_segmentation(source, target)

    assert copied == 3
    assert (target / "Results" / "M0" / "crop_info.csv").is_file()
    for subdir in ("binary_process", "binary_skeleton"):
        assert len(list((target / "Results" / "M2" / "binary_vessel" / subdir).glob("*.png"))) == 3


def test_reuse_segmentation_does_not_disturb_the_source(tmp_path):
    from benchmark.run_vessel import reuse_segmentation

    source = make_source_run(tmp_path)
    before = sorted(p.relative_to(source) for p in source.rglob("*"))

    reuse_segmentation(source, tmp_path / "target")

    assert sorted(p.relative_to(source) for p in source.rglob("*")) == before


def test_reuse_segmentation_reports_a_missing_segmentation(tmp_path):
    from benchmark.run_vessel import reuse_segmentation

    source = tmp_path / "empty"
    (source / "Results" / "M0").mkdir(parents=True)
    (source / "Results" / "M0" / "crop_info.csv").write_text("Name\n")
    with pytest.raises(FileNotFoundError, match="nothing to reuse"):
        reuse_segmentation(source, tmp_path / "target")


def test_reuse_segmentation_reports_missing_crop_info(tmp_path):
    from benchmark.run_vessel import reuse_segmentation

    source = tmp_path / "nocrop"
    (source / "Results" / "M2" / "binary_vessel" / "binary_process").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="crop_info"):
        reuse_segmentation(source, tmp_path / "target")


def test_environment_carries_the_vessel_threshold(tmp_path):
    from benchmark.run import DEFAULT_VESSEL_THRESHOLD, build_environment

    assert DEFAULT_VESSEL_THRESHOLD == 0.2, "must match M2's own default, or runs disagree silently"
    environment = build_environment(tmp_path, "cpu", 0, 8, vessel_threshold=0.35)
    assert environment["AUTOMORPH_VESSEL_THRESHOLD"] == "0.35"


def test_m2_reads_the_threshold_from_the_environment():
    """The value has to reach both places M2 binarises, not just the first."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "M2_Vessel_seg" / "test_outside_integrated.py").read_text()

    assert "AUTOMORPH_VESSEL_THRESHOLD" in source
    assert source.count(">=VESSEL_THRESHOLD") == 2, "resize_binary and raw_binary must agree"
    assert ">=0.5]=1" not in source, "a hardcoded threshold is left somewhere"


def test_m2_and_the_benchmark_agree_on_the_default_threshold():
    """Two defaults for one quantity is a trap: a plain run.sh and a benchmark run must match."""
    from benchmark.run import DEFAULT_VESSEL_THRESHOLD

    root = Path(__file__).resolve().parents[2]
    source = (root / "M2_Vessel_seg" / "test_outside_integrated.py").read_text()
    assert f"os.getenv('AUTOMORPH_VESSEL_THRESHOLD', {DEFAULT_VESSEL_THRESHOLD})" in source


def test_default_run_root_does_not_clobber_the_frozen_baseline():
    """.benchmark_run_vessel holds the pre-fix masks; a default run must not rmtree them."""
    from benchmark.run_vessel import DEFAULT_RUN_ROOT

    assert DEFAULT_RUN_ROOT.name != ".benchmark_run_vessel"
