# ABOUTME: Tests for the benchmark runner's environment setup and per-stage retry behaviour.
# ABOUTME: Uses real subprocesses throughout, so what is tested is what the pipeline will see.

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.run import (
    STAGES,
    Stage,
    build_environment,
    completion_counts,
    count_files,
    count_rows,
    run_stage,
)


def test_environment_pins_the_device(tmp_path):
    environment = build_environment(tmp_path, device="cpu", num_workers=0)
    assert environment["AUTOMORPH_DEVICE"] == "cpu"


def test_environment_points_automorph_data_at_an_absolute_run_root(tmp_path):
    environment = build_environment(tmp_path, device="cpu", num_workers=0)
    assert Path(environment["AUTOMORPH_DATA"]).is_absolute()
    assert Path(environment["AUTOMORPH_DATA"]) == tmp_path.resolve()


def test_environment_sets_worker_count(tmp_path):
    assert build_environment(tmp_path, device="cpu", num_workers=3)["NUM_WORKERS"] == "3"


def test_environment_puts_the_running_interpreter_first_on_path(tmp_path):
    environment = build_environment(tmp_path, device="cpu", num_workers=0)
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)


def test_successful_stage_is_reported_with_a_duration(tmp_path):
    stage = Stage("hello", ".", ["sh", "-c", "echo hi"])
    result = run_stage(stage, Path.cwd(), build_environment(tmp_path, "cpu", 0), tmp_path, attempts=1)

    assert result["status"] == "ok"
    assert result["attempts"] == 1
    assert result["returncode"] == 0
    assert result["seconds"] >= 0


def test_stage_output_is_written_to_a_log(tmp_path):
    stage = Stage("hello", ".", ["sh", "-c", "echo marker-text"])
    result = run_stage(stage, Path.cwd(), build_environment(tmp_path, "cpu", 0), tmp_path, attempts=1)

    assert "marker-text" in Path(result["log"]).read_text()


def test_a_flaky_stage_is_retried_until_it_succeeds(tmp_path):
    """The command fails until a marker file exists, so the retry has to be real to pass."""
    marker = tmp_path / "marker"
    stage = Stage("flaky", ".", ["sh", "-c", f"test -f {marker} || {{ touch {marker}; exit 1; }}"])

    result = run_stage(stage, Path.cwd(), build_environment(tmp_path, "cpu", 0), tmp_path, attempts=3)

    assert result["status"] == "ok"
    assert result["attempts"] == 2


def test_a_failing_stage_is_reported_rather_than_raised(tmp_path):
    stage = Stage("doomed", ".", ["sh", "-c", "echo nope >&2; exit 3"])

    result = run_stage(stage, Path.cwd(), build_environment(tmp_path, "cpu", 0), tmp_path, attempts=2)

    assert result["status"] == "failed"
    assert result["attempts"] == 2
    assert result["returncode"] == 3
    assert "nope" in Path(result["log"]).read_text()


def test_a_missing_command_is_reported_rather_than_raised(tmp_path):
    stage = Stage("absent", ".", ["definitely-not-a-real-command-xyz"])

    result = run_stage(stage, Path.cwd(), build_environment(tmp_path, "cpu", 0), tmp_path, attempts=1)

    assert result["status"] == "failed"


def test_pipeline_stages_cover_every_module():
    directories = {stage.cwd for stage in STAGES}
    for module in (
        "M0_Preprocess",
        "M1_Retinal_Image_quality_EyePACS",
        "M2_Vessel_seg",
        "M2_Artery_vein",
        "M2_lwnet_disc_cup",
        "M3_feature_zone/retipy",
        "M3_feature_whole_pic/retipy",
    ):
        assert module in directories, f"no benchmark stage runs in {module}"


def test_pipeline_stage_names_are_unique():
    names = [stage.name for stage in STAGES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.name)
def test_every_stage_runs_in_a_directory_that_exists(stage):
    root = Path(__file__).resolve().parents[2]
    assert (root / stage.cwd).is_dir()


def test_count_files_ignores_other_suffixes(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert count_files(tmp_path) == 2


def test_count_files_of_a_missing_directory_is_zero(tmp_path):
    assert count_files(tmp_path / "absent") == 0


def test_count_rows_of_a_missing_table_is_zero(tmp_path):
    assert count_rows(tmp_path / "absent.csv") == 0


def test_count_rows_counts_images_not_the_header(tmp_path):
    table = tmp_path / "features.csv"
    table.write_text("Name,value\na.png,1\nb.png,2\n")
    assert count_rows(table) == 2


def test_completion_counts_show_where_images_dropped_out(tmp_path):
    results = tmp_path / "Results"
    for relative, count in [
        ("../images", 32),
        ("M0/images", 30),
        ("M1/Good_quality", 28),
        ("M2/binary_vessel/binary_process", 28),
    ]:
        directory = (results / relative).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"{index}.png").write_bytes(b"x")

    counts = completion_counts(tmp_path, expected=32).set_index("stage")["images"].to_dict()

    assert counts["staged images"] == 32
    assert counts["M0 preprocessed"] == 30
    assert counts["M1 good quality"] == 28
    assert counts["M1 bad quality"] == 0
    assert counts["M3 disc features"] == 0
