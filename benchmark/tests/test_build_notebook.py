# ABOUTME: Tests the notebook generator: both profiles assemble, and shared cells stay shared.
# ABOUTME: Guards the seam — every shared cell must work against names the profile's load cell binds.

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.build_notebook import PROFILES, build, cells_for

#: Names a profile's load cell must bind, because shared cells use them.
CONTRACT = ("RESULTS", "RUN_LABEL", "GATE_ENFORCED", "scores", "selection", "paired", "ok", "order",
            "truth_all", "REJECTED_BY_M1")


def assigned_names(source):
    """Every name bound at the top level of a code cell."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_profile_load_cell_binds_the_shared_contract(profile_name):
    bound = assigned_names(PROFILES[profile_name].load)
    missing = [name for name in CONTRACT if name not in bound]
    assert not missing, f"{profile_name} load cell does not bind {missing}"


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_every_cell_is_syntactically_valid(profile_name):
    for kind, source in cells_for(PROFILES[profile_name]):
        if kind == "code":
            ast.parse(source)


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_notebook_assembles(profile_name):
    notebook = build(PROFILES[profile_name])
    assert len(notebook.cells) > 15
    assert notebook.cells[0].cell_type == "markdown"


def test_the_two_profiles_write_to_different_notebooks():
    assert PROFILES["full"].notebook != PROFILES["vessel"].notebook
    assert PROFILES["vessel"].notebook.name == "analysis_M2_vessels.ipynb"


def test_the_two_profiles_share_the_analysis_cells():
    """The point of the refactor: only intro, load and the gate section may differ."""
    full = [source for _, source in cells_for(PROFILES["full"])]
    vessel = [source for _, source in cells_for(PROFILES["vessel"])]
    shared = set(full) & set(vessel)
    assert len(shared) >= 15, f"only {len(shared)} cells shared — the profiles have diverged"


def test_gate_sections_differ_between_profiles():
    full_gate = [source for _, source in PROFILES["full"].gate_cells]
    vessel_gate = [source for _, source in PROFILES["vessel"].gate_cells]
    assert not set(full_gate) & set(vessel_gate)


def test_vessel_profile_reads_the_vessel_results_directory():
    assert 'RESULTS = REPO_ROOT / "benchmark" / "results" / "M2_vessels"' in PROFILES["vessel"].load


def test_full_profile_does_not_read_the_vessel_results_directory():
    assert "M2_vessels" not in PROFILES["full"].load


def test_vessel_profile_tolerates_a_missing_full_run():
    """Grading the gate needs the full run's scores; absence must not be a crash."""
    assert "is_file()" in PROFILES["vessel"].load
    assert "REJECTED_BY_M1 = set()" in PROFILES["vessel"].load


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_gate_enforced_flag_matches_the_profile(profile_name):
    load = PROFILES[profile_name].load
    expected = "GATE_ENFORCED = True" if profile_name == "full" else "GATE_ENFORCED = False"
    assert expected in load
