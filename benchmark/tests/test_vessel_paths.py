# ABOUTME: Tests that detect_vessel_border returns traced paths, not BFS discovery order.
# ABOUTME: Every tortuosity measure assumes consecutive points are adjacent; these pin that.

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "M3_feature_whole_pic", "retipy"))
sys.path.insert(0, REPO_ROOT)

from retipy import retina as R

ADJACENT = 2**0.5 + 1e-9


def steps(path_x, path_y):
    """Distances between consecutive points of a traced path."""
    return [
        ((path_x[i] - path_x[i + 1]) ** 2 + (path_y[i] - path_y[i + 1]) ** 2) ** 0.5
        for i in range(len(path_x) - 1)
    ]


def test_order_as_path_keeps_every_step_adjacent_on_a_line():
    pixels = [(5, column) for column in range(20)]
    ordered = R.order_as_path(list(reversed(pixels)))
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]

    assert len(ordered) == 20
    assert max(steps(xs, ys)) <= ADJACENT


def test_order_as_path_starts_at_an_endpoint():
    pixels = [(5, column) for column in range(10)]
    ordered = R.order_as_path(pixels)
    assert ordered[0] in {(5, 0), (5, 9)}
    assert ordered[-1] in {(5, 0), (5, 9)}


def test_order_as_path_handles_a_corner():
    pixels = [(0, c) for c in range(10)] + [(r, 9) for r in range(1, 10)]
    ordered = R.order_as_path(pixels)
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]

    assert len(ordered) == len(pixels)
    assert max(steps(xs, ys)) <= ADJACENT


def test_order_as_path_handles_a_diagonal():
    pixels = [(i, i) for i in range(15)]
    ordered = R.order_as_path(pixels)
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    assert max(steps(xs, ys)) <= ADJACENT


def test_order_as_path_visits_each_pixel_once():
    pixels = [(3, c) for c in range(12)]
    ordered = R.order_as_path(pixels)
    assert len(ordered) == len(set(ordered))


def test_order_as_path_is_deterministic():
    pixels = [(4, c) for c in range(10)]
    assert R.order_as_path(pixels) == R.order_as_path(list(reversed(pixels)))


def test_order_as_path_of_a_single_pixel():
    assert R.order_as_path([(1, 1)]) == [(1, 1)]


def test_order_as_path_of_two_pixels():
    assert len(R.order_as_path([(1, 1), (1, 2)])) == 2


def test_order_as_path_of_a_branch_returns_a_simple_path():
    """A T-shape cannot be one path; the trace must stay simple and take the longest run."""
    stem = [(r, 5) for r in range(10)]
    arm = [(9, c) for c in range(6, 12)]
    ordered = R.order_as_path(stem + arm)
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]

    assert len(ordered) == len(set(ordered))
    assert max(steps(xs, ys)) <= ADJACENT
    assert len(ordered) >= 10  # at least the longest straight run


def test_order_as_path_of_a_closed_loop_stays_adjacent():
    loop = [(0, c) for c in range(6)] + [(r, 5) for r in range(1, 6)] + \
           [(5, c) for c in range(4, -1, -1)] + [(r, 0) for r in range(4, 0, -1)]
    ordered = R.order_as_path(loop)
    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    assert max(steps(xs, ys)) <= ADJACENT


class Skeleton:
    """A stand-in for Retina holding only what detect_vessel_border reads.

    Retina's constructor insists on a companion vessel image and a crop_info.csv reachable by string
    surgery on its store path — neither of which detect_vessel_border touches. It reads ``np_image``
    and ``shape``, so those are what this supplies.
    """

    def __init__(self, image):
        self.np_image = image
        self.shape = image.shape
        self._file_name = "synthetic"


def line_image(side=60):
    """A skeleton image holding one long horizontal vessel and one diagonal."""
    image = np.zeros((side, side), dtype=float)
    image[20, 5:55] = 255
    for i in range(30):
        image[i + 25, i + 5] = 255
    return image


def test_detect_vessel_border_returns_adjacent_steps_only():
    """The integration guarantee every tortuosity measure depends on."""
    vessels = R.detect_vessel_border(Skeleton(line_image()))

    assert vessels
    for vessel_x, vessel_y in vessels:
        if len(vessel_x) < 2:
            continue
        assert max(steps(vessel_x, vessel_y)) <= ADJACENT, "non-adjacent jump in a traced vessel"


def test_detect_vessel_border_preserves_a_long_vessel():
    vessels = R.detect_vessel_border(Skeleton(line_image()))
    assert max(len(vessel_x) for vessel_x, _ in vessels) >= 40


def test_traced_line_has_arc_length_equal_to_its_chord():
    """A straight vessel must have distance tortuosity 1.0 — the sharpest test of ordering."""
    from retipy.tortuosity_measures import distance_measure_tortuosity

    image = np.zeros((60, 60), dtype=float)
    image[20, 5:55] = 255

    vessels = [v for v in R.detect_vessel_border(Skeleton(image)) if len(v[0]) >= 40]
    assert vessels
    assert distance_measure_tortuosity(vessels[0][0], vessels[0][1]) == pytest.approx(1.0, abs=1e-6)


def test_a_wavy_vessel_has_tortuosity_above_one():
    """Ordering must not flatten real curvature away either."""
    from retipy.tortuosity_measures import distance_measure_tortuosity

    image = np.zeros((80, 80), dtype=float)
    for column in range(5, 75):
        row = 40 + int(6 * np.sin(column / 5.0))
        image[row, column] = 255

    vessels = [v for v in R.detect_vessel_border(Skeleton(image)) if len(v[0]) >= 40]
    assert vessels
    assert distance_measure_tortuosity(vessels[0][0], vessels[0][1]) > 1.05


def test_minimum_vessel_length_is_fifty():
    from retipy.tortuosity_measures import MIN_VESSEL_LENGTH

    assert MIN_VESSEL_LENGTH == 50


def test_evaluate_window_aggregates_by_median_not_mean():
    """A mean would be dragged up by the single extreme segment; a median must not be."""
    import inspect

    from retipy import tortuosity_measures

    source = inspect.getsource(tortuosity_measures.evaluate_window)
    assert "np.median(t2_values)" in source
    assert "np.median(t4_values)" in source
    assert "np.median(td_values)" in source
    assert "t2 = t2/vessel_count" not in source


def test_retina_stays_identical_between_the_two_retipy_copies():
    """The fix must land in both, or zone and whole-picture features diverge silently."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    zone = (root / "M3_feature_zone" / "retipy" / "retipy" / "retina.py").read_text()
    whole = (root / "M3_feature_whole_pic" / "retipy" / "retipy" / "retina.py").read_text()
    assert zone == whole
    assert "def order_as_path" in zone
