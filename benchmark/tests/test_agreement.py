# ABOUTME: Tests for the agreement statistics: ICC(2,1), bias, MAE, and informational strength.
# ABOUTME: ICC is checked against hand-computed ANOVA values, including the bias case Pearson misses.

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.agreement import agreement_table, icc21, informational_strength


def test_icc_of_identical_measurements_is_one():
    values = np.array([1.0, 3.0, 5.0, 7.0])
    assert icc21(values, values) == pytest.approx(1.0)


def test_icc_matches_hand_computed_anova():
    """n=3, k=2, a constant +1 offset.

    MSR=8, MSC=1.5, MSE=0  ->  ICC = (8-0) / (8 + 0 + 2*(1.5-0)/3) = 8/9.
    """
    first = np.array([1.0, 3.0, 5.0])
    second = np.array([2.0, 4.0, 6.0])
    assert icc21(first, second) == pytest.approx(8 / 9)


def test_icc_penalises_systematic_bias_where_pearson_does_not():
    """The whole reason to prefer ICC(2,1) here: it charges for the offset."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    biased = truth + 2.0

    assert np.corrcoef(truth, biased)[0, 1] == pytest.approx(1.0)
    assert icc21(truth, biased) < 0.75


def test_icc_of_unrelated_measurements_is_near_zero():
    generator = np.random.default_rng(0)
    first = generator.normal(size=400)
    second = generator.normal(size=400)
    assert abs(icc21(first, second)) < 0.2


def test_icc_scales_with_agreement():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    close = truth + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    loose = truth + np.array([1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5])
    assert icc21(truth, close) > icc21(truth, loose)


def test_icc_needs_at_least_two_subjects():
    with pytest.raises(ValueError, match="at least two"):
        icc21(np.array([1.0]), np.array([1.0]))


def test_icc_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        icc21(np.array([1.0, 2.0]), np.array([1.0]))


def test_icc_of_two_constant_columns_is_not_a_crash():
    """Zero variance everywhere leaves ICC undefined; it must be NaN, not a ZeroDivisionError."""
    constant = np.array([2.0, 2.0, 2.0, 2.0])
    assert np.isnan(icc21(constant, constant))


def paired_frame():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    return pd.DataFrame({"width_gt": truth, "width_pred": truth * 1.1 + 0.5})


def test_agreement_table_reports_the_requested_statistics():
    table = agreement_table(paired_frame(), ["width"])
    for column in ("n", "bias", "rel_bias_%", "MAE", "MAPE_%", "pearson_r", "spearman_r", "ICC21"):
        assert column in table.columns


def test_agreement_bias_and_mae_are_signed_and_unsigned():
    frame = pd.DataFrame({"width_gt": [1.0, 2.0, 3.0], "width_pred": [2.0, 1.0, 4.0]})
    row = agreement_table(frame, ["width"]).loc["width"]

    assert row["bias"] == pytest.approx((1.0 - 1.0 + 1.0) / 3)
    assert row["MAE"] == pytest.approx(1.0)


def test_agreement_relative_bias_is_a_percentage_of_the_truth():
    frame = pd.DataFrame({"width_gt": [10.0, 10.0], "width_pred": [11.0, 11.0]})
    assert agreement_table(frame, ["width"]).loc["width", "rel_bias_%"] == pytest.approx(10.0)


def test_agreement_ignores_failed_measurements():
    """retipy writes -1 for a failed measurement; it must not be averaged as a value."""
    frame = pd.DataFrame({"width_gt": [1.0, 2.0, np.nan], "width_pred": [1.0, 2.0, 5.0]})
    assert agreement_table(frame, ["width"]).loc["width", "n"] == 2


def test_informational_strength_is_iqr_over_median():
    frame = pd.DataFrame({"width_gt": [1.0, 2.0, 3.0, 4.0, 5.0]})
    row = informational_strength(frame, ["width"]).loc["width"]

    expected = (np.percentile([1, 2, 3, 4, 5], 75) - np.percentile([1, 2, 3, 4, 5], 25)) / 3.0
    assert row["gt_IQR_over_median"] == pytest.approx(expected)
    assert row["gt_median"] == pytest.approx(3.0)


def test_informational_strength_of_a_constant_feature_is_zero():
    frame = pd.DataFrame({"width_gt": [4.0, 4.0, 4.0, 4.0]})
    assert informational_strength(frame, ["width"]).loc["width", "gt_IQR_over_median"] == 0.0


def test_informational_strength_of_a_zero_median_is_not_a_crash():
    frame = pd.DataFrame({"width_gt": [-1.0, 0.0, 0.0, 1.0]})
    assert np.isnan(informational_strength(frame, ["width"]).loc["width", "gt_IQR_over_median"])
