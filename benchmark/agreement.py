# ABOUTME: Agreement statistics between predicted and ground-truth features, including ICC(2,1).
# ABOUTME: Also the informational strength of a feature, IQR/median over the expert annotations.

"""How well do predicted features agree with the truth, and is the feature worth measuring at all?

Two separate questions, and a feature has to pass both.

**Agreement** — does the predicted value match the true one? Reported as:

======================  =======================================================================
``bias``                mean signed difference. A systematic offset, correctable by calibration.
``rel_bias_%``          the same as a percentage of the truth's mean.
``MAE``                 mean absolute error, in the feature's own units.
``MAPE_%``              mean absolute error as a percentage. Per-image reliability.
``pearson_r``           linear association. Blind to bias: a constant offset still scores 1.0.
``spearman_r``          rank association. What matters if the feature is used to order patients.
``ICC21``               intraclass correlation, two-way random effects, absolute agreement,
                        single measurement — Shrout & Fleiss ICC(2,1). Unlike Pearson it charges
                        for systematic bias, so it is the honest headline number here.
``BA_low``/``BA_high``  Bland-Altman limits, mean difference ± 1.96 SD.
======================  =======================================================================

**Informational strength** — does the feature vary enough between eyes to carry information?
``IQR / median`` over the expert annotations: a robust, unit-free measure of how much spread the
feature has in the population. A feature with almost no spread cannot separate patients however
precisely it is measured, so this is what turns "small error" into "useful measurement": an error
of 2% is only impressive if the feature's own spread is much wider than 2%.
"""

import numpy as np
import pandas as pd

#: Koo & Li's conventional reading of ICC. Boundaries are conventions, not facts.
ICC_BANDS = ((0.90, "excellent"), (0.75, "good"), (0.50, "moderate"), (0.0, "poor"))


def icc21(first, second):
    """Shrout & Fleiss ICC(2,1): two-way random effects, absolute agreement, single measurement.

    Both arguments are one measurement each of the same subjects — here the predicted and the
    ground-truth value of one feature, one entry per image.

    :return: the ICC, or NaN if there is no variance to partition.
    :raises ValueError: if the inputs differ in length or hold fewer than two subjects.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if first.shape != second.shape:
        raise ValueError("icc21 needs both measurements to be the same length")

    matrix = np.column_stack([first, second])
    subjects, raters = matrix.shape
    if subjects < 2:
        raise ValueError("icc21 needs at least two subjects")

    grand_mean = matrix.mean()
    total = ((matrix - grand_mean) ** 2).sum()
    between_subjects = raters * ((matrix.mean(axis=1) - grand_mean) ** 2).sum()
    between_raters = subjects * ((matrix.mean(axis=0) - grand_mean) ** 2).sum()
    residual = total - between_subjects - between_raters

    subject_ms = between_subjects / (subjects - 1)
    rater_ms = between_raters / (raters - 1)
    residual_ms = residual / ((subjects - 1) * (raters - 1))

    denominator = subject_ms + (raters - 1) * residual_ms + raters * (rater_ms - residual_ms) / subjects
    if denominator == 0:
        return float("nan")
    return (subject_ms - residual_ms) / denominator


def icc_interval(first, second, resamples=2000, seed=0):
    """A bootstrap 95% interval for :func:`icc21`, resampling subjects.

    With a couple of dozen images a point estimate on its own overstates what is known, so the
    interval is reported next to it.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    generator = np.random.default_rng(seed)

    estimates = []
    for _ in range(resamples):
        index = generator.integers(0, len(first), len(first))
        if len(np.unique(index)) < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = icc21(first[index], second[index])
        if np.isfinite(estimate):
            estimates.append(estimate)

    if not estimates:
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def icc_label(value):
    """The conventional word for an ICC value."""
    if not np.isfinite(value):
        return "undefined"
    for threshold, label in ICC_BANDS:
        if value >= threshold:
            return label
    return "poor"


def _paired(frame, feature):
    """The rows where both the predicted and the true value of ``feature`` are present."""
    pair = frame[[f"{feature}_pred", f"{feature}_gt"]].dropna()
    return pair[f"{feature}_pred"], pair[f"{feature}_gt"]


def agreement_table(frame, features, interval=False):
    """Agreement statistics per feature.

    :param frame: one row per image, with ``<feature>_pred`` and ``<feature>_gt`` columns.
    :param features: feature names, without the suffix.
    :param interval: also compute bootstrap intervals for the ICC. Slower.
    """
    rows = []
    for feature in features:
        prediction, truth = _paired(frame, feature)
        difference = prediction - truth

        row = {
            "feature": feature,
            "n": len(prediction),
            "gt_mean": truth.mean(),
            "pred_mean": prediction.mean(),
            "bias": difference.mean(),
            "rel_bias_%": 100 * difference.mean() / truth.mean() if truth.mean() else float("nan"),
            "MAE": difference.abs().mean(),
            "MAPE_%": 100 * (difference.abs() / truth.abs()).mean(),
            "pearson_r": prediction.corr(truth),
            "spearman_r": prediction.corr(truth, method="spearman"),
            "ICC21": icc21(prediction, truth) if len(prediction) > 1 else float("nan"),
            "BA_low": difference.mean() - 1.96 * difference.std(),
            "BA_high": difference.mean() + 1.96 * difference.std(),
        }
        row["ICC_reading"] = icc_label(row["ICC21"])

        if interval:
            row["ICC_lo"], row["ICC_hi"] = icc_interval(prediction.to_numpy(), truth.to_numpy())

        rows.append(row)

    return pd.DataFrame(rows).set_index("feature")


def informational_strength(frame, features, suffix="_gt"):
    """Spread of each feature over the ground truth, as ``IQR / median``.

    :param frame: one row per image, holding ``<feature><suffix>`` columns.
    :return: median, IQR, and their ratio. NaN ratio where the median is zero.
    """
    rows = []
    for feature in features:
        values = frame[f"{feature}{suffix}"].dropna()
        first_quartile, median, third_quartile = (
            values.quantile(0.25),
            values.median(),
            values.quantile(0.75),
        )
        spread = third_quartile - first_quartile
        rows.append(
            {
                "feature": feature,
                "n": len(values),
                "gt_median": median,
                "gt_IQR": spread,
                "gt_IQR_over_median": spread / median if median else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def discriminability(agreement, strength):
    """Compare each feature's spread against the noise in measuring it.

    A feature is only useful for telling eyes apart if its between-eye spread is wide relative to
    the error in measuring it. ``spread_to_noise`` is the ground-truth IQR divided by the standard
    deviation of the prediction error — systematic bias is excluded, because a constant offset does
    not stop a feature separating one eye from another. Below about 1 the measurement error covers
    the whole interquartile range and the feature cannot rank eyes reliably.
    """
    noise = (agreement["BA_high"] - agreement["BA_low"]) / (2 * 1.96)
    table = pd.DataFrame(
        {
            "gt_IQR_over_median": strength["gt_IQR_over_median"],
            "error_sd": noise,
            "spread_to_noise": strength["gt_IQR"] / noise,
        }
    )
    return table
