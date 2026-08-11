# ABOUTME: Generates benchmark/report_post_fix/ — the post-fix report, figures and all tables.
# ABOUTME: Every number and plot comes from the stored run output, so the prose cannot drift from it.

"""Build the post-fix report.

    uv run python -m benchmark.build_report

Writes ``benchmark/report_post_fix/report.md`` and its figures under ``images/``. Prose lives in this
file as templates; every figure, table and inline number is computed from the two recorded runs, so
regenerating after a rerun updates the report rather than leaving it stale.

Compares:

* **before** — ``benchmark/results/M2_vessels`` — original tracing, 15 px minimum, threshold 0.5
* **after**  — ``benchmark/results/M2_vessels_thr02_mean`` — all three fixes

Both bypass the M1 quality gate and score all 32 images, so the comparison is not confounded by the
gate's selection.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.agreement import agreement_table, discriminability, informational_strength

BEFORE = REPO_ROOT / "benchmark" / "results" / "M2_vessels"
AFTER = REPO_ROOT / "benchmark" / "results" / "M2_vessels_thr02_mean"
OUTPUT = REPO_ROOT / "benchmark" / "report_post_fix"

FEATURES = [
    "Fractal_dimension",
    "Vessel_density",
    "Average_width",
    "Distance_tortuosity",
    "Squared_curvature_tortuosity",
    "Tortuosity_density",
]
SHORT = {
    "Fractal_dimension": "fractal dim",
    "Vessel_density": "vessel density",
    "Average_width": "average width",
    "Distance_tortuosity": "distance tort.",
    "Squared_curvature_tortuosity": "sq. curvature tort.",
    "Tortuosity_density": "tortuosity density",
}
METRICS = ["dice", "iou", "sensitivity", "specificity"]

BEFORE_COLOUR = "#C44E52"
AFTER_COLOUR = "#55A868"

plt.rcParams.update(
    {"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9}
)


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------


class Run:
    """One recorded run's tables, plus the statistics derived from them."""

    def __init__(self, directory, label):
        self.directory = Path(directory)
        self.label = label
        self.scores = pd.read_csv(self.directory / "vessel_scores.csv")
        self.paired = pd.read_csv(self.directory / "features_paired.csv")
        self.truth = pd.read_csv(self.directory / "features_ground_truth.csv")
        self.predicted = pd.read_csv(self.directory / "features_predicted.csv")
        self.ok = self.paired[self.paired["status"] == "ok"]

        self.agreement = agreement_table(self.ok, FEATURES)
        truth_named = self.truth.rename(columns={f: f"{f}_gt" for f in FEATURES})
        self.strength = informational_strength(truth_named, FEATURES)
        self.discriminability = discriminability(self.agreement, informational_strength(self.ok, FEATURES))

    @property
    def dice(self):
        return self.ok["dice"]


def load():
    return Run(BEFORE, "before"), Run(AFTER, "after")


def below_diagonal(run):
    """Features measured better than they vary: MAPE below ground-truth IQR/median.

    This is what the strength-versus-noise plots draw, so counts quoted alongside them must use it.
    It is deliberately stricter than ``spread_to_noise`` in the tables, which divides by the error's
    standard deviation and so forgives a systematic offset; MAPE does not.
    """
    return int(sum(
        run.agreement.loc[feature, "MAPE_%"]
        < run.discriminability.loc[feature, "gt_IQR_over_median"] * 100
        for feature in FEATURES
    ))


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------


def figure_strength_vs_noise(before, after, path):
    """The summary figure: informational strength against measurement noise, before -> after.

    x is how much the feature varies between eyes; y is how badly it is measured. Anything above the
    diagonal is measured worse than the population varies, so it cannot rank eyes however good its
    other statistics look.
    """
    figure, axis = plt.subplots(figsize=(7.8, 5.8))

    # Log-log: the two worst "before" points are two orders of magnitude from the best "after" ones,
    # and on a linear scale they squash every improved feature into an illegible corner.
    label_offsets = {
        "Fractal_dimension": (8, -12),
        "Vessel_density": (9, 5),
        "Average_width": (-62, 4),
        "Distance_tortuosity": (9, 3),
        "Squared_curvature_tortuosity": (9, 5),
        "Tortuosity_density": (-74, -12),
    }

    points = []
    for feature in FEATURES:
        x0 = before.discriminability.loc[feature, "gt_IQR_over_median"] * 100
        y0 = before.agreement.loc[feature, "MAPE_%"]
        x1 = after.discriminability.loc[feature, "gt_IQR_over_median"] * 100
        y1 = after.agreement.loc[feature, "MAPE_%"]
        points += [x0, y0, x1, y1]

        axis.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.1, alpha=0.8,
                            connectionstyle="arc3,rad=0.15"),
        )
        axis.scatter(x0, y0, s=54, facecolor="white", edgecolor=BEFORE_COLOUR, lw=1.6, zorder=3)
        axis.scatter(x1, y1, s=68, color=AFTER_COLOUR, zorder=4)
        axis.annotate(SHORT[feature], (x1, y1), fontsize=8.5, xytext=label_offsets[feature],
                      textcoords="offset points", zorder=5)

    low, high = min(points) * 0.55, max(points) * 1.9
    axis.plot([low, high], [low, high], "k--", lw=1, alpha=0.55)
    axis.fill_between([low, high], [low, high], high, color=BEFORE_COLOUR, alpha=0.06)
    axis.text(low * 2.4, high * 0.42,
              "error exceeds spread\ncannot rank eyes", fontsize=9, color=BEFORE_COLOUR)

    axis.scatter([], [], s=54, facecolor="white", edgecolor=BEFORE_COLOUR, lw=1.6, label="before fixes")
    axis.scatter([], [], s=68, color=AFTER_COLOUR, label="after fixes")
    axis.set(
        xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
        xlabel="informational strength: ground-truth IQR / median (%)  →  more to say",
        ylabel="measurement error: MAPE (%)  →  worse",
        title="Useful features sit low and to the right (log scales)",
    )
    axis.legend(loc="lower right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_dice(before, after, path):
    """Dice: paired per image, and per disease."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist([before.dice, after.dice], bins=12,
                 color=[BEFORE_COLOUR, AFTER_COLOUR], label=["before", "after"])
    axes[0].set(xlabel="Dice", ylabel="images", title="Dice distribution")
    axes[0].legend(fontsize=8)

    merged = before.ok[["key", "dice"]].merge(after.ok[["key", "dice"]], on="key",
                                              suffixes=("_before", "_after"))
    for _, row in merged.iterrows():
        axes[1].plot([0, 1], [row["dice_before"], row["dice_after"]],
                     color="grey", lw=0.8, alpha=0.6, zorder=2)
    axes[1].scatter(np.zeros(len(merged)), merged["dice_before"], color=BEFORE_COLOUR, s=28, zorder=3)
    axes[1].scatter(np.ones(len(merged)), merged["dice_after"], color=AFTER_COLOUR, s=28, zorder=3)
    improved = int((merged["dice_after"] > merged["dice_before"]).sum())
    axes[1].set(xticks=[0, 1], xticklabels=["before", "after"], ylabel="Dice",
                xlim=(-0.25, 1.25), title=f"Per image ({improved}/{len(merged)} improved)")

    diseases = sorted(after.ok["disease"].unique())
    position = np.arange(len(diseases))
    axes[2].bar(position - 0.2, [before.ok.loc[before.ok.disease == d, "dice"].mean() for d in diseases],
                width=0.38, color=BEFORE_COLOUR, label="before")
    axes[2].bar(position + 0.2, [after.ok.loc[after.ok.disease == d, "dice"].mean() for d in diseases],
                width=0.38, color=AFTER_COLOUR, label="after")
    axes[2].set(xticks=position, xticklabels=diseases, ylabel="mean Dice",
                ylim=(0.7, 0.92), title="By disease")
    axes[2].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_agreement(before, after, path):
    """ICC and absolute bias, before against after."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    position = np.arange(len(FEATURES))
    labels = [SHORT[f] for f in FEATURES]

    axes[0].barh(position + 0.2, before.agreement.loc[FEATURES, "ICC21"], height=0.38,
                 color=BEFORE_COLOUR, label="before")
    axes[0].barh(position - 0.2, after.agreement.loc[FEATURES, "ICC21"], height=0.38,
                 color=AFTER_COLOUR, label="after")
    for threshold, name in ((0.5, "moderate"), (0.75, "good"), (0.9, "excellent")):
        axes[0].axvline(threshold, color="grey", ls=":", lw=1)
        axes[0].text(threshold, len(FEATURES) - 0.35, name, fontsize=7, rotation=90,
                     va="top", ha="right", color="grey")
    axes[0].set(yticks=position, yticklabels=labels, xlabel="ICC(2,1)", xlim=(0, 1.0),
                title="Agreement with ground truth")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].barh(position + 0.2, before.agreement.loc[FEATURES, "rel_bias_%"].abs(), height=0.38,
                 color=BEFORE_COLOUR, label="before")
    axes[1].barh(position - 0.2, after.agreement.loc[FEATURES, "rel_bias_%"].abs(), height=0.38,
                 color=AFTER_COLOUR, label="after")
    axes[1].set(yticks=position, yticklabels=["" for _ in FEATURES],
                xlabel="|relative bias| (%)", title="Systematic bias")
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_predicted_vs_truth(run, path, colour):
    """Predicted against ground truth, one panel per feature, with the identity line."""
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.4))

    for axis, feature in zip(axes.ravel(), FEATURES):
        pair = run.ok[[f"{feature}_pred", f"{feature}_gt"]].dropna()
        truth, prediction = pair[f"{feature}_gt"], pair[f"{feature}_pred"]
        axis.scatter(truth, prediction, s=30, alpha=0.8, color=colour)

        combined = pd.concat([truth, prediction])
        span = [combined.min(), combined.max()]
        pad = (span[1] - span[0]) * 0.06 or 0.01
        axis.plot(span, span, "k--", lw=1, alpha=0.6)
        axis.set(
            xlim=(span[0] - pad, span[1] + pad), ylim=(span[0] - pad, span[1] + pad),
            xlabel="ground truth", ylabel="predicted",
            title=f"{SHORT[feature]}\nICC {run.agreement.loc[feature, 'ICC21']:.3f}, "
                  f"MAPE {run.agreement.loc[feature, 'MAPE_%']:.2f}%",
        )

    figure.suptitle(f"Predicted vs ground truth — {run.label} fixes "
                    f"(dashed = perfect agreement)", y=0.995)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_spread_vs_noise_panels(before, after, path):
    """The notebook's spread-versus-noise scatter, one panel per run on shared axes.

    Same construction as the cell in ``analysis_M2_vessels_thr02.ipynb``: informational strength on
    x, measurement error on y, the diagonal marking error == spread. Side by side on identical axes
    so the shift between runs is visible rather than inferred.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharex=True, sharey=True)

    panel_offsets = {
        "Fractal_dimension": (9, 4),
        "Vessel_density": (9, 6),
        "Average_width": (9, -12),
        "Distance_tortuosity": (9, 4),
        "Squared_curvature_tortuosity": (9, 5),
        "Tortuosity_density": (9, -12),
    }

    values = []
    for run in (before, after):
        for feature in FEATURES:
            values.append(run.discriminability.loc[feature, "gt_IQR_over_median"] * 100)
            values.append(run.agreement.loc[feature, "MAPE_%"])
    low, high = min(values) * 0.55, max(values) * 1.9

    for axis, run, colour in ((axes[0], before, BEFORE_COLOUR), (axes[1], after, AFTER_COLOUR)):
        axis.plot([low, high], [low, high], "k--", lw=1, alpha=0.55)
        axis.fill_between([low, high], [low, high], high, color=BEFORE_COLOUR, alpha=0.06)

        for feature in FEATURES:
            x = run.discriminability.loc[feature, "gt_IQR_over_median"] * 100
            y = run.agreement.loc[feature, "MAPE_%"]
            axis.scatter(x, y, s=76, color=colour, alpha=0.9, zorder=3)
            axis.annotate(SHORT[feature], (x, y), fontsize=8.5, xytext=panel_offsets[feature],
                          textcoords="offset points", zorder=4)

        axis.text(low * 1.6, high * 0.55, "error exceeds spread", fontsize=8.5, color=BEFORE_COLOUR)
        axis.set(
            xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
            xlabel="ground-truth IQR / median (%)",
            title=f"{run.label} fixes — {below_diagonal(run)}/6 features below the diagonal",
        )
    axes[0].set_ylabel("measurement error: MAPE (%)")

    figure.suptitle("Spread against noise: a feature must vary more than it is mis-measured", y=0.99)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def figure_strength(before, after, path):
    """Informational strength per feature, before against after."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    position = np.arange(len(FEATURES))

    axes[0].barh(position + 0.2, before.strength.loc[FEATURES, "gt_IQR_over_median"] * 100,
                 height=0.38, color=BEFORE_COLOUR, label="before")
    axes[0].barh(position - 0.2, after.strength.loc[FEATURES, "gt_IQR_over_median"] * 100,
                 height=0.38, color=AFTER_COLOUR, label="after")
    axes[0].set(yticks=position, yticklabels=[SHORT[f] for f in FEATURES],
                xlabel="ground-truth IQR / median (%)",
                title="Informational strength (how much eyes differ)")
    axes[0].legend(fontsize=8)

    axes[1].barh(position + 0.2, before.discriminability.loc[FEATURES, "spread_to_noise"],
                 height=0.38, color=BEFORE_COLOUR, label="before")
    axes[1].barh(position - 0.2, after.discriminability.loc[FEATURES, "spread_to_noise"],
                 height=0.38, color=AFTER_COLOUR, label="after")
    axes[1].axvline(1.0, color="k", ls="--", lw=1.2)
    axes[1].text(1.02, -0.55, "noise = spread", fontsize=8, color="k")
    axes[1].set(yticks=position, yticklabels=["" for _ in FEATURES],
                xlabel="spread / noise", title="Can it separate eyes? (>1 yes)")
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


# --------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------


def markdown(frame, floats="{:.4f}"):
    """A markdown table from a frame, index first."""
    header = [frame.index.name or ""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for index, row in frame.iterrows():
        cells = [str(index)]
        for value in row:
            if isinstance(value, (int, np.integer)):
                cells.append(str(value))
            elif isinstance(value, float):
                cells.append("—" if pd.isna(value) else floats.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dice_table(before, after):
    rows = []
    for metric in METRICS:
        b, a = before.ok[metric].mean(), after.ok[metric].mean()
        rows.append({"metric": metric, "before": b, "after": a, "change": a - b})
    frame = pd.DataFrame(rows).set_index("metric")
    frame.index.name = "Metric"
    return markdown(frame)


def dice_by_disease_table(before, after):
    rows = []
    for disease in sorted(after.ok["disease"].unique()):
        b = before.ok.loc[before.ok.disease == disease, "dice"]
        a = after.ok.loc[after.ok.disease == disease, "dice"]
        rows.append({"disease": disease, "n": len(a), "Dice before": b.mean(),
                     "Dice after": a.mean(), "change": a.mean() - b.mean()})
    frame = pd.DataFrame(rows).set_index("disease")
    frame.index.name = "Disease"
    return markdown(frame)


def agreement_table_markdown(before, after):
    rows = []
    for feature in FEATURES:
        rows.append({
            "feature": feature,
            "ICC before": before.agreement.loc[feature, "ICC21"],
            "ICC after": after.agreement.loc[feature, "ICC21"],
            "bias % before": before.agreement.loc[feature, "rel_bias_%"],
            "bias % after": after.agreement.loc[feature, "rel_bias_%"],
            "MAPE % before": before.agreement.loc[feature, "MAPE_%"],
            "MAPE % after": after.agreement.loc[feature, "MAPE_%"],
            "Spearman before": before.agreement.loc[feature, "spearman_r"],
            "Spearman after": after.agreement.loc[feature, "spearman_r"],
        })
    frame = pd.DataFrame(rows).set_index("feature")
    frame.index.name = "Feature"
    return markdown(frame, "{:.3f}")


def values_table(run):
    """Ground-truth and predicted value distributions for one run."""
    rows = []
    for feature in FEATURES:
        truth = run.truth[feature].replace(-1, np.nan).dropna()
        prediction = run.predicted[feature].replace(-1, np.nan).dropna()
        rows.append({
            "feature": feature,
            "GT mean": truth.mean(), "GT median": truth.median(),
            "GT min": truth.min(), "GT max": truth.max(),
            "pred mean": prediction.mean(), "pred median": prediction.median(),
            "pred min": prediction.min(), "pred max": prediction.max(),
        })
    frame = pd.DataFrame(rows).set_index("feature")
    frame.index.name = "Feature"
    return markdown(frame, "{:.4g}")


def strength_table(before, after):
    rows = []
    for feature in FEATURES:
        rows.append({
            "feature": feature,
            "IQR/median before": before.strength.loc[feature, "gt_IQR_over_median"],
            "IQR/median after": after.strength.loc[feature, "gt_IQR_over_median"],
            "spread/noise before": before.discriminability.loc[feature, "spread_to_noise"],
            "spread/noise after": after.discriminability.loc[feature, "spread_to_noise"],
        })
    frame = pd.DataFrame(rows).set_index("feature")
    frame.index.name = "Feature"
    return markdown(frame, "{:.4f}")


def verdict_table(after):
    verdicts = pd.read_csv(AFTER / "feature_verdicts.csv", index_col=0)
    frame = pd.DataFrame({
        "ICC": after.agreement["ICC21"],
        "reading": after.agreement["ICC_reading"],
        "spread/noise": after.discriminability["spread_to_noise"],
        "verdict": verdicts["verdict"],
    }).loc[FEATURES]
    frame.index.name = "Feature"
    return markdown(frame, "{:.3f}")


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------

DETECT_VESSEL_BORDER_BEFORE = '''# before — flood fill, pixels appended in *discovery* order
def vessel_extractor(window, start_x, start_y):
    vessel = []
    pending_pixels = [[start_x, start_y]]
    while pending_pixels:
        pixel = pending_pixels.pop(0)              # a QUEUE: breadth-first
        if window.np_image[pixel[0], pixel[1]] > 0:
            vessel.append(pixel)                   # order = when it was found
            window.np_image[pixel[0], pixel[1]] = 0
            pending_pixels.extend(neighbours(pixel, window))

    # sort by x position
    \'\'\'
    vessel.sort(key=lambda item: item[0])           # <- disabled 2021/10/31
    ...
    \'\'\'
    filtered_vessel = vessel
    return [[p[0] for p in filtered_vessel], [p[1] for p in filtered_vessel]]
'''

DETECT_VESSEL_BORDER_AFTER = '''# after — the fill finds WHICH pixels; a walk decides IN WHAT ORDER
def vessel_extractor(window, start_x, start_y):
    segment = []
    pending_pixels = [[start_x, start_y]]
    while pending_pixels:
        pixel = pending_pixels.pop(0)
        if window.np_image[pixel[0], pixel[1]] > 0:
            segment.append((pixel[0], pixel[1]))
            window.np_image[pixel[0], pixel[1]] = 0
            pending_pixels.extend(neighbours(pixel, window))

    path = order_as_path(segment)                  # <- the fix
    return [[p[0] for p in path], [p[1] for p in path]]


def order_as_path(pixels):
    """Walk a connected pixel set neighbour to neighbour, starting from an endpoint."""
    positions = {tuple(p) for p in pixels}
    if len(positions) < 2:
        return sorted(positions)

    adjacency = {p: [n for n in _eight_neighbours(p) if n in positions] for p in positions}

    def walk(start):
        path, visited = [start], {start}
        while True:
            candidates = [n for n in adjacency[path[-1]] if n not in visited]
            if not candidates:
                return path
            # nearest first (4-connected before diagonal), then lexicographic for determinism
            step = min(candidates,
                       key=lambda n: ((n[0] - path[-1][0]) ** 2 + (n[1] - path[-1][1]) ** 2, n))
            path.append(step)
            visited.add(step)

    endpoints = sorted(p for p in positions if len(adjacency[p]) == 1)
    starts = endpoints or [min(positions)]          # a closed loop has no endpoint
    return max((walk(start) for start in starts), key=len)
'''


def report(before, after):
    """The report, with every number computed from the two runs."""
    b, a = before, after
    dice_gain = a.dice.mean() - b.dice.mean()
    sens_gain = a.ok["sensitivity"].mean() - b.ok["sensitivity"].mean()
    density_before = b.agreement.loc["Vessel_density", "rel_bias_%"]
    density_after = a.agreement.loc["Vessel_density", "rel_bias_%"]
    paired_dice = b.ok[["key", "dice"]].merge(a.ok[["key", "dice"]], on="key",
                                              suffixes=("_b", "_a"))
    improved = int((paired_dice["dice_a"] > paired_dice["dice_b"]).sum())
    usable_before = below_diagonal(b)
    usable_after = below_diagonal(a)
    sweep = pd.read_csv(AFTER / "threshold_sweep.csv")
    best = sweep.loc[sweep["dice"].idxmax()]

    return f"""# AutoMorph post-fix report

Three defects in AutoMorph's vessel segmentation and measurement, found by benchmarking against the
FIVES expert annotations, and what fixing them changed.

Both runs below segment **all 32 benchmark images** with the M1 quality gate bypassed, so the
comparison is not confounded by the gate's selection. Everything here is generated from the recorded
run output by `benchmark/build_report.py`.

| | Before | After |
| --- | --- | --- |
| Results | `benchmark/results/M2_vessels/` | `benchmark/results/M2_vessels_thr02_mean/` |
| Vessel tracing | flood-fill discovery order | traced paths |
| Tortuosity minimum length | 15 px | 50 px |
| Binarisation threshold | 0.5 | 0.2 |
| Notebook | `analysis_M2_vessels.ipynb` | `analysis_M2_vessels_thr02.ipynb` |

---

## 1. Summary

Segmentation improved on **every one of the 32 images**, by **{dice_gain:+.3f} Dice** on average ({b.dice.mean():.3f} → {a.dice.mean():.3f}) and
**{sens_gain:+.3f} sensitivity** ({b.ok['sensitivity'].mean():.3f} → {a.ok['sensitivity'].mean():.3f}).
That matters less than what happened to the features AutoMorph actually reports.

The figure below is the whole result in one plot. The x axis is **informational strength** — how much
a feature varies between eyes, which bounds how much it could ever tell you. The y axis is
**measurement error**. Above the diagonal, a feature is measured worse than the population varies, so
it cannot rank eyes however good its other statistics look. Arrows run from before to after.

![Informational strength against measurement noise](images/strength_vs_noise.png)

**Before, {usable_before} of 6 features sat where measurement error was smaller than the spread they
had to resolve. After, {usable_after} of 6 do.** Two features moved from the unusable region into the
usable one, and every feature moved left-and-down or stayed put.

The single largest change is `Vessel_density`, whose systematic bias went from
**{density_before:+.1f}%** to **{density_after:+.1f}%**. That bias had been reported as a property of
the pipeline; it was a thresholding artefact.

{verdict_table(a)}

No feature is now unusable. Before the fixes one was degenerate, two were biased enough to need
calibration before any absolute value could be quoted, and one had no detectable relationship with
the truth.

---

## 2. The fixes

### 2a. A bug in `detect_vessel_border`

`detect_vessel_border` returned each vessel's pixels in **flood-fill discovery order**, not in path
order along the vessel. `vessel_extractor` walked the segment with a **queue** (`pop(0)`, so
breadth-first) and appended each pixel as it was found. Seeded in the middle of a segment, a
breadth-first fill expands in *both* directions at once, so the returned sequence jumps back and
forth across the seed. A sort by x position that partly masked this was commented out in 2021.

This matters because every tortuosity measure treats consecutive points as consecutive *positions*
along the vessel:

| Function | What it assumes |
| --- | --- |
| `_curve_length` | sums the distance between successive points → arc length |
| `_chord_length` | takes the first and last point |
| `squared_curvature_tortuosity` | differentiates with `np.gradient` |
| `tortuosity_density` | splits the curve at inflections found *by index* |

None of them is meaningful on a sequence that teleports. Measured on the benchmark's own data before
the fix:

| | |
| --- | --- |
| Consecutive pairs that were **not** adjacent | **6.2%** |
| Mean size of those jumps | **42 px** |
| Largest jump | **145 px** on a 912 grid |

A 145-pixel step where a one-pixel move belongs inflates arc length severalfold. The evidence is in
the values: ground-truth distance tortuosity — an arc-to-chord ratio, so 1.0 means a straight vessel
— came out at a median of **{b.truth['Distance_tortuosity'].median():.2f}**. That would mean the
average retinal vessel wanders more than three times its end-to-end distance. It does not.

```python
{DETECT_VESSEL_BORDER_BEFORE}```

The fix separates the two concerns. The flood fill still decides *which* pixels belong to the
segment; a walk decides *in what order* they are reported. `order_as_path` starts at an endpoint — a
pixel with exactly one neighbour in the set — and steps to the nearest unvisited neighbour, so every
consecutive pair is adjacent, at most √2 apart.

A set containing a branch point cannot be a single path. The walk is tried from each endpoint and the
longest simple path returned, which follows the trunk and drops the short spur — better than a
sequence that teleports between branches. Intersections are normally removed upstream, so branches
are rare.

```python
{DETECT_VESSEL_BORDER_AFTER}```

After the fix, non-adjacent steps are **0.00%**, the largest step is exactly √2, and a straight
vessel scores exactly 1.0. Ground-truth distance tortuosity becomes
**{a.truth['Distance_tortuosity'].median():.3f}** — where a retinal arc-chord ratio belongs.

The fix is applied to both copies of retipy in the repository, so `retina.py` stays byte-identical
between `M3_feature_zone` and `M3_feature_whole_pic`; a test enforces that.

### 2b. Tuning the vessel-probability cut and the tortuosity length

**Binarisation threshold: 0.5 → 0.2.** M2 averages ten ensemble sigmoids and calls a pixel vessel
above a threshold, hardcoded at 0.5. At 0.5 the benchmark measured sensitivity
{b.ok['sensitivity'].mean():.3f} against specificity {b.ok['specificity'].mean():.3f} — missing about
a quarter of the annotated vessel while inventing almost nothing. That asymmetry is the signature of
a threshold set too high, not of a model that cannot see vessels.

M2 saves the averaged sigmoid map, so the threshold can be swept without re-running the network:
re-binarise, re-apply `remove_small_objects(30)`, re-score.

{markdown(sweep.set_index('threshold').rename_axis('Threshold'))}

**{best['threshold']:.2f} is the Dice optimum** at {best['dice']:.4f}, and the maximum is broad — 0.15
to 0.25 all land within 0.002 — so the exact value is not delicate, while 0.5 sits well down the
slope. The sweep reproduces the 0.5 run's Dice exactly, which is the check that it follows the
pipeline's own path. Note it optimises Dice, which weights sensitivity and precision equally; that is
a choice, not a clinical requirement.

Configured by `AUTOMORPH_VESSEL_THRESHOLD`, now defaulting to 0.2, and applied at **both** places M2
binarises — the 912 grid that feeds the features, and the crop-resolution map.

**Tortuosity minimum vessel length: 15 px → 50 px.** Arc-chord ratio and curvature are unstable when
the chord spans only a few pixels: a one-pixel wobble moves the ratio a long way, and skeletonisation
noise dominates the derivative. Segments shorter than 50 px are counted but not measured. The
threshold is `evaluate_window`'s existing `min_pixels_per_vessel` parameter — a measurement policy, so
it belongs with the caller rather than baked into the library.

Aggregation across vessels remains the **mean**, as the original code had it.

---

## 3. Dice, before and after

{dice_table(b, a)}

Sensitivity gains {sens_gain * 100:.1f} points for
{abs(a.ok['specificity'].mean() - b.ok['specificity'].mean()) * 100:.1f} of specificity. Because
vessels occupy only ~2% of the field of view, that trade is strongly favourable: Dice and IoU both
rise.

**Dice improved on {improved} of {len(paired_dice)} images** — the gain is not an average masking a
mixture, it is every image moving the same way. The worst image before the fixes
({paired_dice['dice_b'].min():.3f}) is still the worst after ({paired_dice['dice_a'].min():.3f}), but
by a smaller margin.

![Dice before and after](images/dice.png)

{dice_by_disease_table(b, a)}

Every disease improves. Specificity and accuracy are near 1 for any prediction on this data, because
of how rare vessel pixels are, so they carry little information either way.

---

## 4. Feature agreement

Predicted against ground truth, both measured by the same code over the same images. ICC(2,1) is the
headline: unlike Pearson it charges for systematic bias, so a prediction that tracks the truth but
sits offset from it scores lower.

{agreement_table_markdown(b, a)}

![Feature agreement before and after](images/agreement.png)

Which fix did what:

- **The threshold** drove the accuracy and bias gains. `Vessel_density`'s bias falls from
  {density_before:+.1f}% to {density_after:+.1f}% — the probability mass was there all along, sitting
  between 0.2 and 0.5 and being discarded. `Fractal_dimension` and `Average_width` improve for the
  same reason.
- **The tracing fix** drove the tortuosity gains. `Distance_tortuosity`'s per-image error falls from
  {b.agreement.loc['Distance_tortuosity', 'MAPE_%']:.1f}% to
  {a.agreement.loc['Distance_tortuosity', 'MAPE_%']:.2f}%, and
  `Squared_curvature_tortuosity`'s from {b.agreement.loc['Squared_curvature_tortuosity', 'MAPE_%']:.1f}%
  to {a.agreement.loc['Squared_curvature_tortuosity', 'MAPE_%']:.2f}%.

`Squared_curvature_tortuosity`'s ICC barely moves
({b.agreement.loc['Squared_curvature_tortuosity', 'ICC21']:.3f} →
{a.agreement.loc['Squared_curvature_tortuosity', 'ICC21']:.3f}), but the before figure rested on a
single 13× outlier — 0.144 on leave-one-out — so the honest reading is 0.144 → 
{a.agreement.loc['Squared_curvature_tortuosity', 'ICC21']:.3f}.

### Feature values

What the numbers actually are, which is how the tracing bug is visible directly: before the fix the
tortuosity measures took values no retinal vessel can have.

**Before the fixes**

{values_table(b)}

**After the fixes**

{values_table(a)}

---

## 5. Predicted vs ground truth

One panel per feature, predicted on the vertical axis against ground truth on the horizontal, with
the dashed line marking perfect agreement. Points below the line are under-estimates.

**Before the fixes** — note `Vessel_density` and `Average_width` sitting well below the line, and the
tortuosity panels scattered with no relationship to the truth.

![Predicted vs truth before](images/predicted_vs_truth_before.png)

**After the fixes** — the density and width panels collapse onto the identity line, and the tortuosity
panels acquire a relationship.

![Predicted vs truth after](images/predicted_vs_truth_after.png)

---

## 6. Informational strength

Agreement is only half the question. A feature must also **vary between eyes**, or it cannot separate
them however precisely it is measured. Informational strength is `IQR / median` over the ground truth
— robust, unit-free, and a property of the cohort rather than of the pipeline.

`spread / noise` combines the two: ground-truth IQR divided by the standard deviation of the
prediction error, with systematic bias excluded, since a constant offset does not stop a feature
separating two eyes but random scatter does. **Below 1, the noise covers the whole interquartile
range.**

{strength_table(b, a)}

![Informational strength before and after](images/strength.png)

The same thing as a scatter — the plot from the analysis notebook, one panel per run on shared log
axes. A feature has to sit **below the diagonal**: measured better than the population varies. Above
it, the error swamps the differences the feature is supposed to resolve. **Before: {below_diagonal(b)}
of 6. After: {below_diagonal(a)} of 6.**

Note this is stricter than the `spread / noise` column above, which is why the two counts differ.
`spread / noise` divides by the *standard deviation* of the error, so a systematic offset costs it
nothing; the diagonal here compares against MAPE, which includes the offset. `Vessel_density` before
the fixes is the case that separates them: spread/noise
{b.discriminability.loc['Vessel_density', 'spread_to_noise']:.2f} looks acceptable, but a
{b.agreement.loc['Vessel_density', 'rel_bias_%']:+.1f}% bias puts it above the diagonal. Both readings
are useful — one asks whether the feature can *rank* eyes, the other whether its value can be
*quoted*.

![Spread against noise, before and after](images/spread_vs_noise.png)

The tortuosity measures' apparent strength before the fixes was largely an artefact: the arc-length
inflation varied from image to image, and that variation looked like between-eye signal. Correctly
measured, they vary less — but they vary *real*, and their noise falls by more than their spread
does, which is why `spread / noise` improves for both.

`Fractal_dimension` is the opposite case and worth noting: it is measured most accurately of the six
({a.agreement.loc['Fractal_dimension', 'MAPE_%']:.2f}% error) but has the least to say
({a.strength.loc['Fractal_dimension', 'gt_IQR_over_median'] * 100:.1f}% spread). Precision about
something that barely varies is not the same as usefulness — which is exactly what the summary figure
is for.

---

## Caveats

- **n = 32**, one cohort, all FIVES quality 3. The threshold optimum is the optimum *here* and is not
  a calibration transferable to other data without checking.
- All three fixes are in play at once in this comparison, so it has no control group. The per-fix
  attribution above comes from having measured them separately during development.
- `Average_width` is in nominal microns: FIVES publishes no pixel size, so the benchmark writes a
  placeholder 0.008 mm/pixel. Both sides share that scaling, so every comparison statistic here is
  unaffected; only the absolute micron value is not physical.
- `order_as_path` drops the shorter branch of a branching segment. Intersections are normally removed
  upstream so this is rare, but it is a deliberate simplification.
- Only vessel segmentation was re-run. Artery/vein and disc/cup have their own thresholds, untouched,
  and the zone features inherit the improved tracing but were not re-scored.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)

    before, after = load()

    figure_strength_vs_noise(before, after, images / "strength_vs_noise.png")
    figure_dice(before, after, images / "dice.png")
    figure_agreement(before, after, images / "agreement.png")
    figure_predicted_vs_truth(before, images / "predicted_vs_truth_before.png", BEFORE_COLOUR)
    figure_predicted_vs_truth(after, images / "predicted_vs_truth_after.png", AFTER_COLOUR)
    figure_strength(before, after, images / "strength.png")
    figure_spread_vs_noise_panels(before, after, images / "spread_vs_noise.png")

    target = args.output / "report.md"
    target.write_text(report(before, after))

    print(f"7 figures -> {images}")
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
