# ABOUTME: Generates benchmark/analysis.ipynb, the Dice and feature-agreement analysis notebook.
# ABOUTME: Keeping the notebook generated means its source is reviewable and diffable as plain code.

"""Build the analysis notebook.

A notebook is the right artefact for exploring these results, but ``.ipynb`` JSON reviews badly and
merges worse. So the notebook is generated from this file: edit here, regenerate, and the diff stays
readable.

    uv run python -m benchmark.build_notebook          # write the notebook
    uv run python -m benchmark.build_notebook --run    # write it and execute it in place
"""

import argparse
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "analysis.ipynb"

CELLS = [
    (
        "markdown",
        """# AutoMorph on FIVES — segmentation accuracy and feature agreement

Two questions about the benchmark run:

1. **How well does M2 reproduce the expert vessel annotation?** Dice, IoU, sensitivity, specificity.
2. **Do the features AutoMorph reports survive that error?** The same six whole-image features are
   measured from the annotation itself, by the same retipy code, and compared image by image.

The second question is the interesting one. A segmentation can lose a quarter of the annotated
pixels and still report an almost-correct fractal dimension, or be pixel-accurate and report a badly
wrong width. Dice alone does not tell you which features you can trust.

Inputs, all produced by the benchmark:

| Path | Contents |
| --- | --- |
| `benchmark/results/vessel_scores.csv` | Per-image Dice/IoU/sensitivity/specificity |
| `benchmark/results/selection.csv` | The 32 chosen images and their FIVES labels |
| `.benchmark_run/Results/M3/Macular_Features.csv` | Features from the **predicted** segmentation |
| `.benchmark_run/Results/M3/Ground_truth_Macular_Features.csv` | Features from the **annotation** |

Regenerate the last one with `uv run python -m benchmark.ground_truth_features`.

> **`Average_width` is in nominal microns.** FIVES publishes no pixel size, so the benchmark writes a
> placeholder 0.008 mm/pixel. Prediction and truth are scaled by the same per-image figure, so the
> *comparison* holds; the absolute value is not a physical measurement.""",
    ),
    (
        "code",
        """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path.cwd() if (Path.cwd() / "benchmark").is_dir() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "benchmark" / "results"
RUN_RESULTS = REPO_ROOT / ".benchmark_run" / "Results"

FEATURES = [
    "Fractal_dimension",
    "Vessel_density",
    "Average_width",
    "Distance_tortuosity",
    "Squared_curvature_tortuosity",
    "Tortuosity_density",
]

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
pd.set_option("display.width", 200)
print(f"repository root: {REPO_ROOT}")""",
    ),
    ("markdown", "## Load and join"),
    (
        "code",
        """def strip_suffix(series):
    return series.str.replace(".png", "", regex=False)


scores = pd.read_csv(RESULTS / "vessel_scores.csv")
selection = pd.read_csv(RESULTS / "selection.csv")

predicted = pd.read_csv(RUN_RESULTS / "M3" / "Macular_Features.csv")[["Name"] + FEATURES]
truth = pd.read_csv(RUN_RESULTS / "M3" / "Ground_truth_Macular_Features.csv")[["Name"] + FEATURES]
predicted["key"] = strip_suffix(predicted["Name"])
truth["key"] = strip_suffix(truth["Name"])

# A -1 in any feature is retipy's failed-measurement marker, not a value.
for frame in (predicted, truth):
    frame[FEATURES] = frame[FEATURES].mask(frame[FEATURES] == -1)

paired = (
    predicted.drop(columns="Name")
    .merge(truth.drop(columns="Name"), on="key", suffixes=("_pred", "_gt"))
    .merge(scores[["key", "disease", "status", "dice", "iou", "sensitivity", "specificity"]], on="key")
)

print(f"{len(selection)} images selected, {(scores['status'] == 'ok').sum()} segmented, {len(paired)} paired")
print()
print(scores["status"].value_counts().to_string())""",
    ),
    (
        "markdown",
        """## 1. Where images dropped out

M1 is a gate, not a report: `merge_quality_assessment.py` sorts images into `Good_quality/` or
`Bad_quality/`, and M2 only reads the former. A rejected image yields no segmentation and no
features at all — so the drop-out belongs in the results, not in a footnote.""",
    ),
    (
        "code",
        """gate = selection[["key", "disease"]].merge(scores[["key", "status"]], on="key")
gate["reached_segmentation"] = gate["status"] == "ok"

summary = gate.groupby("disease").agg(
    selected=("key", "size"),
    segmented=("reached_segmentation", "sum"),
)
summary["rejected"] = summary["selected"] - summary["segmented"]
summary["rejected_pct"] = (100 * summary["rejected"] / summary["selected"]).round(1)
display(summary)

print("rejected images:", ", ".join(sorted(gate.loc[~gate["reached_segmentation"], "key"])))""",
    ),
    (
        "markdown",
        """## 2. Segmentation accuracy

Vessels occupy roughly 2% of the field of view, so **specificity and accuracy sit near 1 for any
prediction** and carry almost no information. Read Dice and IoU; read sensitivity to see which way
the errors fall.""",
    ),
    (
        "code",
        """ok = paired[paired["status"] == "ok"]
metrics = ["dice", "iou", "sensitivity", "specificity"]

display(ok[metrics].describe().round(4))
display(ok.groupby("disease")[metrics].agg(["mean", "std"]).round(4))""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

axes[0].hist(ok["dice"], bins=12, color="#4C72B0", edgecolor="white")
axes[0].axvline(ok["dice"].mean(), color="#C44E52", ls="--", label=f"mean {ok['dice'].mean():.3f}")
axes[0].set(xlabel="Dice", ylabel="images", title="Dice distribution")
axes[0].legend()

order = sorted(ok["disease"].unique())
axes[1].boxplot([ok.loc[ok["disease"] == d, "dice"] for d in order], labels=order)
for index, disease in enumerate(order, start=1):
    values = ok.loc[ok["disease"] == disease, "dice"]
    axes[1].scatter(np.full(len(values), index), values, alpha=0.6, s=18, color="#4C72B0", zorder=3)
axes[1].set(ylabel="Dice", title="Dice by disease")

fig.tight_layout()""",
    ),
    (
        "code",
        """# Sensitivity against specificity: where does the model trade one for the other?
fig, ax = plt.subplots(figsize=(5.2, 4))
for disease in order:
    subset = ok[ok["disease"] == disease]
    ax.scatter(subset["sensitivity"], subset["dice"], label=disease, s=42, alpha=0.8)
ax.set(xlabel="sensitivity (fraction of annotated vessel found)", ylabel="Dice",
       title="Dice is driven by sensitivity")
ax.legend(fontsize=8)
fig.tight_layout()

print("correlation of Dice with sensitivity:", round(ok["dice"].corr(ok["sensitivity"]), 3))
print("correlation of Dice with specificity:", round(ok["dice"].corr(ok["specificity"]), 3))""",
    ),
    (
        "markdown",
        """## 3. Feature agreement

For each feature, predicted against ground truth:

- **bias** — mean signed difference, and the same as a percentage of the truth's mean. A systematic
  offset. Correctable by calibration if it is stable.
- **MAPE** — mean absolute percentage error. Per-image reliability.
- **Pearson r** — linear agreement. **Spearman** — rank agreement, which is what matters if the
  feature is used to order or stratify patients rather than to report an absolute number.
- **Bland–Altman limits** — mean difference ± 1.96 SD, the interval containing ~95% of differences.

A feature can have a large bias and still be useful (rank-preserving), or a small bias and be
useless (uncorrelated). Both columns are needed.""",
    ),
    (
        "code",
        """def agreement(frame, features=FEATURES):
    rows = []
    for feature in features:
        pair = frame[[f"{feature}_pred", f"{feature}_gt"]].dropna()
        prediction, reference = pair[f"{feature}_pred"], pair[f"{feature}_gt"]
        difference = prediction - reference
        rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "gt_mean": reference.mean(),
                "pred_mean": prediction.mean(),
                "bias": difference.mean(),
                "bias_%": 100 * difference.mean() / reference.mean(),
                "MAPE_%": 100 * (difference.abs() / reference.abs()).mean(),
                "pearson_r": prediction.corr(reference),
                "spearman_r": prediction.corr(reference, method="spearman"),
                "BA_low": difference.mean() - 1.96 * difference.std(),
                "BA_high": difference.mean() + 1.96 * difference.std(),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


agreement_table = agreement(ok)
display(agreement_table.round(3))""",
    ),
    (
        "code",
        """# Is the linear agreement statistically distinguishable from zero at n=26?
for feature in FEATURES:
    pair = ok[[f"{feature}_pred", f"{feature}_gt"]].dropna()
    r, p = stats.pearsonr(pair[f"{feature}_pred"], pair[f"{feature}_gt"])
    rho, p_rho = stats.spearmanr(pair[f"{feature}_pred"], pair[f"{feature}_gt"])
    verdict = "" if p < 0.05 else "   <-- not significant"
    print(f"{feature:32s} r={r:6.3f} (p={p:.2g})   rho={rho:6.3f} (p={p_rho:.2g}){verdict}")""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

for ax, feature in zip(axes.ravel(), FEATURES):
    pair = ok[[f"{feature}_pred", f"{feature}_gt", "disease"]].dropna()
    for disease in order:
        subset = pair[pair["disease"] == disease]
        ax.scatter(subset[f"{feature}_gt"], subset[f"{feature}_pred"], label=disease, s=34, alpha=0.8)

    combined = pd.concat([pair[f"{feature}_gt"], pair[f"{feature}_pred"]])
    limits = [combined.min(), combined.max()]
    ax.plot(limits, limits, "k--", lw=1, alpha=0.6)
    r = pair[f"{feature}_pred"].corr(pair[f"{feature}_gt"])
    ax.set(xlabel="ground truth", ylabel="predicted", title=f"{feature}\\nr = {r:.3f}")

axes[0, 0].legend(fontsize=7)
fig.suptitle("Predicted vs ground-truth features (dashed line = perfect agreement)", y=1.0)
fig.tight_layout()""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

for ax, feature in zip(axes.ravel(), FEATURES):
    pair = ok[[f"{feature}_pred", f"{feature}_gt"]].dropna()
    mean = (pair[f"{feature}_pred"] + pair[f"{feature}_gt"]) / 2
    difference = pair[f"{feature}_pred"] - pair[f"{feature}_gt"]

    ax.scatter(mean, difference, s=34, alpha=0.75, color="#4C72B0")
    ax.axhline(0, color="k", lw=1, alpha=0.5)
    ax.axhline(difference.mean(), color="#C44E52", ls="-", lw=1.2, label=f"bias {difference.mean():.3g}")
    for limit in (difference.mean() - 1.96 * difference.std(), difference.mean() + 1.96 * difference.std()):
        ax.axhline(limit, color="#C44E52", ls="--", lw=1, alpha=0.7)
    ax.set(xlabel="mean of the two", ylabel="predicted - truth", title=feature)
    ax.legend(fontsize=7)

fig.suptitle("Bland-Altman: bias and 95% limits of agreement", y=1.0)
fig.tight_layout()""",
    ),
    (
        "markdown",
        """## 4. Does Dice predict feature error?

If it does, Dice is a usable proxy for feature trustworthiness on unlabelled data — which is the
only situation that matters in practice, since you will not have annotations. If it does not, a
good Dice is no guarantee that the reported morphometry is right.""",
    ),
    (
        "code",
        """rows = []
for feature in FEATURES:
    pair = ok[[f"{feature}_pred", f"{feature}_gt", "dice", "sensitivity"]].dropna()
    relative_error = (pair[f"{feature}_pred"] - pair[f"{feature}_gt"]).abs() / pair[f"{feature}_gt"].abs()
    rows.append(
        {
            "feature": feature,
            "corr(dice, |rel err|)": pair["dice"].corr(relative_error),
            "corr(sens, |rel err|)": pair["sensitivity"].corr(relative_error),
            "median_|rel err|_%": 100 * relative_error.median(),
        }
    )

display(pd.DataFrame(rows).set_index("feature").round(3))""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 3, figsize=(13, 7))

for ax, feature in zip(axes.ravel(), FEATURES):
    pair = ok[[f"{feature}_pred", f"{feature}_gt", "dice"]].dropna()
    relative_error = 100 * (pair[f"{feature}_pred"] - pair[f"{feature}_gt"]).abs() / pair[f"{feature}_gt"].abs()
    ax.scatter(pair["dice"], relative_error, s=34, alpha=0.75, color="#55A868")
    ax.set(xlabel="Dice", ylabel="|relative error| (%)",
           title=f"{feature}\\nr = {pair['dice'].corr(relative_error):.3f}")

fig.suptitle("Feature error against segmentation Dice", y=1.0)
fig.tight_layout()""",
    ),
    ("markdown", "## 5. Per-disease agreement\n\nSmall groups — read as indicative, not conclusive."),
    (
        "code",
        """for disease in order:
    subset = ok[ok["disease"] == disease]
    print(f"\\n=== {disease}  (n={len(subset)}, mean Dice {subset['dice'].mean():.3f}) ===")
    display(agreement(subset)[["bias_%", "MAPE_%", "pearson_r", "spearman_r"]].round(3))""",
    ),
    (
        "markdown",
        """## 6. Verdict per feature

The table below is generated from the numbers above, not hand-written, so it cannot drift out of
sync with the data. The thresholds are judgement calls and stated explicitly in the code.""",
    ),
    (
        "code",
        """def verdict(row):
    if row["pearson_r"] < 0.4 or row["MAPE_%"] > 50:
        return "unreliable - do not use per-image"
    if abs(row["bias_%"]) > 10:
        return "rank-usable, biased - calibrate before absolute use"
    if row["MAPE_%"] < 5:
        return "reliable"
    return "usable with care"


verdicts = agreement_table[["bias_%", "MAPE_%", "pearson_r", "spearman_r"]].copy()
verdicts["verdict"] = agreement_table.apply(verdict, axis=1)
display(verdicts.round(3).sort_values("pearson_r", ascending=False))""",
    ),
    (
        "code",
        """# Save the tables next to the rest of the benchmark output.
agreement_table.round(6).to_csv(RESULTS / "feature_agreement.csv")
verdicts.round(6).to_csv(RESULTS / "feature_verdicts.csv")
print(f"wrote feature_agreement.csv and feature_verdicts.csv to {RESULTS}")""",
    ),
]


def build():
    """Assemble the notebook from CELLS."""
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", action="store_true", help="execute the notebook in place")
    args = parser.parse_args(argv)

    notebook = build()

    if args.run:
        from nbclient import NotebookClient

        print("executing")
        NotebookClient(notebook, timeout=1200, kernel_name="python3", resources={
            "metadata": {"path": str(REPO_ROOT)}
        }).execute()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, str(args.output))
    print(f"{len(notebook.cells)} cells -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
