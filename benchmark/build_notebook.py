# ABOUTME: Generates the Dice and feature-agreement notebooks for the full and vessel-only runs.
# ABOUTME: Keeping them generated means their source is reviewable and diffable as plain code.

"""Build an analysis notebook.

A notebook is the right artefact for exploring these results, but ``.ipynb`` JSON reviews badly and
merges worse. So the notebooks are generated from this file: edit here, regenerate, and the diff
stays readable.

    uv run python -m benchmark.build_notebook                          # full run
    uv run python -m benchmark.build_notebook --profile vessel --run   # vessel-only run, executed

Two runs, one analysis
----------------------
The full run and the vessel-only run ask the same questions of different data, so nearly every cell
is shared. The seam is :class:`Profile`: its ``load`` cell establishes a fixed set of names —
``scores``, ``paired``, ``ok``, ``order``, ``truth_all``, ``REJECTED_BY_M1``, ``GATE_ENFORCED`` —
and every shared cell downstream works against those, unaware of which run it is reading.

Only one section genuinely differs. Where the gate is *enforced* the interesting question is what it
dropped; where it is *bypassed* the interesting question is whether dropping it was right.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURE_LIST = """FEATURES = [
    "Fractal_dimension",
    "Vessel_density",
    "Average_width",
    "Distance_tortuosity",
    "Squared_curvature_tortuosity",
    "Tortuosity_density",
]"""


# --------------------------------------------------------------------------------------------------
# Shared preamble
# --------------------------------------------------------------------------------------------------

SETUP = (
    "code",
    f"""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path.cwd() if (Path.cwd() / "benchmark").is_dir() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.agreement import (
    agreement_table,
    discriminability,
    icc21,
    icc_label,
    informational_strength,
)

{FEATURE_LIST}

plt.rcParams.update({{"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9}})
pd.set_option("display.width", 200)
print(f"repository root: {{REPO_ROOT}}")""",
)


# --------------------------------------------------------------------------------------------------
# Section 2 onward: shared analysis
# --------------------------------------------------------------------------------------------------

ACCURACY = [
    (
        "markdown",
        """## 2. Segmentation accuracy

Vessels occupy roughly 2% of the field of view, so **specificity and accuracy sit near 1 for any
prediction** and carry almost no information. Read Dice and IoU; read sensitivity to see which way
the errors fall.""",
    ),
    (
        "code",
        """metrics = ["dice", "iou", "sensitivity", "specificity"]

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

axes[1].boxplot([ok.loc[ok["disease"] == d, "dice"] for d in order], labels=order)
for index, disease in enumerate(order, start=1):
    values = ok.loc[ok["disease"] == disease, "dice"]
    axes[1].scatter(np.full(len(values), index), values, alpha=0.6, s=18, color="#4C72B0", zorder=3)
axes[1].set(ylabel="Dice", title="Dice by disease")

fig.tight_layout()""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(5.2, 4))
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
]

AGREEMENT = [
    (
        "markdown",
        """## 3. Feature agreement

For each feature, predicted against ground truth. Definitions live in `benchmark/agreement.py`,
which is unit-tested — ICC in particular against hand-computed ANOVA values.

- **bias** / **rel_bias_%** — mean signed difference, absolute and as a percentage of the truth's
  mean. A systematic offset, correctable by calibration if it is stable.
- **MAE** / **MAPE_%** — mean absolute error, in the feature's own units and as a percentage.
- **Pearson r** — linear association. Deliberately blind to bias: a constant offset still scores
  1.0. **Spearman** — rank association, which is what matters if the feature orders patients.
- **ICC(2,1)** — intraclass correlation, two-way random effects, absolute agreement, single
  measurement. Unlike Pearson it *charges for systematic bias*, so it is the honest headline
  number. Bands: <0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, >0.9 excellent. Bootstrap 95%
  intervals are shown because n is small.
- **Bland–Altman limits** — mean difference ± 1.96 SD.

The gap between Pearson and ICC is the interesting column: where Pearson is high and ICC much
lower, the prediction tracks the truth but is offset from it.""",
    ),
    (
        "code",
        """agreement = agreement_table(ok, FEATURES, interval=True)

display(
    agreement[
        ["n", "gt_mean", "pred_mean", "bias", "rel_bias_%", "MAE", "MAPE_%",
         "pearson_r", "spearman_r", "ICC21", "ICC_lo", "ICC_hi", "ICC_reading"]
    ].round(3)
)""",
    ),
    (
        "code",
        """# Where does ICC disagree with Pearson? That gap is exactly the systematic bias.
gap = pd.DataFrame(
    {
        "pearson_r": agreement["pearson_r"],
        "ICC21": agreement["ICC21"],
        "pearson_minus_ICC": agreement["pearson_r"] - agreement["ICC21"],
        "rel_bias_%": agreement["rel_bias_%"],
    }
).sort_values("pearson_minus_ICC", ascending=False)

display(gap.round(3))
print("A large gap means the prediction tracks the truth but sits offset from it.")""",
    ),
    (
        "code",
        """# Is the linear agreement statistically distinguishable from zero at this sample size?
for feature in FEATURES:
    pair = ok[[f"{feature}_pred", f"{feature}_gt"]].dropna()
    r, p = stats.pearsonr(pair[f"{feature}_pred"], pair[f"{feature}_gt"])
    rho, p_rho = stats.spearmanr(pair[f"{feature}_pred"], pair[f"{feature}_gt"])
    verdict = "" if p < 0.05 else "   <-- not significant"
    print(f"{feature:32s} n={len(pair):3d}  r={r:6.3f} (p={p:.2g})   rho={rho:6.3f} (p={p_rho:.2g}){verdict}")""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(8, 4))
position = np.arange(len(FEATURES))

ax.barh(position + 0.2, agreement.loc[FEATURES, "pearson_r"], height=0.36,
        label="Pearson r (bias-blind)", color="#8FA8CC")
ax.barh(position - 0.2, agreement.loc[FEATURES, "ICC21"], height=0.36,
        label="ICC(2,1) (charges for bias)", color="#4C72B0")

for threshold, name in ((0.9, "excellent"), (0.75, "good"), (0.5, "moderate")):
    ax.axvline(threshold, color="grey", ls=":", lw=1)
    ax.text(threshold, len(FEATURES) - 0.4, name, fontsize=7, rotation=90,
            va="top", ha="right", color="grey")

ax.set(yticks=position, yticklabels=FEATURES, xlabel="agreement", xlim=(0, 1.02),
       title="Pearson vs ICC(2,1): the gap is systematic bias")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()""",
    ),
]

LEVERAGE = [
    (
        "markdown",
        """### Does any single image carry the result?

At this sample size one extreme value can manufacture a correlation. Leave-one-out on the ICC finds
it: if dropping a single image moves ICC by more than about 0.15, the agreement is that image's
doing, not the model's.

This is not hypothetical — in the vessel-only run `Squared_curvature_tortuosity` scores ICC 0.544
until one 13x outlier is removed, whereupon it collapses to 0.144. Spearman, being rank-based, is
immune to leverage and stays low throughout, which is why both are reported.""",
    ),
    (
        "code",
        """rows = []
for feature in FEATURES:
    pair = ok[["key", f"{feature}_pred", f"{feature}_gt"]].dropna()
    predicted = pair[f"{feature}_pred"].to_numpy()
    reference = pair[f"{feature}_gt"].to_numpy()
    full = icc21(predicted, reference)

    worst_key, worst_icc = None, full
    for position in range(len(pair)):
        keep = np.arange(len(pair)) != position
        without = icc21(predicted[keep], reference[keep])
        if abs(without - full) > abs(worst_icc - full):
            worst_key, worst_icc = pair["key"].iloc[position], without

    rows.append(
        {
            "feature": feature,
            "ICC_all": full,
            "most_influential_image": worst_key,
            "ICC_without_it": worst_icc,
            "swing": worst_icc - full,
            "fragile": abs(worst_icc - full) > 0.15,
        }
    )

leverage = pd.DataFrame(rows).set_index("feature")
display(leverage.round(3))

fragile = leverage[leverage["fragile"]]
if len(fragile):
    print("Leverage-dependent — do not read the ICC point estimate alone:")
    for feature, row in fragile.iterrows():
        print(f"  {feature}: {row['ICC_all']:.3f} -> {row['ICC_without_it']:.3f} "
              f"without {row['most_influential_image']}")
else:
    print("No single image moves any ICC by more than 0.15.")""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

for ax, feature in zip(axes.ravel(), FEATURES):
    pair = ok[[f"{feature}_pred", f"{feature}_gt", "disease"]].dropna()
    for disease in order:
        subset = pair[pair["disease"] == disease]
        ax.scatter(subset[f"{feature}_gt"], subset[f"{feature}_pred"], label=disease, s=34, alpha=0.8)

    combined_values = pd.concat([pair[f"{feature}_gt"], pair[f"{feature}_pred"]])
    limits = [combined_values.min(), combined_values.max()]
    ax.plot(limits, limits, "k--", lw=1, alpha=0.6)
    r = pair[f"{feature}_pred"].corr(pair[f"{feature}_gt"])
    ax.set(xlabel="ground truth", ylabel="predicted",
           title=f"{feature}\\nr = {r:.3f}, ICC = {agreement.loc[feature, 'ICC21']:.3f}")

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
    ax.axhline(difference.mean(), color="#C44E52", lw=1.2, label=f"bias {difference.mean():.3g}")
    for limit in (difference.mean() - 1.96 * difference.std(), difference.mean() + 1.96 * difference.std()):
        ax.axhline(limit, color="#C44E52", ls="--", lw=1, alpha=0.7)
    ax.set(xlabel="mean of the two", ylabel="predicted - truth", title=feature)
    ax.legend(fontsize=7)

fig.suptitle("Bland-Altman: bias and 95% limits of agreement", y=1.0)
fig.tight_layout()""",
    ),
]

DICE_PROXY = [
    (
        "markdown",
        """## 4. Does Dice predict feature error?

If it does, Dice is a usable proxy for feature trustworthiness on unlabelled data — the only
situation that matters in practice, since you will not have annotations. If it does not, a good Dice
is no guarantee that the reported morphometry is right.""",
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

proxy_table = pd.DataFrame(rows).set_index("feature")
display(proxy_table.round(3))""",
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
]

PER_DISEASE = [
    ("markdown", "## 5. Per-disease agreement\n\nSmall groups — read as indicative, not conclusive."),
    (
        "code",
        """for disease in order:
    subset = ok[ok["disease"] == disease]
    print(f"\\n=== {disease}  (n={len(subset)}, mean Dice {subset['dice'].mean():.3f}) ===")
    display(
        agreement_table(subset, FEATURES)[
            ["rel_bias_%", "MAE", "MAPE_%", "pearson_r", "spearman_r", "ICC21"]
        ].round(3)
    )""",
    ),
]

STRENGTH = [
    (
        "markdown",
        """## 6. Informational strength of each feature

Agreement is only half the question. A feature also has to **vary between eyes**, or it cannot
separate them however precisely it is measured.

Informational strength is `IQR / median` over the **ground truth** — a robust, unit-free measure of
how much spread the feature has in the population. It says nothing about AutoMorph; it is a property
of the feature and this cohort, so it is computed over every annotated image, not only the ones that
produced a prediction.""",
    ),
    (
        "code",
        """strength = informational_strength(truth_all, FEATURES).sort_values(
    "gt_IQR_over_median", ascending=False
)
display(strength.round(4))""",
    ),
    (
        "markdown",
        """### Spread against noise

`spread_to_noise` is the ground-truth IQR divided by the standard deviation of the prediction error,
with systematic bias excluded — a constant offset does not stop a feature separating one eye from
another, but random scatter does.

Below about 1 the measurement noise covers the feature's whole interquartile range: even a
"low-error" feature cannot rank eyes if its spread is narrower than its error.""",
    ),
    (
        "code",
        """combined = discriminability(agreement, informational_strength(ok, FEATURES))
combined["MAPE_%"] = agreement["MAPE_%"]
combined["ICC21"] = agreement["ICC21"]
display(combined.sort_values("spread_to_noise", ascending=False).round(3))""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(6.4, 4.8))

for feature in FEATURES:
    ax.scatter(combined.loc[feature, "gt_IQR_over_median"] * 100,
               agreement.loc[feature, "MAPE_%"], s=70, alpha=0.85)
    ax.annotate(feature.replace("_", " "),
                (combined.loc[feature, "gt_IQR_over_median"] * 100, agreement.loc[feature, "MAPE_%"]),
                fontsize=7.5, xytext=(6, 3), textcoords="offset points")

ceiling = max(combined["gt_IQR_over_median"].max() * 100, agreement["MAPE_%"].max()) * 1.15
ax.plot([0, ceiling], [0, ceiling], "k--", lw=1, alpha=0.5)
ax.fill_between([0, ceiling], [0, ceiling], ceiling, color="#C44E52", alpha=0.07)
ax.text(ceiling * 0.35, ceiling * 0.8, "error exceeds spread\\n(feature cannot discriminate)",
        fontsize=8, color="#C44E52")

ax.set(xlabel="informational strength: ground-truth IQR / median (%)",
       ylabel="measurement error, MAPE (%)",
       title="Useful features sit low and to the right")
fig.tight_layout()""",
    ),
]

VERDICTS = [
    (
        "markdown",
        """## 7. Verdict per feature

Generated from the numbers above rather than hand-written, so it cannot drift out of sync with the
data. The thresholds are judgement calls, stated explicitly in the code so they can be argued with.
A feature whose ICC depends on one image is judged on the leave-one-out value, not the headline.""",
    ),
    (
        "code",
        """def verdict(feature):
    icc = agreement.loc[feature, "ICC21"]
    if leverage.loc[feature, "fragile"]:
        icc = leverage.loc[feature, "ICC_without_it"]        # judge on the robust value
    spearman = agreement.loc[feature, "spearman_r"]
    mape = agreement.loc[feature, "MAPE_%"]
    bias = abs(agreement.loc[feature, "rel_bias_%"])
    ratio = combined.loc[feature, "spread_to_noise"]

    if icc < 0.5 and spearman < 0.5:
        return "unusable - no agreement with truth"
    if ratio < 1.0:
        return "error covers the population spread - cannot rank eyes"
    if bias > 10:
        return "rank-usable, biased - calibrate before quoting absolutes"
    if mape < 5 and icc >= 0.75:
        return "reliable"
    return "usable with care"


verdicts = pd.DataFrame(
    {
        "rel_bias_%": agreement["rel_bias_%"],
        "MAE": agreement["MAE"],
        "MAPE_%": agreement["MAPE_%"],
        "pearson_r": agreement["pearson_r"],
        "spearman_r": agreement["spearman_r"],
        "ICC21": agreement["ICC21"],
        "ICC_robust": leverage["ICC_without_it"].where(leverage["fragile"], agreement["ICC21"]),
        "gt_IQR_over_median": combined["gt_IQR_over_median"],
        "spread_to_noise": combined["spread_to_noise"],
    }
)
verdicts["verdict"] = [verdict(feature) for feature in verdicts.index]
display(verdicts.round(3).sort_values("ICC_robust", ascending=False))""",
    ),
]

CONCLUSIONS = [
    (
        "markdown",
        """## 8. Conclusions

Generated from the run below, so the prose cannot drift from the numbers.""",
    ),
    (
        "code",
        """lines = [f"RUN: {RUN_LABEL}", ""]

lines += [
    "SEGMENTATION",
    f"  Dice {ok['dice'].mean():.3f}, sensitivity {ok['sensitivity'].mean():.3f}, "
    f"specificity {ok['specificity'].mean():.3f} over n={len(ok)}.",
    f"  The model under-segments: it misses ~{100 * (1 - ok['sensitivity'].mean()):.0f}% of annotated"
    " vessel and invents almost none.",
    f"  Dice correlates with sensitivity at r={ok['dice'].corr(ok['sensitivity']):.3f}, so here Dice"
    " is largely a restatement of sensitivity.",
    "",
    "QUALITY GATE",
]

if GATE_ENFORCED:
    dropped = int((scores["status"] != "ok").sum())
    lines += [
        f"  Enforced. It rejected {dropped} of {len(scores)} FIVES quality-3 images; rejection is"
        " silent and total.",
        "  Feature results below are conditioned on the survivors, so they are optimistic.",
    ]
elif REJECTED_BY_M1:
    inside = ok[ok["key"].isin(REJECTED_BY_M1)]["dice"]
    outside = ok[~ok["key"].isin(REJECTED_BY_M1)]["dice"]
    _, p_gate = stats.mannwhitneyu(inside, outside)
    lines += [
        f"  Bypassed; all {len(ok)} images segmented.",
        f"  The {len(inside)} images M1 would have rejected score Dice {inside.mean():.3f} against"
        f" {outside.mean():.3f} for the rest (Mann-Whitney p={p_gate:.3f}).",
        f"  Discarding them would raise mean Dice by {outside.mean() - ok['dice'].mean():+.4f} while"
        f" losing {100 * len(inside) / len(ok):.0f}% of the cohort.",
    ]
else:
    lines += [f"  Bypassed; all {len(ok)} images segmented."]

lines += ["", "FEATURE AGREEMENT"]
for feature in verdicts.sort_values("ICC_robust", ascending=False).index:
    row = verdicts.loc[feature]
    note = "  (leverage-adjusted)" if leverage.loc[feature, "fragile"] else ""
    lines.append(
        f"  {feature:30s} ICC={row['ICC_robust']:5.2f} ({icc_label(row['ICC_robust']):9s})"
        f" bias={row['rel_bias_%']:+6.1f}%  MAPE={row['MAPE_%']:5.1f}%"
        f"  spread/noise={row['spread_to_noise']:4.1f}{note}"
    )
    lines.append(f"      -> {row['verdict']}")

proxy = proxy_table["corr(dice, |rel err|)"].sort_values()
lines += [
    "",
    "IS DICE A USABLE PROXY?",
    "  In production there are no annotations, so Dice is the only handle available.",
    f"  It tracks feature error well for: {', '.join(proxy[proxy < -0.5].index) or 'nothing'}",
    f"  It is uninformative about:       {', '.join(proxy[proxy > -0.3].index) or 'nothing'}",
]

print("\\n".join(lines))""",
    ),
    (
        "code",
        """# Save the tables next to the rest of this run's output.
agreement.round(6).to_csv(RESULTS / "feature_agreement.csv")
verdicts.round(6).to_csv(RESULTS / "feature_verdicts.csv")
strength.round(6).to_csv(RESULTS / "feature_informational_strength.csv")
leverage.to_csv(RESULTS / "feature_leverage.csv")
proxy_table.round(6).to_csv(RESULTS / "dice_as_proxy.csv")
Path(RESULTS / "conclusions.txt").write_text("\\n".join(lines) + "\\n")
print(f"wrote agreement, verdicts, informational strength, leverage, proxy and conclusions to {RESULTS}")""",
    ),
]


# --------------------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """One run's identity, data loading, and gate section."""

    name: str
    notebook: Path
    intro: str
    load: str
    gate_cells: list = field(default_factory=list)


FULL_INTRO = """# AutoMorph on FIVES — full pipeline run

The complete 13-stage pipeline over 32 FIVES images, scored against the expert vessel annotations.

Two questions:

1. **How well does M2 reproduce the annotation?** Dice, IoU, sensitivity, specificity.
2. **Do the features AutoMorph reports survive that error?** The same six whole-image features are
   measured from the annotation itself, by the same retipy code, and compared image by image.

The second is the interesting one. A segmentation can lose a quarter of the annotated pixels and
still report an almost-correct fractal dimension, or be pixel-accurate and report a badly wrong
width. Dice alone does not tell you which features to trust.

**The M1 quality gate is enforced in this run**, and it drops 6 of the 32 images before segmentation.
Everything below is therefore conditioned on the survivors. The companion notebook,
`analysis_M2_vessels.ipynb`, bypasses the gate and segments all 32 — read them together.

> `Average_width` is in nominal microns: FIVES publishes no pixel size, so the benchmark writes a
> placeholder 0.008 mm/pixel. Prediction and truth share the same scaling, so every comparison
> statistic here is unaffected; only the absolute micron value is not physical."""

FULL_LOAD = """RESULTS = REPO_ROOT / "benchmark" / "results"
RUN_RESULTS = REPO_ROOT / ".benchmark_run" / "Results"
RUN_LABEL = "full pipeline, M1 quality gate enforced"
GATE_ENFORCED = True


def strip_suffix(series):
    return series.str.replace(".png", "", regex=False)


scores = pd.read_csv(RESULTS / "vessel_scores.csv")
selection = pd.read_csv(RESULTS / "selection.csv")

predicted = pd.read_csv(RUN_RESULTS / "M3" / "Macular_Features.csv")[["Name"] + FEATURES]
truth = pd.read_csv(RUN_RESULTS / "M3" / "Ground_truth_Macular_Features.csv")[["Name"] + FEATURES]
predicted["key"] = strip_suffix(predicted["Name"])
truth["key"] = strip_suffix(truth["Name"])

# retipy writes -1 for a failed measurement; that is a marker, not a value.
for frame in (predicted, truth):
    frame[FEATURES] = frame[FEATURES].mask(frame[FEATURES] == -1)

paired = (
    predicted.drop(columns="Name")
    .merge(truth.drop(columns="Name"), on="key", suffixes=("_pred", "_gt"))
    .merge(scores[["key", "disease", "status", "dice", "iou", "sensitivity", "specificity"]], on="key")
)
ok = paired[paired["status"] == "ok"]
order = sorted(ok["disease"].unique())

# Ground truth exists for every image, gate or no gate.
truth_all = truth.rename(columns={f: f"{f}_gt" for f in FEATURES})

REJECTED_BY_M1 = set(scores.loc[scores["status"] != "ok", "key"])

print(f"{len(selection)} selected, {(scores['status'] == 'ok').sum()} segmented, {len(ok)} paired")
print()
print(scores["status"].value_counts().to_string())"""

FULL_GATE = [
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

print("rejected images:", ", ".join(sorted(gate.loc[~gate["reached_segmentation"], "key"])))
print()
print("The vessel-only run segments these anyway and grades the decision.")""",
    ),
]

VESSEL_INTRO = """# AutoMorph on FIVES — vessel-only run, quality gate bypassed

M0 preprocessing and M2 vessel segmentation over **all 32 images**, with the M1 quality gate
bypassed by feeding it a verdict of "everything passes". See `docs/vessel_run.md`.

Three questions, the first of which the full-pipeline notebook cannot answer:

1. **Was the gate's judgement worth its cost?** It drops 6 of 32 images. Segmenting them anyway
   grades that decision.
2. **How well does M2 reproduce the annotation** across the whole cohort, not just the images that
   passed a filter?
3. **Do the reported features survive that error?** Measured on both sides by the same code.

Because nothing is filtered out first, the numbers here are free of the survivor bias that shapes
`analysis.ipynb` — and comparing the two shows how large that bias is.

> `Average_width` is in nominal microns: FIVES publishes no pixel size, so the benchmark writes a
> placeholder 0.008 mm/pixel. Prediction and truth share the same scaling, so every comparison
> statistic here is unaffected; only the absolute micron value is not physical."""

VESSEL_LOAD = """RESULTS = REPO_ROOT / "benchmark" / "results" / "M2_vessels"
FULL_RESULTS = REPO_ROOT / "benchmark" / "results"
RUN_LABEL = "M0 + M2 vessel only, M1 quality gate bypassed"
GATE_ENFORCED = False

scores = pd.read_csv(RESULTS / "vessel_scores.csv")
selection = pd.read_csv(RESULTS / "selection.csv")

# run_vessel.py already measured both sides and joined them.
paired = pd.read_csv(RESULTS / "features_paired.csv")
predicted = pd.read_csv(RESULTS / "features_predicted.csv")
truth = pd.read_csv(RESULTS / "features_ground_truth.csv")

for frame in (predicted, truth):
    frame[FEATURES] = frame[FEATURES].mask(frame[FEATURES] == -1)

ok = paired[paired["status"] == "ok"]
order = sorted(ok["disease"].unique())
truth_all = truth.rename(columns={f: f"{f}_gt" for f in FEATURES})

# Which images the enforced gate rejected, so its decision can be graded. Optional: the full run may
# not have been executed in this checkout.
full_scores_path = FULL_RESULTS / "vessel_scores.csv"
if full_scores_path.is_file():
    full_scores = pd.read_csv(full_scores_path)
    REJECTED_BY_M1 = set(full_scores.loc[full_scores["status"] != "ok", "key"])
else:
    REJECTED_BY_M1 = set()
    print("no full-run scores found — skipping the gate comparison")

print(f"{len(selection)} selected, {(scores['status'] == 'ok').sum()} segmented, {len(ok)} paired")
print(f"{len(REJECTED_BY_M1)} of them would have been rejected by the M1 gate")"""

VESSEL_GATE = [
    (
        "markdown",
        """## 1. Grading the quality gate

The full run drops 6 of 32 images before segmentation and cannot say whether that was right. This
run segments them, so the gate's decisions can be scored against what the segmentation actually
achieved.

Two things to separate:

- **Is the gate discriminating?** Do rejected images really segment worse than accepted ones?
- **Is the trade worth it?** How much Dice does discarding them buy, against how much data it
  costs?

A gate can be statistically discriminating and still be a bad deal.""",
    ),
    (
        "code",
        """if not REJECTED_BY_M1:
    print("no full-run scores available — cannot grade the gate")
else:
    graded = ok.assign(
        gate=np.where(ok["key"].isin(REJECTED_BY_M1), "M1 rejected", "M1 accepted")
    )
    display(
        graded.groupby("gate").agg(
            n=("key", "size"),
            dice=("dice", "mean"),
            dice_sd=("dice", "std"),
            sensitivity=("sensitivity", "mean"),
        ).round(4)
    )

    rejected = graded[graded["gate"] == "M1 rejected"]
    accepted = graded[graded["gate"] == "M1 accepted"]
    statistic, p_value = stats.mannwhitneyu(rejected["dice"], accepted["dice"])

    print(f"Mann-Whitney p = {p_value:.3f} — the gate {'IS' if p_value < 0.05 else 'is NOT'} "
          "discriminating between better and worse images")
    print()
    print(f"mean Dice over all {len(ok)}:            {ok['dice'].mean():.4f}")
    print(f"mean Dice over the {len(accepted)} accepted:      {accepted['dice'].mean():.4f}")
    print(f"  -> discarding {len(rejected)} images ({100 * len(rejected) / len(ok):.0f}% of the cohort) "
          f"buys {accepted['dice'].mean() - ok['dice'].mean():+.4f} Dice")""",
    ),
    (
        "code",
        """if REJECTED_BY_M1:
    print("The rejected images, worst first — compare against the accepted median:")
    display(
        rejected[["key", "disease", "dice", "sensitivity"]]
        .sort_values("dice")
        .round(4)
        .reset_index(drop=True)
    )
    median_accepted = accepted["dice"].median()
    as_good = (rejected["dice"] >= median_accepted).sum()
    print(f"accepted median Dice = {median_accepted:.4f}")
    print(f"{as_good} of {len(rejected)} rejected images score at or above it — "
          "removed despite segmenting as well as a typical accepted image")""",
    ),
    (
        "code",
        """if REJECTED_BY_M1:
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = ["M1 accepted", "M1 rejected"]
    ax.boxplot([graded.loc[graded["gate"] == g, "dice"] for g in groups], labels=groups,
               widths=0.5)
    for index, group in enumerate(groups, start=1):
        values = graded.loc[graded["gate"] == group, "dice"]
        ax.scatter(np.random.default_rng(0).normal(index, 0.04, len(values)), values,
                   alpha=0.75, s=40, zorder=3,
                   color="#4C72B0" if group == "M1 accepted" else "#C44E52")
    ax.axhline(accepted["dice"].median(), color="grey", ls=":", lw=1,
               label=f"accepted median {accepted['dice'].median():.3f}")
    ax.set(ylabel="Dice", title="What the quality gate discards")
    ax.legend(fontsize=8)
    fig.tight_layout()""",
    ),
]

PROFILES = {
    "full": Profile(
        name="full",
        notebook=REPO_ROOT / "benchmark" / "analysis.ipynb",
        intro=FULL_INTRO,
        load=FULL_LOAD,
        gate_cells=FULL_GATE,
    ),
    "vessel": Profile(
        name="vessel",
        notebook=REPO_ROOT / "benchmark" / "analysis_M2_vessels.ipynb",
        intro=VESSEL_INTRO,
        load=VESSEL_LOAD,
        gate_cells=VESSEL_GATE,
    ),
}


def cells_for(profile):
    """Assemble the cell list for one profile."""
    return [
        ("markdown", profile.intro),
        SETUP,
        ("markdown", "## Load and join"),
        ("code", profile.load),
        *profile.gate_cells,
        *ACCURACY,
        *AGREEMENT,
        *LEVERAGE,
        *DICE_PROXY,
        *PER_DISEASE,
        *STRENGTH,
        *VERDICTS,
        *CONCLUSIONS,
    ]


def build(profile):
    """Assemble a notebook for one profile."""
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in cells_for(profile)
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--output", type=Path, default=None, help="override the notebook path")
    parser.add_argument("--run", action="store_true", help="execute the notebook in place")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    output = args.output or profile.notebook
    notebook = build(profile)

    if args.run:
        from nbclient import NotebookClient

        print(f"executing the {profile.name} notebook")
        NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={
            "metadata": {"path": str(REPO_ROOT)}
        }).execute()

    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, str(output))
    print(f"{len(notebook.cells)} cells -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
