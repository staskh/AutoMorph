# The tortuosity fix

```bash
uv run python -m benchmark.run_vessel \
    --reuse-segmentation-from .benchmark_run_vessel \
    --run-root .benchmark_run_tortuosity \
    --output benchmark/results/M2_vessels_fixed
uv run python -m benchmark.build_notebook --profile fixed --run
```

Three changes to how tortuosity is computed, evaluated on the **same segmentation masks** as the
[vessel-only run](vessel_results.md) — byte-identical, reused rather than recomputed, so anything
that moves is the feature formula and nothing else.

## 1. What was wrong

`detect_vessel_border` returned each vessel's pixels in **flood-fill discovery order**, not path
order. `vessel_extractor` walked the segment with a BFS queue (`pending_pixels.pop(0)`) and appended
pixels as it found them. Seeded in the middle of a segment, BFS expands in both directions at once,
so the returned sequence jumps back and forth across the seed. An `x`-sort used to partly mask this
and was commented out in 2021 ("2021/10/31 remove setting of the sort & x duplication").

This matters because **every tortuosity measure treats consecutive points as consecutive positions
along the vessel**:

| Function | Assumption |
| --- | --- |
| `_curve_length` | sums the distance between successive points → arc length |
| `_chord_length` | takes the first and last point |
| `squared_curvature_tortuosity` | differentiates with `np.gradient` |
| `tortuosity_density` | splits the curve at inflections found *by index* |

Measured on the benchmark's own data, before the fix:

| | |
| --- | --- |
| Consecutive pairs that were **not** adjacent | **6.2%** |
| Mean size of those jumps | **42 px** |
| Largest jump | **145 px** on a 912 grid |

A 145-pixel step in what should be a one-pixel move inflates arc length severalfold, and every
tortuosity value with it. The evidence in the values themselves: ground-truth `Distance_tortuosity`
had a median of **3.38**, meaning the average vessel supposedly wandered more than three times its
end-to-end distance. Retinal vessels do not do that.

## 2. What changed

**`order_as_path`** (in `retipy/retina.py`) walks a segment neighbour to neighbour from an endpoint,
preferring 4-connected steps. Discovery and ordering are now separate concerns: the flood fill
decides *which* pixels belong to the segment, the walk decides *in what order* they are reported.
Branching sets — which cannot be one path — are walked from each endpoint and the longest simple path
is kept, so the trace follows the trunk instead of teleporting between branches.

Result: non-adjacent steps **0.00%**, maximum step exactly √2, and a straight vessel now scores
exactly 1.0.

**Minimum vessel length 50 px.** Arc-chord ratio and curvature are unstable on short fragments: the
chord is a few pixels, so a one-pixel wobble moves the ratio a long way, and skeletonisation noise
dominates the derivative. Shorter segments are counted but not measured.

The threshold stays `evaluate_window`'s existing `min_pixels_per_vessel` parameter, whose retipy
default remains **10**. It is a measurement policy, so it belongs with the caller: the benchmark
passes `benchmark.ground_truth_features.MIN_VESSEL_LENGTH = 50`, while the pipeline's own M3 scripts
keep passing `CONFIG.pixels_per_window` (15) and are unaffected. The one behaviour change for those
callers is that the comparison is now `>=` rather than `>`, so a 15-pixel segment is included where
it previously needed 16.

**Aggregation across vessels** was briefly changed from mean to median, then **reverted to the
original mean** — see [the comparison below](#8-mean-vs-median-across-vessels). The median turned out
to destroy the between-eye variation the feature exists to capture.

The fix is applied to **both** retipy copies so `retina.py` stays identical between them. The zone
copy's `evaluate_window` aggregation is untouched, so zone features are unaffected except through the
improved tracing.

## 3. What it did — the control first

`Fractal_dimension`, `Vessel_density` and `Average_width` do not depend on vessel tracing, so they
must not move. They don't, to the last bit:

| Feature | ICC before | ICC after | MAPE % before | after |
| --- | --- | --- | --- | --- |
| Fractal_dimension | 0.7007 | 0.7007 | 2.024 | 2.024 |
| Vessel_density | 0.4091 | 0.4091 | 22.052 | 22.052 |
| Average_width | 0.3195 | 0.3195 | 12.886 | 12.886 |

That is the evidence the change is confined to tortuosity.

## 4. Agreement improved sharply

| Feature | ICC before | **ICC after** | MAPE % before | **after** | bias % before | **after** |
| --- | --- | --- | --- | --- | --- | --- |
| **Distance_tortuosity** | 0.412 | **0.868** | 27.6 | **0.21** | +10.6 | **−0.09** |
| **Tortuosity_density** | 0.757 | **0.822** | 3.55 | **2.97** | −0.57 | **−0.22** |
| Squared_curvature_tortuosity | 0.544 | −0.031 | 90.2 | **6.25** | +10.2 | −7.1 |

`Distance_tortuosity` goes from *poor* to *good* agreement, and its per-image error falls by a factor
of 130 — from 27.6% to 0.21%. Its median value drops from 3.38 to **1.080**, which is where a retinal
arc-chord ratio belongs. The pre-fix number was not a noisy measurement of tortuosity; it was a
measurement of the tracing bug.

## 5. And it revealed that the spread was artefact

Agreement improved, but the **between-eye spread collapsed**:

| Feature | IQR/median before | **after** | spread/noise before | **after** |
| --- | --- | --- | --- | --- |
| Distance_tortuosity | 0.373 | **0.0039** | 0.96 | 1.22 |
| Squared_curvature_tortuosity | 0.738 | **0.0000** | 0.41 | 0.00 |
| Tortuosity_density | 0.101 | **0.0368** | 2.37 | **0.84** |

Correctly computed, `Distance_tortuosity` spans 1.072–1.105 across all 32 eyes and
`Squared_curvature_tortuosity` is constant to four decimal places.

So the pre-fix "informational strength" of the tortuosity measures — which
[results.md](results.md) reported as the *highest* of the six features (0.37 and 0.74) — **was not
anatomy**. It was per-image variation in how badly the tracing jumped: noise dressed as signal. That
correction matters more than the ICC improvement, because it inverts an earlier conclusion.

**The 50-px threshold is not responsible.** With ordering fixed, ground-truth `Distance_tortuosity`
spread is 0.0024 at a 15-px threshold and 0.0039 at 50 px — the threshold slightly *increases* it.
The collapse is attributable to the ordering fix alone.

## 6. Choosing the threshold: 50 px vs 25 px

Both re-extractions reuse the same masks, so only the threshold differs. The three features that do
not use vessel tracing are identical at both, as they must be.

| Feature | min px | ICC | MAPE % | Spearman | IQR/median | spread/noise |
| --- | --- | --- | --- | --- | --- | --- |
| Distance_tortuosity | **50** | **0.868** good | **0.214** | 0.794 | 0.0039 | **1.22** |
| | 25 | 0.731 moderate | 0.231 | 0.816 | 0.0030 | 0.91 |
| Tortuosity_density | **50** | **0.822** good | **2.97** | 0.680 | 0.0368 | **0.84** |
| | 25 | 0.470 poor | 5.92 | 0.837 | 0.0444 | 0.32 |
| Squared_curvature_tortuosity | 50 | −0.031 | 6.25 | 0.291 | 0.0000 | 0.00 |
| | 25 | *undefined* | 0.000 | 0.354 | 0.0000 | — |

**50 px is the better choice.** `Distance_tortuosity` drops from *good* to *moderate* agreement at
25 px, and its spread-to-noise falls below 1 (0.91), meaning the measurement error covers the
population spread. `Tortuosity_density` degrades much harder — ICC 0.822 → 0.470 and error doubling
from 3.0% to 5.9%. Admitting 25–49 px segments adds fragments whose arc-chord ratio is dominated by
skeletonisation noise, which is exactly what the threshold exists to exclude.

Spearman moves the *other* way for both (0.794 → 0.816 and 0.680 → 0.837), so the shorter threshold
preserves ranking slightly better while measuring absolute values worse. If a downstream use only
needs to order eyes, 25 px is defensible; for anything quoting a value, 50 px is clearly better.

### `Squared_curvature_tortuosity` is degenerate at both thresholds

At 25 px it is **exactly constant** — 0.089443 for all 32 images, on both the prediction and the
ground truth. That produces a MAPE of 0.000%, which reads like perfection and is nothing of the kind:
there is no variation to agree about.

It also exposed a defect in `benchmark/agreement.py`. On a constant column the ICC arithmetic still
returns a number, but it is float noise — the same data gave 0.933, 0.596, 0.170 and 0.0001
depending on perturbations of 1e-15. `icc21` now returns NaN when a column varies by less than one
part in 10⁹ of its scale, and the verdict logic reports *"constant across eyes, no agreement is
definable"*. The 0.933 that the first 25 px run printed was that bug, not a result.

## 7. Net verdict per feature

| Feature | Verdict after the fix |
| --- | --- |
| Distance_tortuosity | **reliable** — ICC 0.87, error 0.2%, spread/noise 1.2 |
| Fractal_dimension | usable with care — unchanged |
| Vessel_density | rank-usable, biased −21% — unchanged |
| Average_width | rank-usable, biased −13% — unchanged |
| Tortuosity_density | **error covers the population spread** — spread/noise 0.84, and its ICC is leverage-dependent (0.82 → 0.48 without one image) |
| Squared_curvature_tortuosity | **unusable** — now constant across eyes, so no agreement is even definable |

`Distance_tortuosity` is the clear win: it moves from "borderline" to the second-most-trustworthy
feature after the fix. `Tortuosity_density` moves the other way once its spread is measured honestly.
`Squared_curvature_tortuosity` was unusable before and remains so, for a different and clearer
reason: not that it disagrees with the truth, but that it has nothing to say.

## Caveats

- **n = 32**, one cohort, all FIVES quality 3. The spread collapse is a statement about *these* eyes;
  a cohort with more tortuous pathology could show real variation.
- `order_as_path` drops the shorter branch of a branching segment. Intersections are normally removed
  upstream so this is rare, but it is a deliberate simplification, not a proof that no pixel is lost.
- The three unchanged features are unchanged *because* they do not use tracing — this fix says nothing
  about whether their own agreement is good.
- Zone features (`M3_feature_zone`) inherit the improved tracing but were not re-run or re-scored, so
  no zone number in these docs reflects the fix.

## See also

- [vessel_results.md](vessel_results.md) — the pre-fix run these numbers are compared against
- `benchmark/analysis_M2_vessels_fixed.ipynb` — the comparison, with plots
- `benchmark/results/M2_vessels_fixed/` — the result tables

## 8. Mean vs median across vessels

`evaluate_window` measures every vessel in an image and reduces them to one number. The original code
took the **mean**. It was briefly changed to the **median**, on the reasoning that a mean is dragged
up by short segments whose arc-chord ratio explodes when the chord is only a few pixels. That
reasoning was wrong twice over: `min_pixels_per_vessel` already excludes those segments, and the
median throws away exactly the information the feature is for.

Same masks (threshold 0.2), same 50 px minimum, only the aggregate differs. The three features that
do not use it are byte-identical, which is the control.

| Feature | agg | ICC | bias % | MAPE % | Spearman | IQR/median | spread/noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Distance_tortuosity | median | **0.762** | −0.18 | **0.26** | 0.809 | 0.0039 | 1.01 |
| | **mean** | 0.556 | −0.46 | 0.61 | **0.864** | **0.0123** | **1.13** |
| Squared_curvature_tortuosity | median | 0.483 | −5.71 | **3.13** | 0.450 | **0.0000** | **0.00** |
| | **mean** | **0.547** | −3.56 | 8.85 | 0.459 | **0.1234** | **1.10** |
| Tortuosity_density | median | 0.819 | −1.41 | **2.78** | 0.720 | 0.0368 | 0.87 |
| | **mean** | **0.828** | −2.85 | 4.43 | **0.858** | **0.0999** | **1.93** |

### The median was making features degenerate

Distinct ground-truth values across the 32 eyes:

| Feature | median | mean |
| --- | --- | --- |
| Distance_tortuosity | 32 | 32 |
| **Squared_curvature_tortuosity** | **3** | **32** |
| Tortuosity_density | 31 | 32 |

Under the median, `Squared_curvature_tortuosity` took **three distinct values across 32 eyes** — and
at a 25 px threshold, [one](#6-choosing-the-threshold-50-px-vs-25-px). A median over ~50 vessels lands
on a discrete, repeated value; the between-eye differences live in the tail, which the median
discards by construction.

Restoring the mean fixes that. Every feature becomes distinct per eye, `IQR/median` rises by 3x to
30x, and **spread-to-noise improves for all three** — from 0.87 to 1.93 for `Tortuosity_density`, and
from 0.00 to 1.10 for `Squared_curvature_tortuosity`, which crosses from "cannot discriminate" to
usable. Spearman improves for all three too.

### The trade

The mean is less *precise*: `Distance_tortuosity` loses ICC (0.762 → 0.556) and its per-image error
doubles (0.26% → 0.61%). That is the honest cost — averaging admits the tail's variance, some of which
is noise.

But precision about a constant is worthless, and that is what the median was delivering. The spread
the mean recovers is larger than the noise it admits, which is exactly what `spread_to_noise` measures
and it improves in all three cases. **The mean is the better aggregate here, and it is what the
original code did.**

The earlier conclusion that "correctly-computed tortuosity barely varies across these eyes" was
therefore partly an artefact of the median, not only of the tracing fix. With the mean, ground-truth
`Distance_tortuosity` spans 1.074–1.148 rather than 1.072–1.105.
