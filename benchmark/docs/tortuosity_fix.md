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

**Median instead of mean** across vessels, so the few short segments whose ratio explodes cannot
dominate the image's value.

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

## 6. Net verdict per feature

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
