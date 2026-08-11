# Three fixes to vessel segmentation and measurement

```bash
uv run python -m benchmark.run_vessel \
    --run-root .benchmark_run_thr02 --output benchmark/results/M2_vessels_thr02_mean
uv run python -m benchmark.build_notebook --profile threshold --run
```

Three defects found by benchmarking AutoMorph against the FIVES expert annotations, and what fixing
them changed. All are in effect by default; the before-and-after is
[`analysis_M2_vessels_thr02.ipynb`](../analysis_M2_vessels_thr02.ipynb), comparing
`benchmark/results/M2_vessels_thr02_mean/` against the pre-fix `benchmark/results/M2_vessels/`.

## 1. Vessels were not traced as paths

`detect_vessel_border` returned each vessel's pixels in **flood-fill discovery order**, not path
order. `vessel_extractor` walked the segment with a BFS queue (`pending_pixels.pop(0)`) and appended
pixels as it found them; seeded mid-segment, BFS expands both directions at once, so the sequence
jumps back and forth across the seed. An `x`-sort that partly masked this was commented out in 2021.

This breaks every tortuosity measure, because all of them treat consecutive points as consecutive
positions along the vessel:

| Function | Assumption |
| --- | --- |
| `_curve_length` | sums the distance between successive points → arc length |
| `_chord_length` | takes the first and last point |
| `squared_curvature_tortuosity` | differentiates with `np.gradient` |
| `tortuosity_density` | splits the curve at inflections found *by index* |

Measured on the benchmark's own data: **6.2% of consecutive pairs were not adjacent**, averaging 42 px
apart and peaking at **145 px** on a 912 grid. A 145-pixel step where a one-pixel move belongs inflates
arc length severalfold. The evidence in the values: the ground-truth arc-chord ratio came out at
**3.38**, meaning the average vessel supposedly wandered more than three times its end-to-end
distance. Retinal vessels do not do that.

**The fix.** `order_as_path` in `retipy/retina.py` walks a segment neighbour to neighbour from an
endpoint, preferring 4-connected steps. Discovery and ordering become separate concerns: the flood
fill decides *which* pixels belong, the walk decides *in what order* they are reported. A branching
set — which cannot be one path — is walked from each endpoint and the longest simple path kept, so the
trace follows the trunk instead of teleporting between branches.

Non-adjacent steps drop to **0.00%**, the maximum step is exactly √2, and a straight vessel now scores
exactly 1.0. Applied to both retipy copies, so `retina.py` stays identical between them.

## 2. Tortuosity was measured on fragments too short to have a shape

Arc-chord ratio and curvature are unstable when the chord spans a few pixels: a one-pixel wobble moves
the ratio a long way, and skeletonisation noise dominates the derivative.

**The fix.** A minimum vessel length of **50 px**, passed as `evaluate_window`'s existing
`min_pixels_per_vessel` parameter — the threshold is a measurement policy, so it belongs with the
caller. `benchmark.ground_truth_features.MIN_VESSEL_LENGTH = 50`; retipy's own default stays 10 and the
pipeline's M3 scripts keep passing `CONFIG.pixels_per_window` (15). Shorter segments are counted but
not measured.

Aggregation across vessels remains the **mean**, as the original code had it.

## 3. Vessel probability was binarised at 0.5

M2 averages ten ensemble sigmoids and calls a pixel vessel above a threshold, hardcoded at 0.5. At
0.5 the benchmark measured sensitivity **0.736** against specificity **0.995** — missing about a
quarter of the annotated vessel while inventing almost nothing. That asymmetry is the signature of a
threshold set too high, not of a model that cannot see vessels, and `Vessel_density` running 21% low
was the same fact downstream.

**The fix.** `AUTOMORPH_VESSEL_THRESHOLD`, defaulting to **0.2**. Applied at both places M2 binarises:
`resize_binary` (the 912 grid, which feeds `filter_frag` and therefore every feature) and `raw_binary`
(crop resolution). `benchmark.run.DEFAULT_VESSEL_THRESHOLD` matches, and a test ties them together so a
plain `run.sh` and a benchmark run cannot disagree silently. Set the variable to 0.5 to restore the old
behaviour.

0.2 is not a guess. M2 saves the averaged sigmoid map in `binary_vessel/resize/`, so the threshold
sweeps without re-running the network — re-binarise, re-apply `remove_small_objects(30)`, re-score:

| threshold | Dice | IoU | sensitivity | specificity |
| --- | --- | --- | --- | --- |
| 0.05 | 0.8131 | 0.6862 | 0.9465 | 0.9539 |
| 0.10 | 0.8376 | 0.7218 | 0.9028 | 0.9696 |
| 0.15 | 0.8536 | 0.7458 | 0.8773 | 0.9786 |
| **0.20** | **0.8554** | **0.7488** | 0.8530 | 0.9832 |
| 0.25 | 0.8544 | 0.7476 | 0.8318 | 0.9864 |
| 0.30 | 0.8512 | 0.7429 | 0.8108 | 0.9890 |
| 0.40 | 0.8406 | 0.7276 | 0.7741 | 0.9923 |
| 0.50 | 0.8249 | 0.7050 | 0.7360 | 0.9948 |
| 0.70 | 0.7776 | 0.6406 | 0.6524 | 0.9978 |

The maximum is broad — 0.15 to 0.25 all within 0.002 — so the exact value is not delicate, while 0.5
sits well down the slope. The sweep reproduces the 0.5 run's Dice (0.8249) exactly, which is the check
that it follows the pipeline's own path. Note it optimises **Dice**, which weights sensitivity and
precision equally; that is a choice, not a clinical requirement.

## What the three changed, together

All 32 images, quality gate bypassed in both runs. `benchmark/results/M2_vessels/` →
`benchmark/results/M2_vessels_thr02_mean/`.

### Segmentation

| | before | **after** | change |
| --- | --- | --- | --- |
| Dice | 0.8249 | **0.8553** | **+0.030** |
| IoU | 0.7050 | **0.7487** | +0.044 |
| Sensitivity | 0.7360 | **0.8521** | **+0.116** |
| Specificity | 0.9948 | 0.9833 | −0.011 |

Eleven points of sensitivity for one of specificity. Because vessels are only ~2% of the field of
view, that trade is strongly favourable.

### Features

| Feature | ICC before | **ICC after** | bias % before | **after** | MAPE % before | **after** | spread/noise before | **after** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Fractal_dimension** | 0.701 | **0.921** | −2.01 | +0.53 | 2.02 | **0.93** | 1.65 | **2.50** |
| **Vessel_density** | 0.409 | **0.883** | −21.42 | **−0.04** | 22.05 | **6.39** | 1.45 | 1.41 |
| Tortuosity_density | 0.757 | **0.828** | −0.57 | −2.85 | 3.55 | 4.43 | 2.37 | 1.93 |
| **Average_width** | 0.319 | **0.641** | −12.94 | **−5.34** | 12.89 | **6.00** | 1.97 | 2.01 |
| **Distance_tortuosity** | 0.412 | **0.556** | +10.64 | **−0.46** | 27.65 | **0.61** | 0.96 | **1.13** |
| **Squared_curvature_tortuosity** | 0.544 | 0.547 | +10.16 | −3.56 | 90.15 | **8.85** | 0.41 | **1.10** |

Which fix did what:

- **The threshold** drove the accuracy and bias gains. `Vessel_density`'s −21.4% bias falls to
  −0.04%: the probability mass was there all along, sitting between 0.2 and 0.5 and being discarded.
  `Fractal_dimension` and `Average_width` improve for the same reason.
- **The tracing fix** drove the tortuosity gains. `Distance_tortuosity`'s per-image error falls from
  27.6% to 0.61% and its bias from +10.6% to −0.46%; its median value drops from the impossible 3.38
  to 1.090, where a retinal arc-chord ratio belongs. `Squared_curvature_tortuosity`'s error falls from
  90% to 8.8%.
- `Squared_curvature_tortuosity`'s ICC barely moves (0.544 → 0.547), but the before figure rested on a
  single 13× outlier — 0.144 on leave-one-out — so the honest reading is 0.144 → 0.547.

### Verdicts under the fixed pipeline

| Feature | ICC | spread/noise | Verdict |
| --- | --- | --- | --- |
| Fractal_dimension | 0.921 | 2.50 | **reliable** |
| Vessel_density | 0.883 | 1.41 | usable with care |
| Tortuosity_density | 0.828 | 1.93 | **reliable** |
| Average_width | 0.641 | 2.01 | usable with care |
| Distance_tortuosity | 0.556 (0.778 robust) | 1.13 | **reliable** |
| Squared_curvature_tortuosity | 0.547 | 1.10 | usable with care |

**No feature is unusable.** Before the fixes one was degenerate, two were biased enough to need
calibration, and one had no detectable relationship with the truth.

## What this supersedes

[results.md](results.md) and [vessel_results.md](vessel_results.md) were recorded before these fixes.
Their findings on the M1 quality gate, on completion, and on the benchmark protocol stand. Two of
their conclusions do not:

- **The under-segmentation is largely a threshold choice, not a model limitation.** Sensitivity 0.736
  there, 0.852 here.
- **`Vessel_density`'s −21% bias is not intrinsic**, so the advice to calibrate it before quoting
  absolutes is superseded by fixing the threshold.

## Caveats

- **n = 32**, one cohort, all FIVES quality 3. 0.2 is the Dice optimum *here* and is not a calibration
  transferable to other data without checking.
- All three changes are in play in the comparison, so it has no control group. Per-fix attribution
  above comes from having measured them separately during development.
- `order_as_path` drops the shorter branch of a branching segment. Intersections are normally removed
  upstream so this is rare, but it is a deliberate simplification.
- Only vessel segmentation was re-run. Artery/vein and disc/cup have their own thresholds, untouched,
  and zone features inherit the improved tracing but were not re-scored.
