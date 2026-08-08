# Results

One run, 32 FIVES images, CPU only. Reproduce with `uv run python -m benchmark.run`.

| | |
| --- | --- |
| Images | 32 (8 each amd/dr/glaucoma/normal, quality 3, FIVES `test` split) |
| Device | CPU (`AUTOMORPH_DEVICE=cpu`), `NUM_WORKERS=0`, M1 batch 8 |
| Machine | Apple silicon, 8 cores, 24 GB |
| Total time | **5.82 h** — all 13 stages succeeded on the first attempt |

## 1. Completion — the quality gate drops 6 of 32

| Stage | Images | of 32 |
| --- | --- | --- |
| Staged | 32 | 32 |
| M0 preprocessed | 32 | 32 |
| **M1 good quality** | **26** | 32 |
| M1 bad quality | 6 | 32 |
| M2 binary vessel | 26 | 32 |
| M2 artery/vein | 26 | 32 |
| M2 disc/cup | 26 | 32 |
| M3 macular features | 26 | 32 |
| M3 disc features | 0 | 32 |

`M3 disc features` being 0 is correct, not a failure: all 26 images are macular-centred, so the
disc-centred branch has nothing to do. `Disc_Features.csv` is legitimately empty for FIVES.

The 6 lost images are the real finding. Every benchmark image is FIVES **quality 3** — the cleanest
grade — yet M1 rejected 6, and rejection is silent and total: no segmentation, no features, no row
in the output.

### The rejections are concentrated in the healthy images

| Disease | Selected | Segmented | Rejected |
| --- | --- | --- | --- |
| amd | 8 | 8 | 0 |
| dr | 8 | 7 | 1 |
| glaucoma | 8 | 8 | 0 |
| **normal** | 8 | **3** | **5** |

Five of eight healthy eyes were discarded while every amd and glaucoma image passed. The M1 model
is the EyePACS quality network, trained for diabetic-retinopathy screening; a domain effect is the
obvious hypothesis, but this run does not establish the cause and 8 images per group is thin
evidence. It is a flag for investigation, not a conclusion.

### Two rejections turn on a hardcoded threshold

The gate passes `Prediction == 0`, passes `Prediction == 1` only when `softmax_bad < 0.25`, and
always fails `Prediction == 2`:

| key | disease | Prediction | softmax_bad | outcome |
| --- | --- | --- | --- | --- |
| test_56_D | dr | 1 | 0.258 | rejected, by 0.008 |
| test_151_N | normal | 1 | 0.282 | rejected, by 0.032 |
| test_167_N | normal | 1 | 0.427 | rejected |
| test_158_N | normal | 2 | 0.544 | rejected outright |
| test_161_N | normal | 2 | 0.575 | rejected outright |
| test_162_N | normal | 2 | 0.635 | rejected outright |

Two of the six sit within 0.032 of the cutoff. For those, the threshold decides, not image quality.

## 2. Vessel segmentation accuracy

Dice/IoU/sensitivity/specificity against the expert annotation, inside the field of view, on the
912 grid. 26 images scored.

| Disease | Dice | IoU | Sensitivity | Specificity | Scored |
| --- | --- | --- | --- | --- | --- |
| amd | 0.853 | 0.745 | 0.776 | 0.995 | 8/8 |
| glaucoma | 0.842 | 0.728 | 0.763 | 0.994 | 8/8 |
| normal | 0.851 | 0.742 | 0.784 | 0.992 | 3/8 |
| **dr** | **0.789** | **0.657** | **0.681** | 0.996 | 7/8 |
| **all** | **0.832** | **0.716** | **0.747** | **0.995** | 26/32 |

**The model under-segments, consistently.** Sensitivity averages 0.747 while specificity is 0.995 —
about a quarter of the annotated vessel is missed, and almost nothing is invented. Dice correlates
with sensitivity at r = 0.987 and with specificity at −0.589: on this data Dice is essentially a
restatement of sensitivity.

**DR is the weakest disease** (Dice 0.789 vs ~0.85), and it loses ground specifically on sensitivity
(0.681). Haemorrhages and exudates obscuring vessels is a plausible reading, consistent with the
error being missed vessel rather than false vessel.

`normal`'s 0.851 rests on 3 images. Do not read into it.

Specificity and accuracy are near 1 for any prediction here, because vessels are ~2% of the field
of view. They are reported for completeness and should not be used to compare methods.

## 3. Do the reported features survive that error?

Dice measures pixels. What AutoMorph actually reports is morphometry. So the same six whole-image
features were measured from the expert annotation, by the same retipy code, through the same
post-processing — see [protocol.md](protocol.md) and `benchmark/ground_truth_features.py`.

Predicted vs ground truth, 26 paired images:

| Feature | bias % | MAPE % | Pearson r | Spearman | Verdict |
| --- | --- | --- | --- | --- | --- |
| Fractal_dimension | −1.9 | 1.9 | 0.938 | 0.914 | **reliable** |
| Tortuosity_density | −0.5 | 3.2 | 0.802 | 0.791 | **reliable** |
| Vessel_density | −20.0 | 20.8 | 0.939 | 0.908 | biased, rank-usable |
| Average_width | −12.3 | 12.3 | 0.789 | 0.838 | biased, rank-usable |
| Distance_tortuosity | +10.6 | 25.8 | 0.465 | 0.642 | weak |
| Squared_curvature_tortuosity | +28.7 | 89.3 | **0.097** (p=0.64) | 0.173 (p=0.4) | **unreliable** |

Three distinct behaviours, and Dice predicts none of them on its own:

**Robust.** `Fractal_dimension` and `Tortuosity_density` come through a 25%-under-segmentation
almost intact — under 4% error, negligible bias. Both are normalised, scale-free quantities, so
losing thin peripheral vessels barely moves them.

**Biased but order-preserving.** `Vessel_density` runs 20% low and `Average_width` 12% low, while
correlating with truth at r ≈ 0.94 and 0.79. The bias is the under-segmentation showing up directly
— a −20% density against a 0.747 sensitivity is the same fact stated twice. These are usable to
rank or stratify, and need calibration before any absolute value is quoted. The Bland–Altman limits
for `Vessel_density` (−0.027, −0.007) exclude zero entirely: the offset is systematic, not noise.

**Unreliable.** `Squared_curvature_tortuosity` has no detectable relationship with the truth
(r = 0.097, p = 0.64; Spearman 0.173, p = 0.4) and a median per-image error of 43%. On this data it
should not be used per image, and its absolute value carries no information about the eye.
`Distance_tortuosity` is weak rather than dead (r = 0.465, p = 0.017) and ranks better than it
measures (Spearman 0.642).

Curvature-based tortuosity is computed from second derivatives along a traced centreline, so it is
far more sensitive to skeleton noise than an area or box-counting measure — which is consistent with
being the feature that breaks first.

### Dice is a useful proxy for some features and useless for others

Correlation of Dice with each feature's absolute relative error:

| Feature | corr(Dice, \|rel err\|) | median \|rel err\| % |
| --- | --- | --- |
| Vessel_density | −0.960 | 18.6 |
| Fractal_dimension | −0.939 | 1.6 |
| Average_width | −0.731 | 11.6 |
| Tortuosity_density | −0.188 | 3.1 |
| Distance_tortuosity | −0.062 | 16.0 |
| Squared_curvature_tortuosity | −0.050 | 43.3 |

This matters because in production there are no annotations, so Dice is the only handle available.
It works for the density and size features: a high Dice really does mean a trustworthy
`Vessel_density` or `Fractal_dimension`. It tells you nothing about either tortuosity measure —
their errors are uncorrelated with segmentation quality, so a good Dice is no reassurance at all.

## Caveats

- **n = 26.** Per-disease groups are 3–8 images. Overall figures are indicative; per-disease
  comparisons are weaker still, and `normal` is 3 images.
- **Micron figures are nominal.** FIVES publishes no pixel size, so a placeholder 0.008 mm/pixel is
  used. `Average_width`'s bias and correlation are meaningful (both sides share the scaling); its
  absolute value in microns is not a physical measurement.
- **Scored at 912, not native.** Both prediction and annotation are evaluated on the pipeline's
  working grid, which costs thin vessels on both sides. Dice here is not comparable to a FIVES
  leaderboard number computed at full resolution.
- **The under-segmentation and the feature bias are one finding, not two.** They should not be
  cited as independent corroboration.
- **Survivor bias in section 3.** Features are compared only on the 26 images that passed the M1
  gate. If the gate preferentially rejects harder images, the feature agreement reported here is
  optimistic relative to the full 32.

## Stage timings

| Stage | Seconds | Note |
| --- | --- | --- |
| M0 preprocess | 61 | 32 images |
| M1 quality assessment | 1712 | 8-model EfficientNet-B4 ensemble |
| M1 quality merge | 1 | |
| M2 vessel segmentation | 6046 | 10-model ensemble at 912² |
| **M2 artery/vein** | **12779** | 8 members × 3 forward passes at 720² — 61% of the run |
| M2 disc/cup | 122 | |
| M3 zone macular B / C | 77 / 143 | |
| M3 zone disc B / C | 2 / 1 | nothing to do, no disc-centred images |
| M3 whole macular | 20 | |
| M3 whole disc | 1 | nothing to do |
| csv merge | <1 | |
| **Total** | **20964 (5.82 h)** | |

Artery/vein alone is 61% of the wall clock and is not used by any metric in this report — worth
skipping if only vessel accuracy is wanted. On a GPU the whole run is minutes.

## Files

Per-image data behind every number above:

| Path | Contents |
| --- | --- |
| `benchmark/results/vessel_scores.csv` | Per-image confusion counts and scores |
| `benchmark/results/vessel_summary.csv` | Scores per disease and overall |
| `benchmark/results/completion.csv` | Images surviving each module |
| `benchmark/results/stage_timings.csv` | Per-stage status and wall clock |
| `benchmark/results/feature_agreement.csv` | Bias, MAPE, correlations, Bland–Altman limits |
| `benchmark/results/feature_verdicts.csv` | The verdict table from section 3 |
| `benchmark/results/selection.csv` | The 32 images and their FIVES labels |
| `.benchmark_run/Results/M3/Ground_truth_Macular_Features.csv` | Features from the annotations |
| `benchmark/analysis.ipynb` | The analysis, with plots |
