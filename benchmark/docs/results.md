# Results

> **Superseded for the tortuosity features.** Every tortuosity number below was computed on
> non-adjacent point pairs, because `detect_vessel_border` returned vessel pixels in flood-fill order
> rather than path order — see [tortuosity_fix.md](tortuosity_fix.md). Two conclusions there change:
> `Distance_tortuosity` becomes the second-most-trustworthy feature, and the high "informational
> strength" credited to the tortuosity measures here was tracing artefact. Dice, completion, and the
> three non-tortuosity features are unaffected.

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

> Segmenting them anyway shows the gate discards 19% of the cohort to gain 0.007 Dice, and that four
> of the six segment as well as the median accepted image. See
> [vessel_results.md](vessel_results.md).

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

Predicted vs ground truth, 26 paired images, ordered by ICC:

| Feature | bias | rel. bias % | MAE | MAPE % | Pearson | Spearman | **ICC(2,1)** | ICC 95% CI | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tortuosity_density | −0.003 | −0.5 | 0.022 | 3.2 | 0.802 | 0.791 | **0.805** | 0.61–0.90 | good |
| Fractal_dimension | −0.028 | −1.9 | 0.028 | 1.9 | 0.938 | 0.914 | **0.747** | 0.56–0.84 | moderate |
| Vessel_density | −0.017 | −20.0 | 0.017 | 20.8 | 0.939 | 0.908 | **0.504** | 0.26–0.65 | moderate |
| Distance_tortuosity | +0.366 | +10.6 | 0.858 | 25.8 | 0.465 | 0.642 | **0.443** | 0.19–0.75 | poor |
| Average_width | −14.04 | −12.3 | 14.04 | 12.3 | 0.789 | 0.838 | **0.353** | 0.22–0.46 | poor |
| Squared_curvature_tortuosity | +6.59 | +28.7 | 16.04 | 89.3 | 0.097 | 0.173 | **0.088** | −0.16–0.50 | poor |

### ICC(2,1) tells a harsher story than Pearson, and it is the right one

Pearson is deliberately blind to systematic bias: shift every prediction by a constant and it still
scores 1.0. ICC(2,1) charges for that offset, so the gap between the two columns *is* the bias:

| Feature | Pearson | ICC(2,1) | gap | rel. bias % |
| --- | --- | --- | --- | --- |
| Average_width | 0.789 | 0.353 | **0.436** | −12.3 |
| Vessel_density | 0.939 | 0.504 | **0.435** | −20.0 |
| Fractal_dimension | 0.938 | 0.747 | 0.191 | −1.9 |
| Tortuosity_density | 0.802 | 0.805 | −0.003 | −0.5 |

`Average_width` and `Vessel_density` look strong by Pearson (0.79, 0.94) and are only *poor* and
*moderate* by ICC. They track the truth closely — they are simply offset from it. That distinction
decides how they may be used: fine for ranking or stratifying, not for quoting an absolute number
without calibration. `Vessel_density`'s Bland–Altman limits (−0.027, −0.007) exclude zero entirely,
confirming the offset is systematic rather than noise.

Only `Tortuosity_density` reaches *good*, and only it has no bias to charge for.

`Squared_curvature_tortuosity` has no detectable relationship with the truth by any measure
(Pearson 0.097, p = 0.64; Spearman 0.173, p = 0.4; ICC 0.088 with a CI spanning zero) and a median
per-image error of 43%. Curvature tortuosity is computed from second derivatives along a traced
centreline, which is far more sensitive to skeleton noise than an area or box-counting measure — so
it is unsurprising that this is the feature that breaks first.

## 4. Informational strength — is the feature worth measuring at all?

Agreement is only half the question. A feature must also **vary between eyes**, or it cannot
separate them however precisely it is measured. `IQR / median` over the ground truth (all 32
annotated images, since spread is a property of the cohort, not of the pipeline):

| Feature | GT median | GT IQR | **IQR/median** |
| --- | --- | --- | --- |
| Squared_curvature_tortuosity | 19.88 | 14.67 | **0.738** |
| Distance_tortuosity | 3.380 | 1.262 | **0.373** |
| Average_width | 114.11 | 11.99 | **0.105** |
| Vessel_density | 0.0845 | 0.0088 | **0.104** |
| Tortuosity_density | 0.6891 | 0.0699 | **0.101** |
| Fractal_dimension | 1.4728 | 0.0336 | **0.023** |

**This inverts the picture.** The two tortuosity measures carry by far the most spread — and are the
two AutoMorph reproduces worst. `Fractal_dimension` is the most accurately reproduced feature and
has the least to say: its entire interquartile range is 2.3% of its median, so all 32 eyes are
squeezed into a very narrow band.

### Putting the halves together

`spread_to_noise` = ground-truth IQR ÷ SD of the prediction error, with systematic bias excluded
(a constant offset does not stop a feature separating two eyes; random scatter does). Below ~1 the
noise covers the whole interquartile range.

| Feature | IQR/median | spread/noise | ICC | Verdict |
| --- | --- | --- | --- | --- |
| Tortuosity_density | 0.114 | **2.90** | 0.805 | **reliable** |
| Vessel_density | 0.126 | 2.07 | 0.504 | rank-usable, biased — calibrate |
| Average_width | 0.105 | 2.03 | 0.353 | rank-usable, biased — calibrate |
| Fractal_dimension | 0.025 | 1.75 | 0.747 | usable with care — very narrow spread |
| Distance_tortuosity | 0.404 | **1.04** | 0.443 | borderline — noise ≈ spread |
| Squared_curvature_tortuosity | 0.737 | **0.60** | 0.088 | **unusable** |

`Fractal_dimension`'s low 1.9% error is much less impressive than it looks: because its spread is
only 2.3%, the error consumes over half the interquartile range. A precise measurement of something
that barely varies is not a useful measurement.

`Distance_tortuosity` sits at exactly 1.04 — the measurement error is about the size of the
population spread. It should be treated as borderline, not as the "weak but usable" its Spearman of
0.64 might suggest. Its ICC is also the least stable here: leave-one-out *raises* it from 0.443 to
0.623 without `test_105_G` (`feature_leverage.csv`), so the "poor" rating is softer than the point
estimate implies — though borderline either way on spread/noise. On all 32 images it drops to 0.96,
below the threshold entirely (see [vessel_results.md](vessel_results.md)).

Only `Tortuosity_density` clears both bars: good ICC, no meaningful bias, and spread comfortably
wider than its noise.

## 5. Dice is a useful proxy for some features and useless for others

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

## Conclusions

1. **The quality gate is the largest single source of data loss** — 6 of 32 clean images, silently,
   and concentrated in healthy eyes (5 of 8). It costs more images than any downstream failure, and
   because it runs before segmentation, nothing later can recover them. It deserves attention before
   any segmentation tuning.

2. **The segmentation under-segments by design, and that bias propagates into the features.** Dice
   0.832 with sensitivity 0.747 and specificity 0.995 is one fact; `Vessel_density` −20% and
   `Average_width` −12% are the same fact seen downstream. Do not cite them as independent
   corroboration.

3. **Judge features by ICC, not Pearson.** Pearson rates `Vessel_density` at 0.94 and
   `Average_width` at 0.79; ICC(2,1) rates them 0.50 and 0.35. Both track the truth well and sit
   offset from it. Reporting Pearson alone would materially overstate their usability.

4. **The features AutoMorph measures best are the ones with least to say.** `Fractal_dimension` has
   1.9% error but only 2.3% interquartile spread; the two tortuosity measures carry the most spread
   and are reproduced worst. Accuracy and informativeness are anti-correlated across this feature
   set, which is the most consequential finding here — a feature table can look excellent on error
   metrics while carrying little usable signal.

5. **Of the six, one is solid, three need calibration, two should not be used per image.**
   `Tortuosity_density` clears both bars. `Vessel_density`, `Average_width` and `Fractal_dimension`
   are rank-usable with caveats. `Distance_tortuosity` is borderline (noise ≈ spread) and
   `Squared_curvature_tortuosity` is unusable (ICC 0.088, CI spanning zero, noise 1.7× the spread).

6. **Dice cannot be used as a blanket quality signal.** It predicts error for the density and size
   features (r ≈ −0.73 to −0.96) and is uninformative about all three tortuosity measures
   (r ≈ −0.05 to −0.19). In production, where no annotation exists, a good Dice tells you nothing
   about the tortuosity numbers you are reporting.

All six conclusions rest on n=26 (n=32 for spread) from one cohort and one run — see the caveats
below. They are strong enough to act on as priorities for investigation, not as published
performance figures.

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
- **Survivor bias in section 3 — now measured, and real.** Features are compared only on the 26
  images that passed the M1 gate. The [vessel-only run](vessel_results.md) segments all 32 and finds
  every ICC lower by 0.03–0.10, so the agreement reported here is **optimistic**. The ordering of the
  features is unchanged. The same applies to the per-disease accuracy above: `normal`'s 0.851 on
  three survivors becomes 0.819 across all eight.

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
skipping if only vessel accuracy is wanted, which is what
[the vessel-only run](vessel_run.md) does. On a GPU the whole run is minutes.

> **These timings are upper bounds, not the cost of the pipeline.** The
> [vessel-only run](vessel_results.md) executed the identical M2 vessel stage — same code, device and
> batch size — at 34s/image against the 232s/image recorded here. The difference is in the inference
> loop rather than post-processing, and this run's per-batch times were erratic (358, 161, 230
> s/image) where the later one was stable at 28–45. That points to resource contention or memory
> pressure over the six-hour run rather than anything in the code, but the cause is not established.
> A clean run is likely much faster than 5.82 h. No accuracy figure is affected.

## Files

Per-image data behind every number above:

| Path | Contents |
| --- | --- |
| `benchmark/results/vessel_scores.csv` | Per-image confusion counts and scores |
| `benchmark/results/vessel_summary.csv` | Scores per disease and overall |
| `benchmark/results/completion.csv` | Images surviving each module |
| `benchmark/results/stage_timings.csv` | Per-stage status and wall clock |
| `benchmark/results/feature_agreement.csv` | bias, rel. bias, MAE, MAPE, Pearson, Spearman, ICC(2,1) + CI, Bland–Altman |
| `benchmark/results/feature_informational_strength.csv` | GT median, IQR, IQR/median |
| `benchmark/results/feature_verdicts.csv` | The combined verdict table |
| `benchmark/results/conclusions.txt` | The conclusions, generated from the run |
| `benchmark/results/selection.csv` | The 32 images and their FIVES labels |
| `.benchmark_run/Results/M3/Ground_truth_Macular_Features.csv` | Features from the annotations |
| `benchmark/analysis.ipynb` | The analysis, with plots |
