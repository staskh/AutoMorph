# Vessel-only run: results

All 32 images, quality gate bypassed. Reproduce with `uv run python -m benchmark.run_vessel`.
See [vessel_run.md](vessel_run.md) for what the run does; [results.md](results.md) is the
full-pipeline counterpart.

| | |
| --- | --- |
| Images | 32 (8 each amd/dr/glaucoma/normal, quality 3, FIVES `test` split) |
| Segmented | **32 of 32** — the gate that dropped 6 in the full run is bypassed |
| Device | CPU (`AUTOMORPH_DEVICE=cpu`), `NUM_WORKERS=0`, batch 8 |
| Total time | **18.2 min** — M0 7s, M2 vessel 1084s, both first attempt |

## 1. Was the quality gate worth it?

This is the question the run exists to answer, and the full run cannot: it never segments the images
M1 rejects. Segmenting all 32 lets the gate's decisions be graded.

| | n | Dice | sd | Sensitivity |
| --- | --- | --- | --- | --- |
| M1 **rejected** | 6 | **0.7936** | 0.044 | 0.6867 |
| M1 accepted | 26 | **0.8321** | 0.053 | 0.7474 |

**The gate is directionally right** — rejected images do segment worse, and the difference survives a
rank test (Mann–Whitney p = 0.025). It is not making random cuts.

**But the trade is poor.** Discarding those 6 images — 19% of the data — raises mean Dice from 0.8249
to 0.8321. **+0.0072 Dice for a fifth of the cohort.**

And the per-image view shows why:

| key | disease | Dice | Sensitivity |
| --- | --- | --- | --- |
| test_167_N | normal | 0.7172 | 0.5807 |
| test_56_D | dr | 0.7643 | 0.6236 |
| test_158_N | normal | 0.8106 | 0.7242 |
| test_151_N | normal | 0.8120 | 0.7114 |
| test_162_N | normal | 0.8255 | 0.7469 |
| test_161_N | normal | 0.8322 | 0.7333 |

Only two are genuinely poor. The other four land between 0.811 and 0.832 — at or above the median of
the images the gate *accepted*, and `test_161_N` at 0.8322 sits exactly on the accepted mean. Two of
those four (`test_56_D`, `test_151_N`) were rejected by margins of 0.008 and 0.032 on a hardcoded
`softmax_bad < 0.25` threshold (see [results.md](results.md)).

So the gate is a blunt instrument: it does identify harder images, but at this threshold it discards
four usable segmentations to remove two weak ones. Anyone processing a small cohort should consider
raising the cutoff, or reviewing rejects rather than dropping them silently.

## 2. Segmentation accuracy on all 32

| Disease | Dice | IoU | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| amd | 0.8530 | 0.7445 | 0.7761 | 0.9948 |
| glaucoma | 0.8420 | 0.7277 | 0.7628 | 0.9941 |
| dr | 0.7856 | 0.6525 | 0.6738 | 0.9965 |
| normal | 0.8188 | 0.6955 | 0.7312 | 0.9936 |
| **all** | **0.8249** | **0.7050** | **0.7360** | **0.9948** |

`amd` and `glaucoma` are identical to the full run — the gate rejected none of them, so they are the
same eight images. `dr` moves 0.7886 → 0.7856 with its eighth image, and **`normal` moves 0.8511 →
0.8188**, which is the honest figure: the full run's 0.8511 rested on the three healthy eyes that
happened to pass a gate biased against healthy eyes.

Under-segmentation is unchanged and confirmed on the full cohort: sensitivity 0.736 against
specificity 0.995. About a quarter of the annotated vessel is missed and almost nothing is invented.
DR remains the weakest disease, losing ground specifically on sensitivity (0.674).

## 3. Feature agreement over all 32

Both sides measured by the same `measure_masks` call — see [vessel_run.md](vessel_run.md).

| Feature | rel. bias % | MAE | MAPE % | Pearson | Spearman | ICC(2,1) | ICC 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tortuosity_density | −0.6 | 0.024 | 3.6 | 0.755 | 0.711 | **0.757** good | 0.55–0.87 |
| Fractal_dimension | −2.0 | 0.030 | 2.0 | 0.926 | 0.869 | **0.701** moderate | 0.49–0.81 |
| Squared_curvature_tortuosity | +10.2 | 20.50 | 90.2 | 0.626 | 0.278 | 0.544 (see below) | −0.05–0.65 |
| Distance_tortuosity | +10.6 | 0.938 | 27.6 | 0.433 | 0.576 | **0.412** poor | 0.17–0.69 |
| Vessel_density | −21.4 | 0.018 | 22.1 | 0.889 | 0.792 | **0.409** poor | 0.18–0.57 |
| Average_width | −12.9 | 14.93 | 12.9 | 0.766 | 0.832 | **0.319** poor | 0.21–0.41 |

Every ICC is **lower** than the full run's, which is what including the harder images should do:

| Feature | ICC, 26 images (gated) | ICC, 32 images | change |
| --- | --- | --- | --- |
| Tortuosity_density | 0.805 | 0.757 | −0.048 |
| Fractal_dimension | 0.747 | 0.701 | −0.046 |
| Vessel_density | 0.504 | 0.409 | −0.095 |
| Distance_tortuosity | 0.443 | 0.412 | −0.031 |
| Average_width | 0.353 | 0.319 | −0.034 |

The full run's feature agreement was therefore **optimistic** — the survivor bias flagged in
[results.md](results.md) is real and worth roughly 0.03–0.10 of ICC. The ordering of the features is
unchanged.

### The one ICC that appears to improve does not

`Squared_curvature_tortuosity` seems to jump from 0.088 to 0.544. It is an artefact of a **single
image**:

| | n | ICC | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| all 32 | 32 | 0.544 | 0.626 | 0.278 |
| excluding `test_158_N` | 31 | **0.144** | 0.168 | 0.206 |

`test_158_N` has a ground-truth value of 267.8 against a cohort median of 19.9 — a 13x outlier, and
one of the six images the gate rejected, so it appears only in this run. One high-leverage point
carries the whole correlation. Its ICC confidence interval spans zero (−0.05 to 0.65), and Spearman
— immune to leverage — stays at 0.278.

**The verdict is unchanged: unusable.** This is a good argument for reporting Spearman and a CI
beside every ICC; on the point estimate alone this would read as a real improvement.

### Leverage is now checked automatically

Because that case was found by hand, both notebooks now run leave-one-out on every ICC and flag any
feature a single image moves by more than 0.15 (`feature_leverage.csv`). Verdicts are computed from
the leave-one-out value, not the headline, so a leverage artefact cannot promote a feature.

| Feature | ICC (all 32) | most influential image | ICC without it | swing | fragile |
| --- | --- | --- | --- | --- | --- |
| Fractal_dimension | 0.701 | test_52_D | 0.612 | −0.089 | |
| Vessel_density | 0.409 | test_52_D | 0.301 | −0.108 | |
| Average_width | 0.319 | test_16_A | 0.296 | −0.023 | |
| Distance_tortuosity | 0.412 | test_105_G | 0.544 | +0.132 | |
| **Squared_curvature_tortuosity** | 0.544 | **test_158_N** | **0.144** | **−0.399** | **yes** |
| Tortuosity_density | 0.757 | test_67_D | 0.716 | −0.041 | |

Only one feature is leverage-dependent, and it is the one already judged unusable. Note also that
`Fractal_dimension` and `Vessel_density` both hinge partly on `test_52_D` (swings −0.09 and −0.11) —
under the 0.15 threshold, but a reminder that n = 32 is small.

## 4. Informational strength and discriminability

Unchanged from the full run for spread, since ground truth does not depend on the gate:

| Feature | IQR/median | spread/noise | ICC | Verdict |
| --- | --- | --- | --- | --- |
| Tortuosity_density | 0.101 | **2.37** | 0.757 | reliable |
| Average_width | 0.105 | 1.97 | 0.319 | biased, rank-usable |
| Fractal_dimension | 0.023 | 1.65 | 0.701 | usable — very narrow spread |
| Vessel_density | 0.104 | 1.45 | 0.409 | biased, rank-usable |
| Distance_tortuosity | 0.373 | **0.96** | 0.412 | noise now exceeds spread |
| Squared_curvature_tortuosity | 0.738 | **0.41** | 0.088* | unusable |

\* excluding the leverage point.

`Distance_tortuosity` has crossed below 1.0 (0.96, from 1.04): on the full cohort its measurement
error is larger than the population spread it is meant to resolve.

## 5. The two runs agree exactly on the images they share

Every difference between this report and [results.md](results.md) comes from *which images were
included* — nothing else. Checked over the 26 images common to both runs:

| Check | Result |
| --- | --- |
| `binary_process`, `binary_skeleton`, `raw_binary` PNGs | byte-identical, 26/26 |
| TP/FP/FN/TN, Dice, IoU, sensitivity, specificity, accuracy | max difference 0 |
| `truth_pixels`, `predicted_pixels`, `fov_pixels`, `crop_radius` | max difference 0 |
| All six predicted features | max difference 0 |
| All six ground-truth features | max difference 0 |
| ICC, bias, MAPE restricted to the same 26 | identical to 6 dp |
| Mean Dice on the 26 | 0.8321 — the full run's figure exactly |

Two things this validates beyond reproducibility:

**`measure_masks` reproduces AutoMorph's own M3 script.** The two runs arrived at identical feature
values by *different code paths* — the full run through
`M3_feature_whole_pic/retipy/create_datasets_macular_centred.py`, this run through
`ground_truth_features.measure_masks`. The inputs were confirmed to be the same masks
(`binary_process` and `macular_centred_binary_process` are byte-identical for all 26). Since the
ground-truth features are measured by that same function, the like-for-like comparison underpinning
the agreement analysis has independent support.

**Batch composition does not affect inference.** 26 images batch as 8/8/8/2 and 32 as 8/8/8/8, so
most images sat in a differently composed batch. Byte-identical output confirms the ensemble is
deterministic in eval mode.

## 6. A timing correction

M2 vessel segmentation took **1084s for 32 images (34s/image)** here. The full run recorded
**6046s for 26 (232s/image)** for the identical stage, identical code, device and batch size.

From the logs the difference is in the inference loop, not post-processing (1078s of 1084s here;
6040s of 6046s there), and the full run's per-batch times were erratic — 358, 161, 230 s/image —
against a stable 28–45 s/image now. That pattern points to resource contention or memory pressure
during the six-hour run rather than anything in the code, but **the cause is not established**.

The practical consequence: the stage timings in [results.md](results.md) should be treated as upper
bounds, and the "~6 hours" figure for the full pipeline is probably well above what a clean run
costs. No accuracy number is affected — only wall clock.

## 7. Conclusions

1. **The quality gate costs more than it buys.** It removes 19% of the cohort to gain 0.007 Dice.
   Four of six rejects segment as well as the median accepted image. It is not random — rejects are
   genuinely worse (p = 0.025) — but the threshold is set far too aggressively for small cohorts.
2. **The full run's `normal` Dice was an artefact of the gate.** 0.851 on 3 survivors versus 0.819 on
   all 8. Any per-disease figure from a gated run inherits the gate's bias.
3. **The full run's feature agreement was optimistic by 0.03–0.10 ICC.** Survivor bias, now measured
   rather than suspected.
4. **Conclusions about *which* features to trust are unchanged.** `Tortuosity_density` alone clears
   both bars; `Vessel_density` and `Average_width` are biased but rank-usable; the two remaining
   tortuosity measures fail, and `Distance_tortuosity` now fails on discriminability too.
5. **Under-segmentation is a property of the model, not of the cohort.** Sensitivity 0.736 across all
   32 mirrors 0.747 on the gated 26.
6. **The two runs are bit-identical on the 26 images they share**, so conclusions 2 and 3 are
   attributable to the gate alone — not to any difference in how the two runs compute. The
   comparison is clean.

## Caveats

- **n = 32**, 8 per disease. Better than the gated 26, still small.
- The gate comparison rests on **6 rejected images**. The direction (p = 0.025) is more trustworthy
  than the magnitude.
- Ground truth is downsampled with `nearest`, matching the full run. `area` would raise Dice by
  roughly 0.03 — see [protocol.md](protocol.md).
- `Average_width` is in nominal microns; its bias and ICC are unaffected by that, its absolute value
  is not physical. See [protocol.md](protocol.md).
- Artery/vein, disc/cup and the M3 zone features were not run and are not reported.
