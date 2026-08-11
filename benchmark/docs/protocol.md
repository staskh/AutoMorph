# Protocol

## What is measured

### 1. Completion

How many of the 32 images each module produced output for. Written to
`benchmark/results/End2End_original/completion.csv`.

This is not a formality. M1 is a **gate**: `merge_quality_assessment.py` sorts every image into
`Results/M1/Good_quality/` or `Results/M1/Bad_quality/`, and M2 reads only the former. An image M1
rejects produces no segmentation, no features, and no row in the final feature tables. Because
every benchmark image is FIVES quality 3, the gate's rejection rate is a result in its own right.

### 2. Vessel segmentation accuracy

M2's binary vessel map against the FIVES expert annotation, on the 912x912 grid the network works
in and M3 consumes. Per image in `benchmark/results/End2End_original/vessel_scores.csv`, averaged per disease and
overall in `vessel_summary.csv`.

With TP/FP/FN/TN counted **inside the field of view only**:

| Score | Definition |
| --- | --- |
| Dice | `2TP / (2TP + FP + FN)` |
| IoU | `TP / (TP + FP + FN)` |
| Sensitivity | `TP / (TP + FN)` — of the annotated vessel, how much was found |
| Specificity | `TN / (TN + FP)` — of the annotated background, how much was left alone |
| Accuracy | `(TP + TN) / all` |

All four are counted **inside the detected field of view only** — `confusion()` masks the
prediction, the annotation and the true-negative count by it. The circular retina covers 78.5% of
the pipeline's square; the remaining 21.5% is black padding the networks never saw, and counting it
as correctly-classified background would flatter the model for free.

How much that restriction actually changes, measured over the 26 scored images:

| | FOV only | whole square | delta |
| --- | --- | --- | --- |
| Dice | 0.8321 | 0.8317 | −0.0003 |
| Specificity | 0.9947 | 0.9959 | +0.0012 |
| Accuracy | 0.9693 | 0.9758 | +0.0065 |

**Dice and IoU are almost immune to it, structurally**: neither formula contains a TN term, so
excluding background can only move them through vessel pixels lying outside the field of view, and
there are barely any — 36 annotated and 14 predicted per image, against ~69,000 vessel pixels. The
restriction earns its keep on specificity and accuracy, the two metrics that do count TN.

Separately, and for a different reason: vessels occupy roughly 2% of the field of view, so
**accuracy and specificity sit near 1 for any prediction** and carry almost no information. That is
prevalence, not padding — restricting to the field of view does not rescue them. Dice and IoU are
the scores worth reading; sensitivity says which way the errors fall.

### 3. Feature agreement and informational strength

Dice measures pixels; AutoMorph reports morphometry. So the six whole-image features are also
measured **from the expert annotation** and compared with the predicted values image by image.
Written to `benchmark/results/End2End_original/feature_agreement.csv` by `benchmark/analysis.ipynb`; the statistics
themselves live in `benchmark/agreement.py`, which is unit-tested — ICC against hand-computed ANOVA
values.

| Statistic | What it adds |
| --- | --- |
| `bias`, `rel_bias_%` | Mean signed difference, absolute and as a percentage. Calibratable if stable |
| `MAE`, `MAPE_%` | Mean absolute error, in the feature's units and as a percentage |
| `pearson_r` | Linear association — **blind to bias**, a constant offset still scores 1.0 |
| `spearman_r` | Rank association: what matters if the feature orders patients |
| `ICC21` | Shrout & Fleiss ICC(2,1): two-way random effects, absolute agreement, single measurement. **Charges for systematic bias**, so it is the honest headline. Bootstrap 95% CI reported |
| `BA_low`, `BA_high` | Bland–Altman limits, mean difference ± 1.96 SD |

ICC is read with Koo & Li's conventional bands (<0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, >0.9
excellent). Those boundaries are conventions, not facts. The **gap between Pearson and ICC is the
bias**, and on this data it is large for two features — reporting Pearson alone would overstate
their usability.

**Informational strength** is the other half of the question: a feature must vary between eyes or it
cannot separate them however precisely it is measured. Reported as `IQR / median` over the ground
truth — robust, unit-free, and a property of the cohort rather than of the pipeline, so it is
computed over all 32 annotated images rather than only the 26 that passed the quality gate.

The two halves combine as `spread_to_noise`: ground-truth IQR ÷ SD of the prediction error, with
systematic bias excluded, since a constant offset does not stop a feature separating two eyes but
random scatter does. Below about 1 the noise covers the whole interquartile range and the feature
cannot rank eyes, regardless of how good its error metrics look.

Comparability is the whole point, so the annotation is put through the identical path M2's output
takes before M3 measures it — aligned to the pipeline crop, resized to 912, restricted to the field
of view, `remove_small_objects(30, connectivity=5)` exactly as `filter_frag` applies, skeletonised,
then measured by the same `retipy.tortuosity_measures.evaluate_window` at window size 912. Only the
first step differs from what the prediction gets, and it has to: the prediction is born at 912 while
the annotation is native.

`benchmark/ground_truth_features.py` does this. Two of retipy's conventions are load-bearing and
easy to break by renaming a directory, so it validates them up front rather than failing later with
a confusing read error:

* it recovers a skeleton's binary map as `store_path + path.split('_skeleton')[1]`, so `_skeleton`
  must appear exactly once in the path;
* it finds `crop_info.csv` as `store_path.split('M2')[0] + 'M0/crop_info.csv'`, so `M2` must appear
  exactly once.

### 4. Wall-clock time

Per stage, with attempt count and exit status, in `benchmark/results/End2End_original/stage_timings.csv`.

## Aligning the annotation

The annotation is at native 2048x2048; the prediction is on a 912x912 grid derived from a crop of
the photograph. Comparing them requires reproducing M0's crop exactly, and "exactly" is the whole
difficulty — M0 does not simply cut a circle. It finds the retina by thresholding, masks outside
it, trims to the bounding box, then pads the result to a square.

Rather than reimplement that, `evaluate.align_annotation` calls **the same function M0 calls**,
`fundus_prep.process_without_gb`, which already accepts a label alongside the image and applies
the identical crop to both. The annotation therefore lands in the pipeline's own frame, driven by
the same photograph, with no chance of drift.

Then both the annotation and the field-of-view mask are taken down to 912 with nearest-neighbour
sampling, which is what keeps them binary.

### Two things worth knowing

**The annotation is cropped but not circle-masked.** M0 masks the *photograph* outside the retina
circle, but only crops the *label* to the bounding box. A little annotation can therefore sit
outside the detected field of view. Those pixels are excluded by the field-of-view restriction, so
they never become false negatives — there is a test pinning exactly that.

**Downsampling costs thin vessels, and how you downsample is a choice.** Taking a 2048-scale
annotation to 912 loses single-pixel vessels. This is a property of the grid the pipeline works in,
not of the scoring: the prediction is produced at 912 too, so both sides are evaluated at the same
scale. It does mean the Dice here is not comparable to a FIVES leaderboard number computed at native
resolution.

`benchmark.geometry.resize_mask` implements both rules and the caller must pick one: `area`
(coverage averaged, kept at ≥ 0.5) or `nearest` (point-sampled).

**Neither is intrinsically the better estimator.** Measured on FIVES they recover the same total
vessel area — 69,186 (nearest) and 69,213 (area) against an area-preserving expectation of 69,232 —
and are equally stable to sub-pixel translation (CV 0.054% vs 0.051%). The plausible argument that
point sampling is noisier because it inspects 1 of the 4.88 native pixels behind each output pixel
does not survive measurement.

**What distinguishes them is consistency with the image.** M2 feeds the network
`Image.resize((912, 912))`, and PIL's resize is antialiased — on a FIVES crop it sits 0.090 grey
levels from `INTER_AREA` and 0.688 from `INTER_NEAREST`. The network is asked *"what dominates this
cell?"*. Ground truth that answers *"what is at this point?"* compares two sampling models and
manufactures boundary disagreement unrelated to model quality.

The rule to follow is therefore **match the operator applied to the image**, which here means
`area`.

| | Dice | Sensitivity |
| --- | --- | --- |
| GT via `nearest` (current) | 0.8321 | 0.7474 |
| GT via `area` | 0.8607 | 0.7731 |

`area` scores higher on 26 of 26 images (mean +0.0286, sd 0.0076). **The store was built with
`area`; scoring currently uses `nearest`** (`evaluate.RESIZE_METHOD`) — which is why recovered
annotation counts differ slightly from the manifest, and it means the reported Dice is conservative
by about 0.03. Switching also changes the ground-truth skeleton and so the feature agreement, most
sharply for the tortuosity measures (`Tortuosity_density` ICC 0.805 → 0.854, `Distance_tortuosity`
0.443 → 0.283) — that sensitivity to a preprocessing choice which leaves vessel *area* untouched is
itself evidence those measures are fragile.

The thin-structure objection to `area` — that a vessel covering under half an output pixel is
deleted — does not apply here: FIVES annotations average 14 px wide at native, 6.3 px at 912.
Where it does apply, the answer is not to pick a rule but to stop downsampling ground truth:
upsample the prediction and score at native resolution.

### Cross-check

The alignment was verified against the store, which was built independently by a different
implementation. For the benchmark images the recovered crop radius and annotated pixel count match
the store's recorded values exactly (e.g. `test_12_A`: radius 1007 vs 1007, 339546 annotated
pixels vs 339546), and the 912 counts agree to within the difference between the two resamplers.

## Pixel resolution — read this before trusting any micron

M0 reads `resolution_information.csv` and multiplies its `res` column by the crop scale to get the
microns-per-pixel figure behind every width, diameter, and calibre metric AutoMorph reports.

**FIVES publishes no pixel size.** There is no correct value to write. The benchmark writes
`0.008` mm/pixel uniformly across all 32 images — the placeholder AutoMorph's own README suggests
for a Topcon 3D-OCT. It is inherited, not derived or measured.

### What it actually affects

Less than "every micron" suggests. Of the six reported features, `resolution` enters exactly one —
it appears once in the whole measurement path:

```python
width = np.sum(vessel_) / np.sum(skeleton) * retina.resolution   # tortuosity_measures.py:73
```

`Fractal_dimension`, `Vessel_density`, `Distance_tortuosity`, `Squared_curvature_tortuosity` and
`Tortuosity_density` are dimensionless. Every segmentation score is computed in pixels. So the
constant reaches `Average_width` (and M3's other calibre metrics, CRAE/CRVE) and nothing else.

**And even there it cancels in every comparison statistic.** Prediction and ground truth are scaled
by the same per-image `scale_resolution`, so `rel_bias_%`, `MAPE`, Pearson, Spearman, ICC(2,1) and
`IQR / median` are all invariant to the choice — `Average_width`'s ICC of 0.353 and its −12.3% bias
would be identical at any resolution. Only the absolute micron value and `MAE` scale linearly.

### Is 0.008 close?

Probably ~10% high, on a geometric estimate that has **not** been confirmed against FIVES'
acquisition details:

| | mm/pixel | implied mean annotated vessel width |
| --- | --- | --- |
| 0.008 (used) | 0.00800 | 115 µm |
| 50° field of view, 0.29 mm/deg | 0.00720 | 104 µm |
| 50° field of view, 0.28–0.30 mm/deg | 0.0070–0.0075 | 100–107 µm |

The retina spans ~2014 px after M0's crop. Both figures sit inside the physiologically plausible
band for retinal vessels (~50–200 µm), so this is weak corroboration that 0.008 is not absurd —
not grounds to replace it. Note also that no single constant can be right for every image:
mm-per-degree varies with axial length, making the true value per-eye.

`benchmark/resolution.py` takes `--resolution` and per-image `--override` if a real figure becomes
available.
