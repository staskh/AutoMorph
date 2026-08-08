# Protocol

## What is measured

### 1. Completion

How many of the 32 images each module produced output for. Written to
`benchmark/results/completion.csv`.

This is not a formality. M1 is a **gate**: `merge_quality_assessment.py` sorts every image into
`Results/M1/Good_quality/` or `Results/M1/Bad_quality/`, and M2 reads only the former. An image M1
rejects produces no segmentation, no features, and no row in the final feature tables. Because
every benchmark image is FIVES quality 3, the gate's rejection rate is a result in its own right.

### 2. Vessel segmentation accuracy

M2's binary vessel map against the FIVES expert annotation, on the 912x912 grid the network works
in and M3 consumes. Per image in `benchmark/results/vessel_scores.csv`, averaged per disease and
overall in `vessel_summary.csv`.

With TP/FP/FN/TN counted **inside the field of view only**:

| Score | Definition |
| --- | --- |
| Dice | `2TP / (2TP + FP + FN)` |
| IoU | `TP / (TP + FP + FN)` |
| Sensitivity | `TP / (TP + FN)` — of the annotated vessel, how much was found |
| Specificity | `TN / (TN + FP)` — of the annotated background, how much was left alone |
| Accuracy | `(TP + TN) / all` |

Restricting to the field of view matters. The pipeline's square is mostly black padding outside
the circular retina; counting that padding as true negative would push specificity and accuracy
towards 1 and make them meaningless.

Vessels occupy roughly 2% of the field of view, so **accuracy and specificity are near 1 for any
prediction** and carry almost no information. Dice and IoU are the scores worth reading;
sensitivity says which way the errors fall.

### 3. Feature agreement

Dice measures pixels; AutoMorph reports morphometry. So the six whole-image features are also
measured **from the expert annotation**, and compared with the predicted values image by image:
bias, MAPE, Pearson and Spearman correlation, Bland–Altman limits. Written to
`benchmark/results/feature_agreement.csv` by `benchmark/analysis.ipynb`.

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

Per stage, with attempt count and exit status, in `benchmark/results/stage_timings.csv`.

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

**Downsampling costs thin vessels.** Taking a 2048-scale annotation to 912 loses single-pixel
vessels. This is a property of the grid the pipeline works in, not of the scoring: the prediction
is produced at 912 too, so both sides are evaluated at the same scale. It does mean the Dice here
is not comparable to a FIVES leaderboard number computed at native resolution.

### Cross-check

The alignment was verified against the store, which was built independently by a different
implementation. For the benchmark images the recovered crop radius and annotated pixel count match
the store's recorded values exactly (e.g. `test_12_A`: radius 1007 vs 1007, 339546 annotated
pixels vs 339546), and the 912 counts agree to within the difference between the two resamplers.

## Pixel resolution — read this before trusting any micron

M0 reads `resolution_information.csv` and multiplies its `res` column by the crop scale to get the
microns-per-pixel figure behind every width, diameter, and calibre metric AutoMorph reports.

**FIVES publishes no pixel size.** There is no correct value to write. The benchmark writes
`0.008` mm/pixel — the placeholder AutoMorph's own README suggests for a Topcon 3D-OCT — uniformly
across all 32 images.

Consequently every micron-denominated feature in the results is **nominal**: internally consistent
and comparable between benchmark images, but not a physical measurement. Dimensionless features
(fractal dimension, vessel density, tortuosity, cup-to-disc ratio) are unaffected, as are all the
segmentation scores above, which are computed in pixels.

`benchmark/resolution.py` takes `--resolution` and per-image `--override` if a real figure ever
becomes available.
