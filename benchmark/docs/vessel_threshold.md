# Binarising vessel probability at 0.2

```bash
uv run python -m benchmark.run_vessel \
    --run-root .benchmark_run_thr02 --vessel-threshold 0.2 --min-vessel-length 50 \
    --output benchmark/results/M2_vessels_thr02
uv run python -m benchmark.build_notebook --profile threshold --run
```

M2 averages ten ensemble sigmoids and calls a pixel vessel above a probability threshold. That
threshold was hardcoded at **0.5**. Lowering it to **0.2** is the single largest improvement measured
anywhere in this benchmark.

Unlike the [tortuosity fix](tortuosity_fix.md), this changes the segmentation itself, so there is no
control group — every number moves, Dice included. Both runs compared here share the tortuosity fix
and a 50 px minimum vessel length, so the threshold is the only difference.

## Why 0.5 was suspect

The [vessel-only run](vessel_results.md) measured sensitivity **0.736** against specificity
**0.995**: the ensemble was missing about a quarter of the annotated vessel while inventing almost
nothing. That asymmetry is the signature of a threshold set too high, not of a model that cannot see
vessels. `Vessel_density` running **21% low** was the same fact showing up downstream.

## What changed

| | at 0.5 | **at 0.2** | change |
| --- | --- | --- | --- |
| Dice | 0.8249 | **0.8553** | **+0.030** |
| IoU | 0.7050 | **0.7487** | +0.044 |
| Sensitivity | 0.7360 | **0.8521** | **+0.116** |
| Specificity | 0.9948 | 0.9833 | −0.011 |

Sensitivity gains eleven points for one point of specificity. Because vessels are only ~2% of the
field of view, that trade is strongly favourable: Dice and IoU both rise.

### The features improve far more than Dice does

| Feature | ICC at 0.5 | **ICC at 0.2** | bias % at 0.5 | **at 0.2** | MAPE % at 0.5 | **at 0.2** |
| --- | --- | --- | --- | --- | --- | --- |
| **Vessel_density** | 0.409 | **0.883** | −21.42 | **−0.04** | 22.05 | **6.39** |
| **Fractal_dimension** | 0.701 | **0.921** | −2.01 | +0.53 | 2.02 | **0.93** |
| **Squared_curvature_tortuosity** | −0.031 | **0.483** | −7.14 | −5.71 | 6.25 | 3.13 |
| **Average_width** | 0.319 | **0.641** | −12.94 | **−5.34** | 12.89 | **6.00** |
| Tortuosity_density | 0.822 | 0.819 | −0.22 | −1.41 | 2.97 | 2.78 |
| Distance_tortuosity | 0.868 | 0.762 | −0.09 | −0.18 | 0.21 | 0.26 |

**`Vessel_density`'s bias essentially vanishes — from −21.4% to −0.04%.** That is the headline. The
bias reported in [results.md](results.md) as a systematic property of the pipeline was a
*thresholding artefact*: the ensemble's probability mass was there all along, sitting between 0.2 and
0.5 and being discarded. `Fractal_dimension` reaches *excellent* agreement (0.921), and
`Average_width`'s bias more than halves.

`Distance_tortuosity` is the one feature that gets slightly worse (ICC 0.868 → 0.762). Admitting
lower-confidence pixels roughens the skeleton, which is what an arc-chord ratio is most sensitive to.
It remains *good*.

## Is 0.2 the right threshold?

Checkable, and checked. M2 saves the averaged sigmoid map in `binary_vessel/resize/`, so the
threshold can be swept without re-running the network: re-binarise, re-apply
`remove_small_objects(30, connectivity=5)`, re-score.

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

**0.2 is the Dice optimum**, with 0.15–0.25 within 0.002 of it — a broad, flat maximum, so the exact
value is not delicate. The default of 0.5 sits well down the slope, costing 0.03 Dice and 0.12
sensitivity. The sweep reproduces the 0.5 run's Dice (0.8249) exactly, which is the check that it
replicates the pipeline's own path.

Note the sweep optimises **Dice**. A use case that cared more about not inventing vessels would pick
a higher threshold, and one calibrating `Vessel_density` would pick whatever zeroes its bias — near
0.2 as it happens.

## How it is configured

**0.2 is now the pipeline default**, not just the benchmark's choice. `AUTOMORPH_VESSEL_THRESHOLD` in
`M2_Vessel_seg/test_outside_integrated.py` defaults to 0.2, so an ordinary `run.sh` gets it; set the
variable to 0.5 to restore the previous behaviour.

`benchmark.run.DEFAULT_VESSEL_THRESHOLD` matches, and a test asserts the two agree — two defaults for
one quantity would let a plain pipeline run and a benchmark run disagree silently. Both runners take
`--vessel-threshold`.

The variable is applied at **both** places M2 binarises: `resize_binary` (the 912 grid, which feeds
`filter_frag` and therefore every feature) and `raw_binary` (crop resolution). A test asserts both
moved, because changing one alone would silently desynchronise the two.

Changing it requires re-running segmentation, unlike a feature-formula change — about 25 minutes on
CPU for 32 images.

> **Results produced before this change are not comparable.** Everything in
> [results.md](results.md), [vessel_results.md](vessel_results.md) and
> [tortuosity_fix.md](tortuosity_fix.md), and the stored tables in `benchmark/results/`,
> `M2_vessels/`, `M2_vessels_fixed/` and `M2_vessels_fixed_25px/`, was produced at 0.5. Re-running
> any of them now will give different — better — numbers. `M2_vessels_thr02/` is the one set that
> reflects the new default.

## What this means for earlier conclusions

- **The under-segmentation is largely a threshold choice, not a model limitation.** Reported as a
  property of the pipeline in [results.md](results.md) and
  [vessel_results.md](vessel_results.md); at 0.2 sensitivity is 0.852.
- **`Vessel_density`'s −21% bias is not intrinsic.** It was the same threshold, and it goes to −0.04%.
  Anywhere those documents advise calibrating it before quoting absolutes, the better fix is the
  threshold.
- **`Fractal_dimension` becomes the most trustworthy feature** — ICC 0.921 with spread/noise 2.5.
- Unchanged: `Squared_curvature_tortuosity` is still unusable (constant across eyes), and
  `Tortuosity_density` still has error covering its population spread.

## Caveats

- **n = 32**, one cohort, all FIVES quality 3. 0.2 is the optimum *here*; it is not a calibration
  transferable to other data without checking.
- The sweep optimises Dice, which weights sensitivity and precision equally. That is a choice, not a
  clinical requirement.
- Only vessel segmentation was re-run. Artery/vein and disc/cup have their own thresholds, untouched.
- Lower thresholds admit more small fragments; `remove_small_objects(30)` absorbs some of that, and
  the sweep applies it at every threshold so the comparison is fair.
