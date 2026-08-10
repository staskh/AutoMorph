# The vessel-only run

```bash
uv run python -m benchmark.run_vessel
```

Preprocessing and M2 vessel segmentation over **all 32 images**, with the M1 quality gate bypassed.
Reports Dice and the six morphometric features measured from *both* the prediction and the expert
annotation. About two hours on CPU; results land in `benchmark/results/M2_vessels/`.

## Why it exists

The [full run](results.md) answers "what does the pipeline do end to end", and its answer is shaped
by the M1 quality gate, which drops 6 of the 32 images before segmentation. That makes it the wrong
instrument for two questions:

1. **How good is segmentation on the images M1 rejects?** The full run cannot say — it never
   segments them. Since 5 of the 6 rejected images are healthy eyes, the full run's accuracy figures
   are conditioned on a non-random subset.
2. **Do the six features survive segmentation error**, measured on both sides by the same code over
   the same images?

This run answers both, on all 32.

## How the gate is skipped

M2 reads `Results/M1/Good_quality/`. The run copies **every** preprocessed image there, so the gate
is fed a verdict of "everything passes":

```
M0 preprocess  ->  Results/M0/images/        (32 images)
                        |
                        |  bypass_quality_gate: copy all
                        v
                   Results/M1/Good_quality/  (32 images)
                        |
                        v
M2 vessel segmentation
```

The gate's **output** is supplied; its **model** is not run and not stubbed. No quality verdict is
invented, and nothing about M1's behaviour is simulated — the quality network simply never executes.

Artery/vein, disc/cup and the six M3 scripts are skipped entirely. They cost about four of the full
run's six hours and none of them feeds a number reported here.

## Isolation from the full run

| | Full run | Vessel-only run |
| --- | --- | --- |
| Run root | `.benchmark_run/` | `.benchmark_run_vessel/` |
| Output | `benchmark/results/` | `benchmark/results/M2_vessels/` |

Separate roots, separate outputs. `run_vessel.py` **refuses to start** if `--run-root` points at the
full run's directory, so the recorded end-to-end results cannot be destroyed by a stray flag. The
two sets of numbers are meant to be read side by side.

## How the features are measured

Both sides go through `ground_truth_features.measure_masks` — the same function, the same retipy
configuration, the same window size — against two directory pairs:

| Side | Skeleton | Binary map | Produced by |
| --- | --- | --- | --- |
| Prediction | `M2/binary_vessel/binary_skeleton/` | `.../binary_process/` | M2's `filter_frag` |
| Ground truth | `M2/ground_truth/ground_truth_binary_skeleton/` | `.../ground_truth_binary_process/` | `build_mask_pair` |

This is what makes the comparison meaningful. M2's `filter_frag` applies
`remove_small_objects(30, connectivity=5)` and `skeletonize` to its own output; `build_mask_pair`
applies exactly the same two operations to the annotation. So the two sides differ **only because
the masks differ** — never because the measurement did.

`Average_width` is scaled by `Scale_resolution` from `M0/crop_info.csv`, which derives from the
nominal 0.008 mm/pixel the benchmark writes (FIVES publishes no pixel size). Both sides get the same
per-image scaling, so their comparison is sound; the absolute microns are not a physical
measurement. See [protocol.md](protocol.md).

Ground-truth masks are downsampled with `nearest`, matching the full run so the two are comparable.
[protocol.md](protocol.md) explains why `area` is the better-justified choice and what switching
would cost.

## Output

Everything under `benchmark/results/M2_vessels/`:

| File | Contents |
| --- | --- |
| `selection.csv` | The 32 images and their FIVES labels |
| `stage_timings.csv` | Status, attempts, exit code and wall clock for the two stages |
| `vessel_scores.csv` | Per image: confusion counts, Dice, IoU, sensitivity, specificity |
| `vessel_summary.csv` | Scores per disease and overall |
| `features_predicted.csv` | The six features measured from M2's segmentation |
| `features_ground_truth.csv` | The six features measured from the expert annotation |
| `features_paired.csv` | Both, joined per image, with that image's Dice |
| `feature_agreement.csv` | bias, rel. bias, MAE, MAPE, Pearson, Spearman, ICC(2,1) + CI, Bland–Altman |
| `feature_informational_strength.csv` | Ground-truth median, IQR, IQR/median |
| `feature_discriminability.csv` | Spread against measurement noise |
| `logs/` | Full stage output |

The notebook adds four more:

| File | Contents |
| --- | --- |
| `feature_verdicts.csv` | The combined per-feature verdict |
| `feature_leverage.csv` | Leave-one-out ICC: which image each result depends on |
| `dice_as_proxy.csv` | Correlation of Dice with each feature's error |
| `conclusions.txt` | Conclusions, generated from the run |

## Notebook

```bash
uv run python -m benchmark.build_notebook --profile vessel --run
```

Writes `benchmark/analysis_M2_vessels.ipynb`, executed in place. It shares nearly all of its cells
with the full run's `analysis.ipynb` — both are generated from `build_notebook.py`, with a `Profile`
supplying the data loading and the one section that genuinely differs: where the gate is enforced,
the question is what it dropped; here, the question is whether dropping it was right.

Edit `build_notebook.py` and regenerate; editing the `.ipynb` directly loses the change.

## Options

Same knobs as the full run — `--device`, `--num-workers`, `--batch-size`, `--per-disease`,
`--quality-score`, `--split`, `--resolution`, `--attempts`, `--vessel-threshold` — plus:

| Option | Default | Notes |
| --- | --- | --- |
| `--skip-pipeline` | off | Re-score an existing run without recomputing the segmentation |
| `--reuse-segmentation-from` | none | Copy masks from another run root, for a feature-formula change |
| `--min-vessel-length` | `50` | Shortest vessel a tortuosity measure is computed on |
| `--vessel-threshold` | `0.2` | Probability above which M2 calls a pixel vessel; changing it needs re-segmentation |

`--skip-pipeline` and `--reuse-segmentation-from` are the ones to use while iterating on the
analysis or a feature formula: both avoid the 25-minute segmentation.

Note that `--quality-score` still selects *which* images enter the run, using FIVES' own labels. It
is the **AutoMorph** quality model that is bypassed, not the dataset's quality filter.

## Results

See [vessel_results.md](vessel_results.md).
