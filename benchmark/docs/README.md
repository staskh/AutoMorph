# AutoMorph FIVES benchmark

A fixed, reproducible run of the whole AutoMorph pipeline over 32 FIVES fundus photographs, scored
against the FIVES expert vessel annotations.

The benchmark answers two questions:

1. **Does the pipeline complete?** How many images survive each module — in particular the M1
   quality gate, which silently drops images before segmentation ever runs.
2. **How good is the vessel segmentation?** Dice, IoU, sensitivity and specificity against the
   expert annotation, per image and per disease.
3. **Do the reported features survive the segmentation error?** The six whole-image morphometric
   features are also measured from the annotation itself, so predicted and true values can be
   compared image by image. Dice measures pixels; this measures what AutoMorph actually reports.

Wall-clock time per stage is recorded alongside all three, so a device or batch-size change can be
compared against a previous run.

**Headline results.** The M1 quality gate rejects 6 of 32 FIVES quality-3 images, 5 of them healthy
eyes, and each rejection is silent and total ([results.md](results.md)) — segmenting them anyway shows
it discards 19% of the cohort to gain 0.007 Dice ([vessel_results.md](vessel_results.md)).

Benchmarking also found three defects in vessel segmentation and measurement, now fixed
([fixes.md](fixes.md)). With them in place segmentation reaches **Dice 0.855** (sensitivity 0.852) and
all six reported features agree with the expert annotations well enough to use — three *reliable*,
three *usable with care*. Before the fixes one feature was degenerate, two were biased enough to need
calibration, and one had no detectable relationship with the truth.

## Documents

| Document | What it covers |
| --- | --- |
| [dataset.md](dataset.md) | FIVES, the store, and how the 32 images are chosen |
| [protocol.md](protocol.md) | What is measured and how the annotation is aligned to the pipeline |
| [running.md](running.md) | How to run the full pipeline benchmark, and what the knobs do |
| [results.md](results.md) | The recorded full run and what it shows |
| [vessel_run.md](vessel_run.md) | The vessel-only run: all 32 images, no quality gate |
| [vessel_results.md](vessel_results.md) | What the vessel-only run shows |
| [fixes.md](fixes.md) | Three defects the benchmark found, and what fixing them changed |

## The recorded runs

One directory per run under `benchmark/results/`, so runs stay distinguishable and cannot overwrite
each other. Each holds the same file set: scores, feature tables, verdicts and `conclusions.txt`.

| Directory | What it is | Stages | Gate | Fixes | Notebook |
| --- | --- | --- | --- | --- | --- |
| `End2End_original/` | The original end-to-end pipeline record | All 13 | Enforced — drops 6 of 32 | none | [`analysis.ipynb`](../analysis.ipynb) |
| `M2_vessels/` | Vessel-only, **frozen pre-fix baseline** | M0 + M2 vessel | Bypassed — all 32 | none | [`analysis_M2_vessels.ipynb`](../analysis_M2_vessels.ipynb) |
| `M2_vessels_thr02_mean/` | Vessel-only, **current defaults** | M0 + M2 vessel | Bypassed — all 32 | all three | [`analysis_M2_vessels_thr02.ipynb`](../analysis_M2_vessels_thr02.ipynb) |

`End2End_original/` and `M2_vessels/` are frozen records: nothing writes to them by default, and
tests pin that. A default `benchmark.run_vessel` writes to `M2_vessels_thr02_mean/`.

Read them together. `End2End_original` shows what a user of the unfixed pipeline actually gets,
`M2_vessels` shows what the segmentation could do once the quality gate stops filtering, and
`M2_vessels_thr02_mean` shows where it stands after [the fixes](fixes.md). The last notebook carries
the before-and-after directly.

| | End-to-end | Vessel-only |
| --- | --- | --- |
| Time (CPU) | ~6 h | ~25 min |
| Answers | What the pipeline does end to end | How good segmentation is on *every* image |

## Quick start

```bash
uv run python -m benchmark.run                    # full pipeline + scores          (~6 h on CPU)
uv run python -m benchmark.ground_truth_features  # measure the annotations         (~1 min)
uv run python -m benchmark.build_notebook --run   # regenerate the analysis         (~1 min)

uv run python -m benchmark.run_vessel             # vessel only, all 32 images      (~25 min on CPU)
uv run python -m benchmark.build_notebook --profile threshold --run
```

The first stages the subset, runs every module pinned to the CPU, scores the output and writes
everything under `benchmark/results/`. The next two add the feature comparison; they need the first
to have run, and both are cheap to repeat.

The last is independent — its own run root, its own output under
`benchmark/results/M2_vessels_thr02_mean/` — and computes Dice plus both feature sets in one go. Its
notebook compares against the frozen pre-fix baseline in `benchmark/results/M2_vessels/`.

## Layout

```
benchmark/
  selection.py             choose the 32 images and stage them for the pipeline
  resolution.py            write the resolution_information.csv that M0 reads
  evaluate.py              score M2's vessel maps against the FIVES annotations
  ground_truth_features.py measure the annotations with the same retipy code M3 uses
  agreement.py             ICC(2,1), bias/MAE/MAPE, informational strength
  run.py                   run every module in order, timed, and score the result
  run_vessel.py            M0 + M2 vessel only, no quality gate, both feature sets
  build_notebook.py        generate both notebooks (edit here, not the .ipynb)
  analysis.ipynb           full-run analysis, with plots
  analysis_M2_vessels.ipynb  vessel-only analysis, incl. grading the quality gate
  analysis_M2_vessels_thr02.ipynb  the same with all fixes applied, plus a before/after section

  fetch_fives.py           download and build the FIVES store
  datasets/fives.py        FIVES archive layout, quality labels, store schema
  geometry.py              move ground truth into the pipeline's 912 frame
  preprocess.py            M0's crop geometry as a reusable transform
  paths.py                 resolve $AUTOMORPH_BENCHMARK_DATA

  tests/                   unit tests for all of the above
  docs/                    this documentation
  results/End2End_original/      the original end-to-end run
  results/M2_vessels/            vessel-only, frozen pre-fix baseline
  results/M2_vessels_thr02_mean/ vessel-only, current defaults
```

The lower group builds the dataset store; the upper group runs and scores the benchmark. Only
`preprocess.py` bridges them, and it delegates every numeric step to AutoMorph's own
`M0_Preprocess/fundus_prep.py` — see [dataset.md](dataset.md) for how that is verified.

`analysis.ipynb` is generated from `build_notebook.py` so its source stays reviewable and diffable —
edit the Python, then regenerate. Editing the notebook directly means losing the change next time.
