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

**Headline results** ([full write-up](results.md)): the M1 quality gate rejected 6 of 32 FIVES
quality-3 images, 5 of them healthy eyes. Segmentation reaches Dice 0.832 and under-segments
(sensitivity 0.747, specificity 0.995). Of the six reported features, only `Tortuosity_density`
clears both agreement and informativeness (ICC 0.81); `Vessel_density` and `Average_width` track the
truth but sit 20% and 12% low, so ICC rates them 0.50 and 0.35 where Pearson says 0.94 and 0.79; and
`Squared_curvature_tortuosity` has no detectable relationship with the truth at all (ICC 0.088).
The features measured most accurately turn out to be the ones with the least spread to measure.

## Documents

| Document | What it covers |
| --- | --- |
| [dataset.md](dataset.md) | FIVES, the store, and how the 32 images are chosen |
| [protocol.md](protocol.md) | What is measured and how the annotation is aligned to the pipeline |
| [running.md](running.md) | How to run the full pipeline benchmark, and what the knobs do |
| [results.md](results.md) | The recorded full run and what it shows |
| [vessel_run.md](vessel_run.md) | The vessel-only run: all 32 images, no quality gate |
| [vessel_results.md](vessel_results.md) | What the vessel-only run shows |
| [tortuosity_fix.md](tortuosity_fix.md) | A tracing bug in `detect_vessel_border`, and what fixing it changed |
| [vessel_threshold.md](vessel_threshold.md) | Binarising at 0.2 instead of 0.5 — the largest single improvement measured |

## Two runs

| | [Full run](running.md) | [Vessel-only run](vessel_run.md) |
| --- | --- | --- |
| Stages | All 13 | M0 + M2 vessel |
| Quality gate | Enforced — drops 6 of 32 | Bypassed — all 32 segmented |
| Time (CPU) | ~6 h | ~2 h |
| Answers | What the pipeline does end to end | How good segmentation is on *every* image |

They write to separate roots and separate output directories, and are meant to be read together:
the full run shows what a user actually gets, the vessel-only run shows what the segmentation can do
when nothing is filtered out first.

## Quick start

```bash
uv run python -m benchmark.run                    # full pipeline + scores          (~6 h on CPU)
uv run python -m benchmark.ground_truth_features  # measure the annotations         (~1 min)
uv run python -m benchmark.build_notebook --run   # regenerate the analysis         (~1 min)

uv run python -m benchmark.run_vessel             # vessel only, all 32 images      (~20 min on CPU)
uv run python -m benchmark.build_notebook --profile vessel --run
```

The first stages the subset, runs every module pinned to the CPU, scores the output and writes
everything under `benchmark/results/`. The next two add the feature comparison; they need the first
to have run, and both are cheap to repeat.

The last is independent — its own run root, its own output under `benchmark/results/M2_vessels/` —
and computes Dice plus both feature sets in one go.

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
  analysis_M2_vessels_fixed.ipynb  the same after the tortuosity fix, with a before/after section
  analysis_M2_vessels_thr02.ipynb  the same at binarisation threshold 0.2, with a sweep

  fetch_fives.py           download and build the FIVES store
  datasets/fives.py        FIVES archive layout, quality labels, store schema
  geometry.py              move ground truth into the pipeline's 912 frame
  preprocess.py            M0's crop geometry as a reusable transform
  paths.py                 resolve $AUTOMORPH_BENCHMARK_DATA

  tests/                   unit tests for all of the above
  docs/                    this documentation
  results/                 run output (csv), regenerated by every run
```

The lower group builds the dataset store; the upper group runs and scores the benchmark. Only
`preprocess.py` bridges them, and it delegates every numeric step to AutoMorph's own
`M0_Preprocess/fundus_prep.py` — see [dataset.md](dataset.md) for how that is verified.

`analysis.ipynb` is generated from `build_notebook.py` so its source stays reviewable and diffable —
edit the Python, then regenerate. Editing the notebook directly means losing the change next time.
