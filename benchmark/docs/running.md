# Running the benchmark

```bash
uv run python -m benchmark.run
```

Everything else is a default. The run stages the 32 images, executes every module pinned to the
CPU, scores the vessel segmentation and writes the results.

Expect roughly an hour on an Apple-silicon laptop. M1 and the three M2 segmentation modules
dominate; M0 and M3 are minutes.

## What it does

1. **Prepare.** Clears the previous `Results/`, selects the 32 images, copies the native-resolution
   originals into the run root's `images/`, and writes `resolution_information.csv` beside them.
2. **Run.** Walks the same module sequence as `run.sh` — M0, M1 (+ quality merge), M2 vessel, M2
   artery/vein, M2 disc/cup, six M3 feature scripts, `csv_merge.py` — timing each.
3. **Score.** Counts what each module produced and scores M2's vessel maps against FIVES.

## Isolation

The run writes to `.benchmark_run/` (git-ignored), passed to the pipeline as `AUTOMORPH_DATA`. It
never touches the repository's own `images/` or `Results/`, so a benchmark cannot clobber a normal
AutoMorph run — or be clobbered by one.

Note that preparation **deletes** the run root's `images/` and `Results/` so a rerun cannot
silently inherit files from a previous one. Point `--run-root` elsewhere if that matters.

## Errors are recorded, not fatal

Each stage gets `--attempts` tries (2 by default). If it still fails, the failure is recorded in
`stage_timings.csv` with its exit code and the run **continues to the next stage**. This is
deliberate: a late module failing should not throw away the hours of segmentation before it, and
the completion counts make the damage visible — a stage that failed shows up as a shortfall in
`completion.csv` rather than as a missing report.

Full stdout and stderr for every stage, every attempt, lands in `benchmark/results/logs/`.

## Options

| Option | Default | Notes |
| --- | --- | --- |
| `--device` | `cpu` | Pins every module via `AUTOMORPH_DEVICE`. `mps` or `cuda:0` to compare |
| `--num-workers` | `0` | Dataloader workers, via `NUM_WORKERS`. 0 keeps it single-threaded |
| `--batch-size` | `8` | M1's batch size, via `AUTOMORPH_BATCH_SIZE`. Does not affect predictions |
| `--vessel-threshold` | `0.2` | Probability above which M2 calls a pixel vessel, via `AUTOMORPH_VESSEL_THRESHOLD`. `0.5` restores pre-benchmark behaviour — see [fixes.md](fixes.md) |
| `--per-disease` | `8` | Images per disease. `--per-disease 1` is a fast 4-image smoke test |
| `--quality-score` | `3` | FIVES quality label required |
| `--split` | `test` | `all` to draw from train and test together |
| `--resolution` | `0.008` | mm/pixel — nominal, see [protocol.md](protocol.md) |
| `--attempts` | `2` | Tries per stage before moving on |
| `--run-root` | `.benchmark_run` | Becomes `AUTOMORPH_DATA` |
| `--store` | `.benchmark_data/fives` | The FIVES store |
| `--output` | `benchmark/results` | Where the csv files go |
| `--skip-pipeline` | off | Re-score an existing run without recomputing it |

`--skip-pipeline` is the one to reach for while iterating on the scoring — it reuses the previous
run's `selection.csv` and segmentation output, so it finishes in seconds.

## Output

| File | Contents |
| --- | --- |
| `selection.csv` | The 32 chosen images with their FIVES labels |
| `stage_timings.csv` | Per stage: status, attempts, exit code, wall-clock seconds, log path |
| `completion.csv` | How many images survived each module |
| `vessel_scores.csv` | Per image: confusion counts and scores, or why it has none |
| `vessel_summary.csv` | Scores averaged per disease and overall |
| `logs/` | Full stage output (git-ignored) |

## CPU only

`--device cpu` sets `AUTOMORPH_DEVICE=cpu`, which every inference module honours through
`automorph_device.select_device`. Without it the modules auto-detect and would pick MPS on a Mac
or CUDA on a GPU box — fine for speed, but it makes runs incomparable and MPS and CPU kernels do
not always agree bit-for-bit.

`AUTOMORPH_DEVICE` is a general AutoMorph environment variable, not benchmark-specific; it works
for `run.sh` too.

## Tests

```bash
uv run python -m pytest benchmark/tests -q
```

Fast — no model inference. They cover selection, the resolution table, the scoring maths, the
crop alignment (against a synthetic fundus), and the runner's retry and environment handling.
