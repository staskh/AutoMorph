# ABOUTME: Vessel-only benchmark: preprocessing and M2 segmentation over all 32 images, no quality gate.
# ABOUTME: Scores Dice and measures the six features from both the prediction and the annotation.

"""The vessel-only run.

The full benchmark (:mod:`benchmark.run`) answers "what does the pipeline do end to end", and its
answer is shaped by the M1 quality gate, which silently drops images before segmentation. That makes
it the wrong instrument for two questions:

* **How good is the segmentation on every image**, including the ones M1 rejects? The full run
  cannot say, because it never segments them.
* **Do the six morphometric features survive segmentation error** — measured on both sides by the
  same code, over the same images?

This run answers both. It executes M0 and M2 vessel segmentation only, **bypassing the quality gate
by making it a pass-through**: every preprocessed image is copied into ``Results/M1/Good_quality/``,
which is where M2 reads from. Nothing about M1 is simulated or stubbed — the gate is simply fed a
verdict of "everything passes".

Skipping artery/vein, disc/cup and the M3 zone scripts removes about four of the six hours the full
run costs, and none of it feeds the numbers reported here.

Isolation
---------
Writes to its **own** run root (``.benchmark_run_vessel``) and its **own** output directory
(``benchmark/results/M2_vessels/``). It never touches ``.benchmark_run`` or the top-level
``benchmark/results`` files, so the recorded full-pipeline run stays intact and the two sets of
numbers can be compared.

The features are measured by :func:`benchmark.ground_truth_features.measure_masks` for both sides —
literally the same call against two directory pairs — so predicted and true values differ only
because the masks differ, never because the measurement did.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import evaluate as evaluation
from benchmark import ground_truth_features as gtf
from benchmark import resolution as resolution_table
from benchmark.agreement import agreement_table, discriminability, informational_strength
from benchmark.run import (
    DEFAULT_VESSEL_THRESHOLD,
    REPO_ROOT,
    Stage,
    build_environment,
    count_files,
    run_stage,
)
from benchmark.selection import load_manifest, select, stage as stage_images

#: Current run root. ``.benchmark_run_vessel`` holds the frozen pre-fix masks, so the default points
#: elsewhere — a default run must not be able to destroy the baseline it is compared against.
DEFAULT_RUN_ROOT = REPO_ROOT / ".benchmark_run_thr02"
DEFAULT_STORE = REPO_ROOT / ".benchmark_data" / "fives"
#: Current results. ``M2_vessels`` holds the frozen pre-fix baseline the notebook compares against,
#: so the default must not point there — see benchmark/docs/fixes.md.
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "M2_vessels_thr02_mean"

#: Only the two stages whose output this run reports on.
STAGES = [
    Stage("M0 preprocess", "M0_Preprocess", ["python", "EyeQ_process_main.py"]),
    Stage("M2 vessel segmentation", "M2_Vessel_seg", ["sh", "test_outside.sh"]),
]

FEATURES = list(gtf.FEATURE_COLUMNS)


def prediction_mask_directories(results_root):
    """Where M2 writes its cleaned binary map and skeleton.

    ``filter_frag`` produces both, already at 912 and already through
    ``remove_small_objects(30, connectivity=5)`` — the same treatment
    :func:`benchmark.ground_truth_features.build_mask_pair` gives the annotation, which is what
    makes the two measurable on the same footing.
    """
    binary_vessel = Path(results_root) / "M2" / "binary_vessel"
    return binary_vessel / "binary_skeleton", binary_vessel / "binary_process"


def reuse_segmentation(source_root, run_root):
    """Copy an existing run's segmentation into ``run_root`` so features can be recomputed.

    Feature extraction depends on the vessel masks and on ``M0/crop_info.csv`` (retipy reads the
    micron scale from it), and on nothing else the pipeline produces. So a change to a feature
    *formula* can be evaluated without spending an hour re-running a segmentation that would come
    out byte-identical anyway.

    The copy is deliberate rather than a symlink or an in-place rerun: the new run root is separate,
    so the previous results stay readable for a before-and-after comparison.

    :return: how many prediction masks were carried over.
    :raises FileNotFoundError: if the source run has no segmentation or no crop_info.csv.
    """
    source = Path(source_root) / "Results"
    target = Path(run_root) / "Results"

    crop_info = source / "M0" / "crop_info.csv"
    masks = source / "M2" / "binary_vessel"
    if not crop_info.is_file():
        raise FileNotFoundError(f"no crop_info.csv at {crop_info} — the source run is incomplete")
    if not (masks / "binary_process").is_dir():
        raise FileNotFoundError(f"no segmentation at {masks} — nothing to reuse")

    (target / "M0").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(crop_info, target / "M0" / "crop_info.csv")

    copied = 0
    for subdir in ("binary_process", "binary_skeleton"):
        destination = target / "M2" / "binary_vessel" / subdir
        destination.mkdir(parents=True, exist_ok=True)
        for png in sorted((masks / subdir).glob("*.png")):
            shutil.copyfile(png, destination / png.name)
            copied += subdir == "binary_process"
    return copied


def bypass_quality_gate(run_root):
    """Make M1 a pass-through: every preprocessed image counts as good quality.

    M2 reads ``Results/M1/Good_quality/``. Populating it directly is what "skip M1" means here —
    the gate's *output* is supplied rather than its *model* being run, so no quality verdict is
    invented and every image reaches segmentation.

    :return: how many images were passed through.
    :raises FileNotFoundError: if M0 has not run.
    :raises ValueError: if M0 produced nothing.
    """
    results = Path(run_root) / "Results"
    source = results / "M0" / "images"
    if not source.is_dir():
        raise FileNotFoundError(f"no M0 output at {source} — preprocessing must run first")

    images = sorted(path for path in source.iterdir() if path.suffix.lower() == ".png")
    if not images:
        raise ValueError(f"no preprocessed images in {source}")

    good = results / "M1" / "Good_quality"
    good.mkdir(parents=True, exist_ok=True)
    for path in images:
        shutil.copyfile(path, good / path.name)
    return len(images)


def measure_both_sides(
    run_root, store, selection, output, size=evaluation.PIPELINE_SIZE,
    min_vessel_length=gtf.MIN_VESSEL_LENGTH,
):
    """Measure the six features from the prediction and from the annotation, and pair them.

    :return: ``(predicted, truth, paired)`` frames.
    """
    results = Path(run_root) / "Results"

    print("measuring predicted masks")
    predicted = gtf.measure_masks(
        *prediction_mask_directories(results), size=size, min_vessel_length=min_vessel_length
    )

    print("building ground-truth masks")
    gtf.build_masks(selection, store, results, size)
    print("measuring ground-truth masks")
    truth = gtf.measure(results, size=size, min_vessel_length=min_vessel_length)

    predicted.to_csv(output / "features_predicted.csv", index=False)
    truth.to_csv(output / "features_ground_truth.csv", index=False)

    left = predicted.assign(key=predicted["Name"].str.replace(".png", "", regex=False))
    right = truth.assign(key=truth["Name"].str.replace(".png", "", regex=False))
    paired = left.drop(columns="Name").merge(
        right.drop(columns="Name"), on="key", suffixes=("_pred", "_gt")
    )
    # retipy writes -1 for a failed measurement; that is a marker, not a value.
    for column in [f"{feature}_{side}" for feature in FEATURES for side in ("pred", "gt")]:
        paired[column] = paired[column].mask(paired[column] == -1)

    return predicted, truth, paired


def main(argv=None):
    """Stage 32 images, preprocess, segment vessels, then score Dice and both feature sets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-disease", type=int, default=8)
    parser.add_argument("--quality-score", type=int, default=3)
    parser.add_argument("--split", default="test")
    parser.add_argument("--resolution", type=float, default=resolution_table.DEFAULT_RESOLUTION_MM)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--vessel-threshold",
        type=float,
        default=DEFAULT_VESSEL_THRESHOLD,
        help="probability above which M2 calls a pixel vessel "
        f"(default {DEFAULT_VESSEL_THRESHOLD}); requires re-running segmentation",
    )
    parser.add_argument(
        "--min-vessel-length",
        type=int,
        default=gtf.MIN_VESSEL_LENGTH,
        help="shortest vessel a tortuosity measure is computed on, in skeleton pixels "
        f"(default {gtf.MIN_VESSEL_LENGTH})",
    )
    parser.add_argument("--skip-pipeline", action="store_true", help="re-score an existing run")
    parser.add_argument(
        "--reuse-segmentation-from",
        type=Path,
        default=None,
        help="copy the vessel masks and crop_info.csv from this run root instead of segmenting, "
        "for evaluating a change to a feature formula without re-running M2",
    )
    args = parser.parse_args(argv)

    if args.run_root.resolve() == (REPO_ROOT / ".benchmark_run").resolve():
        parser.error("refusing to write into the full run's root — pick a different --run-root")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split = None if args.split == "all" else args.split
    results = args.run_root / "Results"

    selection = select(load_manifest(args.store), args.per_disease, args.quality_score, split)
    selection.to_csv(output / "selection.csv", index=False)

    if args.reuse_segmentation_from:
        gtf.validate_paths(results)
        if results.exists():
            shutil.rmtree(results)
        results.mkdir(parents=True)
        carried = reuse_segmentation(args.reuse_segmentation_from, args.run_root)
        print(f"reused {carried} segmentations from {args.reuse_segmentation_from} — not re-running M2")
    elif not args.skip_pipeline:
        gtf.validate_paths(results)
        if results.exists():
            shutil.rmtree(results)
        results.mkdir(parents=True)

        stage_images(selection, args.store, args.run_root / "images")
        resolution_table.write(
            args.run_root / resolution_table.CSV_NAME,
            resolution_table.resolution_table(sorted(f"{k}.png" for k in selection["key"]), args.resolution),
        )
        print(f"staged {len(selection)} images -> {args.run_root}/images")

        environment = build_environment(
            args.run_root, args.device, args.num_workers, args.batch_size, args.vessel_threshold
        )
        print(f"binarising vessel probability at {args.vessel_threshold}")
        timings = []
        for index, stage in enumerate(STAGES, start=1):
            print(f"[{index}/{len(STAGES)}] {stage.name} ... ", end="", flush=True)
            result = run_stage(stage, REPO_ROOT, environment, output / "logs", args.attempts)
            timings.append(result)
            print(f"{result['status']} in {result['seconds']}s (attempts: {result['attempts']})")
            pd.DataFrame(timings).to_csv(output / "stage_timings.csv", index=False)

            if stage.name == "M0 preprocess" and result["status"] == "ok":
                started = time.time()
                passed = bypass_quality_gate(args.run_root)
                print(f"      quality gate bypassed: {passed} images passed through "
                      f"({round(time.time() - started, 1)}s)")

    scores = evaluation.evaluate(selection, args.store, results)
    scores.to_csv(output / "vessel_scores.csv", index=False)
    summary = evaluation.summarise(scores)
    summary.to_csv(output / "vessel_summary.csv")

    print(f"measuring tortuosity on vessels of at least {args.min_vessel_length} px")
    predicted, truth, paired = measure_both_sides(
        args.run_root, args.store, selection, output, min_vessel_length=args.min_vessel_length
    )

    metrics = ["key", "disease", "status", "dice", "iou", "sensitivity", "specificity", "accuracy"]
    paired = paired.merge(scores[metrics], on="key")
    paired.to_csv(output / "features_paired.csv", index=False)

    scored = paired[paired["status"] == "ok"]
    agreement = agreement_table(scored, FEATURES, interval=True)
    agreement.to_csv(output / "feature_agreement.csv")

    strength = informational_strength(
        truth.rename(columns={f: f"{f}_gt" for f in FEATURES}), FEATURES
    )
    strength.to_csv(output / "feature_informational_strength.csv")
    discriminability(agreement, informational_strength(scored, FEATURES)).to_csv(
        output / "feature_discriminability.csv"
    )

    print()
    print(f"segmented {count_files(results / 'M2' / 'binary_vessel' / 'binary_process')} "
          f"of {len(selection)} images")
    print(scores["status"].value_counts().to_string())
    print()
    print(summary.round(4).to_string())
    print()
    print(agreement[["n", "rel_bias_%", "MAE", "MAPE_%", "pearson_r", "spearman_r", "ICC21"]].round(3).to_string())
    print()
    print(f"results -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
