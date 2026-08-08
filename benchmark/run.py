# ABOUTME: Runs the whole AutoMorph pipeline over the FIVES benchmark subset and scores the result.
# ABOUTME: Forces one device, times every stage, retries a failed stage and carries on past it.

"""The benchmark run.

Stages the subset, writes the resolution table, then walks the same module sequence ``run.sh``
does, but with three differences that make it a benchmark rather than a batch job:

* **One device.** ``AUTOMORPH_DEVICE`` pins every module to the same device, so a rerun on the
  same machine is comparable and a Mac does not silently switch to MPS.
* **Timed.** Every stage's wall-clock time is recorded, per stage, alongside its exit status.
* **Not all-or-nothing.** A stage that fails is retried; if it still fails the run carries on to
  the next stage and the failure is recorded. Later stages usually then produce less output, which
  is exactly what the completion counts in the report are there to show.

Everything is written under a run root of its own (``.benchmark_run`` by default), so a benchmark
never touches the ``images/`` and ``Results/`` a normal AutoMorph run uses.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import evaluate as evaluation
from benchmark import resolution as resolution_table
from benchmark.selection import load_manifest, select, stage as stage_images

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_ROOT = REPO_ROOT / ".benchmark_run"
DEFAULT_STORE = REPO_ROOT / ".benchmark_data" / "fives"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results"

#: M1's quality ensemble runs eight EfficientNet-B4 models; a smaller batch keeps peak memory
#: predictable on CPU. Batch size does not change the predictions — every model runs in eval mode.
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class Stage:
    """One pipeline step: a command, and the module directory it must run from."""

    name: str
    cwd: str
    command: list = field(default_factory=list)


#: The module sequence, mirroring run.sh.
STAGES = [
    Stage("M0 preprocess", "M0_Preprocess", ["python", "EyeQ_process_main.py"]),
    Stage("M1 quality assessment", "M1_Retinal_Image_quality_EyePACS", ["sh", "test_outside.sh"]),
    Stage("M1 quality merge", "M1_Retinal_Image_quality_EyePACS", ["python", "merge_quality_assessment.py"]),
    Stage("M2 vessel segmentation", "M2_Vessel_seg", ["sh", "test_outside.sh"]),
    Stage("M2 artery/vein", "M2_Artery_vein", ["sh", "test_outside.sh"]),
    Stage("M2 disc/cup", "M2_lwnet_disc_cup", ["sh", "test_outside.sh"]),
    Stage("M3 zone disc B", "M3_feature_zone/retipy", ["python", "create_datasets_disc_centred_B.py"]),
    Stage("M3 zone disc C", "M3_feature_zone/retipy", ["python", "create_datasets_disc_centred_C.py"]),
    Stage("M3 zone macular B", "M3_feature_zone/retipy", ["python", "create_datasets_macular_centred_B.py"]),
    Stage("M3 zone macular C", "M3_feature_zone/retipy", ["python", "create_datasets_macular_centred_C.py"]),
    Stage("M3 whole macular", "M3_feature_whole_pic/retipy", ["python", "create_datasets_macular_centred.py"]),
    Stage("M3 whole disc", "M3_feature_whole_pic/retipy", ["python", "create_datasets_disc_centred.py"]),
    Stage("csv merge", ".", ["python", "csv_merge.py"]),
]


def build_environment(run_root, device, num_workers, batch_size=DEFAULT_BATCH_SIZE):
    """The environment every stage runs under.

    The running interpreter's directory goes first on PATH because the module shell scripts call
    plain ``python``, which must be this interpreter and not whatever the login shell resolves.
    """
    environment = dict(os.environ)
    environment["AUTOMORPH_DATA"] = str(Path(run_root).resolve())
    environment["AUTOMORPH_DEVICE"] = device
    environment["NUM_WORKERS"] = str(num_workers)
    environment["AUTOMORPH_BATCH_SIZE"] = str(batch_size)
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", "")]
    )
    return environment


def run_stage(stage, repo_root, environment, log_directory, attempts=2):
    """Run one stage, retrying on failure, and report what happened without raising.

    :return: a row holding the status, attempt count, exit code, wall-clock seconds and log path.
    """
    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / (stage.name.replace("/", "-").replace(" ", "_") + ".log")

    started = time.time()
    returncode = None
    attempt = 0

    with log_path.open("w") as log:
        while attempt < attempts:
            attempt += 1
            log.write(f"=== attempt {attempt}: {' '.join(stage.command)} (cwd={stage.cwd})\n")
            log.flush()
            try:
                completed = subprocess.run(
                    stage.command,
                    cwd=Path(repo_root) / stage.cwd,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                returncode = completed.returncode
            except OSError as error:
                log.write(f"could not start the command: {error}\n")
                returncode = None
            log.flush()
            if returncode == 0:
                break

    return {
        "stage": stage.name,
        "status": "ok" if returncode == 0 else "failed",
        "attempts": attempt,
        "returncode": returncode,
        "seconds": round(time.time() - started, 1),
        "log": str(log_path),
    }


def count_files(directory, suffix=".png"):
    """How many result files a module produced, for the completion counts in the report."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for entry in directory.iterdir() if entry.name.endswith(suffix))


def count_rows(csv_path):
    """How many images a feature table covers, or 0 if the module never wrote one."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return 0
    return len(pd.read_csv(csv_path))


def completion_counts(run_root, expected):
    """Per-module output counts, to show where images dropped out of the pipeline."""
    results = Path(run_root) / "Results"
    stages = {
        "staged images": count_files(Path(run_root) / "images"),
        "M0 preprocessed": count_files(results / "M0" / "images"),
        "M1 good quality": count_files(results / "M1" / "Good_quality"),
        "M1 bad quality": count_files(results / "M1" / "Bad_quality"),
        "M2 binary vessel": count_files(results / "M2" / "binary_vessel" / "binary_process"),
        "M2 artery/vein": count_files(results / "M2" / "artery_vein" / "artery_binary_process"),
        "M2 disc/cup": count_files(results / "M2" / "optic_disc_cup" / "raw"),
        "M3 disc features": count_rows(results / "M3" / "Disc_Features.csv"),
        "M3 macular features": count_rows(results / "M3" / "Macular_Features.csv"),
    }
    return pd.DataFrame(
        [{"stage": name, "images": count, "of_expected": expected} for name, count in stages.items()]
    )


def prepare(run_root, store, per_disease, quality_score, split, resolution_mm, output):
    """Stage the subset and write the resolution table, returning the selection."""
    run_root = Path(run_root)
    results = run_root / "Results"
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True)

    chosen = select(load_manifest(store), per_disease, quality_score, split)
    staged = stage_images(chosen, store, run_root / "images")

    resolution_table.write(
        run_root / resolution_table.CSV_NAME, resolution_table.resolution_table(staged, resolution_mm)
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(output / "selection.csv", index=False)
    return chosen


def main(argv=None):
    """Stage, run every module on one device, then score the segmentation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="AUTOMORPH_DATA for the run")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="the FIVES store directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where results are written")
    parser.add_argument("--device", default="cpu", help="torch device every module is pinned to")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="M1 batch size")
    parser.add_argument("--per-disease", type=int, default=8)
    parser.add_argument("--quality-score", type=int, default=3)
    parser.add_argument("--split", default="test", help="FIVES split to draw from ('all' for both)")
    parser.add_argument("--resolution", type=float, default=resolution_table.DEFAULT_RESOLUTION_MM)
    parser.add_argument("--attempts", type=int, default=2, help="tries per stage before moving on")
    parser.add_argument("--skip-pipeline", action="store_true", help="score an existing run only")
    args = parser.parse_args(argv)

    split = None if args.split == "all" else args.split
    output = Path(args.output)

    if args.skip_pipeline:
        chosen = pd.read_csv(output / "selection.csv")
    else:
        chosen = prepare(
            args.run_root, args.store, args.per_disease, args.quality_score, split,
            args.resolution, output,
        )
        print(f"staged {len(chosen)} images -> {args.run_root}/images")

        environment = build_environment(
            args.run_root, args.device, args.num_workers, args.batch_size
        )
        timings = []
        for index, stage in enumerate(STAGES, start=1):
            print(f"[{index}/{len(STAGES)}] {stage.name} ... ", end="", flush=True)
            result = run_stage(stage, REPO_ROOT, environment, output / "logs", args.attempts)
            timings.append(result)
            print(f"{result['status']} in {result['seconds']}s (attempts: {result['attempts']})")
            pd.DataFrame(timings).to_csv(output / "stage_timings.csv", index=False)

    counts = completion_counts(args.run_root, len(chosen))
    counts.to_csv(output / "completion.csv", index=False)

    frame = evaluation.evaluate(chosen, args.store, Path(args.run_root) / "Results")
    frame.to_csv(output / "vessel_scores.csv", index=False)
    summary = evaluation.summarise(frame)
    summary.to_csv(output / "vessel_summary.csv")

    print()
    print(counts.to_string(index=False))
    print()
    print(frame["status"].value_counts().to_string())
    print()
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
