# ABOUTME: Measures the FIVES expert annotations with the same retipy code the pipeline uses on M2.
# ABOUTME: Yields a ground-truth row of the six whole-image vessel features per benchmark image.

"""Ground-truth morphometry.

The benchmark's Dice tells you how well M2 reproduces the expert annotation pixel by pixel. It does
not tell you whether the *features* AutoMorph reports survive that error — a segmentation can lose
a quarter of the annotated pixels and still give an almost-correct fractal dimension, or be
pixel-accurate and give a badly wrong average width.

To answer that, the same six whole-image features are computed from the annotation itself:

    Fractal_dimension  Vessel_density  Average_width
    Distance_tortuosity  Squared_curvature_tortuosity  Tortuosity_density

Comparability is the whole point, so the annotation is put through the identical path M2's output
takes before M3 measures it:

1. aligned to the pipeline's crop and resized to 912 (:mod:`benchmark.evaluate`),
2. restricted to the field of view,
3. ``remove_small_objects(30, connectivity=5)`` — exactly what ``filter_frag`` applies,
4. ``skeletonize``,
5. measured by ``retipy.tortuosity_measures.evaluate_window`` at window size 912.

Only step 1 differs from what the prediction gets, and it has to: the prediction is born at 912
while the annotation is native.

**Average_width is nominal.** retipy scales it by ``Scale_resolution`` from ``M0/crop_info.csv``,
which derives from the placeholder 0.008 mm/pixel the benchmark writes (FIVES publishes no pixel
size). Prediction and ground truth are scaled by the same per-image figure, so their *comparison*
is sound; the absolute microns are not.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.morphology import remove_small_objects, skeletonize

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIPY_ROOT = REPO_ROOT / "M3_feature_whole_pic" / "retipy"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RETIPY_ROOT))

from benchmark.evaluate import PIPELINE_SIZE, align_annotation, resize_mask
from benchmark.selection import original_vessel_path

#: Where the masks live, relative to the run's Results directory. The "M2" component is load-bearing:
#: retipy locates crop_info.csv as ``store_path.split('M2')[0] + 'M0/crop_info.csv'``.
MASK_PARENT = Path("M2") / "ground_truth"

#: Both names are load-bearing too. retipy locates the binary map for a skeleton as
#: ``store_path + skeleton_path.split('_skeleton')[1]``, so "_skeleton" must appear exactly once in
#: the path and the two directories must be siblings with matching file names.
SKELETON_SUBDIR = "ground_truth_binary_skeleton"
PROCESS_SUBDIR = "ground_truth_binary_process"

#: filter_frag's speck filter, mirrored so the truth is measured on the same footing.
MIN_OBJECT_SIZE = 30
OBJECT_CONNECTIVITY = 5

FEATURE_COLUMNS = (
    "Fractal_dimension",
    "Vessel_density",
    "Average_width",
    "Distance_tortuosity",
    "Squared_curvature_tortuosity",
    "Tortuosity_density",
)

OUTPUT_NAME = "Ground_truth_Macular_Features.csv"


def mask_directories(results_root):
    """The skeleton and binary-map directories, as retipy needs them arranged."""
    parent = Path(results_root) / MASK_PARENT
    return parent / SKELETON_SUBDIR, parent / PROCESS_SUBDIR


def _require_whole_picture_retipy(module):
    """Refuse to measure with the zone copy of retipy.

    The repository ships two packages both named ``retipy``, under ``M3_feature_zone`` and
    ``M3_feature_whole_pic``. ``retina.py`` is byte-identical between them, but
    ``tortuosity_measures.evaluate_window`` is not: the zone copy returns 13 values (it adds
    CRAE/CRVE, bifurcation counts and per-vessel widths) where the whole-picture copy returns the 6
    this module unpacks.

    Which one gets imported depends on ``sys.path`` order and on whether anything already imported
    ``retipy`` in this process. If the zone copy wins, every unpack raises and every feature is
    silently recorded as -1 — a table of garbage that looks like a measurement failure. Checking the
    module's own file path turns that into one clear error.

    :raises RuntimeError: if the imported retipy is not the whole-picture copy.
    """
    loaded = Path(getattr(module, "__file__", "") or "")
    if RETIPY_ROOT.resolve() not in loaded.resolve().parents:
        raise RuntimeError(
            f"retipy was imported from {loaded}, not from {RETIPY_ROOT}. The zone copy's "
            "evaluate_window returns 13 values where this module needs 6, so measuring would "
            "produce -1 for every feature. Ensure nothing imports the zone retipy first."
        )


def validate_mask_paths(skeleton_dir, process_dir):
    """Check the assumptions retipy makes about a skeleton/binary-map directory pair.

    retipy recovers the binary map from a skeleton path by string surgery, ``split('_skeleton')``,
    and finds ``crop_info.csv`` by ``split('M2')``. Both silently produce a wrong path if the
    enclosing directories happen to contain those substrings, and the resulting error is a confusing
    "cannot read image" much later. Failing here instead.

    :raises ValueError: if either path would break its split.
    """
    for path, token in ((Path(skeleton_dir), "_skeleton"), (Path(process_dir), "M2")):
        occurrences = str(path.resolve()).count(token)
        if occurrences != 1:
            raise ValueError(
                f"{path} contains {token!r} {occurrences} times; retipy splits on it and needs "
                f"exactly one. Move the run root somewhere without {token!r} in its path."
            )


def validate_paths(results_root):
    """Check retipy's path assumptions for a run's ground-truth mask directories."""
    validate_mask_paths(*mask_directories(results_root))


def build_mask_pair(image, annotation, size=PIPELINE_SIZE):
    """Turn a native annotation into the 912 binary map and skeleton M3 would measure.

    :param image: the native-resolution photograph, RGB uint8 — drives the crop.
    :param annotation: the native-resolution expert annotation.
    :return: ``(processed, skeleton)``, both boolean and ``size`` x ``size``.
    """
    aligned, field_of_view, _ = align_annotation(image, annotation)
    truth = resize_mask(aligned, size) & resize_mask(field_of_view, size)
    processed = remove_small_objects(truth, MIN_OBJECT_SIZE, connectivity=OBJECT_CONNECTIVITY)
    return processed, skeletonize(processed)


def write_mask_pair(key, processed, skeleton, skeleton_dir, process_dir):
    """Write one image's masks as the 0/255 PNGs retipy reads."""
    for directory in (skeleton_dir, process_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    imsave(str(Path(process_dir) / f"{key}.png"), 255 * processed.astype(np.uint8), check_contrast=False)
    imsave(str(Path(skeleton_dir) / f"{key}.png"), 255 * skeleton.astype(np.uint8), check_contrast=False)


def build_masks(selection, store, results_root, size=PIPELINE_SIZE):
    """Build and write the ground-truth masks for every selected image.

    :return: the keys that were written.
    """
    store = Path(store)
    skeleton_dir, process_dir = mask_directories(results_root)

    written = []
    for record in selection.itertuples(index=False):
        image_path = store / "original" / "images" / f"{record.key}.png"
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        annotation = cv2.imread(str(original_vessel_path(store, record.key)), cv2.IMREAD_GRAYSCALE)
        if image is None or annotation is None:
            print(f"  skipping {record.key}: cannot read image or annotation")
            continue

        processed, skeleton = build_mask_pair(image, annotation, size)
        write_mask_pair(record.key, processed, skeleton, skeleton_dir, process_dir)
        written.append(record.key)
        print(f"\r  masks {len(written)}/{len(selection)}  {record.key:16s}", end="", flush=True)
    print()
    return written


def measure_masks(skeleton_dir, process_dir, config_path=None, size=PIPELINE_SIZE):
    """Measure a skeleton/binary-map directory pair with retipy, mirroring M3's whole-image script.

    Works for any such pair, so the identical code measures the expert annotation and M2's own
    ``binary_skeleton``/``binary_process`` output. That is what makes predicted and ground-truth
    features comparable: they are not merely the same formulas, they are the same call.

    A mask that retipy cannot measure gets -1 in every feature, which is the convention the pipeline
    itself uses for a failed measurement.

    :param skeleton_dir: directory of skeleton PNGs; its path must contain ``_skeleton`` once.
    :param process_dir: directory of matching binary maps; its path must contain ``M2`` once.
    :return: one row per mask, with :data:`FEATURE_COLUMNS`.
    """
    from retipy import configuration, retina, tortuosity_measures

    _require_whole_picture_retipy(tortuosity_measures)
    validate_mask_paths(skeleton_dir, process_dir)
    config = configuration.Configuration(str(config_path or RETIPY_ROOT / "resources" / "retipy.config"))
    skeleton_dir, process_dir = Path(skeleton_dir), Path(process_dir)

    rows = []
    for skeleton_path in sorted(Path(skeleton_dir).glob("*.png")):
        name = skeleton_path.name
        try:
            # store_path without a trailing separator here, with one for evaluate_window: that is
            # the convention the pipeline's own scripts use, and retipy depends on the difference.
            segmented = retina.Retina(None, str(skeleton_path), store_path=str(process_dir))
            window = retina.Window(segmented, size, min_pixels=config.pixels_per_window)
            fractal, density, width, distance, squared, density_t = tortuosity_measures.evaluate_window(
                window,
                config.pixels_per_window,
                config.sampling_size,
                config.r_2_threshold,
                store_path=str(process_dir) + os.sep,
            )
            values = (fractal, density, width, distance, squared, density_t)
        except Exception as error:
            print(f"  {name}: measurement failed ({error})")
            values = (-1,) * len(FEATURE_COLUMNS)

        rows.append({"Name": name, **dict(zip(FEATURE_COLUMNS, values))})
        print(f"\r  measured {len(rows)}  {name:20s}", end="", flush=True)
    print()

    return pd.DataFrame(rows)


def measure(results_root, config_path=None, size=PIPELINE_SIZE):
    """Measure the ground-truth masks a run built, via :func:`measure_masks`."""
    return measure_masks(*mask_directories(results_root), config_path, size)


def main(argv=None):
    """Build the ground-truth masks and write their feature table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=REPO_ROOT / ".benchmark_data" / "fives")
    parser.add_argument("--selection-csv", type=Path, default=REPO_ROOT / "benchmark" / "results" / "selection.csv")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / ".benchmark_run")
    parser.add_argument("--size", type=int, default=PIPELINE_SIZE)
    parser.add_argument("--skip-masks", action="store_true", help="measure masks already on disk")
    args = parser.parse_args(argv)

    results_root = args.run_root / "Results"
    validate_paths(results_root)
    selection = pd.read_csv(args.selection_csv)

    if not args.skip_masks:
        print(f"building ground-truth masks for {len(selection)} images")
        build_masks(selection, args.store, results_root, args.size)

    print("measuring with retipy")
    frame = measure(results_root, size=args.size)

    output = results_root / "M3" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"{len(frame)} rows -> {output}")
    failed = frame[frame["Fractal_dimension"] == -1]
    if not failed.empty:
        print(f"{len(failed)} measurements failed: {', '.join(failed['Name'])}")
    print(frame[list(FEATURE_COLUMNS)].describe().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
