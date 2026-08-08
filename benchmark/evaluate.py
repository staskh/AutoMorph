# ABOUTME: Scores AutoMorph's binary vessel segmentation against the FIVES expert annotations.
# ABOUTME: Aligns each annotation to the pipeline's own crop, then reports Dice/IoU/sens/spec per image.

"""Vessel segmentation accuracy on the benchmark subset.

The comparison happens on the 912x912 grid the segmentation network works in, which is what M3
consumes. Getting there takes two steps that must match the pipeline exactly:

1. **Crop.** M0 finds the retina, masks everything outside it, trims to its bounding box and pads
   to a square. :func:`align_annotation` runs the annotation through *the same* function M0 uses,
   driven by the same photograph, so the annotation lands in the pipeline's frame rather than an
   approximation of it.
2. **Resize.** M2 resizes that square to 912. Annotations and the field-of-view mask are taken down
   with nearest-neighbour sampling, which keeps them binary.

Scores are restricted to the field of view: the black padding outside the retina is not a
segmentation result and counting it as true negative would inflate specificity towards 1.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "M0_Preprocess"))
import fundus_prep as prep

from benchmark.selection import original_vessel_path

#: The grid the segmentation network and the feature modules work in.
PIPELINE_SIZE = 912

#: Binary vessel maps M2 writes, after small-object removal. Relative to Results/M2/binary_vessel/.
PREDICTION_SUBDIR = "binary_process"

SCORE_COLUMNS = ("dice", "iou", "sensitivity", "specificity", "accuracy")


def confusion(prediction, truth, inside):
    """Count true/false positives and negatives over the pixels ``inside`` marks.

    :raises ValueError: if the three masks do not share a shape.
    """
    if not (prediction.shape == truth.shape == inside.shape):
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape}, truth {truth.shape}, fov {inside.shape}"
        )
    predicted = prediction & inside
    actual = truth & inside
    return {
        "tp": int(np.count_nonzero(predicted & actual)),
        "fp": int(np.count_nonzero(predicted & ~actual & inside)),
        "fn": int(np.count_nonzero(~predicted & actual)),
        "tn": int(np.count_nonzero(~predicted & ~actual & inside)),
    }


def scores(counts):
    """Turn confusion counts into the reported scores.

    An image with neither predicted nor annotated vessels scores 1 for the overlap measures: the
    prediction is exactly right, and returning NaN there would silently drop the image from the
    averages.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

    def ratio(numerator, denominator, empty=1.0):
        return empty if denominator == 0 else numerator / denominator

    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def resize_mask(mask, size=PIPELINE_SIZE):
    """Take a binary mask down to ``size`` x ``size`` without turning it grey."""
    resized = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def align_annotation(image, annotation):
    """Put ``annotation`` into the frame M0 produces for ``image``.

    :param image: the native-resolution photograph, RGB uint8.
    :param annotation: the native-resolution vessel annotation, same height and width.
    :return: ``(annotation, field_of_view, radius)`` — both masks boolean and square, at the crop's
        own resolution, plus the retina radius M0 measured in native pixels.
    """
    if image.shape[:2] != annotation.shape[:2]:
        raise ValueError(
            f"annotation {annotation.shape[:2]} does not match image {image.shape[:2]}"
        )

    _, _, field_of_view, cropped, radius_list, _, _ = prep.process_without_gb(
        image, annotation, [], [], []
    )
    return cropped > 0, field_of_view > 0, radius_list[0]


def read_prediction(path):
    """Read one of M2's binary vessel maps as a boolean mask."""
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"cannot read prediction {path}")
    return raw > 127


def evaluate_image(key, store, prediction_path, size=PIPELINE_SIZE):
    """Score one image, returning a row of counts and scores."""
    image = prep.imread(str(store / "original" / "images" / f"{key}.png"))
    annotation = cv2.imread(str(original_vessel_path(store, key)), cv2.IMREAD_GRAYSCALE)
    if annotation is None:
        raise FileNotFoundError(f"cannot read annotation for {key}")

    aligned, field_of_view, radius = align_annotation(image, annotation)
    truth = resize_mask(aligned, size)
    inside = resize_mask(field_of_view, size)
    prediction = read_prediction(prediction_path)

    if prediction.shape != (size, size):
        prediction = resize_mask(prediction, size)

    counts = confusion(prediction, truth, inside)
    return {
        "key": key,
        "crop_radius": int(radius),
        "truth_pixels": int(np.count_nonzero(truth & inside)),
        "predicted_pixels": int(np.count_nonzero(prediction & inside)),
        "fov_pixels": int(np.count_nonzero(inside)),
        **counts,
        **scores(counts),
    }


def evaluate(selection, store, results_root, size=PIPELINE_SIZE):
    """Score every selected image that the pipeline produced a segmentation for.

    Images the pipeline dropped — rejected by the quality gate, or failed outright — are reported
    with ``status`` set and no scores, so the completion rate stays visible in the results.
    """
    predictions = Path(results_root) / "M2" / "binary_vessel" / PREDICTION_SUBDIR
    store = Path(store)

    rows = []
    for record in selection.itertuples(index=False):
        path = predictions / f"{record.key}.png"
        if not path.is_file():
            rows.append({"key": record.key, "disease": record.disease, "status": "no_segmentation"})
            continue
        try:
            row = evaluate_image(record.key, store, path, size)
        except Exception as error:  # a single unreadable image must not end the run
            rows.append({"key": record.key, "disease": record.disease, "status": f"error: {error}"})
            continue
        rows.append({**row, "disease": record.disease, "status": "ok"})

    return pd.DataFrame(rows)


def summarise(frame):
    """Average the scores per disease and overall, over the images that produced one.

    ``images_selected`` against ``images_scored`` is the drop-out: images the quality gate rejected
    or the pipeline failed on. Keeping both in the summary stops a high average hiding the fact
    that it was computed over half the images.
    """
    scored = frame[frame["status"] == "ok"]
    columns = list(SCORE_COLUMNS)
    diseases = frame["disease"].drop_duplicates().sort_values()

    per_disease = pd.DataFrame(index=pd.Index(diseases, name="disease"), columns=columns, dtype=float)
    if not scored.empty:
        per_disease.update(scored.groupby("disease")[columns].mean())
    per_disease["images_selected"] = frame.groupby("disease").size()
    per_disease["images_scored"] = scored.groupby("disease").size().reindex(diseases, fill_value=0)

    overall = pd.DataFrame(
        [{**{column: scored[column].mean() if not scored.empty else float("nan") for column in columns},
          "images_selected": len(frame), "images_scored": len(scored)}],
        index=["all"],
    )

    return pd.concat([per_disease, overall])


def main(argv=None):
    """Score a finished pipeline run against the FIVES annotations."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True, help="the FIVES store directory")
    parser.add_argument("--selection-csv", type=Path, required=True, help="the staged selection")
    parser.add_argument("--results", type=Path, required=True, help="the AutoMorph Results directory")
    parser.add_argument("--output", type=Path, required=True, help="where to write the per-image csv")
    parser.add_argument("--size", type=int, default=PIPELINE_SIZE)
    args = parser.parse_args(argv)

    selection = pd.read_csv(args.selection_csv)
    frame = evaluate(selection, args.store, args.results, args.size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    summary = summarise(frame)
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))

    print(frame["status"].value_counts().to_string())
    print()
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
