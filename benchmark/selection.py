# ABOUTME: Chooses the fixed FIVES subset the benchmark runs on and stages it for the pipeline.
# ABOUTME: Eight highest-quality images per disease, always the same ones, copied at native resolution.

"""The benchmark subset.

FIVES labels every image with a disease and a quality score of 0-3. The benchmark uses only
``quality_score == 3`` so that pipeline failures point at the pipeline rather than at an
unreadable photograph, and takes the same count from each disease so that per-disease numbers
are comparable. It draws from FIVES' own held-out test split, which every disease can supply at
this quality.

Selection is by sorted key, not by sampling, so a rerun on the same store picks the same 32
images without carrying a random seed around.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

#: FIVES disease labels, as they appear in the store manifest.
DISEASES = ("amd", "dr", "glaucoma", "normal")

#: Images taken from each disease group.
PER_DISEASE = 8

#: The only quality score the benchmark accepts.
QUALITY_SCORE = 3

#: The FIVES split the benchmark draws from.
SPLIT = "test"

MANIFEST_NAME = "manifest.csv"
ORIGINAL_SUBDIR = "original"
IMAGES_SUBDIR = "images"
VESSEL_SUBDIR = "vessel"


def load_manifest(store):
    """Read the store manifest, which carries one row per FIVES image."""
    path = Path(store) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no FIVES manifest at {path} — build the store first")
    return pd.read_csv(path)


def select(manifest, per_disease=PER_DISEASE, quality_score=QUALITY_SCORE, split=SPLIT):
    """Return the benchmark subset: ``per_disease`` images of each disease, sorted by key.

    :param manifest: the store manifest, or any frame with key/split/disease/quality_score columns.
    :param per_disease: how many images to take from each disease group.
    :param quality_score: the FIVES quality label every chosen image must carry.
    :param split: the FIVES split to draw from, or None to draw from both.
    :return: a frame holding the chosen rows, ordered by disease then key.
    :raises ValueError: if a disease group cannot supply ``per_disease`` images.
    """
    eligible = manifest[manifest["quality_score"] == quality_score]
    if split is not None:
        eligible = eligible[eligible["split"] == split]

    groups = []
    for disease in DISEASES:
        candidates = eligible[eligible["disease"] == disease].sort_values("key")
        if len(candidates) < per_disease:
            raise ValueError(
                f"{disease}: only {len(candidates)} images at quality_score={quality_score}"
                f"{'' if split is None else f' in the {split} split'}, need {per_disease}"
            )
        groups.append(candidates.head(per_disease))

    return pd.concat(groups).reset_index(drop=True)


def original_image_path(store, key):
    """Path of the native-resolution photograph for ``key``."""
    return Path(store) / ORIGINAL_SUBDIR / IMAGES_SUBDIR / f"{key}.png"


def original_vessel_path(store, key):
    """Path of the native-resolution vessel annotation for ``key``."""
    return Path(store) / ORIGINAL_SUBDIR / VESSEL_SUBDIR / f"{key}.png"


def stage(chosen, store, destination):
    """Copy the chosen photographs into ``destination`` as the pipeline's input set.

    The native-resolution originals are copied, never the store's 912 renditions: AutoMorph's M0
    derives the crop and the micron scale from the photograph itself, so feeding it an
    already-cropped square would produce a meaningless scale.

    ``destination`` is emptied first, so a rerun cannot silently include images from a previous
    benchmark.

    :return: the staged file names, in selection order.
    :raises FileNotFoundError: if the store has no native-resolution copy of a chosen image.
    """
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    staged = []
    for key in chosen["key"]:
        source = original_image_path(store, key)
        if not source.is_file():
            raise FileNotFoundError(
                f"no native-resolution original at {source} — the store must be built with "
                "--keep-originals"
            )
        name = f"{key}.png"
        shutil.copyfile(source, destination / name)
        staged.append(name)
    return staged


def main(argv=None):
    """Select the subset and stage it, printing what was chosen."""
    parser = argparse.ArgumentParser(description="Stage the FIVES benchmark subset for AutoMorph.")
    parser.add_argument("--store", type=Path, required=True, help="the FIVES store directory")
    parser.add_argument("--images", type=Path, required=True, help="the AutoMorph images/ directory")
    parser.add_argument("--per-disease", type=int, default=PER_DISEASE)
    parser.add_argument("--quality-score", type=int, default=QUALITY_SCORE)
    parser.add_argument("--split", default=SPLIT, help="FIVES split to draw from ('all' for both)")
    parser.add_argument("--selection-csv", type=Path, default=None, help="write the chosen rows here")
    args = parser.parse_args(argv)

    split = None if args.split == "all" else args.split
    chosen = select(load_manifest(args.store), args.per_disease, args.quality_score, split)
    stage(chosen, args.store, args.images)

    if args.selection_csv:
        args.selection_csv.parent.mkdir(parents=True, exist_ok=True)
        chosen.to_csv(args.selection_csv, index=False)

    print(f"staged {len(chosen)} images -> {args.images}")
    print(chosen.groupby("disease")["key"].count().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
