# ABOUTME: FIVES dataset adapter — raw archive layout, quality labels, and the processed 912 store.
# ABOUTME: See benchmark/docs/dataset.md for provenance, licence and the store schema.

"""FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation.

800 colour fundus images at 2048x2048 with consensus binary vessel annotations, 200 each of AMD,
diabetic retinopathy, glaucoma and normal, plus per-image quality labels.

Source
------
figshare, DOI ``10.6084/m9.figshare.19688169`` — a single ``.rar``. Licence **CC BY 4.0**.
Cite: Jin K. et al., *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel
Segmentation*, Scientific Data 9, 475 (2022). https://doi.org/10.1038/s41597-022-01564-3

Raw layout inside the archive::

    FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/
        Quality Assessment.xlsx        # sheets "Train"/"Test": Disease, Number, IC, Blur, LC
        train/Original/<n>_<D>.png     # 600 images (+ a stray Thumbs.db)
        train/Ground truth/<n>_<D>.png
        test/Original/<n>_<D>.png      # 200 images
        test/Ground truth/<n>_<D>.png

``<D>`` is the disease letter and ``<n>`` restarts per split, so **stems collide across splits** —
every key in the processed store is prefixed with its split.

Processed store
---------------
Only the 912 renditions are kept (see ``benchmark/docs/dataset.md``); the originals are
discarded after processing. ``geometry.to_pipeline_image`` makes that lossless for the component
entry points — the stored square is bit-identical to what the network computes from the
full-resolution crop.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from benchmark.paths import dataset_dir

NAME = "fives"

ARCHIVE_URL = "https://ndownloader.figshare.com/files/34969398"
ARCHIVE_NAME = "fives.rar"
ARCHIVE_DOI = "10.6084/m9.figshare.19688169"
ARCHIVE_LICENCE = "CC BY 4.0"
ARCHIVE_ROOT = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"

SPLITS = ("train", "test")
IMAGE_DIR = "Original"
MASK_DIR = "Ground truth"
QUALITY_FILE = "Quality Assessment.xlsx"
QUALITY_SHEETS = {"train": "Train", "test": "Test"}

#: Filename disease letter -> label. Counts are 200 per disease across both splits.
DISEASES = {"A": "amd", "D": "dr", "G": "glaucoma", "N": "normal"}

#: The three binary quality criteria in the spreadsheet. Their train-split sums (496/510/580)
#: reproduce Table 5 of arXiv:2406.14994, which is what fixes this naming.
QUALITY_CRITERIA = {"IC": "illumination_colour", "Blur": "blur", "LC": "contrast"}

_STEM = re.compile(r"^(?P<number>\d+)_(?P<disease>[ADGN])$")

IMAGES_SUBDIR = "images"
VESSEL_SUBDIR = "vessel"
FOV_SUBDIR = "fov"
MANIFEST_NAME = "manifest.csv"

#: Native-resolution copies kept alongside the 912 store, so preprocessing can be studied as its own
#: effect (see benchmark/docs/dataset.md). Same subdirectory names underneath.
ORIGINAL_SUBDIR = "original"


@dataclass(frozen=True)
class RawRecord:
    """One image/annotation pair in the extracted archive."""

    key: str
    split: str
    number: int
    disease: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class Record:
    """One processed image from the 912 store, with its metadata row."""

    key: str
    split: str
    disease: str
    quality_score: int
    image: np.ndarray
    vessel: np.ndarray
    fov: np.ndarray


def archive_root(staging: Path) -> Path:
    """The dataset directory inside an extracted archive.

    :param staging: directory the archive was extracted into.
    :return: the ``FIVES ...`` directory.
    :raises FileNotFoundError: if it is missing.
    """
    root = staging / ARCHIVE_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"extracted archive root not found: {root}")
    return root


def discover(raw_root: Path) -> list[RawRecord]:
    """Find every image/annotation pair in the extracted archive.

    Files whose stem is not ``<number>_<disease letter>`` are skipped — the archive ships a stray
    ``Thumbs.db``. An image without its annotation is an error, not a skip.

    :param raw_root: the ``FIVES ...`` directory from :func:`archive_root`.
    :return: records sorted by key.
    :raises FileNotFoundError: if a split directory or an annotation is missing.
    """
    records: list[RawRecord] = []
    for split in SPLITS:
        image_dir = raw_root / split / IMAGE_DIR
        mask_dir = raw_root / split / MASK_DIR
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"missing {split}/{IMAGE_DIR} or {split}/{MASK_DIR} under {raw_root}")

        for image_path in sorted(image_dir.glob("*.png")):
            matched = _STEM.match(image_path.stem)
            if matched is None:
                continue
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"annotation missing for {image_path}")
            records.append(
                RawRecord(
                    key=f"{split}_{image_path.stem}",
                    split=split,
                    number=int(matched.group("number")),
                    disease=DISEASES[matched.group("disease")],
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return sorted(records, key=lambda record: record.key)


def read_quality(raw_root: Path) -> dict[str, dict[str, int]]:
    """Read the per-image quality labels, keyed like :func:`discover`.

    Each entry holds the three binary criteria plus ``quality_score``, their sum in ``0..3``
    (0 = poor on every criterion), which is the aggregate the FIVES authors describe.

    :param raw_root: the ``FIVES ...`` directory.
    :return: ``{key: {illumination_colour, blur, contrast, quality_score}}``.
    """
    path = raw_root / QUALITY_FILE
    quality: dict[str, dict[str, int]] = {}
    for split, sheet in QUALITY_SHEETS.items():
        frame = pd.read_excel(path, sheet_name=sheet)
        for row in frame.itertuples(index=False):
            key = f"{split}_{int(row.Number)}_{row.Disease}"
            criteria = {label: int(getattr(row, column)) for column, label in QUALITY_CRITERIA.items()}
            quality[key] = {**criteria, "quality_score": sum(criteria.values())}
    return quality


def read_mask(path: Path) -> np.ndarray:
    """Read an annotation PNG as a boolean mask (they are stored 3-channel 0/255)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"cannot read annotation: {path}")
    return mask > 0


def original_paths(key: str, root: Path | None = None) -> tuple[Path, Path]:
    """Paths to one image's native-resolution copies, kept when the store was built with originals.

    :param key: store key.
    :param root: the dataset directory; resolved from the benchmark data root when ``None``.
    :return: ``(image_path, annotation_path)`` at native resolution.
    :raises FileNotFoundError: if originals were not kept — rebuild with ``--keep-originals``.
    """
    directory = (root if root is not None else dataset_dir(NAME)) / ORIGINAL_SUBDIR
    image = directory / IMAGES_SUBDIR / f"{key}.png"
    annotation = directory / VESSEL_SUBDIR / f"{key}.png"
    if not annotation.is_file():
        raise FileNotFoundError(
            f"native-resolution copy missing: {annotation} — "
            "rebuild with `python -m benchmark.fetch_fives --keep-originals --force`"
        )
    return image, annotation


def manifest(root: Path | None = None) -> pd.DataFrame:
    """Load the processed store's manifest.

    :param root: the dataset directory; resolved from the benchmark data root when ``None``.
    :return: one row per image, indexed by ``key``.
    """
    directory = root if root is not None else dataset_dir(NAME)
    return pd.read_csv(directory / MANIFEST_NAME).set_index("key")


def blank_annotations(root: Path | None = None) -> list[str]:
    """Keys whose source annotation is empty.

    FIVES ships two such images (``train_447_G``, ``train_448_G``): the annotation PNGs contain no
    vessel pixels at all. Scoring against them is meaningless — Dice would be 0 for any non-empty
    prediction — so :func:`load` skips them by default.

    :param root: the dataset directory; resolved from the benchmark data root when ``None``.
    :return: sorted keys with no annotation pixels at native resolution.
    """
    rows = manifest(root if root is not None else dataset_dir(NAME))
    return sorted(rows.index[rows["annotation_pixels_native"] == 0])


#: Keys whose annotation belongs to a different image. ``train_174_D`` carries ``test_54_D``'s mask
#: (Dice 0.946 between the two annotation files); the two are the same eye captured twice, so the
#: borrowed mask is offset from this image's vessels and scores 0.147 against a prediction that sits
#: on the visible vasculature. Documented in benchmark/docs/dataset.md.
MISMATCHED_ANNOTATIONS = ("train_174_D",)


def excluded_keys(root: Path | None = None) -> list[str]:
    """Keys no benchmark should score, because their ground truth cannot support a verdict.

    Two reasons, both properties of the source rather than of anything measured: an empty annotation
    (:func:`blank_annotations`) and an annotation that belongs to another image
    (:data:`MISMATCHED_ANNOTATIONS`). The images stay in the store and remain loadable — what is
    excluded is *scoring* them.

    :param root: the dataset directory; resolved from the benchmark data root when ``None``.
    :return: sorted keys to skip.
    """
    return sorted({*blank_annotations(root), *MISMATCHED_ANNOTATIONS})


def load(root: Path | None = None, keys: list[str] | None = None, include_blank: bool = False):
    """Iterate the processed 912 store.

    :param root: the dataset directory; resolved from the benchmark data root when ``None``.
    :param keys: optional subset of keys, in the order given; defaults to every key in the manifest.
    :param include_blank: include images whose source annotation is empty (see
        :func:`blank_annotations`); excluded by default.
    :yield: :class:`Record` per image.
    """
    directory = root if root is not None else dataset_dir(NAME)
    rows = manifest(directory)
    selected = list(rows.index) if keys is None else keys
    if not include_blank:
        blank = set(rows.index[rows["annotation_pixels_native"] == 0])
        selected = [key for key in selected if key not in blank]
    for key in selected:
        row = rows.loc[key]
        image = cv2.cvtColor(cv2.imread(str(directory / IMAGES_SUBDIR / f"{key}.png")), cv2.COLOR_BGR2RGB)
        yield Record(
            key=key,
            split=row["split"],
            disease=row["disease"],
            quality_score=int(row["quality_score"]),
            image=image,
            vessel=read_mask(directory / VESSEL_SUBDIR / f"{key}.png"),
            fov=read_mask(directory / FOV_SUBDIR / f"{key}.png"),
        )
