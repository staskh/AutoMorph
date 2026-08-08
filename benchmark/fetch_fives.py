# ABOUTME: Downloads FIVES from figshare and builds the processed 912 store; discards the raw data.
# ABOUTME: Run: uv run python -m benchmark.fetch_fives  (see benchmark/docs/dataset.md)

"""Fetch and prepare FIVES.

Download (1.6 GB ``.rar``) -> extract -> crop each image to its field of view and resize to 912 ->
write the store -> delete the raw data. Every step is idempotent: an existing archive is not
re-downloaded, an existing store is not rebuilt unless ``--force`` is given.

Extraction needs ``bsdtar`` (libarchive, ships with macOS and most Linux distributions) because
figshare publishes FIVES as RAR5, which the standard library cannot read.

The store keeps only 912 renditions. That is lossless for the benchmark's component entry points —
``geometry.to_pipeline_image`` reproduces exactly what ``vessel.segment_vessels`` would compute from
the full-resolution crop — but it means the originals are gone: a stored square must never be fed to
``AutoMorph.process()``, which re-runs M0 and would derive a meaningless ``scale``.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from benchmark.datasets import fives
from benchmark.geometry import evaluation_mask, to_pipeline_frame, to_pipeline_image
from benchmark.paths import data_root
from benchmark.preprocess import CANONICAL_SIZE, apply_crop, crop_transform, imread_rgb

#: sha256 of the figshare archive, recorded when the store was first built.
ARCHIVE_SHA256 = "be72f9af286b107bcebcc08a9dae7fc55c3fb0959409b689e14c72f9fdc4ad8e"

STAGING_DIR = "_staging"
_DOWNLOAD_CHUNK = 1 << 20


def _download(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination``, reporting progress on one line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URL, see fives.ARCHIVE_URL
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with partial.open("wb") as handle:
            while chunk := response.read(_DOWNLOAD_CHUNK):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        print()
    partial.rename(destination)


def _checksum(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: Path, into: Path) -> None:
    """Extract a RAR archive with bsdtar."""
    if shutil.which("bsdtar") is None:
        raise RuntimeError(
            "bsdtar not found — needed to read figshare's RAR5 archive. "
            "macOS ships it; on Debian/Ubuntu install libarchive-tools."
        )
    subprocess.run(["bsdtar", "-xf", str(archive)], cwd=into, check=True)


def process_record(record: fives.RawRecord, size: int = CANONICAL_SIZE) -> tuple[dict, dict]:
    """Crop and resize one image/annotation pair into the pipeline frame.

    :param record: a raw record from :func:`benchmark.datasets.fives.discover`.
    :param size: side length of the pipeline grid.
    :return: ``(arrays, row)`` — the three arrays to store, and the manifest row describing them.
        The row records what the downsample cost: annotation pixels and skeleton length at native
        crop resolution and at ``size``, plus any annotation the field-of-view crop discarded.
    """
    image = imread_rgb(str(record.image_path))
    annotation = fives.read_mask(record.mask_path)
    transform = crop_transform(image)

    cropped = apply_crop(annotation, transform)
    framed = to_pipeline_frame(annotation, transform, size)
    inside = evaluation_mask(transform, size)

    arrays = {
        fives.IMAGES_SUBDIR: to_pipeline_image(image, transform, size),
        fives.VESSEL_SUBDIR: framed,
        fives.FOV_SUBDIR: inside,
    }
    row = {
        "key": record.key,
        "split": record.split,
        "number": record.number,
        "disease": record.disease,
        "native_height": image.shape[0],
        "native_width": image.shape[1],
        "crop_side": cropped.shape[0],
        "radius": transform.radius,
        "scale": transform.radius * 2 / size,
        "annotation_pixels_native": int(np.count_nonzero(cropped)),
        "annotation_pixels_912": int(np.count_nonzero(framed)),
        "annotation_pixels_outside_fov": int(np.count_nonzero(annotation)) - int(np.count_nonzero(cropped)),
        "skeleton_length_native": int(np.count_nonzero(skeletonize(cropped))),
        "skeleton_length_912": int(np.count_nonzero(skeletonize(framed))),
        "components_native": int(cv2.connectedComponents(cropped.astype(np.uint8))[0]) - 1,
        "components_912": int(cv2.connectedComponents(framed.astype(np.uint8))[0]) - 1,
        "fov_pixels_912": int(np.count_nonzero(inside)),
    }
    return arrays, row


def _imwrite(path: Path, image: np.ndarray) -> None:
    """Write one PNG, refusing to fail quietly.

    ``cv2.imwrite`` returns False rather than raising — on a missing parent directory, for one — so
    an unchecked call loses an image while the run reports success.
    """
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def _store_subdirs(directory: Path) -> None:
    """Create the three output subdirectories of a store."""
    for subdir in (fives.IMAGES_SUBDIR, fives.VESSEL_SUBDIR, fives.FOV_SUBDIR):
        (directory / subdir).mkdir(parents=True, exist_ok=True)


def _write(arrays: dict, key: str, directory: Path) -> None:
    """Write the three arrays of one record into the store."""
    _imwrite(
        directory / fives.IMAGES_SUBDIR / f"{key}.png",
        cv2.cvtColor(arrays[fives.IMAGES_SUBDIR], cv2.COLOR_RGB2BGR),
    )
    for subdir in (fives.VESSEL_SUBDIR, fives.FOV_SUBDIR):
        _imwrite(directory / subdir / f"{key}.png", arrays[subdir].astype(np.uint8) * 255)


def build_store(
    raw_root: Path,
    directory: Path,
    size: int = CANONICAL_SIZE,
    limit: int | None = None,
    keep_originals: bool = False,
) -> pd.DataFrame:
    """Process every discovered pair into the 912 store and return the manifest.

    :param raw_root: the extracted ``FIVES ...`` directory.
    :param directory: the dataset directory to write into.
    :param size: side length of the pipeline grid.
    :param limit: process only the first N records (for smoke tests).
    :param keep_originals: also copy the native-resolution image and annotation into
        ``original/``, so preprocessing can be studied as an effect in its own right. Costs ~1.6 GB.
    :return: the manifest, one row per image.
    """
    records = fives.discover(raw_root)
    if limit is not None:
        records = records[:limit]
    quality = fives.read_quality(raw_root)

    _store_subdirs(directory)
    if keep_originals:
        for subdir in (fives.IMAGES_SUBDIR, fives.VESSEL_SUBDIR):
            (directory / fives.ORIGINAL_SUBDIR / subdir).mkdir(parents=True, exist_ok=True)

    rows = []
    for index, record in enumerate(records, start=1):
        arrays, row = process_record(record, size)
        _write(arrays, record.key, directory)
        if keep_originals:
            original = directory / fives.ORIGINAL_SUBDIR
            shutil.copyfile(record.image_path, original / fives.IMAGES_SUBDIR / f"{record.key}.png")
            shutil.copyfile(record.mask_path, original / fives.VESSEL_SUBDIR / f"{record.key}.png")
        rows.append({**row, **quality[record.key]})
        print(f"\r  {index} / {len(records)}  {record.key:20s}", end="", flush=True)
    print()

    frame = pd.DataFrame(rows)
    frame.to_csv(directory / fives.MANIFEST_NAME, index=False)
    return frame


#: Quality labels live in the archive's spreadsheet, which the fetch discards. A rebuild carries them
#: over from the manifest it is replacing.
QUALITY_COLUMNS = ("illumination_colour", "blur", "contrast", "quality_score")


def rebuild_from_originals(
    directory: Path, size: int = CANONICAL_SIZE, limit: int | None = None
) -> pd.DataFrame:
    """Rebuild the 912 store from the native-resolution copies already on disk.

    A store built with ``--keep-originals`` holds everything :func:`process_record` needs, so a change
    to the crop or resampling rule can be applied without fetching the 1.6 GB archive again. The
    stored originals are byte-for-byte copies of the archive's own files, so the result is identical
    to a fresh fetch — except that quality labels come from the manifest being replaced, since the
    spreadsheet they were read from is not kept.

    :param directory: the dataset directory to rebuild in place.
    :param size: side length of the pipeline grid.
    :param limit: process only the first N records (for smoke tests).
    :return: the rebuilt manifest, one row per image.
    :raises FileNotFoundError: if there is no manifest to rebuild from, or an original is missing.
    """
    manifest_path = directory / fives.MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path} — nothing to rebuild from")

    _store_subdirs(directory)

    previous = pd.read_csv(manifest_path)
    if limit is not None:
        previous = previous.head(limit)
    original = directory / fives.ORIGINAL_SUBDIR

    rows = []
    for index, entry in enumerate(previous.itertuples(index=False), start=1):
        image_path = original / fives.IMAGES_SUBDIR / f"{entry.key}.png"
        mask_path = original / fives.VESSEL_SUBDIR / f"{entry.key}.png"
        for path in (image_path, mask_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"native-resolution copy missing: {path} — a rebuild needs a store built with "
                    "`--keep-originals`; otherwise re-fetch the archive"
                )
        record = fives.RawRecord(
            key=entry.key, split=entry.split, number=entry.number, disease=entry.disease,
            image_path=image_path, mask_path=mask_path,
        )
        arrays, row = process_record(record, size)
        _write(arrays, entry.key, directory)
        rows.append({**row, **{column: getattr(entry, column) for column in QUALITY_COLUMNS}})
        print(f"\r  {index} / {len(previous)}  {entry.key:20s}", end="", flush=True)
    print()

    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    return frame


def main(argv: list[str] | None = None) -> int:
    """Download, extract, process, then delete the raw data."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", type=Path, default=None, help="override $AUTOMORPH_BENCHMARK_DATA"
    )
    parser.add_argument("--keep-raw", action="store_true", help="keep the archive and extracted originals")
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="also store native-resolution copies under original/ (~1.6 GB), for preprocessing studies",
    )
    parser.add_argument("--force", action="store_true", help="rebuild the store even if it exists")
    parser.add_argument(
        "--from-originals",
        action="store_true",
        help="reprocess the 912 store from the native copies in original/ — no download, for when "
        "the crop or resampling rule changes",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images")
    parser.add_argument("--size", type=int, default=CANONICAL_SIZE, help="pipeline grid side (default 912)")
    args = parser.parse_args(argv)

    directory = data_root(args.data_root) / fives.NAME

    if args.from_originals:
        print(f"reprocessing the {args.size} store from {fives.ORIGINAL_SUBDIR}/ (no download)")
        frame = rebuild_from_originals(directory, args.size, args.limit)
        print(f"{len(frame)} images -> {directory}")
        return 0

    if (directory / fives.MANIFEST_NAME).exists() and not args.force:
        print(f"store already built at {directory} — pass --force to rebuild")
        return 0

    staging = directory / STAGING_DIR
    archive = staging / fives.ARCHIVE_NAME

    if not archive.exists():
        print(f"downloading FIVES ({fives.ARCHIVE_LICENCE}, doi:{fives.ARCHIVE_DOI})")
        _download(fives.ARCHIVE_URL, archive)

    digest = _checksum(archive)
    if digest != ARCHIVE_SHA256:
        print(f"WARNING: archive sha256 {digest} != expected {ARCHIVE_SHA256}", file=sys.stderr)

    if not (staging / fives.ARCHIVE_ROOT).is_dir():
        print("extracting")
        _extract(archive, staging)

    print("processing into the 912 store")
    frame = build_store(
        fives.archive_root(staging), directory, args.size, args.limit, args.keep_originals
    )

    if not args.keep_raw:
        shutil.rmtree(staging)
        print(f"removed raw data ({STAGING_DIR}/)")

    print(f"{len(frame)} images -> {directory}")
    print(frame.groupby(["split", "disease"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
