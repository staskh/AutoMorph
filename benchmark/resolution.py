# ABOUTME: Builds the resolution_information.csv that M0 reads to convert pixels into microns.
# ABOUTME: FIVES ships no pixel-size metadata, so the benchmark writes one nominal value per image.

"""Pixel resolution for the benchmark subset.

M0 reads ``resolution_information.csv`` (columns ``fundus``, ``res``) and multiplies ``res`` by the
crop scale to get the microns-per-pixel figure that every width, diameter and calibre metric is
expressed in.

FIVES publishes no pixel size, so there is no correct value to write. The benchmark writes
:data:`DEFAULT_RESOLUTION_MM`, the placeholder AutoMorph's own README suggests. Every
micron-denominated feature in the results is therefore *nominal*: comparable between benchmark
images, but not a physical measurement. Pass ``--resolution`` if a real figure is known for the
camera, or ``--override`` for individual images.
"""

import argparse
from pathlib import Path

import pandas as pd

#: AutoMorph's documented stand-in resolution in mm/pixel (Topcon 3D-OCT).
DEFAULT_RESOLUTION_MM = 0.008

#: Extensions M0 can read.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

CSV_NAME = "resolution_information.csv"


def discover_images(directory):
    """Return the sorted image file names in ``directory``, ignoring anything M0 cannot read.

    :raises ValueError: if the directory holds no images.
    """
    directory = Path(directory)
    names = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES and not entry.name.startswith(".")
    )
    if not names:
        raise ValueError(f"no images in {directory}")
    return names


def resolution_table(names, resolution=DEFAULT_RESOLUTION_MM, overrides=None):
    """Build the resolution frame for ``names``.

    :param names: image file names, as they appear in the images directory.
    :param resolution: mm per pixel applied to every image without an override.
    :param overrides: optional ``{name: mm_per_pixel}`` for individual images.
    :raises ValueError: if an override names an image that is not in ``names``.
    """
    ordered = sorted(names)
    overrides = overrides or {}

    unknown = set(overrides) - set(ordered)
    if unknown:
        raise ValueError(f"override for images that are not present: {', '.join(sorted(unknown))}")

    return pd.DataFrame(
        {"fundus": ordered, "res": [overrides.get(name, resolution) for name in ordered]}
    )


def write(path, frame):
    """Write the resolution frame where M0 expects to find it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf8")
    return path


def parse_override(text):
    """Parse a ``name=value`` override from the command line."""
    name, _, value = text.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"expected name=mm_per_pixel, got {text!r}")
    return name, float(value)


def main(argv=None):
    """Write resolution_information.csv for the images staged in an AutoMorph data directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, required=True, help="the AutoMorph images/ directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"where to write the csv (default: {CSV_NAME} beside the images directory)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION_MM,
        help=f"mm per pixel (default {DEFAULT_RESOLUTION_MM}, a nominal value — see module docs)",
    )
    parser.add_argument(
        "--override",
        type=parse_override,
        action="append",
        default=[],
        help="per-image resolution, e.g. --override train_1_A.png=0.006 (repeatable)",
    )
    args = parser.parse_args(argv)

    output = args.output or args.images.parent / CSV_NAME
    frame = resolution_table(discover_images(args.images), args.resolution, dict(args.override))
    write(output, frame)

    print(f"{len(frame)} images at {args.resolution} mm/pixel -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
