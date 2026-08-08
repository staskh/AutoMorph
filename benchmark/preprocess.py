# ABOUTME: Exposes AutoMorph M0's crop geometry as a transform that can be reused across arrays.
# ABOUTME: Calls M0_Preprocess/fundus_prep directly, so the numerics are AutoMorph's, not a copy.

"""M0 crop geometry, reusable.

``fundus_prep.process_without_gb`` does two jobs at once: it detects the field of view and it crops.
That is fine for the pipeline, which only ever crops one image, but a benchmark needs to crop a
*second* array — an expert annotation — into exactly the same square, and to know the geometry it
used.

This module separates the two. :func:`crop_transform` detects the field of view and returns the
geometry; :func:`apply_crop` puts any array through it. Cropping an image and its annotation is then
two calls with one transform, and they cannot drift apart.

**The numerics are AutoMorph's.** Every step delegates to ``M0_Preprocess/fundus_prep``:
``get_mask``, ``mask_image``, ``remove_back_area``, ``supplemental_black_area``. Nothing is
reimplemented, so there is no second copy of the crop rule to keep in sync — and
``benchmark/tests/test_preprocess.py`` asserts byte-equality against ``process_without_gb`` itself.

M0 crops to a square at the **original** pixel resolution; it does not resize to 912. The value 912
enters only through ``scale = 2 * radius / 912``, which M3 later uses to convert to microns.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "M0_Preprocess"))
import fundus_prep as prep

#: Reference grid the pipeline's masks live on. M0 does not resize to it; it only expresses scale.
CANONICAL_SIZE = 912


@dataclass(frozen=True)
class CropTransform:
    """The image-derived geometry of M0's crop, reusable for arrays paired with that image.

    ``centre`` keeps AutoMorph's counter-intuitive order: ``centre[0]`` indexes the height axis.
    ``centre`` and ``radius`` are in the original image's coordinate system.
    """

    fov_mask: np.ndarray
    crop_border: np.ndarray
    pad_border: tuple
    centre: tuple
    radius: int

    @property
    def scale(self):
        """M0's ``scale``: how many original pixels one pipeline-grid pixel covers."""
        return self.radius * 2 / CANONICAL_SIZE


def imread_rgb(file_path):
    """Read an image file as an RGB ``uint8`` array.

    ``fundus_prep.imread`` raises a bare string on failure, which is a ``TypeError`` in Python 3
    rather than a readable error, so the missing-file case is checked here.

    :param file_path: path to the image file.
    :return: RGB ``uint8`` array.
    :raises OSError: if the file cannot be read or decoded.
    """
    if not Path(file_path).is_file():
        raise OSError(f"cannot read image: {file_path}")
    return prep.imread(str(file_path))


def crop_transform(image):
    """Detect the field of view and return the crop geometry, without producing the square.

    :param image: RGB ``uint8`` fundus image at original resolution.
    :return: the :class:`CropTransform` for this image.
    """
    mask, bbox, centre, radius = prep.get_mask(image.copy())
    crop_border = np.array(
        (bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3], image.shape[0], image.shape[1]),
        dtype=int,
    )

    cropped_height = crop_border[1] - crop_border[0]
    cropped_width = crop_border[3] - crop_border[2]
    side = max(cropped_height, cropped_width)
    pad_border = (
        int(side / 2 - cropped_height / 2),
        int(side / 2 - cropped_height / 2) + cropped_height,
        int(side / 2 - cropped_width / 2),
        int(side / 2 - cropped_width / 2) + cropped_width,
        side,
    )

    return CropTransform(
        fov_mask=mask,
        crop_border=crop_border,
        pad_border=pad_border,
        centre=(centre[0], centre[1]),
        radius=int(radius),
    )


def apply_crop(array, transform):
    """Put ``array`` through the crop that :func:`crop_transform` describes.

    Blacks out everything outside the field of view, crops to its bounding box, then pads to the
    centred square — the three steps M0 applies to the image itself, so the result is pixel-aligned
    with what the pipeline produced. Dtype is preserved, so a boolean mask stays boolean.

    Note that M0 blacks out the *image* outside the field of view but not the *label* it carries
    alongside. This function masks whatever it is given, which is the stricter behaviour; scoring
    restricts to the field of view anyway, so the two agree everywhere it matters.

    :param array: an array whose first two axes match the image the transform came from.
    :param transform: geometry from :func:`crop_transform` for that image.
    :return: the cropped, padded square.
    """
    values = array.copy()
    values = prep.mask_image(values, transform.fov_mask)
    values, _ = prep.remove_back_area(values, border=transform.crop_border)
    values, _ = prep.supplemental_black_area(values, border=transform.pad_border)
    return values
