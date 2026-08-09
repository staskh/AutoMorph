# ABOUTME: Moves dataset ground truth into the pipeline's 912 coordinate frame for scoring.
# ABOUTME: Forward transform only (native -> 912); the crop itself comes from benchmark.preprocess.

"""Ground-truth geometry.

The pipeline crops each image to its field of view (M0, at original resolution) and every mask it
returns is ``912x912`` in that cropped-square frame. Ground truth arrives at native resolution in
the original frame, so it has to travel the same path before anything can be scored against it::

    crop (M0 field of view) -> pad to square -> resize to 912

Two rules this module enforces:

- **Never pre-resize the pipeline's input.** A pre-cropped square collapses M0's
  ``scale = 2 * radius / 912`` to about 1 and silently changes the micron scaling. The pipeline
  always receives the original image; only the ground truth is transformed. There is deliberately no
  backward (912 -> native) transform here.
- **Downsampling a binary mask is a choice, not a detail.** See :func:`resize_mask`.
"""

import cv2
import numpy as np
from PIL import Image

from benchmark.preprocess import CANONICAL_SIZE, apply_crop

_COVERAGE_THRESHOLD = 0.5

#: Downsampling rule used when building a dataset store. See :func:`resize_mask`.
DEFAULT_METHOD = "area"

METHODS = ("area", "nearest")


def resize_mask(mask, size=CANONICAL_SIZE, method=DEFAULT_METHOD):
    """Resize a binary mask to ``size x size``.

    Equal sizes are a no-op. Upsampling always replicates (``INTER_NEAREST``), which invents no
    structure. Downsampling is where the two rules differ, and they differ *on exactly the
    structures this benchmark cares about*:

    ``area``
        Averages coverage (``INTER_AREA``) and keeps pixels covered by **at least** half. The
        threshold is inclusive so a structure covering exactly half an output pixel survives; an
        exclusive rule erodes, worst at an exact 2x reduction where many cells land on precisely
        0.5. A structure thinner than half an output pixel disappears entirely.
    ``nearest``
        Point-samples. Thin structures survive, thinned and broken, rather than vanishing.

    **The rule should match the operator applied to the image**, not be chosen on its own merits.
    M2 feeds the network ``Image.resize((912, 912))``, and PIL's resize is antialiased: measured
    against a FIVES crop it sits 0.090 grey levels from ``INTER_AREA`` and 0.688 from
    ``INTER_NEAREST``. The network is therefore asked "what dominates this cell?", so ground truth
    answering "what is at this point?" compares two different sampling models and manufactures
    boundary disagreement unrelated to model quality — worth 0.029 Dice on this data.

    Neither rule is intrinsically better as an estimator. Measured on FIVES they recover the same
    total area (69,186 vs 69,213 against an expectation of 69,232) and are equally stable to
    sub-pixel translation (CV 0.054% vs 0.051%). Only the consistency argument distinguishes them.

    ``area`` is the default and what the dataset stores were built with;
    :mod:`benchmark.evaluate` currently scores with ``nearest``, which is why recovered annotation
    counts differ slightly from the manifest.

    Where thin structures genuinely are at risk — finer annotations, or a larger reduction — the
    answer is not to pick a rule but to stop downsampling ground truth: upsample the prediction and
    score at native resolution.

    :param mask: binary mask — bool, 0/255 ``uint8``, or float; anything non-zero is foreground.
    :param size: side length of the square output.
    :param method: ``"area"`` or ``"nearest"``, applied when downsampling.
    :return: boolean mask ``(size, size)``.
    :raises ValueError: if ``method`` is not one of :data:`METHODS`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    values = np.asarray(mask) > 0
    if values.shape == (size, size):
        return values

    downsampling = values.shape[0] > size or values.shape[1] > size
    if downsampling and method == "area":
        coverage = cv2.resize(values.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        return coverage >= _COVERAGE_THRESHOLD

    resized = cv2.resize(values.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def to_pipeline_image(image, transform, size=CANONICAL_SIZE):
    """Crop an RGB image to its field of view and resize it to the pipeline's square grid.

    Uses the same PIL resize the vessel module applies, so a stored square is bit-identical to what
    the network would compute from the full-resolution crop. That is what makes it safe to keep only
    the 912 renditions of a dataset for component-level work.

    It is **not** safe to feed a stored square back through the full pipeline: M0 would re-run and
    derive a meaningless ``scale`` from an already-cropped image.

    :param image: RGB ``uint8`` image at native resolution.
    :param transform: crop geometry from :func:`benchmark.preprocess.crop_transform`.
    :param size: side length of the pipeline grid.
    :return: RGB ``uint8`` array ``(size, size, 3)``.
    """
    square = apply_crop(image, transform)
    return np.array(Image.fromarray(square).resize((size, size)))


def to_pipeline_frame(mask, transform, size=CANONICAL_SIZE, method=DEFAULT_METHOD):
    """Put a native-resolution ground-truth mask into the pipeline's ``size x size`` frame.

    :param mask: ground-truth mask in the original image's coordinate system.
    :param transform: crop geometry from :func:`benchmark.preprocess.crop_transform`.
    :param size: side length of the pipeline grid.
    :param method: downsampling rule, see :func:`resize_mask`.
    :return: boolean mask ``(size, size)``, pixel-aligned with the pipeline's own masks.
    """
    return resize_mask(apply_crop(mask, transform), size, method)


def evaluation_mask(transform, size=CANONICAL_SIZE, method=DEFAULT_METHOD):
    """The region worth scoring: the detected field of view, in the pipeline frame.

    Everything outside it was blacked out before the networks ever saw it, so counting it would
    inflate true negatives and dilute every rate.

    :param transform: crop geometry from :func:`benchmark.preprocess.crop_transform`.
    :param size: side length of the pipeline grid.
    :param method: downsampling rule, see :func:`resize_mask`.
    :return: boolean mask ``(size, size)``.
    """
    return to_pipeline_frame(transform.fov_mask, transform, size, method)
