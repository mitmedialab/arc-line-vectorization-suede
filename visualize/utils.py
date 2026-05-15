from typing import Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import label as cc_label


def stitch(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("images must contain at least one image")

    base_mode = images[0].mode
    normalized_images = [
        img if img.mode == base_mode else img.convert(base_mode) for img in images
    ]

    widths, heights = zip(*(img.size for img in normalized_images))
    total_width = sum(widths)
    max_height = max(heights)
    stitched = Image.new(base_mode, (total_width, max_height))
    x_offset = 0
    for img in normalized_images:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return stitched


def label_tuple(
    input: NDArray,
    structure: Optional[NDArray] = None,
) -> Tuple[NDArray[np.intp], int]:
    """Typed wrapper around ``scipy.ndimage.label`` that always uses
    ``output=None`` and returns a ``(labels, num_features)`` tuple with
    precise static types.

    Enforcing ``output=None`` guarantees scipy returns the labeled array
    directly rather than writing into a caller-supplied buffer, which is
    what makes the ``(ndarray, int)`` return signature unconditional.
    """
    result = cc_label(input, structure=structure, output=None)
    labels, n = result  # type: ignore[misc]
    return np.asarray(labels, dtype=np.intp), int(n)
