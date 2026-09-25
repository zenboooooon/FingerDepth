"""Image decoding helpers, including HEIF/HEIC fallback support."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError


def read_bgr(path: Path) -> np.ndarray | None:
    """Read an image as uint8 BGR, falling back to Pillow for HEIF/HEIC.

    OpenCV wheels are commonly built without a HEIF decoder. Pillow-heif is
    registered lazily so ordinary OpenCV-readable images retain their fast path.
    EXIF orientation is applied before converting to the array used downstream.
    """

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr

    try:
        pillow_heif.register_heif_opener()
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return None

    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
