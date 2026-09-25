from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image

from fingertip_depth.image_io import read_bgr


def test_read_bgr_uses_pillow_fallback(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "fallback.png"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(path)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)

    decoded = read_bgr(path)

    assert decoded is not None
    assert decoded.shape == (2, 3, 3)
    assert decoded.dtype == np.uint8
    assert decoded[0, 0].tolist() == [30, 20, 10]


def test_read_bgr_returns_none_for_unreadable_file(tmp_path: Path) -> None:
    assert read_bgr(tmp_path / "missing.heic") is None


def test_read_bgr_decodes_actual_heic(tmp_path: Path) -> None:
    pillow_heif.register_heif_opener()
    path = tmp_path / "fixture.heic"
    Image.new("RGB", (8, 6), (210, 40, 20)).save(path, format="HEIF", quality=100)

    decoded = read_bgr(path)

    assert decoded is not None
    assert decoded.shape == (6, 8, 3)
    assert float(np.mean(decoded[..., 2])) > 180
    assert float(np.mean(decoded[..., 0])) < 50


def test_read_bgr_applies_exif_orientation(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:, :20] = (255, 0, 0)
    rgb[:, 20:] = (0, 0, 255)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rgb).save(path, quality=100, subsampling=0, exif=exif)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)

    decoded = read_bgr(path)

    assert decoded is not None
    assert decoded.shape == (40, 20, 3)
    assert decoded[0, 0, 2] > 240
    assert decoded[-1, 0, 0] > 240
