import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingertip_depth.video_cache import (
    VideoFrameCache,
    pixel_sha256,
    sha256_file,
)


def _cache_fixture(tmp_path: Path) -> tuple[Path, Path, str, np.ndarray]:
    source = tmp_path / "source.mov"
    source.write_bytes(b"source video identity")
    frames_dir = tmp_path / "cache" / "frames"
    frames_dir.mkdir(parents=True)
    bgr = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    frame_path = frames_dir / "frame_000000.png"
    assert cv2.imwrite(str(frame_path), bgr)
    baseline_sha256 = "a" * 64
    manifest = {
        "format": "fingertip-depth-lossless-video-frame-cache",
        "format_version": 1,
        "source_sha256": sha256_file(source),
        "baseline_records_sha256": baseline_sha256,
        "frame_count": 1,
        "fps": 30.0,
        "source_decoder": {"opencv_version": "test"},
        "frames": [
            {
                "frame_index": 0,
                "relative_path": "frames/frame_000000.png",
                "width": 5,
                "height": 4,
                "png_sha256": sha256_file(frame_path),
                "bgr_pixel_sha256": pixel_sha256(bgr),
            }
        ],
    }
    manifest_path = tmp_path / "cache" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path, baseline_sha256, bgr


def test_frame_cache_verifies_manifest_file_and_decoded_pixels(tmp_path: Path) -> None:
    source, manifest_path, baseline_sha256, expected = _cache_fixture(tmp_path)
    manifest_sha256 = sha256_file(manifest_path)

    cache = VideoFrameCache.load(
        manifest_path,
        input_path=source,
        baseline_records_sha256=baseline_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    observed, observed_hash = cache.read(0)

    assert np.array_equal(observed, expected)
    assert observed_hash == pixel_sha256(expected)
    assert cache.manifest_sha256 == manifest_sha256


def test_frame_cache_rejects_unexpected_manifest_hash(tmp_path: Path) -> None:
    source, manifest_path, baseline_sha256, _expected = _cache_fixture(tmp_path)

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        VideoFrameCache.load(
            manifest_path,
            input_path=source,
            baseline_records_sha256=baseline_sha256,
            expected_manifest_sha256="0" * 64,
        )


def test_frame_cache_rejects_changed_png(tmp_path: Path) -> None:
    source, manifest_path, baseline_sha256, _expected = _cache_fixture(tmp_path)
    cache = VideoFrameCache.load(
        manifest_path,
        input_path=source,
        baseline_records_sha256=baseline_sha256,
        expected_manifest_sha256=sha256_file(manifest_path),
    )
    (manifest_path.parent / "frames" / "frame_000000.png").write_bytes(b"changed")

    with pytest.raises(ValueError, match="cached PNG SHA-256 mismatch"):
        cache.read(0)
