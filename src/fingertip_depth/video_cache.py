'動画を一度だけデコードして可逆なフレーム画像として保存し、動画・画素のハッシュと件数を使ってキャッシュの完全性を検証します。'

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .artifacts import write_json

_FORMAT_VERSION = 1


# 指定したファイルの内容からSHA-256を計算します。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 画像配列の画素値を正規化してSHA-256を計算します。
def pixel_sha256(bgr: np.ndarray) -> str:
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("cached video frame must be uint8 BGR with shape (H, W, 3)")
    return hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()


# フレームごとの画素ハッシュを順番に結合したSHA-256を計算します。
def pixel_hash_sequence_sha256(digests: Iterable[str]) -> str:
    """Hash ordered pixel digests using the model-comparison audit encoding."""

    sequence = hashlib.sha256()
    for index, value in enumerate(digests):
        digest = str(value)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "pixel digest at sequence index "
                f"{index} must be 64 lowercase hexadecimal characters"
            )
        sequence.update(digest.encode("ascii"))
    return sequence.hexdigest()


# 基準手法の記録ファイルを読み込み、比較対象レコードを返します。
def _baseline_records(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            "baseline records SHA-256 mismatch while creating frame cache: "
            f"expected {expected_sha256}, observed {observed}"
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            record = json.loads(line)
            if not isinstance(record, dict) or int(record.get("frame_index", -1)) != index:
                raise ValueError("baseline records must contain contiguous frame indices")
            records.append(record)
    if not records:
        raise ValueError("baseline records are empty")
    return records


# 動画を読み込み、フレーム画像・時刻・ハッシュを持つキャッシュを作成します。
def create_video_frame_cache(
    *,
    input_path: Path,
    baseline_records_path: Path,
    baseline_records_sha256: str,
    output_dir: Path,
) -> Path:
    """Decode once and store displayed BGR frames as lossless, hashed PNG files."""

    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if manifest_path.is_file():
            cache = VideoFrameCache.load(
                manifest_path,
                input_path=input_path,
                baseline_records_sha256=baseline_records_sha256,
            )
            cache.verify_all()
            return manifest_path
        raise FileExistsError(
            f"frame-cache directory is non-empty but has no manifest: {output_dir}"
        )

    baseline = _baseline_records(baseline_records_path, baseline_records_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"input is not a readable video: {input_path}")
    orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    orientation_metadata_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    orientation_set = False
    orientation_enabled: bool | None = None
    orientation_degrees: float | None = None
    if orientation_property is not None:
        orientation_set = bool(capture.set(orientation_property, 1.0))
        orientation_enabled = capture.get(orientation_property) >= 0.5
    if orientation_metadata_property is not None:
        value = float(capture.get(orientation_metadata_property))
        if math.isfinite(value):
            orientation_degrees = value
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("video does not report a valid FPS")

    frame_entries: list[dict[str, Any]] = []
    try:
        for index, expected in enumerate(baseline):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise ValueError(f"video ended at frame {index}; expected {len(baseline)} frames")
            expected_shape = (int(expected["height"]), int(expected["width"]))
            if bgr.shape[:2] != expected_shape:
                raise ValueError(
                    f"decoded frame {index} shape {bgr.shape[:2]} != baseline {expected_shape}"
                )
            bgr_sha256 = pixel_sha256(bgr)
            expected_bgr_sha256 = expected.get("input_bgr_pixel_sha256")
            if (
                expected_bgr_sha256 is not None
                and not hmac.compare_digest(str(expected_bgr_sha256), bgr_sha256)
            ):
                raise ValueError(
                    f"decoded frame {index} BGR pixel SHA-256 differs from baseline"
                )
            relative_path = Path("frames") / f"frame_{index:06d}.png"
            frame_path = output_dir / relative_path
            if not cv2.imwrite(
                str(frame_path),
                bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                raise OSError(f"failed to write cached frame: {frame_path}")
            frame_entries.append(
                {
                    "frame_index": index,
                    "timestamp_ms": int(expected["timestamp_ms"]),
                    "relative_path": relative_path.as_posix(),
                    "width": bgr.shape[1],
                    "height": bgr.shape[0],
                    "png_sha256": sha256_file(frame_path),
                    "bgr_pixel_sha256": bgr_sha256,
                }
            )
        ok, extra = capture.read()
        if ok and extra is not None:
            raise ValueError("video contains more frames than the fixed baseline records")
    finally:
        capture.release()

    manifest: dict[str, Any] = {
        "format": "fingertip-depth-lossless-video-frame-cache",
        "format_version": _FORMAT_VERSION,
        "source": str(input_path.resolve()),
        "source_sha256": sha256_file(input_path),
        "baseline_records": str(baseline_records_path.resolve()),
        "baseline_records_sha256": baseline_records_sha256,
        "frame_count": len(frame_entries),
        "fps": fps,
        "pixel_format": "uint8 BGR",
        "storage": "lossless PNG",
        "source_decoder": {
            "opencv_version": cv2.__version__,
            "orientation_auto_set_succeeded": orientation_set,
            "orientation_auto_enabled": orientation_enabled,
            "orientation_metadata_deg": orientation_degrees,
        },
        "frames": frame_entries,
    }
    write_json(manifest_path, manifest)
    return manifest_path


# 動画のデコード済みフレームを管理し、読出しと完全性検証を行います。
@dataclass(frozen=True, slots=True)
class VideoFrameCache:
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    # キャッシュのマニフェストを読み、動画・フレーム情報を検証して開きます。
    @classmethod
    def load(
        cls,
        manifest_path: Path,
        *,
        input_path: Path,
        baseline_records_sha256: str,
        expected_manifest_sha256: str | None = None,
    ) -> VideoFrameCache:
        observed_manifest_sha256 = sha256_file(manifest_path)
        if (
            expected_manifest_sha256 is not None
            and observed_manifest_sha256 != expected_manifest_sha256
        ):
            raise ValueError(
                "frame-cache manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, observed {observed_manifest_sha256}"
            )
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict):
            raise TypeError("frame-cache manifest must be a JSON object")
        if manifest.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported frame-cache format version")
        if manifest.get("source_sha256") != sha256_file(input_path):
            raise ValueError("frame-cache source video SHA-256 mismatch")
        if manifest.get("baseline_records_sha256") != baseline_records_sha256:
            raise ValueError("frame-cache baseline records SHA-256 mismatch")
        frames = manifest.get("frames")
        if not isinstance(frames, list) or len(frames) != int(manifest.get("frame_count", -1)):
            raise ValueError("frame-cache manifest has an invalid frame list")
        for index, entry in enumerate(frames):
            if not isinstance(entry, dict) or int(entry.get("frame_index", -1)) != index:
                raise ValueError("frame-cache indices must be contiguous from zero")
        fps = float(manifest.get("fps", 0.0))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("frame-cache FPS must be finite and positive")
        return cls(
            manifest_path=manifest_path.resolve(),
            manifest=manifest,
            manifest_sha256=observed_manifest_sha256,
        )

    # 元動画のフレームレートをキャッシュ情報から返します。
    @property
    def fps(self) -> float:
        return float(self.manifest["fps"])

    # キャッシュに記録されたフレーム総数を返します。
    @property
    def frame_count(self) -> int:
        return int(self.manifest["frame_count"])

    # 元動画を展開したデコーダーの識別情報を返します。
    @property
    def source_decoder(self) -> dict[str, Any]:
        return dict(self.manifest["source_decoder"])

    # 指定フレームを読み込み、画素ハッシュと寸法を照合して返します。
    def read(self, index: int) -> tuple[np.ndarray, str]:
        if not 0 <= index < self.frame_count:
            raise IndexError(f"cached frame index out of range: {index}")
        entry = self.manifest["frames"][index]
        frame_path = (self.manifest_path.parent / entry["relative_path"]).resolve()
        if not frame_path.is_relative_to(self.manifest_path.parent):
            raise ValueError("cached frame path escapes the cache directory")
        if sha256_file(frame_path) != entry["png_sha256"]:
            raise ValueError(f"cached PNG SHA-256 mismatch at frame {index}")
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cached PNG is not readable at frame {index}")
        expected_shape = (int(entry["height"]), int(entry["width"]))
        if bgr.shape[:2] != expected_shape:
            raise ValueError(f"cached PNG shape mismatch at frame {index}")
        observed_pixels = pixel_sha256(bgr)
        if observed_pixels != entry["bgr_pixel_sha256"]:
            raise ValueError(f"cached BGR pixel SHA-256 mismatch at frame {index}")
        return bgr, observed_pixels

    # キャッシュ内の全フレームを読み直し、記録済みハッシュと一致するか検証します。
    def verify_all(self) -> None:
        for index in range(self.frame_count):
            self.read(index)


__all__ = [
    "VideoFrameCache",
    "create_video_frame_cache",
    "pixel_hash_sequence_sha256",
    "pixel_sha256",
    "sha256_file",
]
