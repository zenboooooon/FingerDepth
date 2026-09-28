"""Build audited Depth Pro pseudo-label datasets from prepared hand frames."""

from __future__ import annotations

import csv
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .artifacts import write_json
from .camera import CameraIntrinsics
from .constants import DEFAULT_TARGET_LANDMARK_INDEX, HAND_LANDMARK_NAMES
from .coordinates import lookup_depth
from .geometry import CAMERA_COORDINATE_CONVENTION, CameraPoint3D, backproject_pixel
from .trajectory import (
    TrajectoryPoint,
    write_trajectory_csv,
    write_trajectory_ply,
    write_trajectory_views_png,
)
from .video_cache import pixel_hash_sequence_sha256, pixel_sha256, sha256_file

PREPARED_INPUT_FORMAT = "fingertip-depth-pseudo-label-inputs"
PREPARED_INPUT_FORMAT_VERSION = 1
PSEUDO_LABEL_DATASET_FORMAT = "fingertip-depth-depth-pro-pseudo-label-dataset"
PSEUDO_LABEL_DATASET_FORMAT_VERSION = 1
DEPTH_PRO_TEACHER_CONDITION_ID = "depth_pro__approx_focal"
DEPTH_SAMPLING_METHOD = "single_pixel"
PSEUDO_LABEL_PROGRESS_FILENAME = ".depth-pro-progress.jsonl"
PSEUDO_LABEL_PROGRESS_TEMP_FILENAME = ".depth-pro-progress.tmp.jsonl"
PSEUDO_LABEL_PROGRESS_FORMAT = "fingertip-depth-depth-pro-pseudo-label-progress"
PSEUDO_LABEL_PROGRESS_FORMAT_VERSION = 1
_SEQUENCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset({"train", "validation", "test", "unassigned"})


class PredictionLike(Protocol):
    depth_m: np.ndarray
    inference_ms: float
    device: str
    extras: Mapping[str, Any]


class TeacherEstimator(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> PredictionLike: ...


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def _write(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
                target.write("\n")

    _atomic_generate_file(path, _write)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _checksummed_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["record_sha256"] = _record_sha256(record)
    return record


def _progress_line(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_output_directory(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise RuntimeError(
                f"pseudo-label output directory is already in use: {path}"
            ) from error
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _temporary_sibling_path(path: Path) -> Path:
    suffix = path.suffix
    basename = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f".{basename}.tmp{suffix}")


def _discard_incomplete_temp(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"incomplete temporary file must not be a symbolic link: {path}")
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError(f"incomplete temporary path is not a regular file: {path}")
    path.unlink()
    _fsync_directory(path.parent)


def _fsync_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"atomic output is missing or unsafe: {path}")
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _atomic_generate_file(
    path: Path,
    writer: Callable[[Path], None],
    *,
    durability_root: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError(f"atomic output parent must not be a symbolic link: {path.parent}")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"atomic output path is unsafe: {path}")
    temp_path = _temporary_sibling_path(path)
    _discard_incomplete_temp(temp_path)
    writer(temp_path)
    _fsync_regular_file(temp_path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"atomic output path became unsafe: {path}")
    os.replace(temp_path, path)
    _fsync_directory(path.parent)

    if durability_root is None:
        return
    if not path.is_relative_to(durability_root):
        raise ValueError(f"atomic output is outside its durability root: {path}")
    current = path.parent
    while current != durability_root:
        current = current.parent
        _fsync_directory(current)


def _atomic_write_text(
    path: Path,
    value: str,
    *,
    durability_root: Path | None = None,
) -> None:
    _atomic_generate_file(
        path,
        lambda temp_path: temp_path.write_text(value, encoding="utf-8"),
        durability_root=durability_root,
    )


def _write_all(target: Any, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = target.write(remaining)
        if written is None or written <= 0:
            raise OSError("progress journal write made no progress")
        remaining = remaining[written:]


def _create_progress_journal(path: Path, header: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"progress journal already exists: {path}")
    temp_path = path.with_name(PSEUDO_LABEL_PROGRESS_TEMP_FILENAME)
    if temp_path.is_symlink() or temp_path.exists():
        raise FileExistsError(f"progress journal temporary file already exists: {temp_path}")
    with temp_path.open("xb", buffering=0) as target:
        _write_all(target, _progress_line(header))
        os.fsync(target.fileno())
    try:
        os.link(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temp_path.is_file() and not temp_path.is_symlink():
            temp_path.unlink()
            _fsync_directory(path.parent)
        raise
    temp_path.unlink()
    _fsync_directory(path.parent)


def _append_progress_record(path: Path, record: Mapping[str, Any]) -> None:
    _append_progress_records(path, (record,))


def _append_progress_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    if not records:
        return
    with path.open("ab", buffering=0) as target:
        for record in records:
            _write_all(target, _progress_line(record))
        os.fsync(target.fileno())


def _truncate_progress_journal(path: Path, size: int) -> None:
    with path.open("r+b", buffering=0) as target:
        target.truncate(size)
        os.fsync(target.fileno())


def _decode_progress_record(line: bytes, *, line_number: int) -> dict[str, Any]:
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"progress line {line_number} is not a JSON object")
    claimed = value.get("record_sha256")
    if not isinstance(claimed, str) or not _SHA256_PATTERN.fullmatch(claimed):
        raise ValueError(f"progress line {line_number} has no valid record SHA-256")
    observed = _record_sha256(value)
    if observed != claimed:
        raise ValueError(f"progress line {line_number} record SHA-256 mismatch")
    return value


def _load_progress_journal(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"resume progress journal is missing or unsafe: {path}")
    raw = path.read_bytes()
    committed_size = raw.rfind(b"\n") + 1
    if committed_size == 0:
        raise ValueError("progress journal has no durable header")
    if committed_size != len(raw):
        _truncate_progress_journal(path, committed_size)
        raw = raw[:committed_size]

    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            record = _decode_progress_record(line[:-1], line_number=index + 1)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"progress journal is corrupt at line {index + 1}: {error}") from error
        records.append(record)

    if not records:
        raise ValueError("progress journal has no valid header")
    header = records[0]
    if header.get("record_type") != "header":
        raise ValueError("progress journal first record is not a header")
    previous_sha256 = header["record_sha256"]
    outcomes: list[dict[str, Any]] = []
    for line_number, record in enumerate(records[1:], start=2):
        if record.get("record_type") != "outcome":
            raise ValueError(f"progress line {line_number} is not an outcome")
        if record.get("previous_record_sha256") != previous_sha256:
            raise ValueError(f"progress hash chain is broken at line {line_number}")
        outcomes.append(record)
        previous_sha256 = record["record_sha256"]
    return header, outcomes


def _recover_torn_initial_progress_journal(
    path: Path,
    *,
    expected_header: Mapping[str, Any],
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    raw = path.read_bytes()
    if b"\n" in raw:
        return False
    expected = _progress_line(expected_header)
    if not expected.startswith(raw):
        raise ValueError("unterminated progress header is not a prefix of the expected header")
    unexpected = [entry for entry in path.parent.iterdir() if entry != path]
    if unexpected:
        raise ValueError(
            "cannot recover an unterminated progress header alongside other workspace files"
        )
    path.unlink()
    _fsync_directory(path.parent)
    _create_progress_journal(path, expected_header)
    return True


def _prepared_row_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(row))).hexdigest()


def _stable_teacher_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(metadata)
    identity.pop("checkpoint_path", None)
    return identity


def _relative_file(root: Path, relative_path: object) -> Path:
    path = (root / str(relative_path)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"relative path escapes its dataset root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_identifier(value: str, *, field: str) -> str:
    if not _SEQUENCE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must match {_SEQUENCE_ID_PATTERN.pattern!r}; got {value!r}")
    return value


def _camera_intrinsics(value: object) -> CameraIntrinsics:
    if not isinstance(value, Mapping):
        raise TypeError("prepared camera_intrinsics must be an object")
    return CameraIntrinsics(
        fx_px=float(value["fx_px"]),
        fy_px=float(value["fy_px"]),
        cx_px=float(value["cx_px"]),
        cy_px=float(value["cy_px"]),
    )


def _selected_hand(row: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_index = row.get("selected_hand_index")
    hands = row.get("hands")
    if not isinstance(hands, list):
        raise TypeError("prepared row hands must be a list")
    for hand in hands:
        if isinstance(hand, Mapping) and hand.get("hand_index") == selected_index:
            return hand
    raise ValueError("accepted prepared row does not contain selected_hand_index")


def _feature_landmarks(
    hand: Mapping[str, Any],
    *,
    expected_indices: Sequence[int],
) -> list[dict[str, Any]]:
    raw = hand.get("landmarks", hand.get("feature_landmarks"))
    if not isinstance(raw, list):
        raise TypeError("selected hand landmarks must be a list")
    by_index: dict[int, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("hand landmark must be an object")
        index = int(item["landmark_index"] if "landmark_index" in item else item["index"])
        if index in by_index:
            raise ValueError(f"duplicate landmark index in prepared row: {index}")
        by_index[index] = item
    if set(by_index) != set(expected_indices):
        raise ValueError(
            "prepared feature landmarks differ from manifest: "
            f"expected {list(expected_indices)}, got {sorted(by_index)}"
        )
    return [dict(by_index[index]) for index in expected_indices]


def _target_landmark(row: Mapping[str, Any], *, expected_index: int) -> dict[str, Any]:
    value = row.get("target_landmark")
    if not isinstance(value, Mapping):
        raise TypeError("accepted prepared row target_landmark must be an object")
    index = int(value["landmark_index"] if "landmark_index" in value else value["index"])
    if index != expected_index:
        raise ValueError(
            f"prepared target landmark index mismatch: expected {expected_index}, got {index}"
        )
    if not bool(value.get("in_frame", True)):
        raise ValueError("accepted prepared row target landmark is outside the image")
    if value.get("u_px") is None or value.get("v_px") is None:
        raise ValueError("accepted prepared row target landmark has no pixel coordinate")
    return dict(value)


def _transfer_frame(source: Path, target: Path, *, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        try:
            os.link(source, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source, target)
    else:
        raise ValueError("frame_transfer_mode must be 'hardlink' or 'copy'")


def _timing_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "mean": None, "p95": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _teacher_selection_evidence(
    path: Path | None,
    *,
    prepared_source: Mapping[str, Any],
    prepared_pixel_sequence_sha256: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    report = _read_json(path)
    conditions = report.get("conditions")
    if not isinstance(conditions, list):
        raise TypeError("teacher selection report has no conditions list")
    match = next(
        (
            item
            for item in conditions
            if isinstance(item, Mapping) and item.get("id") == DEPTH_PRO_TEACHER_CONDITION_ID
        ),
        None,
    )
    if not isinstance(match, Mapping):
        raise TypeError(f"teacher selection report lacks {DEPTH_PRO_TEACHER_CONDITION_ID}")
    band = match.get("band")
    if not isinstance(band, Mapping):
        raise TypeError("teacher selection report condition has no band metrics")
    violation = band.get("band_violation_m")
    if not isinstance(violation, Mapping):
        raise TypeError("teacher selection report condition has no violation metrics")
    audit = report.get("audit")
    if not isinstance(audit, Mapping):
        raise TypeError("teacher selection report has no audit object")
    audit_source = audit.get("source")
    audit_frame_cache = audit.get("frame_cache")
    audit_pixels = audit.get("input_pixels")
    if not all(
        isinstance(value, Mapping) for value in (audit_source, audit_frame_cache, audit_pixels)
    ):
        raise TypeError("teacher selection report audit provenance is incomplete")
    assert isinstance(audit_source, Mapping)
    assert isinstance(audit_frame_cache, Mapping)
    assert isinstance(audit_pixels, Mapping)
    expected_video_sha256 = str(prepared_source.get("video_sha256", ""))
    if str(audit_source.get("sha256", "")) != expected_video_sha256:
        raise ValueError("teacher selection report source video SHA-256 mismatch")
    prepared_cache_sha256 = prepared_source.get("frame_cache_manifest_sha256")
    if prepared_cache_sha256 is not None and str(
        audit_frame_cache.get("manifest_sha256", "")
    ) != str(prepared_cache_sha256):
        raise ValueError("teacher selection report frame-cache manifest SHA-256 mismatch")
    if str(audit_pixels.get("sequence_sha256", "")) != prepared_pixel_sequence_sha256:
        raise ValueError("teacher selection report input pixel sequence SHA-256 mismatch")
    expected_range = report.get("expected_range_m")
    return {
        "report_sha256": sha256_file(path),
        "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
        "metric": "approximate_range_band_violation_mean_m",
        "value": float(violation["mean"]),
        "expected_range_m": expected_range,
        "source_video_sha256": expected_video_sha256,
        "input_pixel_sequence_sha256": prepared_pixel_sequence_sha256,
        "provenance_verified": True,
        "interpretation": (
            "model-selection evidence only; this is not focal length and not "
            "per-frame ground-truth error"
        ),
    }


def _implementation_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parents[1]
    candidates = {
        "pseudo_labels.py": Path(__file__),
        "pseudo_label_inputs.py": package_dir / "pseudo_label_inputs.py",
        "alternative_depth.py": package_dir / "alternative_depth.py",
        "camera.py": package_dir / "camera.py",
        "constants.py": package_dir / "constants.py",
        "coordinates.py": package_dir / "coordinates.py",
        "geometry.py": package_dir / "geometry.py",
        "hands.py": package_dir / "hands.py",
        "trajectory.py": package_dir / "trajectory.py",
        "video_cache.py": package_dir / "video_cache.py",
        "generate_depth_pro_pseudo_labels.py": project_dir
        / "scripts"
        / "generate_depth_pro_pseudo_labels.py",
        "prepare_pseudo_label_inputs.py": project_dir
        / "scripts"
        / "prepare_pseudo_label_inputs.py",
        "root_pyproject.toml": project_dir / "pyproject.toml",
        "root_uv_lock": project_dir / "uv.lock",
        "depth_pro_pyproject.toml": project_dir / "environments" / "depth_pro" / "pyproject.toml",
        "depth_pro_uv_lock": project_dir / "environments" / "depth_pro" / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in candidates.items() if path.is_file()}


def _prepared_rejection_record(row: Mapping[str, Any], *, sequence_id: str) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "frame_index": int(row["frame_index"]),
        "timestamp_ms": int(row["timestamp_ms"]),
        "stage": "hand_landmark_preparation",
        "reasons": list(row.get("rejection_reasons", ["prepared_rejection"])),
    }


def _teacher_rejection_record(
    row: Mapping[str, Any], *, sequence_id: str, detail: str
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "frame_index": int(row["frame_index"]),
        "timestamp_ms": int(row["timestamp_ms"]),
        "stage": "teacher_label",
        "reasons": ["invalid_teacher_depth"],
        "detail": detail,
    }


def _sample_record(
    row: Mapping[str, Any],
    *,
    source_image: Path,
    sequence_id: str,
    split: str,
    feature_indices: Sequence[int],
    target_index: int,
    intrinsics: CameraIntrinsics,
    z_teacher_m: float,
    point: CameraPoint3D,
    output_focal_px: object,
) -> dict[str, Any]:
    hand = _selected_hand(row)
    features = _feature_landmarks(hand, expected_indices=feature_indices)
    target = _target_landmark(row, expected_index=target_index)
    frame_index = int(row["frame_index"])
    relative_image = Path("frames") / sequence_id / source_image.name
    sample_id = f"{sequence_id}:{frame_index:06d}:hand{int(hand['hand_index'])}"
    return {
        "schema_version": PSEUDO_LABEL_DATASET_FORMAT_VERSION,
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "split": split,
        "frame_index": frame_index,
        "timestamp_ms": int(row["timestamp_ms"]),
        "image": {
            "relative_path": relative_image.as_posix(),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "png_sha256": row["png_sha256"],
            "bgr_pixel_sha256": row["bgr_pixel_sha256"],
        },
        "camera": {
            **intrinsics.as_dict(),
            "model": "centered_pinhole_approximation",
            "intrinsics_source": ("35mm-equivalent diagonal-FOV approximation; not calibrated K"),
        },
        "hand": {
            "hand_index": int(hand["hand_index"]),
            "handedness": hand.get("handedness"),
            "handedness_score": hand.get("handedness_score"),
            "feature_landmarks": features,
        },
        "target": {
            "landmark_index": target_index,
            "landmark_name": target.get("landmark_name", target.get("name")),
            "u_px": int(target["u_px"]),
            "v_px": int(target["v_px"]),
            "z_teacher_m": z_teacher_m,
            "camera_xyz_m": point.as_dict(),
            "depth_sampling": DEPTH_SAMPLING_METHOD,
            "depth_valid": True,
        },
        "teacher": {
            "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
            "camera_mode": "approx_focal",
            "input_focal_px": intrinsics.fx_px,
            "output_focal_px": output_focal_px,
            "confidence": None,
            "confidence_available": False,
            "depth_unit": "metre",
            "fitted_scale_or_offset_applied": False,
        },
    }


def _trajectory_point_from_sample(sample: Mapping[str, Any]) -> TrajectoryPoint:
    target = sample.get("target")
    if not isinstance(target, Mapping):
        raise TypeError("progress sample target must be an object")
    xyz = target.get("camera_xyz_m")
    if not isinstance(xyz, Mapping):
        raise TypeError("progress sample camera_xyz_m must be an object")
    return TrajectoryPoint(
        frame_index=int(sample["frame_index"]),
        timestamp_ms=int(sample["timestamp_ms"]),
        u_px=float(target["u_px"]),
        v_px=float(target["v_px"]),
        x_m=float(xyz["x_m"]),
        y_m=float(xyz["y_m"]),
        z_m=float(xyz["z_m"]),
    )


def _validate_transferred_frame(
    target: Path,
    *,
    output_dir: Path,
    row: Mapping[str, Any],
) -> None:
    output_root = output_dir.resolve()
    if output_dir.is_symlink():
        raise ValueError("pseudo-label output directory must not be a symbolic link")
    for parent in (target.parent.parent, target.parent):
        if parent.is_symlink():
            raise ValueError(f"transferred frame parent must not be a symbolic link: {parent}")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"transferred frame is missing or unsafe: {target}")
    if not target.resolve().is_relative_to(output_root):
        raise ValueError(f"transferred frame escapes output directory: {target}")
    if sha256_file(target) != row["png_sha256"]:
        raise ValueError(f"transferred PNG SHA-256 mismatch at frame {row['frame_index']}")
    transferred_bgr = cv2.imread(str(target), cv2.IMREAD_COLOR)
    if transferred_bgr is None or pixel_sha256(transferred_bgr) != row["bgr_pixel_sha256"]:
        raise ValueError(f"transferred BGR pixel mismatch at frame {row['frame_index']}")


def _transfer_or_validate_frame(
    source: Path,
    target: Path,
    *,
    output_dir: Path,
    row: Mapping[str, Any],
    mode: str,
) -> None:
    temp_path = _temporary_sibling_path(target)
    if target.exists() or target.is_symlink():
        _validate_transferred_frame(target, output_dir=output_dir, row=row)
        _discard_incomplete_temp(temp_path)
    else:
        _discard_incomplete_temp(temp_path)
        _transfer_frame(source, temp_path, mode=mode)
        _validate_transferred_frame(temp_path, output_dir=output_dir, row=row)
        _fsync_regular_file(temp_path)
        if target.exists() or target.is_symlink():
            _validate_transferred_frame(target, output_dir=output_dir, row=row)
            _discard_incomplete_temp(temp_path)
        else:
            os.replace(temp_path, target)
            _fsync_directory(target.parent)
    _validate_transferred_frame(target, output_dir=output_dir, row=row)
    _fsync_regular_file(target)
    _fsync_directory(target.parent)
    _fsync_directory(target.parent.parent)
    _fsync_directory(output_dir)


def _validate_resumed_sample(
    sample: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    source_image: Path,
    output_dir: Path,
    sequence_id: str,
    split: str,
    feature_indices: Sequence[int],
    target_index: int,
) -> dict[str, Any]:
    target = sample.get("target")
    teacher = sample.get("teacher")
    if not isinstance(target, Mapping) or not isinstance(teacher, Mapping):
        raise TypeError("progress sample target and teacher must be objects")
    raw_depth = target.get("z_teacher_m")
    if isinstance(raw_depth, bool) or not isinstance(raw_depth, (int, float)):
        raise TypeError("progress sample teacher depth must be numeric")
    z_teacher_m = float(raw_depth)
    if not math.isfinite(z_teacher_m) or z_teacher_m <= 0.0:
        raise ValueError("progress sample teacher depth must be finite and positive")
    prepared_target = _target_landmark(row, expected_index=target_index)
    intrinsics = _camera_intrinsics(row["camera_intrinsics"])
    point = backproject_pixel(
        u_px=int(prepared_target["u_px"]),
        v_px=int(prepared_target["v_px"]),
        z_m=z_teacher_m,
        intrinsics=intrinsics,
    )
    output_focal_px = teacher.get("output_focal_px")
    if output_focal_px is not None and (
        isinstance(output_focal_px, bool)
        or not isinstance(output_focal_px, (int, float))
        or not math.isfinite(float(output_focal_px))
        or float(output_focal_px) <= 0.0
    ):
        raise ValueError("progress sample output focal length is invalid")
    expected = _sample_record(
        row,
        source_image=source_image,
        sequence_id=sequence_id,
        split=split,
        feature_indices=feature_indices,
        target_index=target_index,
        intrinsics=intrinsics,
        z_teacher_m=z_teacher_m,
        point=point,
        output_focal_px=output_focal_px,
    )
    if dict(sample) != expected:
        raise ValueError(f"progress sample mismatch at frame {row['frame_index']}")
    image = expected["image"]
    target_image = output_dir / str(image["relative_path"])
    _validate_transferred_frame(target_image, output_dir=output_dir, row=row)
    return dict(sample)


def _validate_progress_outcomes(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    rows: Sequence[Mapping[str, Any]],
    verified_images: Mapping[int, Path],
    output_dir: Path,
    sequence_id: str,
    split: str,
    feature_indices: Sequence[int],
    target_index: int,
    max_frames: int | None,
    initial_teacher_metadata: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[TrajectoryPoint],
    list[float],
    int,
    dict[str, Any] | None,
]:
    if len(outcomes) > len(rows):
        raise ValueError("progress journal contains more outcomes than prepared rows")
    samples: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    trajectory: list[TrajectoryPoint] = []
    inference_times: list[float] = []
    attempted_accepted_frames = 0
    loaded_teacher_metadata: dict[str, Any] | None = None

    for position, outcome in enumerate(outcomes):
        row = rows[position]
        frame_index = int(row["frame_index"])
        raw_frame_index = outcome.get("frame_index")
        if (
            isinstance(raw_frame_index, bool)
            or not isinstance(raw_frame_index, int)
            or raw_frame_index != frame_index
        ):
            raise ValueError(f"progress frame index mismatch at position {position}")
        if outcome.get("prepared_row_sha256") != _prepared_row_sha256(row):
            raise ValueError(f"progress prepared-row SHA-256 mismatch at frame {frame_index}")
        status = row.get("status")
        if outcome.get("prepared_status") != status:
            raise ValueError(f"progress prepared status mismatch at frame {frame_index}")
        teacher_attempted = status == "accepted"
        if outcome.get("teacher_attempted") is not teacher_attempted:
            raise ValueError(f"progress teacher-attempt flag mismatch at frame {frame_index}")
        if teacher_attempted:
            if max_frames is not None and attempted_accepted_frames >= max_frames:
                raise ValueError("progress journal contains outcomes beyond max_frames")
            attempted_accepted_frames += 1
            raw_inference_ms = outcome.get("inference_ms")
            if (
                isinstance(raw_inference_ms, bool)
                or not isinstance(raw_inference_ms, (int, float))
                or not math.isfinite(float(raw_inference_ms))
                or float(raw_inference_ms) < 0.0
            ):
                raise ValueError(f"progress inference time is invalid at frame {frame_index}")
            inference_times.append(float(raw_inference_ms))
            metadata = outcome.get("teacher_metadata")
            if not isinstance(metadata, Mapping):
                raise TypeError(f"progress teacher metadata is missing at frame {frame_index}")
            metadata_dict = dict(metadata)
            if any(
                metadata_dict.get(key) != value for key, value in initial_teacher_metadata.items()
            ):
                raise ValueError(f"progress teacher identity mismatch at frame {frame_index}")
            if loaded_teacher_metadata is None:
                loaded_teacher_metadata = metadata_dict
            elif metadata_dict != loaded_teacher_metadata:
                raise ValueError("progress teacher metadata changed between frames")

            kind = outcome.get("outcome")
            if kind == "sample":
                if outcome.get("rejection") is not None:
                    raise ValueError(f"progress sample also has rejection at frame {frame_index}")
                sample = outcome.get("sample")
                if not isinstance(sample, Mapping):
                    raise TypeError(f"progress sample is missing at frame {frame_index}")
                validated = _validate_resumed_sample(
                    sample,
                    row=row,
                    source_image=verified_images[frame_index],
                    output_dir=output_dir,
                    sequence_id=sequence_id,
                    split=split,
                    feature_indices=feature_indices,
                    target_index=target_index,
                )
                samples.append(validated)
                trajectory.append(_trajectory_point_from_sample(validated))
            elif kind == "rejection":
                if outcome.get("sample") is not None:
                    raise ValueError(f"progress rejection also has sample at frame {frame_index}")
                rejection = outcome.get("rejection")
                if not isinstance(rejection, Mapping):
                    raise TypeError(f"progress rejection is missing at frame {frame_index}")
                detail = rejection.get("detail")
                if not isinstance(detail, str) or not detail:
                    raise ValueError(
                        f"progress teacher rejection detail is invalid at frame {frame_index}"
                    )
                expected = _teacher_rejection_record(row, sequence_id=sequence_id, detail=detail)
                if dict(rejection) != expected:
                    raise ValueError(f"progress teacher rejection mismatch at frame {frame_index}")
                rejections.append(dict(rejection))
            else:
                raise ValueError(f"progress outcome type is invalid at frame {frame_index}")
        else:
            if (
                outcome.get("inference_ms") is not None
                or outcome.get("teacher_metadata") is not None
            ):
                raise ValueError(f"prepared rejection has teacher data at frame {frame_index}")
            if outcome.get("outcome") != "rejection" or outcome.get("sample") is not None:
                raise ValueError(f"prepared rejection outcome is invalid at frame {frame_index}")
            rejection = outcome.get("rejection")
            expected = _prepared_rejection_record(row, sequence_id=sequence_id)
            if not isinstance(rejection, Mapping) or dict(rejection) != expected:
                raise ValueError(f"progress prepared rejection mismatch at frame {frame_index}")
            rejections.append(dict(rejection))

    return (
        samples,
        rejections,
        trajectory,
        inference_times,
        attempted_accepted_frames,
        loaded_teacher_metadata,
    )


def _trajectory_summary(points: Sequence[TrajectoryPoint]) -> dict[str, Any]:
    if not points:
        return {"point_count": 0}
    xyz = np.asarray([[point.x_m, point.y_m, point.z_m] for point in points])
    return {
        "point_count": len(points),
        "x_m": {"min": float(xyz[:, 0].min()), "max": float(xyz[:, 0].max())},
        "y_m": {"min": float(xyz[:, 1].min()), "max": float(xyz[:, 1].max())},
        "z_m": {"min": float(xyz[:, 2].min()), "max": float(xyz[:, 2].max())},
    }


def generate_depth_pro_pseudo_labels(
    *,
    prepared_manifest_path: Path,
    output_dir: Path,
    estimator: TeacherEstimator,
    sequence_id: str,
    split: str = "train",
    frame_transfer_mode: str = "hardlink",
    expected_prepared_manifest_sha256: str | None = None,
    teacher_selection_report_path: Path | None = None,
    max_frames: int | None = None,
    checkpoint_interval_frames: int = 100,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate Phase 5-7 artifacts using fixed Depth Pro approximate-focal labels."""

    if output_dir.is_symlink():
        raise ValueError("pseudo-label output directory must not be a symbolic link")
    output_dir.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output_dir.parent)
    with _exclusive_output_directory(output_dir):
        return _generate_depth_pro_pseudo_labels_locked(
            prepared_manifest_path=prepared_manifest_path,
            output_dir=output_dir,
            estimator=estimator,
            sequence_id=sequence_id,
            split=split,
            frame_transfer_mode=frame_transfer_mode,
            expected_prepared_manifest_sha256=expected_prepared_manifest_sha256,
            teacher_selection_report_path=teacher_selection_report_path,
            max_frames=max_frames,
            checkpoint_interval_frames=checkpoint_interval_frames,
            resume=resume,
        )


def _generate_depth_pro_pseudo_labels_locked(
    *,
    prepared_manifest_path: Path,
    output_dir: Path,
    estimator: TeacherEstimator,
    sequence_id: str,
    split: str = "train",
    frame_transfer_mode: str = "hardlink",
    expected_prepared_manifest_sha256: str | None = None,
    teacher_selection_report_path: Path | None = None,
    max_frames: int | None = None,
    checkpoint_interval_frames: int = 100,
    resume: bool = False,
) -> dict[str, Any]:
    sequence_id = _validate_identifier(sequence_id, field="sequence_id")
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}")
    if frame_transfer_mode not in {"hardlink", "copy"}:
        raise ValueError("frame_transfer_mode must be 'hardlink' or 'copy'")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if (
        isinstance(checkpoint_interval_frames, bool)
        or not isinstance(checkpoint_interval_frames, int)
        or checkpoint_interval_frames <= 0
    ):
        raise ValueError("checkpoint_interval_frames must be a positive integer")
    progress_path = output_dir / PSEUDO_LABEL_PROGRESS_FILENAME
    progress_temp_path = output_dir / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
    resume_existing_progress = resume
    if output_dir.is_symlink():
        raise ValueError("pseudo-label output directory must not be a symbolic link")
    if resume_existing_progress:
        entries = list(output_dir.iterdir())
        if not progress_path.is_file():
            if len(entries) == 1 and entries[0] == progress_temp_path:
                _discard_incomplete_temp(progress_temp_path)
                resume_existing_progress = False
            else:
                raise FileNotFoundError(
                    f"resume requires an existing progress journal: {progress_path}"
                )
        elif progress_temp_path.exists() or progress_temp_path.is_symlink():
            _discard_incomplete_temp(progress_temp_path)
    elif output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries:
            if len(entries) == 1 and entries[0] == progress_temp_path:
                _discard_incomplete_temp(progress_temp_path)
            else:
                raise FileExistsError(f"pseudo-label output directory is not empty: {output_dir}")

    prepared_manifest_sha256 = sha256_file(prepared_manifest_path)
    if (
        expected_prepared_manifest_sha256 is not None
        and prepared_manifest_sha256 != expected_prepared_manifest_sha256
    ):
        raise ValueError(
            "prepared manifest SHA-256 mismatch: expected "
            f"{expected_prepared_manifest_sha256}, observed {prepared_manifest_sha256}"
        )
    prepared = _read_json(prepared_manifest_path)
    if prepared.get("format") != PREPARED_INPUT_FORMAT:
        raise ValueError("unsupported prepared pseudo-label input format")
    if prepared.get("format_version") != PREPARED_INPUT_FORMAT_VERSION:
        raise ValueError("unsupported prepared pseudo-label input version")
    prepared_root = prepared_manifest_path.resolve().parent
    frames_jsonl = prepared.get("frames_jsonl")
    if not isinstance(frames_jsonl, Mapping):
        raise TypeError("prepared manifest frames_jsonl must be an object")
    records_path = _relative_file(prepared_root, frames_jsonl["relative_path"])
    observed_records_sha256 = sha256_file(records_path)
    if observed_records_sha256 != frames_jsonl.get("sha256"):
        raise ValueError("prepared frames.jsonl SHA-256 mismatch")
    rows = _read_jsonl(records_path)
    if len(rows) != int(prepared.get("frame_count", -1)):
        raise ValueError("prepared frame count does not match frames.jsonl")

    feature_meta = prepared.get("feature_landmarks")
    target_meta = prepared.get("target_landmark")
    if not isinstance(feature_meta, Mapping) or not isinstance(target_meta, Mapping):
        raise TypeError("prepared landmark metadata is missing")
    feature_indices = tuple(int(value) for value in feature_meta["indices"])
    target_index = int(target_meta["index"])
    if not feature_indices or len(set(feature_indices)) != len(feature_indices):
        raise ValueError("prepared feature landmark indices must be non-empty and unique")
    if any(not 0 <= index < len(HAND_LANDMARK_NAMES) for index in feature_indices):
        raise ValueError("prepared feature landmark index is outside 0..20")
    expected_feature_names = [HAND_LANDMARK_NAMES[index] for index in feature_indices]
    if list(feature_meta.get("names", [])) != expected_feature_names:
        raise ValueError("prepared feature landmark names do not match their indices")
    if target_index != DEFAULT_TARGET_LANDMARK_INDEX:
        raise ValueError(f"prepared target landmark must be {DEFAULT_TARGET_LANDMARK_INDEX}")
    if target_meta.get("name") != HAND_LANDMARK_NAMES[target_index]:
        raise ValueError("prepared target landmark name does not match its index")
    if target_meta.get("depth_sampling") != DEPTH_SAMPLING_METHOD:
        raise ValueError("prepared target depth sampling must be single_pixel")
    frames_meta = prepared.get("frames")
    if not isinstance(frames_meta, Mapping):
        raise TypeError("prepared manifest frames must be an object")
    expected_pixel_sequence_sha256 = str(frames_meta.get("pixel_sequence_sha256", ""))
    camera_meta = prepared.get("camera")
    if not isinstance(camera_meta, Mapping):
        raise TypeError("prepared manifest camera must be an object")
    global_intrinsics = _camera_intrinsics(camera_meta.get("intrinsics"))
    image_size = camera_meta.get("image_size_px")
    if not isinstance(image_size, Mapping):
        raise TypeError("prepared camera image_size_px must be an object")
    global_width = int(image_size["width"])
    global_height = int(image_size["height"])

    statuses = [row.get("status") for row in rows]
    if any(status not in {"accepted", "rejected"} for status in statuses):
        raise ValueError("prepared row status must be accepted or rejected")
    accepted_count = statuses.count("accepted")
    rejected_count = statuses.count("rejected")
    if accepted_count != int(prepared.get("accepted_frame_count", -1)):
        raise ValueError("prepared accepted frame count does not match frames.jsonl")
    if rejected_count != int(prepared.get("rejected_frame_count", -1)):
        raise ValueError("prepared rejected frame count does not match frames.jsonl")

    verified_images: dict[int, Path] = {}
    pixel_hash_sequence: list[str] = []
    for expected_index, row in enumerate(rows):
        if int(row.get("frame_index", -1)) != expected_index:
            raise ValueError("prepared frame indices must be contiguous from zero")
        if (int(row["width"]), int(row["height"])) != (global_width, global_height):
            raise ValueError(f"prepared image dimensions disagree at frame {expected_index}")
        if _camera_intrinsics(row["camera_intrinsics"]) != global_intrinsics:
            raise ValueError(f"prepared camera intrinsics disagree at frame {expected_index}")
        reasons = row.get("rejection_reasons")
        if not isinstance(reasons, list):
            raise TypeError("prepared rejection_reasons must be a list")
        if (row["status"] == "accepted" and reasons) or (
            row["status"] == "rejected" and not reasons
        ):
            raise ValueError(f"prepared row status/reasons disagree at frame {expected_index}")
        image_path = _relative_file(prepared_root, row["image_path"])
        if sha256_file(image_path) != row.get("png_sha256"):
            raise ValueError(f"prepared PNG SHA-256 mismatch at frame {expected_index}")
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"prepared PNG is unreadable at frame {expected_index}")
        if bgr.shape[:2] != (int(row["height"]), int(row["width"])):
            raise ValueError(f"prepared PNG shape mismatch at frame {expected_index}")
        observed_pixel_hash = pixel_sha256(bgr)
        if observed_pixel_hash != row.get("bgr_pixel_sha256"):
            raise ValueError(f"prepared BGR pixel SHA-256 mismatch at frame {expected_index}")
        verified_images[expected_index] = image_path
        pixel_hash_sequence.append(observed_pixel_hash)
    observed_pixel_sequence_sha256 = pixel_hash_sequence_sha256(pixel_hash_sequence)
    if observed_pixel_sequence_sha256 != expected_pixel_sequence_sha256:
        raise ValueError("prepared BGR pixel hash sequence SHA-256 mismatch")

    implementation_hashes = _implementation_hashes()
    initial_teacher_metadata = dict(estimator.metadata)
    initial_teacher_identity = _stable_teacher_identity(initial_teacher_metadata)
    teacher_selection_report_sha256 = (
        None
        if teacher_selection_report_path is None
        else sha256_file(teacher_selection_report_path)
    )
    progress_identity = {
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "frames_jsonl_sha256": observed_records_sha256,
        "pixel_sequence_sha256": observed_pixel_sequence_sha256,
        "sequence_id": sequence_id,
        "split": split,
        "frame_transfer_mode": frame_transfer_mode,
        "max_frames": max_frames,
        "checkpoint_interval_frames": checkpoint_interval_frames,
        "teacher_selection_report_sha256": teacher_selection_report_sha256,
        "teacher_condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
        "camera_mode": "approx_focal",
        "depth_sampling": DEPTH_SAMPLING_METHOD,
        "teacher_metadata": initial_teacher_identity,
        "implementation_sha256": implementation_hashes,
    }
    expected_header = _checksummed_record(
        {
            "record_type": "header",
            "format": PSEUDO_LABEL_PROGRESS_FORMAT,
            "format_version": PSEUDO_LABEL_PROGRESS_FORMAT_VERSION,
            "identity": progress_identity,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_existing_progress:
        _recover_torn_initial_progress_journal(
            progress_path,
            expected_header=expected_header,
        )
        observed_header, progress_outcomes = _load_progress_journal(progress_path)
        if observed_header != expected_header:
            raise ValueError(
                "progress journal identity mismatch; prepared inputs, configuration, "
                "teacher, or implementation changed"
            )
    else:
        _create_progress_journal(progress_path, expected_header)
        progress_outcomes = []

    (
        samples,
        rejections,
        trajectory,
        inference_times,
        attempted_accepted_frames,
        loaded_teacher_metadata,
    ) = _validate_progress_outcomes(
        progress_outcomes,
        rows=rows,
        verified_images=verified_images,
        output_dir=output_dir,
        sequence_id=sequence_id,
        split=split,
        feature_indices=feature_indices,
        target_index=target_index,
        max_frames=max_frames,
        initial_teacher_metadata=initial_teacher_identity,
    )
    considered_source_frames = len(progress_outcomes)
    previous_record_sha256 = (
        progress_outcomes[-1]["record_sha256"]
        if progress_outcomes
        else expected_header["record_sha256"]
    )
    pending_progress_records: list[dict[str, Any]] = []

    def _checkpoint_record(record: dict[str, Any]) -> None:
        pending_progress_records.append(record)
        if len(pending_progress_records) >= checkpoint_interval_frames:
            _append_progress_records(progress_path, pending_progress_records)
            pending_progress_records.clear()

    for row in rows[considered_source_frames:]:
        frame_index = int(row["frame_index"])
        if (
            row.get("status") == "accepted"
            and max_frames is not None
            and attempted_accepted_frames >= max_frames
        ):
            break
        if row.get("status") != "accepted":
            rejection = _prepared_rejection_record(row, sequence_id=sequence_id)
            record = _checksummed_record(
                {
                    "record_type": "outcome",
                    "previous_record_sha256": previous_record_sha256,
                    "frame_index": frame_index,
                    "prepared_row_sha256": _prepared_row_sha256(row),
                    "prepared_status": "rejected",
                    "teacher_attempted": False,
                    "inference_ms": None,
                    "outcome": "rejection",
                    "sample": None,
                    "rejection": rejection,
                    "teacher_metadata": None,
                }
            )
            _checkpoint_record(record)
            previous_record_sha256 = record["record_sha256"]
            rejections.append(rejection)
            considered_source_frames += 1
            continue
        attempted_accepted_frames += 1

        source_image = verified_images[frame_index]
        bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if bgr is None:  # already verified, but retain a local invariant
            raise ValueError(f"prepared PNG became unreadable at frame {frame_index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        intrinsics = _camera_intrinsics(row["camera_intrinsics"])
        prediction = estimator.predict(
            rgb,
            intrinsics=intrinsics,
            camera_mode="approx_focal",
        )
        inference_ms = prediction.inference_ms
        if (
            isinstance(inference_ms, bool)
            or not isinstance(inference_ms, (int, float))
            or not math.isfinite(float(inference_ms))
            or float(inference_ms) < 0.0
        ):
            raise ValueError(f"teacher inference time is invalid at frame {frame_index}")
        inference_ms = float(inference_ms)
        if prediction.depth_m.shape != bgr.shape[:2]:
            raise ValueError(
                f"teacher depth shape mismatch at frame {frame_index}: "
                f"{prediction.depth_m.shape} vs {bgr.shape[:2]}"
            )
        current_teacher_metadata = dict(estimator.metadata)
        if any(
            current_teacher_metadata.get(key) != value
            for key, value in initial_teacher_identity.items()
        ):
            raise ValueError("teacher identity changed after model load")
        if loaded_teacher_metadata is None:
            loaded_teacher_metadata = current_teacher_metadata
        elif current_teacher_metadata != loaded_teacher_metadata:
            raise ValueError("teacher metadata changed between frames or resume attempts")
        target = _target_landmark(row, expected_index=target_index)
        u_px = int(target["u_px"])
        v_px = int(target["v_px"])
        try:
            z_teacher_m = lookup_depth(prediction.depth_m, u_px, v_px)
            point = backproject_pixel(
                u_px=u_px,
                v_px=v_px,
                z_m=z_teacher_m,
                intrinsics=intrinsics,
            )
        except (IndexError, ValueError) as error:
            rejection = _teacher_rejection_record(row, sequence_id=sequence_id, detail=str(error))
            record = _checksummed_record(
                {
                    "record_type": "outcome",
                    "previous_record_sha256": previous_record_sha256,
                    "frame_index": frame_index,
                    "prepared_row_sha256": _prepared_row_sha256(row),
                    "prepared_status": "accepted",
                    "teacher_attempted": True,
                    "inference_ms": inference_ms,
                    "outcome": "rejection",
                    "sample": None,
                    "rejection": rejection,
                    "teacher_metadata": current_teacher_metadata,
                }
            )
            _checkpoint_record(record)
            previous_record_sha256 = record["record_sha256"]
            inference_times.append(inference_ms)
            rejections.append(rejection)
            considered_source_frames += 1
            continue

        sample = _sample_record(
            row,
            source_image=source_image,
            sequence_id=sequence_id,
            split=split,
            feature_indices=feature_indices,
            target_index=target_index,
            intrinsics=intrinsics,
            z_teacher_m=z_teacher_m,
            point=point,
            output_focal_px=prediction.extras.get("output_focal_px"),
        )
        target_image = output_dir / str(sample["image"]["relative_path"])
        _transfer_or_validate_frame(
            source_image,
            target_image,
            output_dir=output_dir,
            row=row,
            mode=frame_transfer_mode,
        )
        record = _checksummed_record(
            {
                "record_type": "outcome",
                "previous_record_sha256": previous_record_sha256,
                "frame_index": frame_index,
                "prepared_row_sha256": _prepared_row_sha256(row),
                "prepared_status": "accepted",
                "teacher_attempted": True,
                "inference_ms": inference_ms,
                "outcome": "sample",
                "sample": sample,
                "rejection": None,
                "teacher_metadata": current_teacher_metadata,
            }
        )
        _checkpoint_record(record)
        previous_record_sha256 = record["record_sha256"]
        inference_times.append(inference_ms)
        samples.append(sample)
        trajectory.append(_trajectory_point_from_sample(sample))
        considered_source_frames += 1

    _append_progress_records(progress_path, pending_progress_records)
    pending_progress_records.clear()

    samples_path = output_dir / "samples.jsonl"
    rejections_path = output_dir / "rejections.jsonl"
    targets_csv = output_dir / "targets.csv"
    _write_jsonl(samples_path, samples)
    _write_jsonl(rejections_path, rejections)
    write_teacher_depth_summary_csv(targets_csv, samples=samples)
    split_path = output_dir / "splits" / f"{split}.txt"
    _atomic_write_text(
        split_path,
        "".join(f"{row['sample_id']}\n" for row in samples),
        durability_root=output_dir,
    )

    trajectories_dir = output_dir / "trajectories"
    trajectory_csv = trajectories_dir / f"{sequence_id}.csv"
    trajectory_ply = trajectories_dir / f"{sequence_id}.ply"
    trajectory_png = output_dir / "visualizations" / f"{sequence_id}_views.png"
    _atomic_generate_file(
        trajectory_csv,
        lambda temp_path: write_trajectory_csv(temp_path, trajectory),
        durability_root=output_dir,
    )
    _atomic_generate_file(
        trajectory_ply,
        lambda temp_path: write_trajectory_ply(temp_path, trajectory),
        durability_root=output_dir,
    )
    _atomic_generate_file(
        trajectory_png,
        lambda temp_path: write_trajectory_views_png(temp_path, trajectory),
        durability_root=output_dir,
    )

    prepared_source = prepared.get("source")
    if not isinstance(prepared_source, Mapping):
        raise TypeError("prepared manifest source must be an object")
    if _implementation_hashes() != implementation_hashes:
        raise RuntimeError("pseudo-label implementation changed during generation")
    if teacher_selection_report_path is not None and (
        sha256_file(teacher_selection_report_path) != teacher_selection_report_sha256
    ):
        raise ValueError("teacher selection report changed during generation")
    teacher_selection = _teacher_selection_evidence(
        teacher_selection_report_path,
        prepared_source=prepared_source,
        prepared_pixel_sequence_sha256=observed_pixel_sequence_sha256,
    )
    teacher_metadata = (
        initial_teacher_metadata if loaded_teacher_metadata is None else loaded_teacher_metadata
    )
    manifest: dict[str, Any] = {
        "format": PSEUDO_LABEL_DATASET_FORMAT,
        "format_version": PSEUDO_LABEL_DATASET_FORMAT_VERSION,
        "label_type": "pseudo_label",
        "ground_truth": False,
        "pseudo_label_notice": (
            "Labels reproduce the selected Depth Pro teacher and inherit its systematic "
            "errors and approximate-camera assumptions. Student-vs-teacher evaluation is "
            "distillation fidelity, not real-world depth accuracy."
        ),
        "sequence": {
            "id": sequence_id,
            "split": split,
            "split_policy": (
                "explicit sequence-level split; frames from one video must never be "
                "randomly divided across train/validation/test"
            ),
        },
        "prepared_inputs": {
            "manifest_sha256": prepared_manifest_sha256,
            "frames_jsonl_sha256": observed_records_sha256,
            "pixel_hash_sequence_sha256": observed_pixel_sequence_sha256,
            "pixel_hash_sequence_encoding": frames_meta.get("pixel_sequence_encoding"),
            "all_png_and_bgr_verified": True,
            "source": prepared.get("source"),
        },
        "feature_landmarks": feature_meta,
        "target_landmark": target_meta,
        "teacher": {
            "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
            "camera_mode": "approx_focal",
            "depth_sampling": DEPTH_SAMPLING_METHOD,
            "metric_depth_unit": "metre",
            "depth_semantics_assumption": "optical-axis Z",
            "fitted_scale_or_offset_applied": False,
            "confidence_available": False,
            "model": teacher_metadata,
            "selection_evidence": teacher_selection,
        },
        "camera": prepared.get("camera"),
        "camera_coordinate_system": {
            "convention": CAMERA_COORDINATE_CONVENTION,
            "x_formula": "(u_px - cx_px) * z_m / fx_px",
            "y_formula": "(v_px - cy_px) * z_m / fy_px",
            "z_formula": "teacher optical-axis depth in metres",
        },
        "filtering": {
            "depth_extraction": "single pixel only; Phase 3 ROI comparison skipped",
            "temporal_filtering": "none; Phase 4 temporal filtering skipped",
            "label_clipping": "none",
            "structural_checks_only": True,
        },
        "counts": {
            "source_frames_total": len(rows),
            "source_frames_considered": considered_source_frames,
            "source_frames_unprocessed": len(rows) - considered_source_frames,
            "prepared_accepted_frames_total": int(prepared.get("accepted_frame_count", -1)),
            "attempted_teacher_frames": attempted_accepted_frames,
            "accepted_samples": len(samples),
            "rejected_frames_considered": len(rejections),
        },
        "inference_ms": _timing_summary(inference_times),
        "trajectory": _trajectory_summary(trajectory),
        "artifacts": {
            "samples_jsonl": {
                "relative_path": "samples.jsonl",
                "sha256": sha256_file(samples_path),
            },
            "rejections_jsonl": {
                "relative_path": "rejections.jsonl",
                "sha256": sha256_file(rejections_path),
            },
            "targets_csv": {
                "relative_path": "targets.csv",
                "sha256": sha256_file(targets_csv),
            },
            "split": {
                "relative_path": split_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(split_path),
            },
            "trajectory_csv": {
                "relative_path": trajectory_csv.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(trajectory_csv),
            },
            "trajectory_ply": {
                "relative_path": trajectory_ply.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(trajectory_ply),
            },
            "trajectory_views_png": {
                "relative_path": trajectory_png.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(trajectory_png),
            },
        },
        "provenance": {
            "implementation_sha256": implementation_hashes,
            "frame_transfer_mode": frame_transfer_mode,
            "checkpoint_interval_frames": checkpoint_interval_frames,
        },
    }
    _atomic_generate_file(
        output_dir / "dataset_manifest.json",
        lambda temp_path: write_json(temp_path, manifest),
        durability_root=output_dir,
    )
    return manifest


def write_teacher_depth_summary_csv(
    path: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
) -> None:
    """Write a compact per-frame target table for inspection and external tooling."""

    def _write(temp_path: Path) -> None:
        with temp_path.open("w", newline="", encoding="utf-8") as target_file:
            writer = csv.DictWriter(
                target_file,
                fieldnames=[
                    "sample_id",
                    "sequence_id",
                    "split",
                    "frame_index",
                    "timestamp_ms",
                    "u_px",
                    "v_px",
                    "x_m",
                    "y_m",
                    "z_teacher_m",
                ],
            )
            writer.writeheader()
            for sample in samples:
                label = sample["target"]
                xyz = label["camera_xyz_m"]
                writer.writerow(
                    {
                        "sample_id": sample["sample_id"],
                        "sequence_id": sample["sequence_id"],
                        "split": sample["split"],
                        "frame_index": sample["frame_index"],
                        "timestamp_ms": sample["timestamp_ms"],
                        "u_px": label["u_px"],
                        "v_px": label["v_px"],
                        "x_m": xyz["x_m"],
                        "y_m": xyz["y_m"],
                        "z_teacher_m": label["z_teacher_m"],
                    }
                )

    _atomic_generate_file(path, _write)


__all__ = [
    "DEPTH_PRO_TEACHER_CONDITION_ID",
    "DEPTH_SAMPLING_METHOD",
    "PREPARED_INPUT_FORMAT",
    "PSEUDO_LABEL_DATASET_FORMAT",
    "PSEUDO_LABEL_PROGRESS_FILENAME",
    "PSEUDO_LABEL_PROGRESS_TEMP_FILENAME",
    "generate_depth_pro_pseudo_labels",
    "write_teacher_depth_summary_csv",
]
