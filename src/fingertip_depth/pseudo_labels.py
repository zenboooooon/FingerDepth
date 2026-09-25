"""Build audited Depth Pro pseudo-label datasets from prepared hand frames."""

from __future__ import annotations

import csv
import errno
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .artifacts import write_json
from .camera import CameraIntrinsics
from .constants import DEFAULT_TARGET_LANDMARK_INDEX, HAND_LANDMARK_NAMES
from .coordinates import lookup_depth
from .geometry import CAMERA_COORDINATE_CONVENTION, backproject_pixel
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
_SEQUENCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            target.write("\n")


def _relative_file(root: Path, relative_path: object) -> Path:
    path = (root / str(relative_path)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"relative path escapes its dataset root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_identifier(value: str, *, field: str) -> str:
    if not _SEQUENCE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must match {_SEQUENCE_ID_PATTERN.pattern!r}; got {value!r}"
        )
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
            if isinstance(item, Mapping)
            and item.get("id") == DEPTH_PRO_TEACHER_CONDITION_ID
        ),
        None,
    )
    if not isinstance(match, Mapping):
        raise TypeError(
            f"teacher selection report lacks {DEPTH_PRO_TEACHER_CONDITION_ID}"
        )
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
    if not all(isinstance(value, Mapping) for value in (audit_source, audit_frame_cache, audit_pixels)):
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
        "depth_pro_pyproject.toml": project_dir
        / "environments"
        / "depth_pro"
        / "pyproject.toml",
        "depth_pro_uv_lock": project_dir / "environments" / "depth_pro" / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in candidates.items() if path.is_file()}


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
) -> dict[str, Any]:
    """Generate Phase 5-7 artifacts using fixed Depth Pro approximate-focal labels."""

    sequence_id = _validate_identifier(sequence_id, field="sequence_id")
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}")
    if frame_transfer_mode not in {"hardlink", "copy"}:
        raise ValueError("frame_transfer_mode must be 'hardlink' or 'copy'")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
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
        raise ValueError(
            f"prepared target landmark must be {DEFAULT_TARGET_LANDMARK_INDEX}"
        )
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
            raise ValueError(
                f"prepared BGR pixel SHA-256 mismatch at frame {expected_index}"
            )
        verified_images[expected_index] = image_path
        pixel_hash_sequence.append(observed_pixel_hash)
    observed_pixel_sequence_sha256 = pixel_hash_sequence_sha256(pixel_hash_sequence)
    if observed_pixel_sequence_sha256 != expected_pixel_sequence_sha256:
        raise ValueError("prepared BGR pixel hash sequence SHA-256 mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    trajectory: list[TrajectoryPoint] = []
    inference_times: list[float] = []
    attempted_accepted_frames = 0
    considered_source_frames = 0

    for row in rows:
        frame_index = int(row["frame_index"])
        if (
            row.get("status") == "accepted"
            and max_frames is not None
            and attempted_accepted_frames >= max_frames
        ):
            break
        considered_source_frames += 1
        if row.get("status") != "accepted":
            rejections.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "stage": "hand_landmark_preparation",
                    "reasons": list(row.get("rejection_reasons", ["prepared_rejection"])),
                }
            )
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
        inference_times.append(float(prediction.inference_ms))
        if prediction.depth_m.shape != bgr.shape[:2]:
            raise ValueError(
                f"teacher depth shape mismatch at frame {frame_index}: "
                f"{prediction.depth_m.shape} vs {bgr.shape[:2]}"
            )
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
            rejections.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "stage": "teacher_label",
                    "reasons": ["invalid_teacher_depth"],
                    "detail": str(error),
                }
            )
            continue

        hand = _selected_hand(row)
        features = _feature_landmarks(hand, expected_indices=feature_indices)
        sample_id = f"{sequence_id}:{frame_index:06d}:hand{int(hand['hand_index'])}"
        relative_image = Path("frames") / sequence_id / source_image.name
        target_image = output_dir / relative_image
        _transfer_frame(source_image, target_image, mode=frame_transfer_mode)
        if sha256_file(target_image) != row["png_sha256"]:
            raise ValueError(f"transferred PNG SHA-256 mismatch at frame {frame_index}")
        transferred_bgr = cv2.imread(str(target_image), cv2.IMREAD_COLOR)
        if transferred_bgr is None or pixel_sha256(transferred_bgr) != row["bgr_pixel_sha256"]:
            raise ValueError(f"transferred BGR pixel mismatch at frame {frame_index}")

        timestamp_ms = int(row["timestamp_ms"])
        sample = {
            "schema_version": PSEUDO_LABEL_DATASET_FORMAT_VERSION,
            "sample_id": sample_id,
            "sequence_id": sequence_id,
            "split": split,
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
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
                "intrinsics_source": (
                    "35mm-equivalent diagonal-FOV approximation; not calibrated K"
                ),
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
                "u_px": u_px,
                "v_px": v_px,
                "z_teacher_m": z_teacher_m,
                "camera_xyz_m": point.as_dict(),
                "depth_sampling": DEPTH_SAMPLING_METHOD,
                "depth_valid": True,
            },
            "teacher": {
                "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
                "camera_mode": "approx_focal",
                "input_focal_px": intrinsics.fx_px,
                "output_focal_px": prediction.extras.get("output_focal_px"),
                "confidence": None,
                "confidence_available": False,
                "depth_unit": "metre",
                "fitted_scale_or_offset_applied": False,
            },
        }
        samples.append(sample)
        trajectory.append(
            TrajectoryPoint.from_camera_point(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                u_px=u_px,
                v_px=v_px,
                point=point,
            )
        )

    samples_path = output_dir / "samples.jsonl"
    rejections_path = output_dir / "rejections.jsonl"
    targets_csv = output_dir / "targets.csv"
    _write_jsonl(samples_path, samples)
    _write_jsonl(rejections_path, rejections)
    write_teacher_depth_summary_csv(targets_csv, samples=samples)
    split_path = output_dir / "splits" / f"{split}.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("".join(f"{row['sample_id']}\n" for row in samples), encoding="utf-8")

    trajectories_dir = output_dir / "trajectories"
    trajectory_csv = trajectories_dir / f"{sequence_id}.csv"
    trajectory_ply = trajectories_dir / f"{sequence_id}.ply"
    trajectory_png = output_dir / "visualizations" / f"{sequence_id}_views.png"
    write_trajectory_csv(trajectory_csv, trajectory)
    write_trajectory_ply(trajectory_ply, trajectory)
    write_trajectory_views_png(trajectory_png, trajectory)

    prepared_source = prepared.get("source")
    if not isinstance(prepared_source, Mapping):
        raise TypeError("prepared manifest source must be an object")
    teacher_selection = _teacher_selection_evidence(
        teacher_selection_report_path,
        prepared_source=prepared_source,
        prepared_pixel_sequence_sha256=observed_pixel_sequence_sha256,
    )
    teacher_metadata = dict(estimator.metadata)
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
            "prepared_accepted_frames_total": int(
                prepared.get("accepted_frame_count", -1)
            ),
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
            "implementation_sha256": _implementation_hashes(),
            "frame_transfer_mode": frame_transfer_mode,
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def write_teacher_depth_summary_csv(
    path: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
) -> None:
    """Write a compact per-frame target table for inspection and external tooling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target_file:
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


__all__ = [
    "DEPTH_PRO_TEACHER_CONDITION_ID",
    "DEPTH_SAMPLING_METHOD",
    "PREPARED_INPUT_FORMAT",
    "PSEUDO_LABEL_DATASET_FORMAT",
    "generate_depth_pro_pseudo_labels",
    "write_teacher_depth_summary_csv",
]
