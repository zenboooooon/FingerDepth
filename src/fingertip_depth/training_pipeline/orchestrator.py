"""Immutable, resumable orchestration for video-to-student training.

The orchestrator deliberately owns only control flow.  Frame/landmark
preparation, student-dataset construction, and training reuse the existing
Python APIs.  Depth Pro remains the sole subprocess boundary because its
NumPy/OpenCV requirements conflict with the root project.

The minimal duck-typed contract shared with ``config`` and ``discovery`` is:

* ``PipelineConfig`` exposes ``project_root``, ``output``, ``prepare``,
  ``teacher``, ``dataset``, and ``training``.
* ``output`` exposes ``processed_dir``, ``dataset_dir``, and ``runs_dir``.
* ``prepare`` exposes ``hand_model_path``, ``landmark_indices``,
  ``frame_transfer_mode``, and ``max_frames``.
* ``teacher`` exposes ``depth_pro_project``, ``device``,
  ``frame_transfer_mode``, ``max_frames``, and ``teacher_selection_report``.
* ``dataset`` exposes ``frame_transfer_mode`` and
  ``validation_tail_fraction``.
* ``training`` exposes ``device`` plus ``model_config()``,
  ``training_config()``, and ``spike_filter_config()``.
* ``DiscoveredVideo`` exposes ``path``, ``relative_path``, ``split``,
  ``sequence_id``, ``source_sha256``, ``size_bytes``, and
  ``focal_35mm_mm``.

``discover_videos(config)`` is imported lazily so tests and other callers can
inject already-discovered videos without coupling to filesystem discovery.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fingertip_depth.pseudo_label_inputs import (
    PREPARED_INPUT_FORMAT,
    PREPARED_INPUT_FORMAT_VERSION,
    prepare_pseudo_label_inputs,
)
from fingertip_depth.pseudo_labels import (
    PSEUDO_LABEL_DATASET_FORMAT,
    PSEUDO_LABEL_DATASET_FORMAT_VERSION,
    PSEUDO_LABEL_PROGRESS_FILENAME,
    PSEUDO_LABEL_PROGRESS_TEMP_FILENAME,
)
from fingertip_depth.student_dataset import (
    STUDENT_DATASET_FORMAT,
    STUDENT_DATASET_FORMAT_VERSION,
    build_student_dataset,
)
from fingertip_depth.student_training import (
    TRAINING_RUN_FORMAT,
    TRAINING_RUN_FORMAT_VERSION,
    train_student_transformer,
)
from fingertip_depth.video_cache import sha256_file

COMPLETE_FILENAME = "complete.json"
COMPLETE_FORMAT = "fingertip-depth-training-pipeline-complete"
COMPLETE_FORMAT_VERSION = 2
VIDEO_WORKSPACE_FILENAME = ".pipeline-workspace.json"
VIDEO_WORKSPACE_FORMAT = "fingertip-depth-training-pipeline-video-workspace"
VIDEO_WORKSPACE_FORMAT_VERSION = 1
VIDEO_WORKSPACE_SUFFIX = ".work"
_SEQUENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class OutputConfigProtocol(Protocol):
    processed_dir: Path
    dataset_dir: Path
    runs_dir: Path


@runtime_checkable
class PrepareConfigProtocol(Protocol):
    hand_model_path: Path
    landmark_indices: Sequence[int]
    frame_transfer_mode: str
    max_frames: int | None


@runtime_checkable
class TeacherConfigProtocol(Protocol):
    depth_pro_project: Path
    device: str
    frame_transfer_mode: str
    max_frames: int | None
    checkpoint_interval_frames: int
    teacher_selection_report: Path | None


@runtime_checkable
class DatasetConfigProtocol(Protocol):
    frame_transfer_mode: str
    validation_tail_fraction: float | None


@runtime_checkable
class TrainingConfigProtocol(Protocol):
    device: str
    model: object
    optimizer: object
    spike_filter: object

    def model_config(self) -> object: ...

    def training_config(self) -> object: ...

    def spike_filter_config(self) -> object: ...


@runtime_checkable
class PipelineConfigProtocol(Protocol):
    project_root: Path
    output: OutputConfigProtocol
    prepare: PrepareConfigProtocol
    teacher: TeacherConfigProtocol
    dataset: DatasetConfigProtocol
    training: TrainingConfigProtocol


@runtime_checkable
class DiscoveredVideoProtocol(Protocol):
    path: Path
    relative_path: Path
    split: str
    sequence_id: str
    source_sha256: str
    size_bytes: int
    focal_35mm_mm: float


class PipelineError(RuntimeError):
    """Raised when an immutable input or generated artifact cannot be trusted."""


CommandRunner = Callable[..., object]


def _default_discover(config: PipelineConfigProtocol) -> Sequence[DiscoveredVideoProtocol]:
    from fingertip_depth.training_pipeline.discovery import discover_videos

    return discover_videos(config)  # type: ignore[arg-type, no-any-return]


def _default_command_runner(
    command: Sequence[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class PipelineDependencies:
    """Injectable boundaries used by unit tests and alternate front-ends."""

    discover: Callable[[PipelineConfigProtocol], Sequence[DiscoveredVideoProtocol]] = (
        _default_discover
    )
    prepare: Callable[..., object] = prepare_pseudo_label_inputs
    builder: Callable[..., object] = build_student_dataset
    trainer: Callable[..., object] = train_student_transformer
    command_runner: CommandRunner = _default_command_runner
    hash_file: Callable[[Path], str] = sha256_file
    nonce_factory: Callable[[], str] = lambda: uuid.uuid4().hex
    migration_checkpoint: Callable[[str], None] = lambda _phase: None


@dataclass(frozen=True)
class VideoCacheStatus:
    sequence_id: str
    split: str
    source_path: Path
    source_sha256: str
    cache_key: str
    cache_dir: Path
    cache_hit: bool
    prepared_manifest: Path | None = None
    prepared_manifest_sha256: str | None = None
    teacher_manifest: Path | None = None
    teacher_manifest_sha256: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "split": self.split,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "cache_key": self.cache_key,
            "cache_dir": str(self.cache_dir),
            "cache_hit": self.cache_hit,
            "prepared_manifest": (
                None if self.prepared_manifest is None else str(self.prepared_manifest)
            ),
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "teacher_manifest": (
                None if self.teacher_manifest is None else str(self.teacher_manifest)
            ),
            "teacher_manifest_sha256": self.teacher_manifest_sha256,
        }


@dataclass(frozen=True)
class PipelineSummary:
    operation: str
    status: str
    dry_run: bool
    force_train: bool
    videos: tuple[VideoCacheStatus, ...]
    dataset_key: str
    dataset_cache_hit: bool
    dataset_dir: Path
    dataset_manifest: Path | None
    dataset_manifest_sha256: str | None
    run_key: str
    run_cache_hit: bool
    run_dir: Path
    run_manifest: Path | None
    run_manifest_sha256: str | None
    actions: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def videos_total(self) -> int:
        return len(self.videos)

    @property
    def video_cache_hits(self) -> int:
        return sum(item.cache_hit for item in self.videos)

    @property
    def video_cache_misses(self) -> int:
        return self.videos_total - self.video_cache_hits

    def as_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "status": self.status,
            "dry_run": self.dry_run,
            "force_train": self.force_train,
            "videos_total": self.videos_total,
            "video_cache_hits": self.video_cache_hits,
            "video_cache_misses": self.video_cache_misses,
            "videos": [item.as_dict() for item in self.videos],
            "dataset": {
                "cache_key": self.dataset_key,
                "cache_hit": self.dataset_cache_hit,
                "directory": str(self.dataset_dir),
                "manifest": (None if self.dataset_manifest is None else str(self.dataset_manifest)),
                "manifest_sha256": self.dataset_manifest_sha256,
            },
            "run": {
                "cache_key": self.run_key,
                "cache_hit": self.run_cache_hit,
                "directory": str(self.run_dir),
                "manifest": None if self.run_manifest is None else str(self.run_manifest),
                "manifest_sha256": self.run_manifest_sha256,
            },
            "actions": list(self.actions),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _Artifact:
    path: Path
    sha256: str


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _content_key(payload: object) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"JSON artifact must contain an object: {path}")
    return value


def _require_manifest_format(
    manifest: Mapping[str, object],
    *,
    expected_format: str,
    expected_version: int,
    artifact: Path,
) -> None:
    observed_version = manifest.get("format_version")
    if (
        manifest.get("format") != expected_format
        or not isinstance(observed_version, int)
        or isinstance(observed_version, bool)
        or observed_version != expected_version
    ):
        raise PipelineError(f"unexpected manifest format or version: {artifact}")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value)
    if not _SHA256.fullmatch(digest):
        raise PipelineError(f"{field} is not a lowercase SHA-256 digest")
    return digest


def _stable_file_hash(
    path: Path,
    hash_file: Callable[[Path], str],
    *,
    field: str,
) -> str:
    before = path.stat()
    digest = _require_sha256(
        hash_file(path),
        field=field,
    )
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise PipelineError(f"{field} changed while it was being hashed: {path}")
    return digest


def _relative_artifact(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise PipelineError(f"complete marker artifact path must be relative: {value}")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root):
        raise PipelineError(f"complete marker artifact escapes its cache directory: {value}")
    return path


def _validate_declared_artifact(
    root: Path,
    descriptor: object,
    *,
    field: str,
    hash_file: Callable[[Path], str],
) -> _Artifact:
    """Validate one manifest-declared, cache-local regular file."""

    if not isinstance(descriptor, Mapping):
        raise PipelineError(f"{field} descriptor must be an object")
    raw_relative_path = descriptor.get("relative_path")
    if not isinstance(raw_relative_path, str) or not raw_relative_path:
        raise PipelineError(f"{field}.relative_path must be a non-empty string")
    relative_path = Path(raw_relative_path)
    if relative_path.is_absolute():
        raise PipelineError(f"{field}.relative_path must be relative: {raw_relative_path}")
    resolved_root = root.resolve()
    path = _relative_artifact(resolved_root, raw_relative_path)
    canonical = path.relative_to(resolved_root).as_posix()
    if canonical != raw_relative_path:
        raise PipelineError(f"{field}.relative_path is not canonical: {raw_relative_path}")
    unresolved_path = resolved_root / relative_path
    if unresolved_path.is_symlink():
        raise PipelineError(f"{field} must not be a symbolic link: {unresolved_path}")
    if not path.is_file():
        raise PipelineError(f"{field} is missing or is not a regular file: {path}")
    expected = _require_sha256(descriptor.get("sha256"), field=f"{field} SHA-256")
    observed = _stable_file_hash(path, hash_file, field=f"{field} SHA-256")
    if observed != expected:
        raise PipelineError(
            f"{field} SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return _Artifact(path=path, sha256=observed)


def _validate_manifest_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    required_names: Sequence[str],
    hash_file: Callable[[Path], str],
) -> dict[str, _Artifact]:
    """Validate physical-file descriptors in an artifact map."""

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise PipelineError(f"manifest has no artifacts object: {manifest_path}")
    missing = [name for name in required_names if name not in raw_artifacts]
    if missing:
        raise PipelineError(f"manifest lacks required artifacts {missing}: {manifest_path}")

    validated: dict[str, _Artifact] = {}
    for raw_name, descriptor in raw_artifacts.items():
        name = str(raw_name)
        if not isinstance(descriptor, Mapping):
            if name in required_names:
                raise PipelineError(
                    f"required artifact {name!r} descriptor is invalid: {manifest_path}"
                )
            # Student-dataset manifests store aggregate digests and their
            # encoding beside physical-file descriptors in this object.
            continue
        validated[name] = _validate_declared_artifact(
            manifest_path.parent,
            descriptor,
            field=f"manifest artifact {name!r}",
            hash_file=hash_file,
        )
    return validated


def _iter_jsonl_objects(path: Path, *, field: str) -> Iterator[Mapping[str, object]]:
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PipelineError(
                        f"cannot parse {field} line {line_number}: {path}"
                    ) from error
                if not isinstance(row, Mapping):
                    raise PipelineError(f"{field} line {line_number} must be an object")
                yield row
    except (OSError, UnicodeDecodeError) as error:
        raise PipelineError(f"cannot read {field} {path}: {error}") from error


def _validate_sample_images(
    manifest_path: Path,
    samples: _Artifact,
    *,
    field: str,
    hash_file: Callable[[Path], str],
) -> None:
    for line_number, row in enumerate(
        _iter_jsonl_objects(samples.path, field=field),
        start=1,
    ):
        image = row.get("image")
        if not isinstance(image, Mapping):
            raise PipelineError(f"{field} line {line_number} has no image object")
        _validate_declared_artifact(
            manifest_path.parent,
            {
                "relative_path": image.get("relative_path"),
                "sha256": image.get("png_sha256"),
            },
            field=f"{field} line {line_number} image",
            hash_file=hash_file,
        )


def _validate_prepared_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    hash_file: Callable[[Path], str],
) -> None:
    """Validate the prepared JSONL descriptor and every frame PNG it declares."""

    records = _validate_declared_artifact(
        manifest_path.parent,
        manifest.get("frames_jsonl"),
        field="prepared frames_jsonl",
        hash_file=hash_file,
    )
    frames = manifest.get("frames")
    if not isinstance(frames, Mapping):
        raise PipelineError(f"prepared manifest has no frames object: {manifest_path}")
    raw_frames_directory = frames.get("relative_directory")
    if not isinstance(raw_frames_directory, str) or not raw_frames_directory:
        raise PipelineError("prepared frames.relative_directory must be a non-empty string")
    frames_directory = _relative_artifact(manifest_path.parent, raw_frames_directory)
    canonical_directory = frames_directory.relative_to(manifest_path.parent.resolve()).as_posix()
    if canonical_directory != raw_frames_directory:
        raise PipelineError(
            f"prepared frames.relative_directory is not canonical: {raw_frames_directory}"
        )
    unresolved_directory = manifest_path.parent.resolve() / Path(raw_frames_directory)
    if unresolved_directory.is_symlink() or not frames_directory.is_dir():
        raise PipelineError(
            f"prepared frames directory is missing, invalid, or a symbolic link: {frames_directory}"
        )

    raw_frame_count = manifest.get("frame_count")
    if (
        not isinstance(raw_frame_count, int)
        or isinstance(raw_frame_count, bool)
        or raw_frame_count <= 0
    ):
        raise PipelineError("prepared frame_count must be a positive integer")
    observed_paths: set[Path] = set()
    observed_count = 0
    for row in _iter_jsonl_objects(records.path, field="prepared frames JSONL"):
        raw_index = row.get("frame_index")
        if (
            not isinstance(raw_index, int)
            or isinstance(raw_index, bool)
            or raw_index != observed_count
        ):
            raise PipelineError("prepared frame indices must be contiguous from zero")
        image = _validate_declared_artifact(
            manifest_path.parent,
            {
                "relative_path": row.get("image_path"),
                "sha256": row.get("png_sha256"),
            },
            field=f"prepared frame {observed_count} PNG",
            hash_file=hash_file,
        )
        if not image.path.is_relative_to(frames_directory):
            raise PipelineError(
                f"prepared frame is outside the declared frames directory: {image.path}"
            )
        if image.path in observed_paths:
            raise PipelineError(f"prepared frame path is repeated: {image.path}")
        observed_paths.add(image.path)
        observed_count += 1
    if observed_count != raw_frame_count:
        raise PipelineError(
            "prepared frame_count differs from frames JSONL: "
            f"expected {raw_frame_count}, observed {observed_count}"
        )


def _validate_teacher_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    hash_file: Callable[[Path], str],
) -> None:
    artifacts = _validate_manifest_artifacts(
        manifest_path,
        manifest,
        required_names=(
            "samples_jsonl",
            "rejections_jsonl",
            "targets_csv",
            "split",
            "trajectory_csv",
            "trajectory_ply",
            "trajectory_views_png",
        ),
        hash_file=hash_file,
    )
    _validate_sample_images(
        manifest_path,
        artifacts["samples_jsonl"],
        field="teacher samples JSONL",
        hash_file=hash_file,
    )


def _validate_dataset_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    hash_file: Callable[[Path], str],
) -> None:
    artifacts = _validate_manifest_artifacts(
        manifest_path,
        manifest,
        required_names=(
            "samples_jsonl",
            "targets_csv",
            "train_split",
            "validation_split",
        ),
        hash_file=hash_file,
    )
    _validate_sample_images(
        manifest_path,
        artifacts["samples_jsonl"],
        field="student-dataset samples JSONL",
        hash_file=hash_file,
    )


def _validate_run_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, object],
    *,
    hash_file: Callable[[Path], str],
) -> None:
    _validate_manifest_artifacts(
        manifest_path,
        manifest,
        required_names=(
            "best_checkpoint",
            "last_checkpoint",
            "history",
            "validation_predictions_filtered",
            "validation_predictions_raw",
            "teacher_spike_filter",
            "identity_decisions",
            "included_train",
            "included_validation",
        ),
        hash_file=hash_file,
    )


def _cache_file_inventory(root: Path) -> dict[str, Path]:
    resolved_root = root.resolve()
    inventory: dict[str, Path] = {}
    for candidate in sorted(resolved_root.rglob("*"), key=lambda path: path.as_posix()):
        if candidate == resolved_root / COMPLETE_FILENAME:
            continue
        if candidate.is_symlink():
            raise PipelineError(f"immutable cache must not contain symbolic links: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise PipelineError(f"immutable cache contains a special file: {candidate}")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise PipelineError(f"immutable cache file escapes its root: {candidate}")
        relative = resolved.relative_to(resolved_root).as_posix()
        inventory[relative] = resolved
    return inventory


def _inspect_complete(
    root: Path,
    *,
    kind: str,
    cache_key: str,
    artifact_names: Sequence[str],
    hash_file: Callable[[Path], str],
) -> dict[str, _Artifact] | None:
    if not root.exists():
        return None
    if not root.is_dir():
        raise PipelineError(f"immutable cache path is not a directory: {root}")
    marker_path = root / COMPLETE_FILENAME
    if not marker_path.is_file():
        raise PipelineError(f"immutable cache is incomplete (missing {COMPLETE_FILENAME}): {root}")
    marker = _read_json(marker_path)
    if (
        marker.get("format") != COMPLETE_FORMAT
        or marker.get("format_version") != COMPLETE_FORMAT_VERSION
    ):
        raise PipelineError(f"unsupported complete marker: {marker_path}")
    if marker.get("kind") != kind or marker.get("cache_key") != cache_key:
        raise PipelineError(f"complete marker identity mismatch: {marker_path}")
    descriptors = marker.get("artifacts")
    if not isinstance(descriptors, Mapping):
        raise PipelineError(f"complete marker has no artifacts object: {marker_path}")
    artifacts: dict[str, _Artifact] = {}
    for name in artifact_names:
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, Mapping):
            raise PipelineError(f"complete marker lacks artifact {name!r}: {marker_path}")
        path = _relative_artifact(root, descriptor.get("relative_path"))
        if not path.is_file():
            raise PipelineError(f"completed artifact is missing: {path}")
        expected = _require_sha256(descriptor.get("sha256"), field=f"{name} SHA-256")
        observed = _stable_file_hash(path, hash_file, field=f"{name} SHA-256")
        if observed != expected:
            raise PipelineError(
                f"completed artifact SHA-256 mismatch for {path}: "
                f"expected {expected}, observed {observed}"
            )
        artifacts[name] = _Artifact(path=path, sha256=observed)

    file_descriptors = marker.get("files")
    if not isinstance(file_descriptors, Mapping):
        raise PipelineError(f"complete marker has no file inventory: {marker_path}")
    expected_paths = {str(path) for path in file_descriptors}
    actual_files = _cache_file_inventory(root)
    actual_paths = set(actual_files)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise PipelineError(
            f"immutable cache file inventory mismatch for {root}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative_path in sorted(expected_paths):
        descriptor = file_descriptors.get(relative_path)
        if not isinstance(descriptor, Mapping):
            raise PipelineError(f"complete marker file descriptor is invalid: {relative_path}")
        path = _relative_artifact(root, relative_path)
        canonical = path.relative_to(root.resolve()).as_posix()
        if canonical != relative_path:
            raise PipelineError(f"complete marker file path is not canonical: {relative_path}")
        expected_size = descriptor.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or path.stat().st_size != expected_size
        ):
            raise PipelineError(f"completed cache file size mismatch: {path}")
        expected = _require_sha256(
            descriptor.get("sha256"), field=f"cache file {relative_path} SHA-256"
        )
        observed = _stable_file_hash(path, hash_file, field=f"cache file {relative_path} SHA-256")
        if observed != expected:
            raise PipelineError(
                f"completed cache file SHA-256 mismatch for {path}: "
                f"expected {expected}, observed {observed}"
            )
    return artifacts


def _write_complete(
    root: Path,
    *,
    kind: str,
    cache_key: str,
    artifacts: Mapping[str, Path],
    metadata: Mapping[str, object],
    hash_file: Callable[[Path], str],
) -> None:
    descriptors: dict[str, object] = {}
    for name, path in artifacts.items():
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            raise PipelineError(f"cannot complete cache with invalid artifact {name}: {path}")
        descriptors[name] = {
            "relative_path": resolved.relative_to(root.resolve()).as_posix(),
            "sha256": _stable_file_hash(resolved, hash_file, field=f"{name} SHA-256"),
        }
    file_descriptors = {
        relative_path: {
            "size_bytes": path.stat().st_size,
            "sha256": _stable_file_hash(
                path, hash_file, field=f"cache file {relative_path} SHA-256"
            ),
        }
        for relative_path, path in _cache_file_inventory(root).items()
    }
    _write_json(
        root / COMPLETE_FILENAME,
        {
            "format": COMPLETE_FORMAT,
            "format_version": COMPLETE_FORMAT_VERSION,
            "kind": kind,
            "cache_key": cache_key,
            "artifacts": descriptors,
            "files": file_descriptors,
            "metadata": _jsonable(metadata),
        },
    )


def _new_staging_dir(final_dir: Path, dependencies: PipelineDependencies) -> Path:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    nonce = re.sub(r"[^A-Za-z0-9]", "", dependencies.nonce_factory())[:24] or "stage"
    staging = final_dir.parent / f".{final_dir.name}.staging-{nonce}"
    try:
        staging.mkdir()
    except FileExistsError as error:
        raise PipelineError(f"staging directory already exists: {staging}") from error
    return staging


def _publish_staging(staging: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise PipelineError(f"refusing to replace immutable cache directory: {final_dir}")
    try:
        staging.rename(final_dir)
    except OSError as error:
        raise PipelineError(f"failed to publish {staging} as {final_dir}: {error}") from error


def _cleanup_staging(staging: Path) -> None:
    if staging.is_dir() and staging.name.startswith(".") and ".staging-" in staging.name:
        shutil.rmtree(staging, ignore_errors=True)


def _video_workspace_path(final_dir: Path) -> Path:
    return final_dir.parent / f".{final_dir.name}{VIDEO_WORKSPACE_SUFFIX}"


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _video_workspace_payload(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    cache_key: str,
    dependencies: PipelineDependencies,
) -> dict[str, object]:
    hand_model = Path(config.prepare.hand_model_path)
    return {
        "format": VIDEO_WORKSPACE_FORMAT,
        "format_version": VIDEO_WORKSPACE_FORMAT_VERSION,
        "cache_key": cache_key,
        "sequence_id": str(video.sequence_id),
        "split": str(video.split),
        "source_sha256": str(video.source_sha256),
        "focal_35mm_mm": float(video.focal_35mm_mm),
        "prepare": {
            "hand_model_sha256": _stable_file_hash(
                hand_model,
                dependencies.hash_file,
                field="hand landmarker model SHA-256",
            ),
            "landmark_indices": list(config.prepare.landmark_indices),
            "frame_transfer_mode": str(config.prepare.frame_transfer_mode),
            "max_frames": config.prepare.max_frames,
        },
        "teacher": {
            "device": str(config.teacher.device),
            "frame_transfer_mode": str(config.teacher.frame_transfer_mode),
            "max_frames": config.teacher.max_frames,
            "checkpoint_interval_frames": config.teacher.checkpoint_interval_frames,
        },
    }


def _write_video_workspace_marker(
    workspace: Path,
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    cache_key: str,
    dependencies: PipelineDependencies,
) -> None:
    marker_path = workspace / VIDEO_WORKSPACE_FILENAME
    temporary = workspace / f"{VIDEO_WORKSPACE_FILENAME}.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    _write_json(
        temporary,
        _video_workspace_payload(config, video, cache_key, dependencies),
    )
    with temporary.open("rb") as marker_file:
        os.fsync(marker_file.fileno())
    temporary.replace(marker_path)
    _fsync_directory(workspace)


def _validate_video_workspace_marker(
    workspace: Path,
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    cache_key: str,
    dependencies: PipelineDependencies,
) -> None:
    marker_path = workspace / VIDEO_WORKSPACE_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise PipelineError(f"video workspace has no identity marker: {workspace}")
    marker = _read_json(marker_path)
    expected = _video_workspace_payload(config, video, cache_key, dependencies)
    if marker != expected:
        raise PipelineError(f"video workspace identity mismatch: {workspace}")


def _quarantine_legacy_workspace(
    candidate: Path,
    dependencies: PipelineDependencies,
) -> None:
    if not candidate.exists():
        return
    nonce = re.sub(r"[^A-Za-z0-9]", "", dependencies.nonce_factory())[:24] or "legacy"
    stem = candidate.name.lstrip(".").replace(".staging-", "-staging-")
    quarantine = candidate.parent / f".{stem}.legacy-ignored-{nonce}"
    if quarantine.exists():
        raise PipelineError(f"legacy workspace quarantine already exists: {quarantine}")
    candidate.rename(quarantine)
    _fsync_directory(candidate.parent)
    dependencies.migration_checkpoint("legacy-quarantined")


def _preparation_implementation_hashes(
    project_root: Path,
    dependencies: PipelineDependencies,
) -> dict[str, str]:
    package_dir = Path(__file__).resolve().parents[1]
    candidates = {
        "pseudo_label_inputs.py": package_dir / "pseudo_label_inputs.py",
        "hands.py": package_dir / "hands.py",
        "constants.py": package_dir / "constants.py",
        "camera.py": package_dir / "camera.py",
        "sample_experiment.py": package_dir / "sample_experiment.py",
        "video_cache.py": package_dir / "video_cache.py",
        "prepare_pseudo_label_inputs.py": project_root
        / "scripts"
        / "prepare_pseudo_label_inputs.py",
        "pyproject.toml": project_root / "pyproject.toml",
        "uv.lock": project_root / "uv.lock",
    }
    missing = [path for path in candidates.values() if not path.is_file()]
    if missing:
        raise PipelineError(
            "preparation runtime is incomplete: " + ", ".join(str(path) for path in missing)
        )
    return {
        name: _stable_file_hash(
            path,
            dependencies.hash_file,
            field=f"preparation implementation {name} SHA-256",
        )
        for name, path in candidates.items()
    }


def _validate_workspace_prepared(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    manifest_path: Path,
    dependencies: PipelineDependencies,
) -> tuple[dict[str, Any], str]:
    if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
        raise PipelineError(f"prepared workspace manifest path is unsafe: {manifest_path}")
    prepared = _read_json(manifest_path)
    _require_manifest_format(
        prepared,
        expected_format=PREPARED_INPUT_FORMAT,
        expected_version=PREPARED_INPUT_FORMAT_VERSION,
        artifact=manifest_path,
    )
    _validate_prepared_artifacts(
        manifest_path,
        prepared,
        hash_file=dependencies.hash_file,
    )
    prepared_source = prepared.get("source")
    if not isinstance(prepared_source, Mapping) or (
        prepared_source.get("mode") != "direct_video_decode"
        or prepared_source.get("video_sha256") != video.source_sha256
    ):
        raise PipelineError("prepared workspace source video mismatch")
    hand_landmarker = prepared.get("hand_landmarker")
    expected_hand_sha256 = _stable_file_hash(
        Path(config.prepare.hand_model_path),
        dependencies.hash_file,
        field="hand landmarker model SHA-256",
    )
    if not isinstance(hand_landmarker, Mapping) or (
        hand_landmarker.get("model_sha256") != expected_hand_sha256
    ):
        raise PipelineError("prepared workspace hand model hash mismatch")
    feature_landmarks = prepared.get("feature_landmarks")
    if not isinstance(feature_landmarks, Mapping) or feature_landmarks.get("indices") != list(
        config.prepare.landmark_indices
    ):
        raise PipelineError("prepared workspace landmark configuration mismatch")
    camera = prepared.get("camera")
    raw_focal = None if not isinstance(camera, Mapping) else camera.get("focal_35mm_equivalent_mm")
    if (
        not isinstance(raw_focal, (int, float))
        or isinstance(raw_focal, bool)
        or not math.isfinite(float(raw_focal))
        or float(raw_focal) != float(video.focal_35mm_mm)
    ):
        raise PipelineError("prepared workspace focal length mismatch")
    processing = prepared.get("processing")
    if not isinstance(processing, Mapping) or (
        processing.get("max_frames") != config.prepare.max_frames
    ):
        raise PipelineError("prepared workspace frame limit mismatch")
    provenance = prepared.get("provenance")
    implementation = (
        None if not isinstance(provenance, Mapping) else provenance.get("implementation_sha256")
    )
    expected_implementation = _preparation_implementation_hashes(
        Path(config.project_root), dependencies
    )
    if not isinstance(implementation, Mapping) or dict(implementation) != (expected_implementation):
        raise PipelineError("prepared workspace implementation provenance mismatch")
    return prepared, _stable_file_hash(
        manifest_path,
        dependencies.hash_file,
        field="prepared manifest SHA-256",
    )


def _legacy_staging_candidates(parent: Path) -> list[Path]:
    return sorted(
        (
            candidate
            for candidate in parent.iterdir()
            if candidate.name.startswith(".")
            and ".staging-" in candidate.name
            and candidate.is_dir()
            and not candidate.is_symlink()
        ),
        key=lambda path: path.name,
    )


def _recover_legacy_teacher(
    workspace: Path,
    candidates: Sequence[Path],
    dependencies: PipelineDependencies,
) -> None:
    target = workspace / "teacher"
    target_progress_temp: Path | None = None
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise PipelineError(f"teacher workspace is unsafe: {target}")
        target_entries = list(target.iterdir())
        if target_entries:
            progress = target / PSEUDO_LABEL_PROGRESS_FILENAME
            progress_temp = target / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
            if progress.exists() or progress.is_symlink():
                if progress.is_symlink() or not progress.is_file():
                    raise PipelineError(f"teacher progress journal is unsafe: {progress}")
                return
            if progress_temp.exists() or progress_temp.is_symlink():
                if progress_temp.is_symlink() or not progress_temp.is_file():
                    raise PipelineError(
                        f"teacher progress temporary file is unsafe: {progress_temp}"
                    )
                if target_entries != [progress_temp]:
                    raise PipelineError(
                        "teacher progress temporary file is not the only interrupted "
                        f"artifact: {target}"
                    )
                target_progress_temp = progress_temp
            else:
                return
        else:
            target.rmdir()
            _fsync_directory(workspace)

    journaled: list[tuple[Path, Path]] = []
    for candidate in candidates:
        teacher_dir = candidate / "teacher"
        journal = teacher_dir / PSEUDO_LABEL_PROGRESS_FILENAME
        if not (journal.exists() or journal.is_symlink()):
            continue
        if teacher_dir.is_symlink() or not teacher_dir.is_dir():
            raise PipelineError(f"legacy teacher workspace is unsafe: {teacher_dir}")
        if journal.is_symlink() or not journal.is_file():
            raise PipelineError(f"legacy teacher progress journal is unsafe: {journal}")
        journaled.append((candidate, teacher_dir))
    if len(journaled) > 1:
        raise PipelineError(
            "multiple interrupted teacher journals require manual review: "
            + ", ".join(str(teacher) for _, teacher in journaled)
        )
    if not journaled:
        return

    if target_progress_temp is not None:
        target_progress_temp.unlink()
        _fsync_directory(target)
        target.rmdir()
        _fsync_directory(workspace)

    legacy, teacher_dir = journaled[0]
    legacy_teacher_manifest = teacher_dir / "dataset_manifest.json"
    quarantined_manifest = legacy / "teacher_dataset_manifest.unvalidated.json"
    if quarantined_manifest.exists() or quarantined_manifest.is_symlink():
        if quarantined_manifest.is_symlink() or not quarantined_manifest.is_file():
            raise PipelineError(
                f"quarantined legacy teacher manifest is unsafe: {quarantined_manifest}"
            )
        if legacy_teacher_manifest.exists() or legacy_teacher_manifest.is_symlink():
            raise PipelineError(
                f"legacy teacher has both active and quarantined manifests: {teacher_dir}"
            )
    elif legacy_teacher_manifest.exists() or legacy_teacher_manifest.is_symlink():
        if legacy_teacher_manifest.is_symlink() or not legacy_teacher_manifest.is_file():
            raise PipelineError(f"legacy teacher manifest is unsafe: {legacy_teacher_manifest}")
        legacy_teacher_manifest.rename(quarantined_manifest)
        _fsync_directory(teacher_dir)
        _fsync_directory(legacy)
        dependencies.migration_checkpoint("teacher-manifest-quarantined")

    teacher_dir.rename(target)
    _fsync_directory(legacy)
    _fsync_directory(workspace)
    dependencies.migration_checkpoint("teacher-moved")


def _acquire_video_workspace(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    status: VideoCacheStatus,
    dependencies: PipelineDependencies,
) -> Path:
    workspace = _video_workspace_path(status.cache_dir)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    workspace_existed = workspace.exists() or workspace.is_symlink()
    if workspace_existed:
        if workspace.is_symlink() or not workspace.is_dir():
            raise PipelineError(f"video workspace path is invalid: {workspace}")
        marker_path = workspace / VIDEO_WORKSPACE_FILENAME
        if marker_path.is_file():
            _validate_video_workspace_marker(
                workspace, config, video, status.cache_key, dependencies
            )
            temporary = workspace / f"{VIDEO_WORKSPACE_FILENAME}.tmp"
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        else:
            allowed = {f"{VIDEO_WORKSPACE_FILENAME}.tmp"}
            unexpected = [item for item in workspace.iterdir() if item.name not in allowed]
            if unexpected:
                raise PipelineError(f"video workspace has no identity marker: {workspace}")
            temporary = workspace / f"{VIDEO_WORKSPACE_FILENAME}.tmp"
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            _write_video_workspace_marker(workspace, config, video, status.cache_key, dependencies)

        legacy_candidates = _legacy_staging_candidates(workspace.parent)
        prepared_manifest = workspace / "prepared" / "manifest.json"
        if prepared_manifest.is_file():
            _validate_workspace_prepared(config, video, prepared_manifest, dependencies)
            _recover_legacy_teacher(workspace, legacy_candidates, dependencies)
            for candidate in legacy_candidates:
                _quarantine_legacy_workspace(candidate, dependencies)
            return workspace

        marker_path = workspace / VIDEO_WORKSPACE_FILENAME
        if any(item != marker_path for item in workspace.iterdir()):
            return workspace
    else:
        legacy_candidates = _legacy_staging_candidates(workspace.parent)

    valid_candidates: list[tuple[Path, str]] = []
    for candidate in legacy_candidates:
        manifest_path = candidate / "prepared" / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            _, prepared_sha256 = _validate_workspace_prepared(
                config, video, manifest_path, dependencies
            )
        except (OSError, TypeError, ValueError, PipelineError):
            continue
        else:
            valid_candidates.append((candidate, prepared_sha256))
    if len(valid_candidates) > 1:
        distinct_hashes = {digest for _, digest in valid_candidates}
        if len(distinct_hashes) > 1:
            raise PipelineError(
                "multiple differing valid interrupted video workspaces require "
                "manual review: " + ", ".join(str(path) for path, _ in valid_candidates)
            )
        valid_candidates = valid_candidates[:1]
    if not workspace_existed:
        workspace.mkdir()
        _write_video_workspace_marker(workspace, config, video, status.cache_key, dependencies)
        if legacy_candidates:
            dependencies.migration_checkpoint("workspace-created")
    if valid_candidates:
        legacy = valid_candidates[0][0]
        (legacy / "prepared").rename(workspace / "prepared")
        _fsync_directory(legacy)
        _fsync_directory(workspace)
        dependencies.migration_checkpoint("prepared-moved")
        _recover_legacy_teacher(workspace, legacy_candidates, dependencies)
    for candidate in legacy_candidates:
        _quarantine_legacy_workspace(candidate, dependencies)
    return workspace


def _verify_source_video(
    video: DiscoveredVideoProtocol,
    dependencies: PipelineDependencies,
    *,
    context: str,
) -> None:
    path = Path(video.path)
    if not path.is_file():
        raise PipelineError(f"{context}: source video is missing: {path}")
    before = path.stat()
    if before.st_size <= 0:
        raise PipelineError(f"{context}: source video is empty: {path}")
    observed_sha256 = dependencies.hash_file(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise PipelineError(f"{context}: source video changed while it was being hashed: {path}")
    if int(video.size_bytes) != after.st_size:
        raise PipelineError(f"{context}: source video size changed: {path}")
    expected_sha256 = _require_sha256(video.source_sha256, field="source video SHA-256")
    if observed_sha256 != expected_sha256:
        raise PipelineError(
            f"{context}: source video SHA-256 changed for {path}: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )


def _validated_videos(
    config: PipelineConfigProtocol,
    videos: Sequence[DiscoveredVideoProtocol] | None,
    dependencies: PipelineDependencies,
) -> tuple[DiscoveredVideoProtocol, ...]:
    discovered = tuple(dependencies.discover(config) if videos is None else videos)
    if not discovered:
        raise PipelineError("no training videos were discovered")
    sequence_ids: set[str] = set()
    source_hashes: set[str] = set()
    validated: list[DiscoveredVideoProtocol] = []
    for video in discovered:
        path = Path(video.path)
        sequence_id = str(video.sequence_id)
        split = str(video.split)
        expected_sha256 = _require_sha256(video.source_sha256, field="source video SHA-256")
        if not _SEQUENCE_ID.fullmatch(sequence_id):
            raise PipelineError(f"invalid sequence ID from discovery: {sequence_id!r}")
        if sequence_id in sequence_ids:
            raise PipelineError(f"duplicate sequence ID from discovery: {sequence_id}")
        if expected_sha256 in source_hashes:
            raise PipelineError(f"duplicate source video SHA-256 from discovery: {expected_sha256}")
        if split not in {"train", "validation"}:
            raise PipelineError(f"unsupported video split {split!r}: {path}")
        _verify_source_video(video, dependencies, context="discovery validation")
        focal = float(video.focal_35mm_mm)
        if not math.isfinite(focal) or focal <= 0:
            raise PipelineError(f"video focal length must be finite and positive: {path}")
        sequence_ids.add(sequence_id)
        source_hashes.add(expected_sha256)
        validated.append(video)
    return tuple(validated)


def _implementation_fingerprint(
    project_root: Path,
    project_paths: Sequence[str],
    hash_file: Callable[[Path], str],
) -> str:
    """Hash all imported package code plus required project entrypoints and locks."""

    package_root = Path(__file__).resolve().parents[1]
    package_files = sorted(package_root.rglob("*.py"), key=lambda path: path.as_posix())
    if not package_files:
        raise PipelineError(f"fingertip_depth source package is missing: {package_root}")
    project_files: dict[str, str] = {}
    missing: list[Path] = []
    for relative_path in project_paths:
        path = project_root / relative_path
        if not path.is_file():
            missing.append(path)
            continue
        project_files[relative_path] = _stable_file_hash(
            path, hash_file, field=f"project file {relative_path} SHA-256"
        )
    if missing:
        raise PipelineError(
            "pipeline runtime is incomplete: " + ", ".join(str(path) for path in missing)
        )
    package_hashes = {
        path.relative_to(package_root).as_posix(): _stable_file_hash(
            path, hash_file, field=f"package file {path.name} SHA-256"
        )
        for path in package_files
    }
    return _content_key(
        {
            "package_root": "fingertip_depth",
            "package_files": package_hashes,
            "project_files": project_files,
        }
    )


def _video_cache_key(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    dependencies: PipelineDependencies,
) -> str:
    project_root = Path(config.project_root)
    hand_model = Path(config.prepare.hand_model_path)
    if not hand_model.is_file():
        raise PipelineError(f"hand landmarker model is missing: {hand_model}")
    depth_project = Path(config.teacher.depth_pro_project)
    teacher_script = project_root / "scripts" / "generate_depth_pro_pseudo_labels.py"
    depth_pyproject = depth_project / "pyproject.toml"
    depth_lock = depth_project / "uv.lock"
    missing = [path for path in (teacher_script, depth_pyproject, depth_lock) if not path.is_file()]
    if missing:
        raise PipelineError(
            "Depth Pro runtime is incomplete: " + ", ".join(str(path) for path in missing)
        )
    payload = {
        "source": {
            "sha256": video.source_sha256,
            "size_bytes": int(video.size_bytes),
            "sequence_id": video.sequence_id,
            "split": video.split,
            "focal_35mm_mm": float(video.focal_35mm_mm),
        },
        "prepare": {
            "hand_model_sha256": dependencies.hash_file(hand_model),
            "landmark_indices": list(config.prepare.landmark_indices),
            "frame_transfer_mode": config.prepare.frame_transfer_mode,
            "max_frames": config.prepare.max_frames,
        },
        "teacher": {
            "device": config.teacher.device,
            "frame_transfer_mode": config.teacher.frame_transfer_mode,
            "max_frames": config.teacher.max_frames,
            "checkpoint_interval_frames": config.teacher.checkpoint_interval_frames,
            # Selection evidence is intentionally not inherited by newly
            # discovered videos; the configured path is excluded and unused.
            "selection_report": None,
            "depth_pro_pyproject_sha256": dependencies.hash_file(depth_pyproject),
            "depth_pro_lock_sha256": dependencies.hash_file(depth_lock),
        },
        "implementation_sha256": _implementation_fingerprint(
            project_root,
            (
                "pyproject.toml",
                "uv.lock",
                "scripts/generate_depth_pro_pseudo_labels.py",
                "scripts/prepare_pseudo_label_inputs.py",
            ),
            dependencies.hash_file,
        ),
    }
    return _content_key(payload)


def _dataset_cache_key(
    config: PipelineConfigProtocol,
    video_statuses: Sequence[VideoCacheStatus],
    dependencies: PipelineDependencies,
) -> str:
    return _content_key(
        {
            "videos": [
                {
                    "sequence_id": item.sequence_id,
                    "split": item.split,
                    "source_sha256": item.source_sha256,
                    "video_cache_key": item.cache_key,
                }
                for item in sorted(video_statuses, key=lambda value: value.sequence_id)
            ],
            "dataset": {
                "frame_transfer_mode": config.dataset.frame_transfer_mode,
                "validation_tail_fraction": config.dataset.validation_tail_fraction,
            },
            "implementation_sha256": _implementation_fingerprint(
                Path(config.project_root),
                (
                    "pyproject.toml",
                    "uv.lock",
                    "scripts/build_student_dataset.py",
                ),
                dependencies.hash_file,
            ),
        }
    )


def _run_cache_key(
    config: PipelineConfigProtocol,
    dataset_key: str,
    dependencies: PipelineDependencies,
) -> str:
    return _content_key(
        {
            "dataset_cache_key": dataset_key,
            "training": {
                "device": config.training.device,
                "model": config.training.model,
                "optimizer": config.training.optimizer,
                "spike_filter": config.training.spike_filter,
            },
            "implementation_sha256": _implementation_fingerprint(
                Path(config.project_root),
                (
                    "pyproject.toml",
                    "uv.lock",
                    "scripts/train_student_transformer.py",
                ),
                dependencies.hash_file,
            ),
        }
    )


def _inspect_video_cache(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    cache_key: str,
    dependencies: PipelineDependencies,
) -> VideoCacheStatus:
    cache_dir = Path(config.output.processed_dir) / "videos" / str(video.sequence_id) / cache_key
    artifacts = _inspect_complete(
        cache_dir,
        kind="video",
        cache_key=cache_key,
        artifact_names=("prepared_manifest", "teacher_manifest"),
        hash_file=dependencies.hash_file,
    )
    if artifacts is None:
        return VideoCacheStatus(
            sequence_id=str(video.sequence_id),
            split=str(video.split),
            source_path=Path(video.path),
            source_sha256=str(video.source_sha256),
            cache_key=cache_key,
            cache_dir=cache_dir,
            cache_hit=False,
        )
    prepared = _read_json(artifacts["prepared_manifest"].path)
    _require_manifest_format(
        prepared,
        expected_format=PREPARED_INPUT_FORMAT,
        expected_version=PREPARED_INPUT_FORMAT_VERSION,
        artifact=artifacts["prepared_manifest"].path,
    )
    _validate_prepared_artifacts(
        artifacts["prepared_manifest"].path,
        prepared,
        hash_file=dependencies.hash_file,
    )
    prepared_source = prepared.get("source")
    if (
        not isinstance(prepared_source, Mapping)
        or prepared_source.get("video_sha256") != video.source_sha256
    ):
        raise PipelineError(f"prepared manifest source video mismatch: {cache_dir}")
    teacher = _read_json(artifacts["teacher_manifest"].path)
    _require_manifest_format(
        teacher,
        expected_format=PSEUDO_LABEL_DATASET_FORMAT,
        expected_version=PSEUDO_LABEL_DATASET_FORMAT_VERSION,
        artifact=artifacts["teacher_manifest"].path,
    )
    _validate_teacher_artifacts(
        artifacts["teacher_manifest"].path,
        teacher,
        hash_file=dependencies.hash_file,
    )
    sequence = teacher.get("sequence")
    prepared_inputs = teacher.get("prepared_inputs")
    if not isinstance(sequence, Mapping) or (
        sequence.get("id") != video.sequence_id or sequence.get("split") != video.split
    ):
        raise PipelineError(f"teacher manifest sequence/split mismatch: {cache_dir}")
    if (
        not isinstance(prepared_inputs, Mapping)
        or prepared_inputs.get("manifest_sha256") != artifacts["prepared_manifest"].sha256
    ):
        raise PipelineError(f"teacher manifest prepared-input hash mismatch: {cache_dir}")
    return VideoCacheStatus(
        sequence_id=str(video.sequence_id),
        split=str(video.split),
        source_path=Path(video.path),
        source_sha256=str(video.source_sha256),
        cache_key=cache_key,
        cache_dir=cache_dir,
        cache_hit=True,
        prepared_manifest=artifacts["prepared_manifest"].path,
        prepared_manifest_sha256=artifacts["prepared_manifest"].sha256,
        teacher_manifest=artifacts["teacher_manifest"].path,
        teacher_manifest_sha256=artifacts["teacher_manifest"].sha256,
    )


def _invoke_teacher(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    *,
    prepared_manifest: Path,
    prepared_manifest_sha256: str,
    output_dir: Path,
    resume: bool,
    dependencies: PipelineDependencies,
) -> None:
    command = [
        "uv",
        "run",
        "--locked",
        "--project",
        str(config.teacher.depth_pro_project),
        "python",
        str(Path(config.project_root) / "scripts" / "generate_depth_pro_pseudo_labels.py"),
        "--prepared-manifest",
        str(prepared_manifest),
        "--expected-prepared-manifest-sha256",
        prepared_manifest_sha256,
        "--output-dir",
        str(output_dir),
        "--sequence-id",
        str(video.sequence_id),
        "--split",
        str(video.split),
        "--frame-transfer-mode",
        str(config.teacher.frame_transfer_mode),
        "--device",
        str(config.teacher.device),
        "--checkpoint-interval-frames",
        str(config.teacher.checkpoint_interval_frames),
    ]
    if resume:
        command.append("--resume")
    if config.teacher.max_frames is not None:
        command.extend(("--max-frames", str(config.teacher.max_frames)))
    # teacher_selection_report is intentionally never forwarded.  Evidence
    # scoped to an older video must not be relabelled as evidence for a new one.
    try:
        result = dependencies.command_runner(command, cwd=Path(config.project_root))
    except (OSError, subprocess.SubprocessError) as error:
        raise PipelineError(f"Depth Pro subprocess could not run: {error}") from error
    returncode = int(getattr(result, "returncode", 0))
    if returncode != 0:
        stderr = str(getattr(result, "stderr", "")).strip()
        detail = f": {stderr[-2000:]}" if stderr else ""
        raise PipelineError(f"Depth Pro subprocess failed with exit code {returncode}{detail}")


def _remove_workspace_snapshot(path: Path) -> None:
    if path.is_symlink():
        raise PipelineError(f"video workspace snapshot must not be a symbolic link: {path}")
    if path.exists():
        if not path.is_file():
            raise PipelineError(f"video workspace snapshot must be a regular file: {path}")
        path.unlink()


def _ensure_workspace_prepared(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    workspace: Path,
    dependencies: PipelineDependencies,
) -> tuple[Path, dict[str, Any], str]:
    prepared_dir = workspace / "prepared"
    prepared_manifest = prepared_dir / "manifest.json"
    source_path = Path(video.path)
    source_suffix = source_path.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", source_suffix):
        source_suffix = ".video"
    snapshot_path = workspace / f"source_video{source_suffix}"
    hand_model_path = Path(config.prepare.hand_model_path)
    hand_model_snapshot = workspace / "hand_landmarker.task"

    if prepared_manifest.is_file():
        prepared, prepared_sha256 = _validate_workspace_prepared(
            config, video, prepared_manifest, dependencies
        )
        _verify_source_video(
            video,
            dependencies,
            context="before prepared-input workspace reuse",
        )
        _remove_workspace_snapshot(snapshot_path)
        _remove_workspace_snapshot(hand_model_snapshot)
        return prepared_manifest, prepared, prepared_sha256

    if prepared_dir.exists() or prepared_dir.is_symlink():
        if prepared_dir.is_symlink() or not prepared_dir.is_dir():
            raise PipelineError(f"incomplete prepared workspace is invalid: {prepared_dir}")
        shutil.rmtree(prepared_dir)
    teacher_dir = workspace / "teacher"
    if teacher_dir.exists() and (not teacher_dir.is_dir() or any(teacher_dir.iterdir())):
        raise PipelineError(
            f"teacher progress exists without a completed prepared workspace: {teacher_dir}"
        )
    _remove_workspace_snapshot(snapshot_path)
    _remove_workspace_snapshot(hand_model_snapshot)

    shutil.copyfile(hand_model_path, hand_model_snapshot)
    hand_model_sha256 = _stable_file_hash(
        hand_model_path,
        dependencies.hash_file,
        field="hand landmarker model SHA-256",
    )
    if (
        _stable_file_hash(
            hand_model_snapshot,
            dependencies.hash_file,
            field="hand landmarker snapshot SHA-256",
        )
        != hand_model_sha256
    ):
        raise PipelineError("hand landmarker model changed while creating its processing snapshot")
    shutil.copyfile(source_path, snapshot_path)
    snapshot_size = snapshot_path.stat().st_size
    snapshot_sha256 = _stable_file_hash(
        snapshot_path,
        dependencies.hash_file,
        field="source video snapshot SHA-256",
    )
    if snapshot_size != int(video.size_bytes) or snapshot_sha256 != video.source_sha256:
        raise PipelineError(
            f"source video changed while creating the processing snapshot: {source_path}"
        )

    dependencies.prepare(
        output_dir=prepared_dir,
        hand_model_path=hand_model_snapshot,
        focal_35mm_equivalent_mm=float(video.focal_35mm_mm),
        feature_landmark_indices=tuple(config.prepare.landmark_indices),
        input_video_path=snapshot_path,
        frame_cache_manifest_path=None,
        expected_frame_cache_manifest_sha256=None,
        frame_transfer_mode=str(config.prepare.frame_transfer_mode),
        max_frames=config.prepare.max_frames,
    )
    if not prepared_manifest.is_file():
        raise PipelineError(f"prepare stage did not create {prepared_manifest}")
    prepared = _read_json(prepared_manifest)
    prepared_source = prepared.get("source")
    if not isinstance(prepared_source, Mapping) or (
        prepared_source.get("video_sha256") != video.source_sha256
    ):
        raise PipelineError("prepare stage returned the wrong source video hash")
    hand_landmarker = prepared.get("hand_landmarker")
    if not isinstance(hand_landmarker, Mapping) or (
        hand_landmarker.get("model_sha256") != hand_model_sha256
    ):
        raise PipelineError("prepare stage returned the wrong hand model hash")
    rewritten_hand_landmarker = dict(hand_landmarker)
    rewritten_hand_landmarker["model_path"] = str(hand_model_path.resolve())
    prepared["hand_landmarker"] = rewritten_hand_landmarker
    rewritten_source = dict(prepared_source)
    rewritten_source["video_path"] = str(source_path.resolve())
    prepared["source"] = rewritten_source
    _write_json(prepared_manifest, prepared)
    prepared, prepared_sha256 = _validate_workspace_prepared(
        config, video, prepared_manifest, dependencies
    )
    _remove_workspace_snapshot(snapshot_path)
    _remove_workspace_snapshot(hand_model_snapshot)
    _verify_source_video(
        video,
        dependencies,
        context="after prepared-input generation",
    )
    return prepared_manifest, prepared, prepared_sha256


def _validate_workspace_teacher(
    video: DiscoveredVideoProtocol,
    teacher_manifest: Path,
    prepared_manifest_sha256: str,
    dependencies: PipelineDependencies,
) -> dict[str, Any]:
    if teacher_manifest.is_symlink() or teacher_manifest.parent.is_symlink():
        raise PipelineError(f"teacher workspace manifest path is unsafe: {teacher_manifest}")
    teacher = _read_json(teacher_manifest)
    _require_manifest_format(
        teacher,
        expected_format=PSEUDO_LABEL_DATASET_FORMAT,
        expected_version=PSEUDO_LABEL_DATASET_FORMAT_VERSION,
        artifact=teacher_manifest,
    )
    _validate_teacher_artifacts(
        teacher_manifest,
        teacher,
        hash_file=dependencies.hash_file,
    )
    sequence = teacher.get("sequence")
    prepared_inputs = teacher.get("prepared_inputs")
    if not isinstance(sequence, Mapping) or (
        sequence.get("id") != video.sequence_id or sequence.get("split") != video.split
    ):
        raise PipelineError("Depth Pro stage returned the wrong sequence or split")
    if not isinstance(prepared_inputs, Mapping) or (
        prepared_inputs.get("manifest_sha256") != prepared_manifest_sha256
    ):
        raise PipelineError("Depth Pro stage returned the wrong prepared manifest hash")
    return teacher


def _teacher_workspace_resume_state(teacher_dir: Path) -> bool:
    if not teacher_dir.exists():
        return False
    if teacher_dir.is_symlink() or not teacher_dir.is_dir():
        raise PipelineError(f"teacher workspace is not a directory: {teacher_dir}")
    entries = list(teacher_dir.iterdir())
    if not entries:
        return False
    progress = teacher_dir / PSEUDO_LABEL_PROGRESS_FILENAME
    progress_temp = teacher_dir / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
    if progress_temp.exists() or progress_temp.is_symlink():
        if progress_temp.is_symlink() or not progress_temp.is_file():
            raise PipelineError(f"teacher progress temporary file is unsafe: {progress_temp}")
        if progress.exists() or progress.is_symlink():
            if progress.is_symlink() or not progress.is_file():
                raise PipelineError(f"teacher progress journal is unsafe: {progress}")
            return True
        if entries == [progress_temp]:
            return False
        raise PipelineError(
            f"teacher progress temporary file is not the only interrupted artifact: {teacher_dir}"
        )
    if progress.is_symlink() or not progress.is_file():
        raise PipelineError(
            f"non-empty teacher workspace has no recognized progress journal: {teacher_dir}"
        )
    return True


def _build_video_cache(
    config: PipelineConfigProtocol,
    video: DiscoveredVideoProtocol,
    status: VideoCacheStatus,
    dependencies: PipelineDependencies,
) -> VideoCacheStatus:
    workspace = _acquire_video_workspace(config, video, status, dependencies)
    if _video_cache_key(config, video, dependencies) != status.cache_key:
        raise PipelineError(
            "pipeline implementation or dependencies changed before video processing"
        )
    prepared_manifest, _prepared, prepared_sha256 = _ensure_workspace_prepared(
        config, video, workspace, dependencies
    )
    teacher_dir = workspace / "teacher"
    teacher_manifest = teacher_dir / "dataset_manifest.json"
    teacher_complete = False
    resume_teacher = False
    if teacher_manifest.exists() or teacher_manifest.is_symlink():
        try:
            _validate_workspace_teacher(video, teacher_manifest, prepared_sha256, dependencies)
        except (OSError, TypeError, ValueError, PipelineError):
            resume_teacher = _teacher_workspace_resume_state(teacher_dir)
            if not resume_teacher:
                raise
        else:
            teacher_complete = True
    if not teacher_complete:
        if not resume_teacher:
            resume_teacher = _teacher_workspace_resume_state(teacher_dir)
        _invoke_teacher(
            config,
            video,
            prepared_manifest=prepared_manifest,
            prepared_manifest_sha256=prepared_sha256,
            output_dir=teacher_dir,
            resume=resume_teacher,
            dependencies=dependencies,
        )
        if not teacher_manifest.is_file():
            raise PipelineError(f"Depth Pro stage did not create {teacher_manifest}")
        _validate_workspace_teacher(video, teacher_manifest, prepared_sha256, dependencies)
    _verify_source_video(
        video,
        dependencies,
        context="before immutable video-cache publication",
    )
    if _video_cache_key(config, video, dependencies) != status.cache_key:
        raise PipelineError(
            "pipeline implementation or dependencies changed during video processing"
        )
    _write_complete(
        workspace,
        kind="video",
        cache_key=status.cache_key,
        artifacts={
            "prepared_manifest": prepared_manifest,
            "teacher_manifest": teacher_manifest,
        },
        metadata={
            "sequence_id": video.sequence_id,
            "split": video.split,
            "source_sha256": video.source_sha256,
            "teacher_selection_report_used": False,
        },
        hash_file=dependencies.hash_file,
    )
    _inspect_complete(
        workspace,
        kind="video",
        cache_key=status.cache_key,
        artifact_names=("prepared_manifest", "teacher_manifest"),
        hash_file=dependencies.hash_file,
    )
    _publish_staging(workspace, status.cache_dir)
    return _inspect_video_cache(config, video, status.cache_key, dependencies)


def _inspect_dataset_cache(
    dataset_dir: Path,
    dataset_key: str,
    dependencies: PipelineDependencies,
) -> _Artifact | None:
    artifacts = _inspect_complete(
        dataset_dir,
        kind="dataset",
        cache_key=dataset_key,
        artifact_names=("dataset_manifest",),
        hash_file=dependencies.hash_file,
    )
    if artifacts is None:
        return None
    manifest = _read_json(artifacts["dataset_manifest"].path)
    _require_manifest_format(
        manifest,
        expected_format=STUDENT_DATASET_FORMAT,
        expected_version=STUDENT_DATASET_FORMAT_VERSION,
        artifact=artifacts["dataset_manifest"].path,
    )
    _validate_dataset_artifacts(
        artifacts["dataset_manifest"].path,
        manifest,
        hash_file=dependencies.hash_file,
    )
    return artifacts["dataset_manifest"]


def _build_dataset_cache(
    config: PipelineConfigProtocol,
    statuses: Sequence[VideoCacheStatus],
    dataset_dir: Path,
    dataset_key: str,
    dependencies: PipelineDependencies,
) -> _Artifact:
    if any(
        item.teacher_manifest is None or item.teacher_manifest_sha256 is None for item in statuses
    ):
        raise PipelineError("cannot build student dataset before every teacher cache is complete")
    staging = _new_staging_dir(dataset_dir, dependencies)
    try:
        dependencies.builder(
            source_manifest_paths=[item.teacher_manifest for item in statuses],
            output_dir=staging,
            frame_transfer_mode=str(config.dataset.frame_transfer_mode),
            expected_source_manifest_sha256s=[item.teacher_manifest_sha256 for item in statuses],
            validation_tail_fraction=config.dataset.validation_tail_fraction,
        )
        manifest_path = staging / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise PipelineError(f"dataset builder did not create {manifest_path}")
        manifest = _read_json(manifest_path)
        if manifest.get("format") != STUDENT_DATASET_FORMAT:
            raise PipelineError(f"dataset builder returned an unexpected format: {manifest_path}")
        _require_manifest_format(
            manifest,
            expected_format=STUDENT_DATASET_FORMAT,
            expected_version=STUDENT_DATASET_FORMAT_VERSION,
            artifact=manifest_path,
        )
        _validate_dataset_artifacts(
            manifest_path,
            manifest,
            hash_file=dependencies.hash_file,
        )
        if _dataset_cache_key(config, statuses, dependencies) != dataset_key:
            raise PipelineError(
                "pipeline implementation or dependencies changed during dataset construction"
            )
        _write_complete(
            staging,
            kind="dataset",
            cache_key=dataset_key,
            artifacts={"dataset_manifest": manifest_path},
            metadata={
                "video_cache_keys": [item.cache_key for item in statuses],
                "source_manifest_sha256s": [item.teacher_manifest_sha256 for item in statuses],
            },
            hash_file=dependencies.hash_file,
        )
        _inspect_complete(
            staging,
            kind="dataset",
            cache_key=dataset_key,
            artifact_names=("dataset_manifest",),
            hash_file=dependencies.hash_file,
        )
        _publish_staging(staging, dataset_dir)
    except Exception:
        _cleanup_staging(staging)
        raise
    artifact = _inspect_dataset_cache(dataset_dir, dataset_key, dependencies)
    if artifact is None:  # pragma: no cover - publish and inspect are adjacent.
        raise PipelineError(f"published dataset cache disappeared: {dataset_dir}")
    return artifact


def _inspect_run_cache(
    run_dir: Path,
    run_key: str,
    dataset_manifest_sha256: str | None,
    dependencies: PipelineDependencies,
) -> _Artifact | None:
    artifacts = _inspect_complete(
        run_dir,
        kind="run",
        cache_key=run_key,
        artifact_names=("run_manifest",),
        hash_file=dependencies.hash_file,
    )
    if artifacts is None:
        return None
    manifest = _read_json(artifacts["run_manifest"].path)
    dataset = manifest.get("dataset")
    _require_manifest_format(
        manifest,
        expected_format=TRAINING_RUN_FORMAT,
        expected_version=TRAINING_RUN_FORMAT_VERSION,
        artifact=artifacts["run_manifest"].path,
    )
    _validate_run_artifacts(
        artifacts["run_manifest"].path,
        manifest,
        hash_file=dependencies.hash_file,
    )
    if dataset_manifest_sha256 is not None and (
        not isinstance(dataset, Mapping)
        or dataset.get("manifest_sha256") != dataset_manifest_sha256
    ):
        raise PipelineError(f"training run references the wrong dataset: {run_dir}")
    return artifacts["run_manifest"]


def _build_run(
    config: PipelineConfigProtocol,
    *,
    dataset_manifest: _Artifact,
    dataset_key: str,
    base_run_key: str,
    run_dir: Path,
    run_key: str,
    dependencies: PipelineDependencies,
) -> _Artifact:
    staging = _new_staging_dir(run_dir, dependencies)
    try:
        dependencies.trainer(
            dataset_manifest_path=dataset_manifest.path,
            expected_dataset_manifest_sha256=dataset_manifest.sha256,
            output_dir=staging,
            model_config=config.training.model_config(),
            training_config=config.training.training_config(),
            spike_filter_config=config.training.spike_filter_config(),
            device_name=str(config.training.device),
            progress=None,
        )
        manifest_path = staging / "run_manifest.json"
        if not manifest_path.is_file():
            raise PipelineError(f"trainer did not create {manifest_path}")
        manifest = _read_json(manifest_path)
        dataset = manifest.get("dataset")
        _require_manifest_format(
            manifest,
            expected_format=TRAINING_RUN_FORMAT,
            expected_version=TRAINING_RUN_FORMAT_VERSION,
            artifact=manifest_path,
        )
        _validate_run_artifacts(
            manifest_path,
            manifest,
            hash_file=dependencies.hash_file,
        )
        if (
            not isinstance(dataset, Mapping)
            or dataset.get("manifest_sha256") != dataset_manifest.sha256
        ):
            raise PipelineError("trainer run manifest references the wrong dataset hash")
        if _run_cache_key(config, dataset_key, dependencies) != base_run_key:
            raise PipelineError(
                "pipeline implementation or dependencies changed during model training"
            )
        _write_complete(
            staging,
            kind="run",
            cache_key=run_key,
            artifacts={"run_manifest": manifest_path},
            metadata={"dataset_manifest_sha256": dataset_manifest.sha256},
            hash_file=dependencies.hash_file,
        )
        _inspect_complete(
            staging,
            kind="run",
            cache_key=run_key,
            artifact_names=("run_manifest",),
            hash_file=dependencies.hash_file,
        )
        _publish_staging(staging, run_dir)
    except Exception:
        _cleanup_staging(staging)
        raise
    artifact = _inspect_run_cache(
        run_dir,
        run_key,
        dataset_manifest.sha256,
        dependencies,
    )
    if artifact is None:  # pragma: no cover - publish and inspect are adjacent.
        raise PipelineError(f"published run disappeared: {run_dir}")
    return artifact


def _warnings(config: PipelineConfigProtocol) -> tuple[str, ...]:
    warnings: list[str] = []
    if config.teacher.teacher_selection_report is not None:
        warnings.append(
            "teacher_selection_report is ignored: model-selection evidence is scoped "
            "to its original video and is never reused for newly discovered videos"
        )
    if config.dataset.validation_tail_fraction is not None:
        warnings.append(
            "validation_tail_fraction evaluates nearby frames from the same videos; "
            "it does not measure held-out-video generalization"
        )
    return tuple(warnings)


def _planned_actions(
    statuses: Sequence[VideoCacheStatus],
    *,
    dataset_hit: bool,
    run_hit: bool,
    force_train: bool,
) -> tuple[str, ...]:
    if run_hit and not force_train:
        return ("reuse completed training run",)
    actions: list[str] = []
    if dataset_hit:
        actions.append("reuse completed student dataset")
    else:
        for item in statuses:
            if not item.cache_hit:
                actions.append(f"prepare and pseudo-label video {item.sequence_id}")
        actions.append("build student dataset")
    actions.append("train student model" if not force_train else "force a new student training run")
    return tuple(actions)


def _orchestrate(
    config: PipelineConfigProtocol,
    videos: Sequence[DiscoveredVideoProtocol] | None,
    *,
    operation: str,
    dry_run: bool,
    force_train: bool,
    dependencies: PipelineDependencies,
) -> PipelineSummary:
    discovered = _validated_videos(config, videos, dependencies)
    statuses = tuple(
        _inspect_video_cache(
            config,
            video,
            _video_cache_key(config, video, dependencies),
            dependencies,
        )
        for video in discovered
    )
    dataset_key = _dataset_cache_key(config, statuses, dependencies)
    dataset_dir = Path(config.output.dataset_dir) / dataset_key
    dataset_artifact = _inspect_dataset_cache(dataset_dir, dataset_key, dependencies)
    dataset_cache_hit = dataset_artifact is not None
    run_key = _run_cache_key(config, dataset_key, dependencies)
    base_run_dir = Path(config.output.runs_dir) / run_key
    base_run_artifact = _inspect_run_cache(
        base_run_dir,
        run_key,
        None if dataset_artifact is None else dataset_artifact.sha256,
        dependencies,
    )
    actions = _planned_actions(
        statuses,
        dataset_hit=dataset_artifact is not None,
        run_hit=base_run_artifact is not None,
        force_train=force_train,
    )

    if operation == "status" or dry_run:
        status_name = (
            "ready"
            if operation == "status" and base_run_artifact is not None
            else "needs-work"
            if operation == "status"
            else "dry-run"
        )
        return PipelineSummary(
            operation=operation,
            status=status_name,
            dry_run=dry_run,
            force_train=force_train,
            videos=statuses,
            dataset_key=dataset_key,
            dataset_cache_hit=dataset_cache_hit,
            dataset_dir=dataset_dir,
            dataset_manifest=(None if dataset_artifact is None else dataset_artifact.path),
            dataset_manifest_sha256=(None if dataset_artifact is None else dataset_artifact.sha256),
            run_key=run_key,
            run_cache_hit=base_run_artifact is not None,
            run_dir=base_run_dir,
            run_manifest=(None if base_run_artifact is None else base_run_artifact.path),
            run_manifest_sha256=(None if base_run_artifact is None else base_run_artifact.sha256),
            actions=actions,
            warnings=_warnings(config),
        )

    if base_run_artifact is not None and not force_train:
        return PipelineSummary(
            operation="run",
            status="cached",
            dry_run=False,
            force_train=False,
            videos=statuses,
            dataset_key=dataset_key,
            dataset_cache_hit=dataset_cache_hit,
            dataset_dir=dataset_dir,
            dataset_manifest=(None if dataset_artifact is None else dataset_artifact.path),
            dataset_manifest_sha256=(None if dataset_artifact is None else dataset_artifact.sha256),
            run_key=run_key,
            run_cache_hit=True,
            run_dir=base_run_dir,
            run_manifest=base_run_artifact.path,
            run_manifest_sha256=base_run_artifact.sha256,
            actions=actions,
            warnings=_warnings(config),
        )

    materialized_statuses = list(statuses)
    if dataset_artifact is None:
        for index, (video, status) in enumerate(zip(discovered, statuses, strict=True)):
            if not status.cache_hit:
                built_status = _build_video_cache(config, video, status, dependencies)
                materialized_statuses[index] = dataclasses.replace(built_status, cache_hit=False)
        dataset_artifact = _build_dataset_cache(
            config,
            materialized_statuses,
            dataset_dir,
            dataset_key,
            dependencies,
        )

    actual_run_key = run_key
    run_dir = base_run_dir
    if force_train and base_run_dir.exists():
        nonce = dependencies.nonce_factory()
        actual_run_key = _content_key({"base_run_key": run_key, "forced_nonce": nonce})
        run_dir = Path(config.output.runs_dir) / actual_run_key
    run_artifact = _build_run(
        config,
        dataset_manifest=dataset_artifact,
        run_dir=run_dir,
        dataset_key=dataset_key,
        base_run_key=run_key,
        run_key=actual_run_key,
        dependencies=dependencies,
    )
    return PipelineSummary(
        operation="run",
        status="trained",
        dry_run=False,
        force_train=force_train,
        videos=tuple(materialized_statuses),
        dataset_key=dataset_key,
        dataset_cache_hit=dataset_cache_hit,
        dataset_dir=dataset_dir,
        dataset_manifest=dataset_artifact.path,
        dataset_manifest_sha256=dataset_artifact.sha256,
        run_key=actual_run_key,
        run_cache_hit=False,
        run_dir=run_dir,
        run_manifest=run_artifact.path,
        run_manifest_sha256=run_artifact.sha256,
        actions=actions,
        warnings=_warnings(config),
    )


def run_pipeline(
    config: PipelineConfigProtocol,
    videos: Sequence[DiscoveredVideoProtocol] | None = None,
    *,
    dry_run: bool = False,
    force_train: bool = False,
    dependencies: PipelineDependencies | None = None,
) -> PipelineSummary:
    """Discover/process videos and train, or report the work with ``dry_run``."""

    return _orchestrate(
        config,
        videos,
        operation="run",
        dry_run=dry_run,
        force_train=force_train,
        dependencies=dependencies or PipelineDependencies(),
    )


def pipeline_status(
    config: PipelineConfigProtocol,
    videos: Sequence[DiscoveredVideoProtocol] | None = None,
    *,
    dependencies: PipelineDependencies | None = None,
) -> PipelineSummary:
    """Inspect and hash-verify caches without creating directories or running work."""

    return _orchestrate(
        config,
        videos,
        operation="status",
        dry_run=False,
        force_train=False,
        dependencies=dependencies or PipelineDependencies(),
    )


__all__ = [
    "COMPLETE_FILENAME",
    "PipelineDependencies",
    "PipelineError",
    "PipelineSummary",
    "VideoCacheStatus",
    "pipeline_status",
    "run_pipeline",
]
