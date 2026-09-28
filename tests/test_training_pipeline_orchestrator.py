from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fingertip_depth.pseudo_label_inputs import (
    PREPARED_INPUT_FORMAT,
    PREPARED_INPUT_FORMAT_VERSION,
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
)
from fingertip_depth.student_training import (
    TRAINING_RUN_FORMAT,
    TRAINING_RUN_FORMAT_VERSION,
)
from fingertip_depth.training_pipeline.orchestrator import (
    COMPLETE_FILENAME,
    VIDEO_WORKSPACE_SUFFIX,
    PipelineDependencies,
    PipelineError,
    _preparation_implementation_hashes,
    pipeline_status,
    run_pipeline,
)
from fingertip_depth.video_cache import sha256_file


@dataclass(frozen=True)
class _Video:
    path: Path
    relative_path: Path
    split: str
    sequence_id: str
    source_sha256: str
    size_bytes: int
    focal_35mm_mm: float = 36.0


class _Training:
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.model = {"encoder": "tiny"}
        self.optimizer = {"epochs": 1}
        self.spike_filter = {"enabled": True}

    def model_config(self) -> object:
        return self.model

    def training_config(self) -> object:
        return self.optimizer

    def spike_filter_config(self) -> object:
        return self.spike_filter


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _artifact(root: Path, relative_path: str, content: bytes) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"relative_path": relative_path, "sha256": sha256_file(path)}


def _fixture(
    tmp_path: Path, *, selection_report: bool = False
) -> tuple[object, tuple[_Video, ...]]:
    hand_model = tmp_path / "assets" / "hand.task"
    hand_model.parent.mkdir(parents=True)
    hand_model.write_bytes(b"hand-model")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake-root'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    depth_project = tmp_path / "environments" / "depth_pro"
    depth_project.mkdir(parents=True)
    (depth_project / "pyproject.toml").write_text("[project]\nname='depth-pro'\n")
    (depth_project / "uv.lock").write_text("version = 1\n")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for script_name in (
        "prepare_pseudo_label_inputs.py",
        "generate_depth_pro_pseudo_labels.py",
        "build_student_dataset.py",
        "train_student_transformer.py",
    ):
        (scripts_dir / script_name).write_text(f"# fake {script_name}\n")
    report = tmp_path / "old-selection-report.json"
    if selection_report:
        report.write_text("{}")

    videos: list[_Video] = []
    for split, name, content in (
        ("train", "train.mov", b"train-video"),
        ("validation", "validation.mov", b"validation-video"),
    ):
        path = tmp_path / "data" / "videos" / split / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        videos.append(
            _Video(
                path=path,
                relative_path=path.relative_to(tmp_path),
                split=split,
                sequence_id=f"{Path(name).stem}-{sha256_file(path)[:12]}",
                source_sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )

    config = SimpleNamespace(
        project_root=tmp_path,
        output=SimpleNamespace(
            processed_dir=tmp_path / "data" / "processed",
            dataset_dir=tmp_path / "data" / "datasets",
            runs_dir=tmp_path / "outputs" / "training",
        ),
        prepare=SimpleNamespace(
            hand_model_path=hand_model,
            landmark_indices=(5, 6, 7, 8),
            frame_transfer_mode="hardlink",
            max_frames=None,
        ),
        teacher=SimpleNamespace(
            depth_pro_project=depth_project,
            device="cuda:0",
            frame_transfer_mode="hardlink",
            max_frames=None,
            checkpoint_interval_frames=100,
            teacher_selection_report=report if selection_report else None,
        ),
        dataset=SimpleNamespace(
            frame_transfer_mode="hardlink",
            validation_tail_fraction=None,
        ),
        training=_Training(),
    )
    return config, tuple(videos)


class _Fakes:
    def __init__(self, *, missing_stage_payload: str | None = None) -> None:
        self.prepare_calls: list[dict[str, Any]] = []
        self.commands: list[tuple[list[str], Path]] = []
        self.builder_calls: list[dict[str, Any]] = []
        self.trainer_calls: list[dict[str, Any]] = []
        self.missing_stage_payload = missing_stage_payload
        self._nonce = 0
        self.project_root: Path | None = None
        self.pipeline_dependencies: PipelineDependencies | None = None

    def nonce(self) -> str:
        self._nonce += 1
        return f"nonce{self._nonce}"

    def prepare(self, **kwargs: Any) -> dict[str, Any]:
        self.prepare_calls.append(kwargs)
        assert self.project_root is not None
        assert self.pipeline_dependencies is not None
        source = Path(kwargs["input_video_path"])
        output = Path(kwargs["output_dir"])
        frame = _artifact(output, "frames/frame_000000.png", b"prepared-frame")
        frame_record = {
            "frame_index": 0,
            "image_path": frame["relative_path"],
            "png_sha256": frame["sha256"],
        }
        frames_jsonl = _artifact(
            output,
            "frames.jsonl",
            (json.dumps(frame_record, sort_keys=True) + "\n").encode(),
        )
        manifest = {
            "format": PREPARED_INPUT_FORMAT,
            "format_version": PREPARED_INPUT_FORMAT_VERSION,
            "source": {
                "mode": "direct_video_decode",
                "video_path": str(source),
                "video_sha256": sha256_file(source),
            },
            "feature_landmarks": {"indices": list(kwargs["feature_landmark_indices"])},
            "hand_landmarker": {
                "model_path": str(kwargs["hand_model_path"]),
                "model_sha256": sha256_file(Path(kwargs["hand_model_path"])),
            },
            "frames_jsonl": frames_jsonl,
            "frames": {"relative_directory": "frames"},
            "frame_count": 1,
            "camera": {"focal_35mm_equivalent_mm": kwargs["focal_35mm_equivalent_mm"]},
            "processing": {"max_frames": kwargs["max_frames"]},
            "provenance": {
                "implementation_sha256": _preparation_implementation_hashes(
                    self.project_root, self.pipeline_dependencies
                ),
            },
        }
        _write_json(output / "manifest.json", manifest)
        if self.missing_stage_payload == "prepared":
            (output / str(frame["relative_path"])).unlink()
        return manifest

    def command(self, command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        self.commands.append((command, cwd))

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        prepared = Path(value("--prepared-manifest"))
        output = Path(value("--output-dir"))
        sequence_id = value("--sequence-id")
        split = value("--split")
        image_artifact = _artifact(
            output,
            f"frames/{sequence_id}/frame_000000.png",
            b"teacher-frame",
        )
        image = {
            "relative_path": image_artifact["relative_path"],
            "png_sha256": image_artifact["sha256"],
        }
        sample = {
            "sample_id": f"{sequence_id}:000000:hand0",
            "sequence_id": sequence_id,
            "split": split,
            "image": image,
        }
        artifacts = {
            "samples_jsonl": _artifact(
                output,
                "samples.jsonl",
                (json.dumps(sample, sort_keys=True) + "\n").encode(),
            ),
            "rejections_jsonl": _artifact(output, "rejections.jsonl", b""),
            "targets_csv": _artifact(output, "targets.csv", b"sample_id,z_teacher_m\n"),
            "split": _artifact(
                output,
                f"splits/{split}.txt",
                f"{sample['sample_id']}\n".encode(),
            ),
            "trajectory_csv": _artifact(
                output,
                f"trajectories/{sequence_id}.csv",
                b"frame_index,x_m,y_m,z_m\n",
            ),
            "trajectory_ply": _artifact(
                output,
                f"trajectories/{sequence_id}.ply",
                b"ply\n",
            ),
            "trajectory_views_png": _artifact(
                output,
                f"visualizations/{sequence_id}_views.png",
                b"teacher-visualization",
            ),
        }
        manifest = {
            "format": PSEUDO_LABEL_DATASET_FORMAT,
            "format_version": PSEUDO_LABEL_DATASET_FORMAT_VERSION,
            "sequence": {"id": sequence_id, "split": split},
            "prepared_inputs": {"manifest_sha256": sha256_file(prepared)},
            "artifacts": artifacts,
        }
        _write_json(output / "dataset_manifest.json", manifest)
        if self.missing_stage_payload == "teacher":
            (output / str(image["relative_path"])).unlink()
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    def builder(self, **kwargs: Any) -> dict[str, Any]:
        self.builder_calls.append(kwargs)
        sources = [Path(path) for path in kwargs["source_manifest_paths"]]
        assert all(".staging-" not in str(path) for path in sources)
        assert [sha256_file(path) for path in sources] == list(
            kwargs["expected_source_manifest_sha256s"]
        )
        output = Path(kwargs["output_dir"])
        source_manifests = [json.loads(path.read_text()) for path in sources]
        rows: list[dict[str, object]] = []
        student_image_paths: list[Path] = []
        for source in source_manifests:
            sequence_id = str(source["sequence"]["id"])
            image_artifact = _artifact(
                output,
                f"images/{sequence_id}/frame_000000.png",
                f"student-frame:{sequence_id}".encode(),
            )
            image = {
                "relative_path": image_artifact["relative_path"],
                "png_sha256": image_artifact["sha256"],
            }
            student_image_paths.append(output / str(image["relative_path"]))
            rows.append(
                {
                    "sample_id": f"student:{source['sequence']['id']}",
                    "source_sequence_id": source["sequence"]["id"],
                    "split": source["sequence"]["split"],
                    "image": image,
                }
            )
        samples_jsonl = _artifact(
            output,
            "samples.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(),
        )
        train_ids = [row["sample_id"] for row in rows if row["split"] == "train"]
        validation_ids = [row["sample_id"] for row in rows if row["split"] == "validation"]
        artifacts = {
            "samples_jsonl": samples_jsonl,
            "targets_csv": _artifact(output, "targets.csv", b"sample_id,z_teacher_m\n"),
            "train_split": _artifact(
                output,
                "splits/train.txt",
                "".join(f"{sample_id}\n" for sample_id in train_ids).encode(),
            ),
            "validation_split": _artifact(
                output,
                "splits/validation.txt",
                "".join(f"{sample_id}\n" for sample_id in validation_ids).encode(),
            ),
        }
        manifest = {
            "format": STUDENT_DATASET_FORMAT,
            "format_version": STUDENT_DATASET_FORMAT_VERSION,
            "sources": source_manifests,
            "artifacts": artifacts,
        }
        _write_json(output / "dataset_manifest.json", manifest)
        if self.missing_stage_payload == "dataset":
            student_image_paths[0].unlink()
        return manifest

    def trainer(self, **kwargs: Any) -> dict[str, Any]:
        self.trainer_calls.append(kwargs)
        dataset_manifest = Path(kwargs["dataset_manifest_path"])
        assert ".staging-" not in str(dataset_manifest)
        assert sha256_file(dataset_manifest) == kwargs["expected_dataset_manifest_sha256"]
        output = Path(kwargs["output_dir"])
        artifacts = {
            "best_checkpoint": _artifact(output, "best_checkpoint.pt", b"best-checkpoint"),
            "last_checkpoint": _artifact(output, "last_checkpoint.pt", b"last-checkpoint"),
            "history": _artifact(output, "history.jsonl", b'{"epoch":1}\n'),
            "validation_predictions_filtered": _artifact(
                output,
                "validation_predictions_filtered.csv",
                b"sample_id,prediction_m\n",
            ),
            "validation_predictions_raw": _artifact(
                output,
                "validation_predictions_raw.csv",
                b"sample_id,prediction_m\n",
            ),
            "teacher_spike_filter": _artifact(
                output,
                "teacher_spike_filter.json",
                b"{}\n",
            ),
            "identity_decisions": _artifact(
                output,
                "teacher_spike_filter_decisions.jsonl",
                b"",
            ),
            "included_train": _artifact(output, "included_train.txt", b"train\n"),
            "included_validation": _artifact(
                output,
                "included_validation.txt",
                b"validation\n",
            ),
        }
        manifest = {
            "format": TRAINING_RUN_FORMAT,
            "format_version": TRAINING_RUN_FORMAT_VERSION,
            "dataset": {
                "manifest_path": str(dataset_manifest),
                "manifest_sha256": kwargs["expected_dataset_manifest_sha256"],
            },
            "artifacts": artifacts,
        }
        _write_json(output / "run_manifest.json", manifest)
        if self.missing_stage_payload == "run":
            (output / str(artifacts["best_checkpoint"]["relative_path"])).unlink()
        return manifest

    def dependencies(self, videos: tuple[_Video, ...]) -> PipelineDependencies:
        dependencies = PipelineDependencies(
            discover=lambda _config: videos,
            prepare=self.prepare,
            builder=self.builder,
            trainer=self.trainer,
            command_runner=self.command,
            nonce_factory=self.nonce,
        )
        project_root = videos[0].path.parents[3]
        self.project_root = project_root
        self.pipeline_dependencies = dependencies
        return dependencies


def _cache_payload(summary: Any, kind: str) -> Path:
    if kind == "prepared":
        return summary.videos[0].cache_dir / "prepared" / "frames" / "frame_000000.png"
    if kind == "teacher":
        return summary.videos[0].cache_dir / "teacher" / "samples.jsonl"
    if kind == "dataset":
        return summary.dataset_dir / "samples.jsonl"
    if kind == "run":
        return summary.run_dir / "best_checkpoint.pt"
    raise AssertionError(f"unsupported cache kind: {kind}")


def test_full_pipeline_is_atomic_incremental_and_noops_on_second_run(tmp_path: Path) -> None:
    config, videos = _fixture(tmp_path, selection_report=True)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    first = run_pipeline(config, dependencies=dependencies)

    assert first.status == "trained"
    assert first.video_cache_hits == 0
    assert first.dataset_cache_hit is False
    assert first.dataset_manifest_sha256 == sha256_file(first.dataset_manifest)  # type: ignore[arg-type]
    assert first.run_manifest_sha256 == sha256_file(first.run_manifest)  # type: ignore[arg-type]
    assert len(fakes.prepare_calls) == 2
    assert len(fakes.commands) == 2
    assert len(fakes.builder_calls) == 1
    assert len(fakes.trainer_calls) == 1
    assert all("--teacher-selection-report" not in command for command, _cwd in fakes.commands)
    assert all(
        command[command.index("--checkpoint-interval-frames") + 1] == "100"
        for command, _cwd in fakes.commands
    )
    assert all((item.cache_dir / COMPLETE_FILENAME).is_file() for item in first.videos)
    assert (first.dataset_dir / COMPLETE_FILENAME).is_file()
    assert (first.run_dir / COMPLETE_FILENAME).is_file()
    assert not list(tmp_path.rglob("*.staging-*"))
    assert any("teacher_selection_report is ignored" in warning for warning in first.warnings)

    second = run_pipeline(config, dependencies=dependencies)

    assert second.status == "cached"
    assert second.video_cache_hits == 2
    assert second.dataset_cache_hit is True
    assert second.run_cache_hit is True
    assert len(fakes.prepare_calls) == 2
    assert len(fakes.commands) == 2
    assert len(fakes.builder_calls) == 1
    assert len(fakes.trainer_calls) == 1

    third_path = tmp_path / "data" / "videos" / "train" / "third.mp4"
    third_path.write_bytes(b"third-video")
    third = _Video(
        path=third_path,
        relative_path=third_path.relative_to(tmp_path),
        split="train",
        sequence_id=f"third-{sha256_file(third_path)[:12]}",
        source_sha256=sha256_file(third_path),
        size_bytes=third_path.stat().st_size,
    )

    third_run = run_pipeline(
        config,
        dependencies=fakes.dependencies((videos[0], third, videos[1])),
    )

    assert third_run.status == "trained"
    assert third_run.video_cache_hits == 2
    assert third_run.video_cache_misses == 1
    assert third_run.dataset_cache_hit is False
    assert len(fakes.prepare_calls) == 3
    assert len(fakes.commands) == 3
    assert len(fakes.builder_calls) == 2
    assert len(fakes.trainer_calls) == 2


def test_dry_run_and_status_do_not_create_output_directories(tmp_path: Path) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    dry_run = run_pipeline(config, dry_run=True, dependencies=dependencies)
    status = pipeline_status(config, dependencies=dependencies)

    assert dry_run.status == "dry-run"
    assert status.status == "needs-work"
    assert dry_run.video_cache_misses == 2
    assert dry_run.actions[-1] == "train student model"
    assert not Path(config.output.processed_dir).exists()
    assert not Path(config.output.dataset_dir).exists()
    assert not Path(config.output.runs_dir).exists()
    assert not fakes.prepare_calls
    assert not fakes.commands


def test_force_train_reuses_data_and_publishes_a_new_immutable_run(tmp_path: Path) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)
    original = run_pipeline(config, dependencies=dependencies)

    forced = run_pipeline(config, force_train=True, dependencies=dependencies)

    assert forced.status == "trained"
    assert forced.force_train is True
    assert forced.run_dir != original.run_dir
    assert forced.run_manifest_sha256 == original.run_manifest_sha256
    assert len(fakes.prepare_calls) == 2
    assert len(fakes.commands) == 2
    assert len(fakes.builder_calls) == 1
    assert len(fakes.trainer_calls) == 2


def test_completed_manifest_is_rehashed_before_cache_reuse(tmp_path: Path) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)
    first = run_pipeline(config, dependencies=dependencies)
    assert first.dataset_manifest is not None
    first.dataset_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(PipelineError, match="SHA-256 mismatch"):
        pipeline_status(config, dependencies=dependencies)


@pytest.mark.parametrize("kind", ("prepared", "teacher", "dataset", "run"))
def test_completed_cache_rejects_missing_payload(tmp_path: Path, kind: str) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)
    completed = run_pipeline(config, dependencies=dependencies)
    payload = _cache_payload(completed, kind)
    assert payload.is_file()

    payload.unlink()

    with pytest.raises(PipelineError, match="file inventory mismatch"):
        pipeline_status(config, dependencies=dependencies)


@pytest.mark.parametrize("kind", ("prepared", "teacher", "dataset", "run"))
def test_completed_cache_rejects_same_size_payload_tamper(
    tmp_path: Path,
    kind: str,
) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)
    completed = run_pipeline(config, dependencies=dependencies)
    payload = _cache_payload(completed, kind)
    original = bytearray(payload.read_bytes())
    assert original

    original[0] ^= 0x01
    payload.write_bytes(original)

    with pytest.raises(PipelineError, match="SHA-256 mismatch"):
        pipeline_status(config, dependencies=dependencies)


def test_source_video_change_during_processing_aborts_before_publication(
    tmp_path: Path,
) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    original_prepare = fakes.prepare

    def prepare_then_mutate_source(**kwargs: Any) -> dict[str, Any]:
        manifest = original_prepare(**kwargs)
        videos[0].path.write_bytes(b"TRAIN-video")
        return manifest

    dependencies = replace(
        fakes.dependencies(videos),
        prepare=prepare_then_mutate_source,
    )

    with pytest.raises(
        PipelineError,
        match="after prepared-input generation: source video SHA-256 changed",
    ):
        run_pipeline(config, dependencies=dependencies)

    assert not list(Path(config.output.processed_dir).rglob(COMPLETE_FILENAME))


def test_prepare_stage_rejects_wrong_source_video_sha256(tmp_path: Path) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    original_prepare = fakes.prepare

    def prepare_with_wrong_source(**kwargs: Any) -> dict[str, Any]:
        manifest = original_prepare(**kwargs)
        manifest["source"]["video_sha256"] = "0" * 64
        _write_json(Path(kwargs["output_dir"]) / "manifest.json", manifest)
        return manifest

    dependencies = replace(
        fakes.dependencies(videos),
        prepare=prepare_with_wrong_source,
    )

    with pytest.raises(PipelineError, match="wrong source video hash"):
        run_pipeline(config, dependencies=dependencies)

    assert not list(Path(config.output.processed_dir).rglob(COMPLETE_FILENAME))


@pytest.mark.parametrize(
    (
        "script_name",
        "expected_video_hits",
        "expected_dataset_hit",
        "expected_prepare_calls",
        "expected_teacher_calls",
        "expected_builder_calls",
    ),
    (
        ("prepare_pseudo_label_inputs.py", 0, False, 4, 4, 2),
        ("generate_depth_pro_pseudo_labels.py", 0, False, 4, 4, 2),
        ("build_student_dataset.py", 2, False, 2, 2, 2),
        ("train_student_transformer.py", 2, True, 2, 2, 1),
    ),
)
def test_wrapper_script_change_invalidates_affected_cache_keys(
    tmp_path: Path,
    script_name: str,
    expected_video_hits: int,
    expected_dataset_hit: bool,
    expected_prepare_calls: int,
    expected_teacher_calls: int,
    expected_builder_calls: int,
) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)
    first = run_pipeline(config, dependencies=dependencies)
    wrapper = Path(config.project_root) / "scripts" / script_name
    wrapper.write_text(wrapper.read_text() + "# implementation changed\n")

    second = run_pipeline(config, dependencies=dependencies)

    assert second.status == "trained"
    assert second.video_cache_hits == expected_video_hits
    assert second.dataset_cache_hit is expected_dataset_hit
    assert second.run_cache_hit is False
    assert len(fakes.prepare_calls) == expected_prepare_calls
    assert len(fakes.commands) == expected_teacher_calls
    assert len(fakes.builder_calls) == expected_builder_calls
    assert len(fakes.trainer_calls) == 2
    assert second.run_key != first.run_key
    assert (second.dataset_key == first.dataset_key) is expected_dataset_hit
    assert (
        [item.cache_key for item in second.videos] == [item.cache_key for item in first.videos]
    ) is (expected_video_hits == len(videos))


@pytest.mark.parametrize(
    ("stage", "output_name"),
    (
        ("prepared", "processed_dir"),
        ("teacher", "processed_dir"),
        ("dataset", "dataset_dir"),
        ("run", "runs_dir"),
    ),
)
def test_stage_rejects_declared_missing_payload_before_complete_publication(
    tmp_path: Path,
    stage: str,
    output_name: str,
) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes(missing_stage_payload=stage)
    dependencies = fakes.dependencies(videos)

    with pytest.raises(PipelineError, match="missing or is not a regular file"):
        run_pipeline(config, dependencies=dependencies)

    failed_output = Path(getattr(config.output, output_name))
    assert not list(failed_output.rglob(COMPLETE_FILENAME))
    assert not list(failed_output.rglob("*.staging-*"))
    if stage in {"dataset", "run"}:
        assert len(list(Path(config.output.processed_dir).rglob(COMPLETE_FILENAME))) == len(videos)
    if stage == "run":
        assert len(list(Path(config.output.dataset_dir).rglob(COMPLETE_FILENAME))) == 1


def test_failed_teacher_subprocess_never_publishes_partial_cache(tmp_path: Path) -> None:
    config, videos = _fixture(tmp_path)
    fakes = _Fakes()

    def fail(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        del cwd
        return subprocess.CompletedProcess(command, 17, stdout="", stderr="teacher failed")

    dependencies = replace(
        fakes.dependencies(videos),
        command_runner=fail,
    )
    with pytest.raises(PipelineError, match="exit code 17"):
        run_pipeline(config, dependencies=dependencies)

    processed = Path(config.output.processed_dir)
    assert not list(processed.rglob(COMPLETE_FILENAME))
    assert not list(processed.rglob("*.staging-*"))
    assert len(fakes.prepare_calls) == 1
    workspaces = [
        path
        for path in processed.rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    ]
    assert len(workspaces) == 1
    assert (workspaces[0] / "prepared" / "manifest.json").is_file()

    completed = run_pipeline(config, dependencies=fakes.dependencies(videos))

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == len(videos)
    assert not [
        path
        for path in processed.rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    ]


def test_teacher_keyboard_interrupt_reuses_prepared_and_resumes_workspace(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_with_progress(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / PSEUDO_LABEL_PROGRESS_FILENAME).write_text(
            '{"fake":"progress"}\n', encoding="utf-8"
        )
        raise KeyboardInterrupt

    interrupted = replace(dependencies, command_runner=interrupt_with_progress)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(config, dependencies=interrupted)

    workspaces = [
        path
        for path in Path(config.output.processed_dir).rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    ]
    assert len(workspaces) == 1
    assert (workspaces[0] / "prepared" / "manifest.json").is_file()
    assert len(fakes.prepare_calls) == 1
    first_output = fakes.commands[0][0][fakes.commands[0][0].index("--output-dir") + 1]
    assert "--resume" not in fakes.commands[0][0]

    completed = run_pipeline(
        config,
        dependencies=replace(dependencies, command_runner=fakes.command),
    )

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2
    assert "--resume" in fakes.commands[1][0]
    assert fakes.commands[1][0][fakes.commands[1][0].index("--output-dir") + 1] == (first_output)
    assert not [
        path
        for path in Path(config.output.processed_dir).rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    ]


def test_teacher_header_temp_only_retries_fresh_with_reused_prepared(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_during_header_publish(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME).write_text(
            '{"fake":"incomplete-header"}\n', encoding="utf-8"
        )
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=interrupt_during_header_publish,
            ),
        )

    def finish_fresh(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert "--resume" not in command
        output = Path(command[command.index("--output-dir") + 1])
        (output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME).unlink()
        return fakes.command(command, cwd=cwd)

    completed = run_pipeline(
        config,
        dependencies=replace(dependencies, command_runner=finish_fresh),
    )

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2


def test_teacher_published_header_with_linked_temp_resumes_and_cleans_temp(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_after_header_link(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
        progress_temp.write_text('{"fake":"valid-header"}\n', encoding="utf-8")
        os.link(progress_temp, output / PSEUDO_LABEL_PROGRESS_FILENAME)
        _artifact(output, "frames/partial.png", b"journaled-frame")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=interrupt_after_header_link,
            ),
        )
    first_output = Path(fakes.commands[0][0][fakes.commands[0][0].index("--output-dir") + 1])

    def finish_resume(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert "--resume" in command
        output = Path(command[command.index("--output-dir") + 1])
        assert output == first_output
        (output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME).unlink()
        return fakes.command(command, cwd=cwd)

    completed = run_pipeline(
        config,
        dependencies=replace(dependencies, command_runner=finish_resume),
    )

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2


@pytest.mark.parametrize("unsafe_kind", ("symlink", "coexisting-file"))
def test_teacher_header_temp_unsafe_states_fail_closed(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_with_unsafe_temp(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
        if unsafe_kind == "symlink":
            progress_temp.symlink_to("missing-progress-header")
        else:
            progress_temp.write_text('{"fake":"header"}\n', encoding="utf-8")
            (output / "unexpected.bin").write_bytes(b"unknown")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(dependencies, command_runner=interrupt_with_unsafe_temp),
        )
    with pytest.raises(PipelineError, match="progress temporary file"):
        run_pipeline(config, dependencies=dependencies)

    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 1


def test_incomplete_teacher_manifest_is_rebuilt_from_progress_journal(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_during_manifest_write(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / PSEUDO_LABEL_PROGRESS_FILENAME).write_text(
            '{"fake":"progress"}\n', encoding="utf-8"
        )
        (output / "dataset_manifest.json").write_text('{"format":', encoding="utf-8")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=interrupt_during_manifest_write,
            ),
        )

    completed = run_pipeline(config, dependencies=dependencies)

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2
    assert "--resume" not in fakes.commands[0][0]
    assert "--resume" in fakes.commands[1][0]


def test_incomplete_teacher_manifest_without_journal_fails_closed(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_during_manifest_write(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "dataset_manifest.json").write_text('{"format":', encoding="utf-8")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=interrupt_during_manifest_write,
            ),
        )
    with pytest.raises(PipelineError, match="no recognized progress journal"):
        run_pipeline(config, dependencies=dependencies)

    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 1


def test_tampered_prepared_workspace_fails_closed_without_repreparing(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_with_progress(command: list[str], *, cwd: Path) -> None:
        fakes.commands.append((command, cwd))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / PSEUDO_LABEL_PROGRESS_FILENAME).write_text(
            '{"fake":"progress"}\n', encoding="utf-8"
        )
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=interrupt_with_progress,
            ),
        )
    workspace = next(
        path
        for path in Path(config.output.processed_dir).rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    )
    (workspace / "prepared" / "frames" / "frame_000000.png").write_bytes(b"tampered-frame")

    with pytest.raises(PipelineError, match="SHA-256 mismatch"):
        run_pipeline(
            config,
            dependencies=replace(dependencies, command_runner=fakes.command),
        )

    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 1


def test_keyboard_interrupt_during_prepare_rebuilds_only_incomplete_preparation(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def interrupt_prepare(**kwargs: Any) -> None:
        fakes.prepare_calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        _artifact(output, "frames/frame_000000.png", b"partial-frame")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(dependencies, prepare=interrupt_prepare),
        )

    completed = run_pipeline(config, dependencies=dependencies)

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 2
    assert len(fakes.commands) == 1


def test_legacy_staging_adopts_only_prepared_and_quarantines_unjournaled_teacher(
    tmp_path: Path,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def complete_teacher_then_interrupt(command: list[str], *, cwd: Path) -> None:
        fakes.command(command, cwd=cwd)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=complete_teacher_then_interrupt,
            ),
        )
    workspace = next(
        path
        for path in Path(config.output.processed_dir).rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    )
    legacy_valid = workspace.parent / ".old-key.staging-valid"
    workspace.rename(legacy_valid)
    legacy_partial = workspace.parent / ".older-key.staging-partial"
    _artifact(legacy_partial, "prepared/frames/frame_000000.png", b"partial")

    completed = run_pipeline(config, dependencies=dependencies)

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2
    assert "--resume" not in fakes.commands[1][0]
    assert not list(workspace.parent.glob(".*.staging-*"))
    quarantines = list(workspace.parent.glob("*.legacy-ignored-*"))
    assert len(quarantines) == 2
    assert any((path / "teacher" / "dataset_manifest.json").is_file() for path in quarantines)


def test_legacy_journal_forces_current_teacher_resume_validation(tmp_path: Path) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def complete_teacher_with_journal_then_interrupt(command: list[str], *, cwd: Path) -> None:
        fakes.command(command, cwd=cwd)
        output = Path(command[command.index("--output-dir") + 1])
        progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
        progress_temp.write_text('{"fake":"progress"}\n', encoding="utf-8")
        os.link(progress_temp, output / PSEUDO_LABEL_PROGRESS_FILENAME)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=complete_teacher_with_journal_then_interrupt,
            ),
        )
    workspace = next(
        path
        for path in Path(config.output.processed_dir).rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    )
    workspace.rename(workspace.parent / ".old-key.staging-journaled")

    def finish_migrated_resume(
        command: list[str], *, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        assert "--resume" in command
        output = Path(command[command.index("--output-dir") + 1])
        (output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME).unlink()
        return fakes.command(command, cwd=cwd)

    completed = run_pipeline(
        config,
        dependencies=replace(dependencies, command_runner=finish_migrated_resume),
    )

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2
    assert "--resume" in fakes.commands[1][0]
    quarantines = list(workspace.parent.glob("*.legacy-ignored-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "teacher_dataset_manifest.unvalidated.json").is_file()


@pytest.mark.parametrize(
    "interrupted_phase",
    (
        "workspace-created",
        "prepared-moved",
        "teacher-manifest-quarantined",
        "teacher-moved",
        "legacy-quarantined",
    ),
)
def test_legacy_journal_migration_resumes_after_every_interruption_boundary(
    tmp_path: Path,
    interrupted_phase: str,
) -> None:
    config, all_videos = _fixture(tmp_path)
    videos = (all_videos[0],)
    fakes = _Fakes()
    dependencies = fakes.dependencies(videos)

    def complete_teacher_with_journal_then_interrupt(command: list[str], *, cwd: Path) -> None:
        fakes.command(command, cwd=cwd)
        output = Path(command[command.index("--output-dir") + 1])
        (output / PSEUDO_LABEL_PROGRESS_FILENAME).write_text(
            '{"fake":"progress"}\n', encoding="utf-8"
        )
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                command_runner=complete_teacher_with_journal_then_interrupt,
            ),
        )
    workspace = next(
        path
        for path in Path(config.output.processed_dir).rglob("*")
        if path.is_dir() and path.name.endswith(VIDEO_WORKSPACE_SUFFIX)
    )
    workspace.rename(workspace.parent / ".old-key.staging-journaled")

    def interrupt_migration(phase: str) -> None:
        if phase == interrupted_phase:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            config,
            dependencies=replace(
                dependencies,
                migration_checkpoint=interrupt_migration,
            ),
        )

    completed = run_pipeline(config, dependencies=dependencies)

    assert completed.status == "trained"
    assert len(fakes.prepare_calls) == 1
    assert len(fakes.commands) == 2
    assert "--resume" not in fakes.commands[0][0]
    assert "--resume" in fakes.commands[1][0]
    assert not list(workspace.parent.glob(".*.staging-*"))
