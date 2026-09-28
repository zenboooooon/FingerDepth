import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import fingertip_depth.pseudo_labels as pseudo_labels_module
from fingertip_depth.artifacts import write_json
from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.pseudo_labels import (
    DEPTH_PRO_TEACHER_CONDITION_ID,
    PSEUDO_LABEL_PROGRESS_FILENAME,
    PSEUDO_LABEL_PROGRESS_TEMP_FILENAME,
    generate_depth_pro_pseudo_labels,
)
from fingertip_depth.video_cache import (
    pixel_hash_sequence_sha256,
    pixel_sha256,
    sha256_file,
)


class _FakeDepthPro:
    def __init__(self) -> None:
        self.calls: list[tuple[CameraIntrinsics | None, str]] = []

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model": "fake Depth Pro",
            "checkpoint_sha256": "a" * 64,
        }

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> SimpleNamespace:
        self.calls.append((intrinsics, camera_mode))
        depth = np.full(rgb.shape[:2], 0.25, dtype=np.float32)
        depth[1, 2] = 0.30
        return SimpleNamespace(
            depth_m=depth,
            inference_ms=4.0,
            device="test",
            extras={"output_focal_px": intrinsics.fx_px if intrinsics else None},
        )


class _InterruptingDepthPro(_FakeDepthPro):
    def __init__(self, *, successful_calls: int) -> None:
        super().__init__()
        self.successful_calls = successful_calls
        self.attempts = 0

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> SimpleNamespace:
        if self.attempts >= self.successful_calls:
            self.attempts += 1
            raise RuntimeError("injected interruption")
        self.attempts += 1
        return super().predict(rgb, intrinsics=intrinsics, camera_mode=camera_mode)


class _RejectThenInterruptDepthPro(_FakeDepthPro):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> SimpleNamespace:
        if self.attempts == 1:
            self.attempts += 1
            raise RuntimeError("injected interruption")
        self.attempts += 1
        prediction = super().predict(rgb, intrinsics=intrinsics, camera_mode=camera_mode)
        prediction.depth_m[1, 2] = 0.0
        return prediction


class _DifferentDepthPro(_FakeDepthPro):
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model": "different fake Depth Pro",
            "checkpoint_sha256": "d" * 64,
        }


def _landmark(index: int, *, u_px: int, v_px: int) -> dict[str, object]:
    names = {
        5: "INDEX_FINGER_MCP",
        6: "INDEX_FINGER_PIP",
        7: "INDEX_FINGER_DIP",
        8: "INDEX_FINGER_TIP",
    }
    return {
        "landmark_index": index,
        "landmark_name": names[index],
        "x_normalized": u_px / 5,
        "y_normalized": v_px / 4,
        "z_mediapipe_relative": -0.01 * index,
        "u_px": u_px,
        "v_px": v_px,
        "in_frame": True,
    }


def _prepared_fixture(
    tmp_path: Path,
    *,
    statuses: tuple[str, ...] = ("accepted", "rejected"),
) -> Path:
    prepared = tmp_path / "prepared"
    frames = prepared / "frames"
    frames.mkdir(parents=True)
    rows = []
    for frame_index, status in enumerate(statuses):
        if status not in {"accepted", "rejected"}:
            raise ValueError(f"unsupported test status: {status}")
        value = 40 + frame_index * 10
        bgr = np.full((4, 5, 3), value, dtype=np.uint8)
        image_path = frames / f"frame_{frame_index:06d}.png"
        assert cv2.imwrite(str(image_path), bgr)
        base = {
            "frame_index": frame_index,
            "timestamp_ms": frame_index * 33,
            "image_path": f"frames/{image_path.name}",
            "width": 5,
            "height": 4,
            "png_sha256": sha256_file(image_path),
            "bgr_pixel_sha256": pixel_sha256(bgr),
            "camera_intrinsics": {
                "fx_px": 100.0,
                "fy_px": 100.0,
                "cx_px": 2.0,
                "cy_px": 1.5,
            },
        }
        if status == "accepted":
            features = [_landmark(index, u_px=2, v_px=1) for index in (5, 6, 7, 8)]
            base.update(
                {
                    "hand_detected": True,
                    "status": "accepted",
                    "rejection_reasons": [],
                    "hands": [
                        {
                            "hand_index": 0,
                            "handedness": "Right",
                            "handedness_score": 0.95,
                            "landmarks": features,
                        }
                    ],
                    "selected_hand_index": 0,
                    "target_landmark": features[-1],
                }
            )
        else:
            base.update(
                {
                    "hand_detected": False,
                    "status": "rejected",
                    "rejection_reasons": ["no_hand"],
                    "hands": [],
                    "selected_hand_index": None,
                    "target_landmark": None,
                }
            )
        rows.append(base)

    records_path = prepared / "frames.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "format": "fingertip-depth-pseudo-label-inputs",
        "format_version": 1,
        "source": {
            "mode": "test",
            "video_sha256": "b" * 64,
            "frame_cache_manifest_sha256": "c" * 64,
        },
        "frames_jsonl": {
            "relative_path": "frames.jsonl",
            "sha256": sha256_file(records_path),
        },
        "frames": {
            "relative_directory": "frames",
            "pixel_sequence_sha256": pixel_hash_sequence_sha256(
                str(row["bgr_pixel_sha256"]) for row in rows
            ),
            "pixel_sequence_encoding": (
                "sha256 of concatenated lowercase hexadecimal BGR pixel SHA-256 "
                "digests encoded as ASCII"
            ),
        },
        "frame_count": len(statuses),
        "accepted_frame_count": statuses.count("accepted"),
        "rejected_frame_count": statuses.count("rejected"),
        "feature_landmarks": {
            "indices": [5, 6, 7, 8],
            "names": [
                "INDEX_FINGER_MCP",
                "INDEX_FINGER_PIP",
                "INDEX_FINGER_DIP",
                "INDEX_FINGER_TIP",
            ],
        },
        "target_landmark": {
            "index": 8,
            "name": "INDEX_FINGER_TIP",
            "depth_sampling": "single_pixel",
        },
        "camera": {
            "focal_35mm_equivalent_mm": 36.0,
            "image_size_px": {"width": 5, "height": 4},
            "intrinsics": rows[0]["camera_intrinsics"],
        },
    }
    manifest_path = prepared / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _selection_report(path: Path, prepared: Path) -> None:
    manifest = json.loads(prepared.read_text(encoding="utf-8"))
    write_json(
        path,
        {
            "expected_range_m": {"min": 0.2, "max": 0.3},
            "audit": {
                "source": {"sha256": manifest["source"]["video_sha256"]},
                "frame_cache": {
                    "manifest_sha256": manifest["source"]["frame_cache_manifest_sha256"]
                },
                "input_pixels": {"sequence_sha256": manifest["frames"]["pixel_sequence_sha256"]},
            },
            "conditions": [
                {
                    "id": DEPTH_PRO_TEACHER_CONDITION_ID,
                    "band": {"band_violation_m": {"mean": 0.029}},
                }
            ],
        },
    )


def _output_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_generates_single_pixel_labels_xyz_and_sequence_level_split(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    report = tmp_path / "selection.json"
    _selection_report(report, prepared)
    estimator = _FakeDepthPro()
    output = tmp_path / "dataset"

    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=estimator,
        sequence_id="recording_01",
        split="train",
        frame_transfer_mode="copy",
        expected_prepared_manifest_sha256=sha256_file(prepared),
        teacher_selection_report_path=report,
    )

    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 1
    sample = samples[0]
    assert [item["landmark_index"] for item in sample["hand"]["feature_landmarks"]] == [
        5,
        6,
        7,
        8,
    ]
    assert sample["target"]["z_teacher_m"] == pytest.approx(0.3)
    assert sample["target"]["camera_xyz_m"] == pytest.approx(
        {"x_m": 0.0, "y_m": -0.0015, "z_m": 0.3}
    )
    assert sample["target"]["depth_sampling"] == "single_pixel"
    assert sample["split"] == "train"
    assert estimator.calls and estimator.calls[0][1] == "approx_focal"
    assert estimator.calls[0][0] is not None
    assert manifest["ground_truth"] is False
    assert manifest["teacher"]["condition_id"] == DEPTH_PRO_TEACHER_CONDITION_ID
    assert manifest["teacher"]["selection_evidence"]["value"] == pytest.approx(0.029)
    assert manifest["teacher"]["selection_evidence"]["provenance_verified"] is True
    assert manifest["counts"]["accepted_samples"] == 1
    assert manifest["counts"]["rejected_frames_considered"] == 1
    assert manifest["counts"]["source_frames_considered"] == 2
    prepared_rows = [
        json.loads(line) for line in (prepared.parent / "frames.jsonl").read_text().splitlines()
    ]
    assert manifest["prepared_inputs"]["pixel_hash_sequence_sha256"] == (
        pixel_hash_sequence_sha256(str(row["bgr_pixel_sha256"]) for row in prepared_rows)
    )
    assert (output / "trajectories" / "recording_01.csv").is_file()
    assert (output / "trajectories" / "recording_01.ply").is_file()
    assert (output / "visualizations" / "recording_01_views.png").is_file()
    with (output / "targets.csv").open(encoding="utf-8", newline="") as source:
        target_rows = list(csv.DictReader(source))
    assert len(target_rows) == 1
    assert target_rows[0]["sample_id"] == "recording_01:000000:hand0"
    assert float(target_rows[0]["y_m"]) == pytest.approx(-0.0015)
    assert float(target_rows[0]["z_teacher_m"]) == pytest.approx(0.3)
    assert manifest["artifacts"]["targets_csv"]["sha256"] == sha256_file(output / "targets.csv")
    assert (output / "splits" / "train.txt").read_text() == ("recording_01:000000:hand0\n")


def test_rejects_tampered_prepared_frames_jsonl(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    records = prepared.parent / "frames.jsonl"
    records.write_text(records.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frames.jsonl SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / "dataset",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
        )


def test_rejects_unexpected_prepared_manifest_hash(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)

    with pytest.raises(ValueError, match="prepared manifest SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / "dataset",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            expected_prepared_manifest_sha256="0" * 64,
        )


def test_rejects_mismatched_prepared_pixel_hash_sequence(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    manifest = json.loads(prepared.read_text(encoding="utf-8"))
    manifest["frames"]["pixel_sequence_sha256"] = "0" * 64
    write_json(prepared, manifest)

    with pytest.raises(ValueError, match="pixel hash sequence SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / "dataset",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
        )


@pytest.mark.parametrize("retry_with_resume", (False, True))
def test_atomic_header_temp_is_recovered_by_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_with_resume: bool,
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    output = tmp_path / "dataset"
    real_write_all = pseudo_labels_module._write_all

    def _interrupt_header(target: object, payload: bytes) -> None:
        target.write(payload[:17])
        raise RuntimeError("injected header interruption")

    monkeypatch.setattr(pseudo_labels_module, "_write_all", _interrupt_header)
    with pytest.raises(RuntimeError, match="header interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
        )

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
    assert not progress.exists()
    assert progress_temp.read_bytes()
    assert list(output.iterdir()) == [progress_temp]

    monkeypatch.setattr(pseudo_labels_module, "_write_all", real_write_all)
    estimator = _FakeDepthPro()
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=estimator,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        resume=retry_with_resume,
    )
    assert len(estimator.calls) == 1
    assert progress.read_bytes().endswith(b"\n")
    assert not progress_temp.exists()


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_fresh_retry_rejects_unsafe_header_temp(tmp_path: Path, kind: str) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    output = tmp_path / "dataset"
    output.mkdir()
    progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
    if kind == "symlink":
        progress_temp.symlink_to(prepared)
    else:
        progress_temp.mkdir()

    estimator = _FakeDepthPro()
    with pytest.raises(ValueError, match="temporary"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
        )
    assert estimator.calls == []
    assert progress_temp.is_symlink() if kind == "symlink" else progress_temp.is_dir()


def test_resume_preserves_header_temp_when_other_artifacts_exist(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    output = tmp_path / "dataset"
    output.mkdir()
    progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
    progress_temp.write_bytes(b"partial header")
    unexpected = output / "unexpected.txt"
    unexpected.write_text("preserve me", encoding="utf-8")
    estimator = _FakeDepthPro()

    with pytest.raises(FileNotFoundError, match="existing progress journal"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
            resume=True,
        )
    assert estimator.calls == []
    assert progress_temp.read_bytes() == b"partial header"
    assert unexpected.read_text(encoding="utf-8") == "preserve me"


def test_resume_recovers_only_expected_torn_initial_headers(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    reference = tmp_path / "reference"
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=reference,
        estimator=_FakeDepthPro(),
        sequence_id="recording_01",
        frame_transfer_mode="copy",
    )
    header = (reference / PSEUDO_LABEL_PROGRESS_FILENAME).read_bytes().splitlines(keepends=True)[0]

    for name, residue in (("empty", b""), ("partial", header[: len(header) // 2])):
        output = tmp_path / name
        output.mkdir()
        progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
        progress.write_bytes(residue)
        estimator = _FakeDepthPro()
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            resume=True,
        )
        assert len(estimator.calls) == 1
        assert progress.read_bytes().endswith(b"\n")

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    unsafe_progress = unsafe / PSEUDO_LABEL_PROGRESS_FILENAME
    unsafe_progress.write_bytes(header[:10])
    (unsafe / "unexpected.txt").write_text("preserve me", encoding="utf-8")
    estimator = _FakeDepthPro()
    with pytest.raises(ValueError, match="alongside other workspace files"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=unsafe,
            estimator=estimator,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            resume=True,
        )
    assert estimator.calls == []
    assert unsafe_progress.read_bytes() == header[:10]


def test_torn_header_recovery_survives_a_second_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    reference = tmp_path / "reference"
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=reference,
        estimator=_FakeDepthPro(),
        sequence_id="recording_01",
        frame_transfer_mode="copy",
    )
    header = (reference / PSEUDO_LABEL_PROGRESS_FILENAME).read_bytes().splitlines(keepends=True)[0]

    output = tmp_path / "dataset"
    output.mkdir()
    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    progress.write_bytes(header[:23])
    real_write_all = pseudo_labels_module._write_all

    def _interrupt_repair(target: object, payload: bytes) -> None:
        target.write(payload[:11])
        raise RuntimeError("injected second header interruption")

    monkeypatch.setattr(pseudo_labels_module, "_write_all", _interrupt_repair)
    with pytest.raises(RuntimeError, match="second header interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            resume=True,
        )

    progress_temp = output / PSEUDO_LABEL_PROGRESS_TEMP_FILENAME
    assert not progress.exists()
    assert progress_temp.is_file()
    assert list(output.iterdir()) == [progress_temp]

    monkeypatch.setattr(pseudo_labels_module, "_write_all", real_write_all)
    estimator = _FakeDepthPro()
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=estimator,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
    )
    assert len(estimator.calls) == 1
    assert progress.is_file()
    assert not progress_temp.exists()


def test_output_directory_lock_and_existing_journal_fail_closed(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    output = tmp_path / "locked"
    output.mkdir()
    estimator = _FakeDepthPro()
    with (
        pseudo_labels_module._exclusive_output_directory(output),
        pytest.raises(RuntimeError, match="already in use"),
    ):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
        )
    assert estimator.calls == []
    assert list(output.iterdir()) == []

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    progress.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
        )
    assert progress.read_bytes() == b"do not overwrite"


def test_checkpoint_interval_replays_only_uncheckpointed_suffix(tmp_path: Path) -> None:
    prepared = _prepared_fixture(
        tmp_path,
        statuses=(
            "accepted",
            "rejected",
            "accepted",
            "accepted",
            "rejected",
            "accepted",
            "accepted",
        ),
    )
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_InterruptingDepthPro(successful_calls=3),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=3,
        )

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    checkpointed = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["frame_index"] for row in checkpointed[1:]] == [0, 1, 2]

    resumed = _FakeDepthPro()
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        checkpoint_interval_frames=3,
        resume=True,
    )
    assert len(resumed.calls) == 3
    assert manifest["counts"]["source_frames_considered"] == 7
    completed = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["frame_index"] for row in completed[1:]] == list(range(7))


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_checkpoint_interval_requires_a_positive_integer(tmp_path: Path, value: object) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    with pytest.raises(ValueError, match="checkpoint_interval_frames"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / f"invalid-{value}",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            checkpoint_interval_frames=value,  # type: ignore[arg-type]
        )


def test_resume_skips_valid_prefix_and_rebuilds_identical_outputs(tmp_path: Path) -> None:
    prepared = _prepared_fixture(
        tmp_path,
        statuses=("accepted", "rejected", "accepted", "accepted"),
    )
    output = tmp_path / "resumed"
    interrupted = _InterruptingDepthPro(successful_calls=1)

    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=interrupted,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
        )

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    progress_rows = [json.loads(line) for line in progress.read_text().splitlines()]
    assert len(progress_rows) == 3
    assert progress_rows[1]["outcome"] == "sample"
    assert progress_rows[2]["prepared_status"] == "rejected"
    assert progress_rows[2]["teacher_attempted"] is False

    resumed = _FakeDepthPro()
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        checkpoint_interval_frames=1,
        resume=True,
    )
    assert len(resumed.calls) == 2
    assert manifest["counts"] == {
        "source_frames_total": 4,
        "source_frames_considered": 4,
        "source_frames_unprocessed": 0,
        "prepared_accepted_frames_total": 3,
        "attempted_teacher_frames": 3,
        "accepted_samples": 3,
        "rejected_frames_considered": 1,
    }

    reference = tmp_path / "reference"
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=reference,
        estimator=_FakeDepthPro(),
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        checkpoint_interval_frames=1,
    )
    assert _output_files(output) == _output_files(reference)

    finalize_again = _FakeDepthPro()
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=finalize_again,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        checkpoint_interval_frames=1,
        resume=True,
    )
    assert finalize_again.calls == []
    assert _output_files(output) == _output_files(reference)


def test_resume_checkpoints_teacher_rejection(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "accepted"))
    output = tmp_path / "dataset"

    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_RejectThenInterruptDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
        )

    resumed = _FakeDepthPro()
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        checkpoint_interval_frames=1,
        resume=True,
    )
    assert len(resumed.calls) == 1
    assert manifest["counts"]["attempted_teacher_frames"] == 2
    assert manifest["counts"]["accepted_samples"] == 1
    assert manifest["counts"]["rejected_frames_considered"] == 1
    rejection = json.loads((output / "rejections.jsonl").read_text().strip())
    assert rejection["stage"] == "teacher_label"
    assert rejection["reasons"] == ["invalid_teacher_depth"]


def test_resume_truncates_only_unterminated_tail(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "accepted"))
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_InterruptingDepthPro(successful_calls=1),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
        )

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    with progress.open("ab") as target:
        target.write(b'{"record_type":"outcome","frame_index":1')

    resumed = _FakeDepthPro()
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        checkpoint_interval_frames=1,
        resume=True,
    )
    assert len(resumed.calls) == 1
    assert progress.read_bytes().endswith(b"\n")
    assert b'"frame_index":1' in progress.read_bytes()


def test_resume_rejects_interior_progress_corruption_before_inference(
    tmp_path: Path,
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "rejected", "accepted"))
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_InterruptingDepthPro(successful_calls=1),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
        )

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    lines = progress.read_bytes().splitlines(keepends=True)
    assert len(lines) == 3
    lines[1] = b'{"corrupt":true}\n'
    progress.write_bytes(b"".join(lines))
    estimator = _FakeDepthPro()
    with pytest.raises(ValueError, match="corrupt at line 2"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
            resume=True,
        )
    assert estimator.calls == []


def test_resume_rejects_newline_terminated_corrupt_final_record(
    tmp_path: Path,
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "accepted"))
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_InterruptingDepthPro(successful_calls=1),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
        )

    progress = output / PSEUDO_LABEL_PROGRESS_FILENAME
    lines = progress.read_bytes().splitlines(keepends=True)
    lines[-1] = b'{"corrupt":true}\n'
    progress.write_bytes(b"".join(lines))
    estimator = _FakeDepthPro()
    with pytest.raises(ValueError, match="corrupt at line 2"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
            resume=True,
        )
    assert estimator.calls == []


def test_resume_reuses_verified_image_left_before_journal_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    output = tmp_path / "dataset"
    real_append = pseudo_labels_module._append_progress_records

    def _interrupt_append(path: Path, records: list[dict[str, object]]) -> None:
        if any(record.get("outcome") == "sample" for record in records):
            raise RuntimeError("injected append interruption")
        real_append(path, records)

    monkeypatch.setattr(pseudo_labels_module, "_append_progress_records", _interrupt_append)
    with pytest.raises(RuntimeError, match="append interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode="hardlink",
            checkpoint_interval_frames=1,
        )
    transferred = output / "frames" / "recording_01" / "frame_000000.png"
    assert transferred.is_file()
    assert len((output / PSEUDO_LABEL_PROGRESS_FILENAME).read_text().splitlines()) == 1

    monkeypatch.setattr(pseudo_labels_module, "_append_progress_records", real_append)
    resumed = _FakeDepthPro()
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="hardlink",
        checkpoint_interval_frames=1,
        resume=True,
    )
    assert len(resumed.calls) == 1


@pytest.mark.parametrize("frame_transfer_mode", ("copy", "hardlink"))
def test_resume_rebuilds_only_incomplete_frame_transfer_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_transfer_mode: str,
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted",))
    output = tmp_path / "dataset"
    real_copy2 = pseudo_labels_module.shutil.copy2
    real_link = pseudo_labels_module.os.link

    if frame_transfer_mode == "hardlink":

        def _cross_device_link(source: Path, target: Path) -> None:
            if target.name == PSEUDO_LABEL_PROGRESS_FILENAME:
                real_link(source, target)
                return
            raise OSError(pseudo_labels_module.errno.EXDEV, "injected cross-device link")

        monkeypatch.setattr(pseudo_labels_module.os, "link", _cross_device_link)

    def _interrupt_copy(source: Path, target: Path) -> None:
        target.write_bytes(source.read_bytes()[:8])
        raise RuntimeError("injected partial copy interruption")

    monkeypatch.setattr(pseudo_labels_module.shutil, "copy2", _interrupt_copy)
    with pytest.raises(RuntimeError, match="partial copy interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode=frame_transfer_mode,
        )

    transferred = output / "frames" / "recording_01" / "frame_000000.png"
    transfer_temp = pseudo_labels_module._temporary_sibling_path(transferred)
    assert not transferred.exists()
    assert transfer_temp.is_file()
    assert len((output / PSEUDO_LABEL_PROGRESS_FILENAME).read_text().splitlines()) == 1

    monkeypatch.setattr(pseudo_labels_module.shutil, "copy2", real_copy2)
    resumed = _FakeDepthPro()
    generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode=frame_transfer_mode,
        resume=True,
    )
    prepared_row = json.loads((prepared.parent / "frames.jsonl").read_text().splitlines()[0])
    assert len(resumed.calls) == 1
    assert transferred.is_file()
    assert sha256_file(transferred) == prepared_row["png_sha256"]
    assert not transfer_temp.exists()


def test_resume_reconstructs_final_artifacts_without_reinference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "accepted"))
    output = tmp_path / "dataset"
    real_write_jsonl = pseudo_labels_module._write_jsonl

    def _interrupt_finalization(_path: Path, _rows: list[dict[str, object]]) -> None:
        raise RuntimeError("injected finalization interruption")

    monkeypatch.setattr(pseudo_labels_module, "_write_jsonl", _interrupt_finalization)
    with pytest.raises(RuntimeError, match="finalization interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
        )

    progress_rows = (output / PSEUDO_LABEL_PROGRESS_FILENAME).read_text().splitlines()
    assert len(progress_rows) == 3
    monkeypatch.setattr(pseudo_labels_module, "_write_jsonl", real_write_jsonl)
    resumed = _FakeDepthPro()
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        resume=True,
    )
    assert resumed.calls == []
    assert manifest["counts"]["accepted_samples"] == 2
    assert len((output / "samples.jsonl").read_text().splitlines()) == 2


def test_resume_recovers_interrupted_atomic_manifest_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "accepted"))
    output = tmp_path / "dataset"
    real_replace = pseudo_labels_module.os.replace

    def _interrupt_manifest_replace(source: Path, target: Path) -> None:
        if Path(target).name == "dataset_manifest.json":
            raise RuntimeError("injected manifest publish interruption")
        real_replace(source, target)

    monkeypatch.setattr(pseudo_labels_module.os, "replace", _interrupt_manifest_replace)
    with pytest.raises(RuntimeError, match="manifest publish interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
        )

    manifest_path = output / "dataset_manifest.json"
    manifest_temp = pseudo_labels_module._temporary_sibling_path(manifest_path)
    assert not manifest_path.exists()
    assert json.loads(manifest_temp.read_text())["format"] == (
        pseudo_labels_module.PSEUDO_LABEL_DATASET_FORMAT
    )
    assert len((output / PSEUDO_LABEL_PROGRESS_FILENAME).read_text().splitlines()) == 3

    monkeypatch.setattr(pseudo_labels_module.os, "replace", real_replace)
    resumed = _FakeDepthPro()
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        resume=True,
    )
    assert resumed.calls == []
    assert json.loads(manifest_path.read_text()) == manifest
    assert not manifest_temp.exists()
    assert not list(output.rglob(".*.tmp.*"))


def test_resume_preserves_max_frames_prefix_semantics(tmp_path: Path) -> None:
    prepared = _prepared_fixture(
        tmp_path,
        statuses=("accepted", "rejected", "accepted", "rejected"),
    )
    output = tmp_path / "dataset"
    first = _FakeDepthPro()
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=first,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        max_frames=1,
    )
    assert len(first.calls) == 1
    assert manifest["counts"]["source_frames_considered"] == 2
    assert manifest["counts"]["source_frames_unprocessed"] == 2
    assert len((output / PSEUDO_LABEL_PROGRESS_FILENAME).read_text().splitlines()) == 3

    resumed = _FakeDepthPro()
    resumed_manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=resumed,
        sequence_id="recording_01",
        frame_transfer_mode="copy",
        max_frames=1,
        resume=True,
    )
    assert resumed.calls == []
    assert resumed_manifest == manifest


def test_resume_identity_and_completed_image_tamper_fail_closed(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, statuses=("accepted", "accepted"))
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=_InterruptingDepthPro(successful_calls=1),
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
        )

    header = json.loads((output / PSEUDO_LABEL_PROGRESS_FILENAME).read_text().splitlines()[0])
    identity = header["identity"]
    assert identity["prepared_manifest_sha256"] == sha256_file(prepared)
    assert identity["frames_jsonl_sha256"] == sha256_file(prepared.parent / "frames.jsonl")
    assert identity["pixel_sequence_sha256"]
    assert identity["sequence_id"] == "recording_01"
    assert identity["split"] == "train"
    assert identity["max_frames"] is None
    assert identity["checkpoint_interval_frames"] == 1
    assert identity["frame_transfer_mode"] == "copy"
    assert identity["teacher_metadata"]["checkpoint_sha256"] == "a" * 64
    assert identity["implementation_sha256"]

    different = _DifferentDepthPro()
    with pytest.raises(ValueError, match="identity mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=different,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
            resume=True,
        )
    assert different.calls == []

    for overrides in (
        {"sequence_id": "different_recording"},
        {"split": "validation"},
        {"max_frames": 1},
        {"frame_transfer_mode": "hardlink"},
        {"checkpoint_interval_frames": 2},
    ):
        estimator = _FakeDepthPro()
        arguments: dict[str, object] = {
            "prepared_manifest_path": prepared,
            "output_dir": output,
            "estimator": estimator,
            "sequence_id": "recording_01",
            "split": "train",
            "frame_transfer_mode": "copy",
            "checkpoint_interval_frames": 1,
            "resume": True,
        }
        arguments.update(overrides)
        with pytest.raises(ValueError, match="identity mismatch"):
            generate_depth_pro_pseudo_labels(**arguments)  # type: ignore[arg-type]
        assert estimator.calls == []

    transferred = output / "frames" / "recording_01" / "frame_000000.png"
    transferred.write_bytes(b"tampered")
    estimator = _FakeDepthPro()
    with pytest.raises(ValueError, match="transferred PNG SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=output,
            estimator=estimator,
            sequence_id="recording_01",
            frame_transfer_mode="copy",
            checkpoint_interval_frames=1,
            resume=True,
        )
    assert estimator.calls == []
