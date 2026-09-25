import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_iphone_samples.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_iphone_samples_for_test", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load {_SCRIPT_PATH}")
evaluate_iphone_samples = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_iphone_samples)


def test_phase1_only_does_not_keep_stale_phase2_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for name, _distance in evaluate_iphone_samples._DISTANCE_SAMPLES:
        (input_dir / name).write_bytes(b"fixture")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps({"phase2_finger_movement": {"stale": True}}),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"aggregate": {"ok": True}}

    monkeypatch.setattr(
        evaluate_iphone_samples,
        "evaluate_known_distance_images",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_iphone_samples.py",
            "--phase",
            "1",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--photo-focal-35mm-mm",
            "27.5",
        ],
    )

    assert evaluate_iphone_samples.main() == 0

    combined = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(combined) == {"phase1_known_distance"}
    assert calls[0]["focal_35mm_mm"] == 27.5


def test_phase2_accepts_explicit_video_path(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    custom_video = tmp_path / "finger_movement_2030.MOV"
    custom_video.write_bytes(b"video fixture")
    hand_model = tmp_path / "hand_landmarker.task"
    hand_model.write_bytes(b"model fixture")
    output_dir = tmp_path / "outputs"
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"movement_evaluation": {"ok": True}}

    monkeypatch.setattr(
        evaluate_iphone_samples,
        "evaluate_finger_movement_video",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_iphone_samples.py",
            "--phase",
            "2",
            "--input-dir",
            str(input_dir),
            "--video",
            str(custom_video),
            "--hand-model",
            str(hand_model),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert evaluate_iphone_samples.main() == 0
    assert calls[0]["input_path"] == custom_video
