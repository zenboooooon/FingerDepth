import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_depth_model_comparison.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_depth_model_comparison_for_test",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load {_SCRIPT_PATH}")
evaluate_depth_model_comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_depth_model_comparison)


def test_validate_inputs_accepts_explicit_phase2_video(tmp_path: Path) -> None:
    custom_video = tmp_path / "finger_movement_2030.MOV"
    custom_video.write_bytes(b"video fixture")
    baseline_frames = tmp_path / "frames.jsonl"
    baseline_frames.write_text("", encoding="utf-8")
    frame_cache_manifest = tmp_path / "manifest.json"
    frame_cache_manifest.write_text("{}", encoding="utf-8")

    args = evaluate_depth_model_comparison.build_parser().parse_args(
        [
            "--backend",
            "unidepth",
            "--phase",
            "2",
            "--input-dir",
            str(tmp_path / "unused-input-dir"),
            "--video",
            str(custom_video),
            "--baseline-frames",
            str(baseline_frames),
            "--frame-cache-manifest",
            str(frame_cache_manifest),
        ]
    )

    _samples, video_path = evaluate_depth_model_comparison._validate_inputs(args)

    assert video_path == custom_video


def test_validate_inputs_keeps_default_video_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    default_video = input_dir / "finger_movement.MOV"
    default_video.write_bytes(b"video fixture")
    baseline_frames = tmp_path / "frames.jsonl"
    baseline_frames.write_text("", encoding="utf-8")
    frame_cache_manifest = tmp_path / "manifest.json"
    frame_cache_manifest.write_text("{}", encoding="utf-8")

    args = evaluate_depth_model_comparison.build_parser().parse_args(
        [
            "--backend",
            "unidepth",
            "--phase",
            "2",
            "--input-dir",
            str(input_dir),
            "--baseline-frames",
            str(baseline_frames),
            "--frame-cache-manifest",
            str(frame_cache_manifest),
        ]
    )

    _samples, video_path = evaluate_depth_model_comparison._validate_inputs(args)

    assert video_path == default_video
