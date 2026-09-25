"""Evaluate UniDepth V2-L or Depth Pro on the fixed iPhone comparison set.

The script intentionally runs one backend per process.  UniDepth and Depth Pro
have incompatible optional dependency constraints, so invoke this script with
the matching uv dependency group.  Each invocation loads the selected model
once and evaluates its two camera-input conditions sequentially.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_DISTANCE_SAMPLES = (
    ("03image.HEIC", 0.3),
    ("05image.HEIC", 0.5),
    ("07image.HEIC", 0.7),
    ("10image.HEIC", 1.0),
    ("15image.HEIC", 1.5),
)

_BASELINE_RECORDS_SHA256 = (
    "3c469db2b7be4f8b7d55721a690b4ff16130e8dc31fb8aa6b9256ec2f7cd5a10"
)
_FRAME_CACHE_MANIFEST_SHA256 = (
    "e86ae1ab0793499a05255d79942e357af26306d2fa9a9b0c8d030703794a9068"
)
_CONDITIONS: dict[str, tuple[dict[str, str], ...]] = {
    "unidepth": (
        {
            "id": "unidepth_v2_l__approx_k",
            "model_family": "UniDepth V2-L",
            "camera_mode": "approx_k",
            "comparison_role": "first_candidate",
            "camera_input": "current diagonal-FOV approximate K",
        },
        {
            "id": "unidepth_v2_l__no_camera",
            "model_family": "UniDepth V2-L",
            "camera_mode": "no_camera",
            "comparison_role": "first_candidate",
            "camera_input": "none; model camera prediction",
        },
    ),
    "depth-pro": (
        {
            "id": "depth_pro__approx_focal",
            "model_family": "Depth Pro",
            "camera_mode": "approx_focal",
            "comparison_role": "second_candidate",
            "camera_input": "current diagonal-FOV approximate focal length in pixels",
        },
        {
            "id": "depth_pro__estimated_focal",
            "model_family": "Depth Pro",
            "camera_mode": "estimated_focal",
            "comparison_role": "second_candidate",
            "camera_input": "none; model focal-length prediction",
        },
    ),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run --group unidepth python scripts/evaluate_depth_model_comparison.py "
            "--backend unidepth\n"
            "  uv run --project environments/depth_pro python "
            "scripts/evaluate_depth_model_comparison.py --backend depth-pro"
        ),
    )
    parser.add_argument("--backend", choices=tuple(_CONDITIONS), required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("phase1_2_sample"),
        help="Directory containing the five HEIC images and finger_movement.MOV.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Phase 2 video path. Defaults to "
            "<input-dir>/finger_movement.MOV when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/depth_model_comparison"),
    )
    parser.add_argument(
        "--baseline-frames",
        type=Path,
        default=Path("outputs/iphone_phase1_2/phase2_finger_movement/frames.jsonl"),
        help="Metric3D run whose MediaPipe fingertip coordinates are reused for Phase 2.",
    )
    parser.add_argument(
        "--baseline-records-sha256",
        default=_BASELINE_RECORDS_SHA256,
        help=(
            "Expected SHA-256 of --baseline-frames; fixed to the audited baseline by default."
        ),
    )
    parser.add_argument(
        "--frame-cache-manifest",
        type=Path,
        default=Path("outputs/depth_model_comparison/shared_video_frames/manifest.json"),
        help="Lossless frame manifest generated once by cache_comparison_video_frames.py.",
    )
    parser.add_argument(
        "--frame-cache-manifest-sha256",
        default=_FRAME_CACHE_MANIFEST_SHA256,
        help="Expected SHA-256 of the audited shared frame-cache manifest.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("all", "1", "2"), default="all")
    parser.add_argument(
        "--max-video-frames",
        type=_positive_int,
        help="Limit Phase 2 for a smoke run; omit for the complete video.",
    )
    parser.add_argument(
        "--photo-focal-35mm-mm",
        type=_positive_float,
        default=26.0,
        help="Photo 35mm-equivalent focal length (default: 26).",
    )
    parser.add_argument(
        "--video-focal-35mm-mm",
        type=_positive_float,
        default=36.0,
        help="Video 35mm-equivalent focal length (default: 36).",
    )
    return parser


def _validate_inputs(args: argparse.Namespace) -> tuple[list[tuple[Path, float]], Path]:
    samples = [(args.input_dir / name, distance) for name, distance in _DISTANCE_SAMPLES]
    if args.phase in ("all", "1"):
        missing = [str(path) for path, _distance in samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing known-distance inputs: {missing}")

    video_path = (
        args.video if args.video is not None else args.input_dir / "finger_movement.MOV"
    )
    if args.phase in ("all", "2"):
        if not video_path.is_file():
            raise FileNotFoundError(f"missing movement video: {video_path}")
        if not args.baseline_frames.is_file():
            raise FileNotFoundError(
                "missing baseline MediaPipe frame records: "
                f"{args.baseline_frames}; run evaluate_iphone_samples.py Phase 2 first"
            )
        if not args.frame_cache_manifest.is_file():
            raise FileNotFoundError(
                "missing shared lossless frame cache: "
                f"{args.frame_cache_manifest}; run cache_comparison_video_frames.py first"
            )
    return samples, video_path


def _make_estimator(backend: str, device: str) -> Any:
    # Lazy imports keep --help and parser tests independent of heavyweight,
    # backend-specific optional dependencies.
    if backend == "unidepth":
        from fingertip_depth.alternative_depth import UniDepthV2L

        return UniDepthV2L(device=device)
    if backend == "depth-pro":
        from fingertip_depth.alternative_depth import DepthProEstimator

        return DepthProEstimator(device=device)
    raise ValueError(f"unsupported backend: {backend}")


def _brief_result(result: dict[str, Any]) -> dict[str, Any]:
    brief: dict[str, Any] = {}
    phase1 = result.get("phase1")
    if isinstance(phase1, dict):
        brief["phase1_aggregate"] = phase1.get("aggregate")
    phase2 = result.get("phase2")
    if isinstance(phase2, dict):
        brief["phase2_movement_evaluation"] = phase2.get("movement_evaluation")
    return brief


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples, video_path = _validate_inputs(args)

    # The isolated Depth Pro uv project intentionally does not install the
    # root project (its NumPy/OpenCV constraints conflict), so expose only the
    # root source tree to Python while keeping third-party dependencies isolated.
    project_src = Path(__file__).resolve().parents[1] / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

    runner_phase = {"all": "all", "1": "phase1", "2": "phase2"}[args.phase]
    from fingertip_depth.artifacts import write_json
    from fingertip_depth.model_comparison import ComparisonCondition, evaluate_condition

    print(f"Loading {args.backend} once for two camera-input conditions...", flush=True)
    estimator = _make_estimator(args.backend, args.device)
    condition_results: dict[str, dict[str, Any]] = {}

    for spec in _CONDITIONS[args.backend]:
        condition = ComparisonCondition(
            id=spec["id"],
            model_family=spec["model_family"],
            camera_mode=spec["camera_mode"],
            metadata={
                "comparison_role": spec["comparison_role"],
                "camera_input": spec["camera_input"],
            },
        )
        print(f"Evaluating {condition.id} ({args.phase=})...", flush=True)
        result = evaluate_condition(
            condition=condition,
            estimator=estimator,
            samples=samples,
            video_path=video_path,
            baseline_records_path=args.baseline_frames,
            baseline_records_sha256=args.baseline_records_sha256,
            output_dir=args.output_dir / condition.id,
            frame_cache_manifest_path=args.frame_cache_manifest,
            frame_cache_manifest_sha256=args.frame_cache_manifest_sha256,
            photo_focal_35mm_mm=args.photo_focal_35mm_mm,
            video_focal_35mm_mm=args.video_focal_35mm_mm,
            max_video_frames=args.max_video_frames,
            phase=runner_phase,
        )
        condition_results[condition.id] = result
        print(json.dumps(_brief_result(result), indent=2), flush=True)

    summary: dict[str, Any] = {
        "experiment": "Alternative monocular metric-depth model comparison",
        "backend": args.backend,
        "phase": runner_phase,
        "phase_argument": args.phase,
        "conditions": condition_results,
        "configuration": {
            "input_dir": str(args.input_dir.resolve()),
            "video": str(video_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "baseline_frames": str(args.baseline_frames.resolve()),
            "baseline_records_expected_sha256": args.baseline_records_sha256,
            "frame_cache_manifest": str(args.frame_cache_manifest.resolve()),
            "frame_cache_manifest_expected_sha256": args.frame_cache_manifest_sha256,
            "photo_focal_35mm_equivalent_mm": args.photo_focal_35mm_mm,
            "video_focal_35mm_equivalent_mm": args.video_focal_35mm_mm,
            "max_video_frames": args.max_video_frames,
            "device_request": args.device,
            "phase2_coordinate_policy": (
                "reuse fixed MediaPipe fingertip coordinates from the Metric3D baseline"
            ),
        },
    }
    summary_name = (
        "unidepth_summary.json" if args.backend == "unidepth" else "depth_pro_summary.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / summary_name
    write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
