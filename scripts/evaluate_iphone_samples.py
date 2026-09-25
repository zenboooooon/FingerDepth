"""Run the supplied iPhone known-distance and finger-movement experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingertip_depth.artifacts import write_json
from fingertip_depth.sample_experiment import (
    evaluate_finger_movement_video,
    evaluate_known_distance_images,
)

_DISTANCE_SAMPLES = (
    ("03image.HEIC", 0.3),
    ("05image.HEIC", 0.5),
    ("07image.HEIC", 0.7),
    ("10image.HEIC", 1.0),
    ("15image.HEIC", 1.5),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("phase1_2_sample"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iphone_phase1_2"),
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path("assets/hand_landmarker.task"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Phase 2 video path. Defaults to "
            "<input-dir>/finger_movement.MOV when omitted."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("all", "1", "2"), default="all")
    parser.add_argument("--max-video-frames", type=int)
    parser.add_argument(
        "--photo-focal-35mm-mm",
        type=float,
        default=26.0,
        help="Photo 35mm-equivalent focal length from metadata (default: 26).",
    )
    parser.add_argument(
        "--video-focal-35mm-mm",
        type=float,
        default=36.0,
        help="User-supplied video 35mm-equivalent focal length (default: 36).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary: dict[str, object] = {}

    if args.phase in ("all", "1"):
        samples = [(args.input_dir / name, distance) for name, distance in _DISTANCE_SAMPLES]
        missing = [str(path) for path, _distance in samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing known-distance inputs: {missing}")
        print("Phase 1: evaluating five known-distance HEIC images...", flush=True)
        phase1 = evaluate_known_distance_images(
            samples=samples,
            output_dir=args.output_dir / "phase1_known_distance",
            focal_35mm_mm=args.photo_focal_35mm_mm,
            device=args.device,
        )
        summary["phase1_known_distance"] = phase1
        print(json.dumps(phase1["aggregate"], indent=2), flush=True)

    if args.phase in ("all", "2"):
        video = args.video if args.video is not None else args.input_dir / "finger_movement.MOV"
        if not video.is_file():
            raise FileNotFoundError(f"missing movement video: {video}")
        if not args.hand_model.is_file():
            raise FileNotFoundError(f"missing hand model: {args.hand_model}")
        print(f"Phase 2: evaluating {video}...", flush=True)
        phase2 = evaluate_finger_movement_video(
            input_path=video,
            output_dir=args.output_dir / "phase2_finger_movement",
            hand_model_path=args.hand_model,
            focal_35mm_mm=args.video_focal_35mm_mm,
            device=args.device,
            max_frames=args.max_video_frames,
        )
        summary["phase2_finger_movement"] = phase2
        print(json.dumps(phase2["movement_evaluation"], indent=2), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
