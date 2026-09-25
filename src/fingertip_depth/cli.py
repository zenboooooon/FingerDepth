"""Command-line entry point for phase 1 and phase 2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .pipeline import run_phase1_image, run_phase2_image, run_video

_IMAGE_SUFFIXES = {
    ".bmp",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fx-px",
        type=float,
        required=True,
        help="Calibrated focal length fx in pixels for the original input frame.",
    )
    parser.add_argument("--fy-px", type=float, help="Focal length fy; defaults to fx.")
    parser.add_argument("--cx-px", type=float, help="Principal point cx; defaults to image center.")
    parser.add_argument("--cy-px", type=float, help="Principal point cy; defaults to image center.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="RGB image or video path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-type",
        choices=("auto", "image", "video"),
        default="auto",
        help="Override image/video detection.",
    )
    parser.add_argument("--device", default="auto", help="auto or cuda:0")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames.")
    parser.add_argument(
        "--save-depth-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save processed video depth maps as .npy. Defaults to on for phase1 and off for phase2."
        ),
    )
    _add_camera_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingertip-depth",
        description="Metric3D v2 phase-1 depth and phase-2 fingertip-depth PoC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1 = subparsers.add_parser("phase1", help="Run Metric3D v2 metric-depth inference.")
    _add_common_arguments(phase1)
    phase1.add_argument("--ground-truth-depth", type=Path, help="Optional GT depth image.")
    phase1.add_argument(
        "--ground-truth-scale",
        type=float,
        default=1.0,
        help="Raw GT units per metre, e.g. 256 for the Metric3D KITTI demo.",
    )

    phase2 = subparsers.add_parser(
        "phase2", help="Run Metric3D v2 plus MediaPipe INDEX_FINGER_TIP lookup."
    )
    _add_common_arguments(phase2)
    phase2.add_argument(
        "--hand-model",
        type=Path,
        default=Path("assets/hand_landmarker.task"),
        help="Path to the fixed MediaPipe Hand Landmarker .task file.",
    )
    return parser


def _input_kind(path: Path, override: str) -> str:
    if override != "auto":
        return override
    return "image" if path.suffix.lower() in _IMAGE_SUFFIXES else "video"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist or is not a file: {args.input}")
    kind = _input_kind(args.input, args.input_type)
    if kind == "video" and args.command == "phase1" and args.ground_truth_depth is not None:
        parser.error("--ground-truth-depth is only supported for image input")
    camera = {
        "fx_px": args.fx_px,
        "fy_px": args.fy_px,
        "cx_px": args.cx_px,
        "cy_px": args.cy_px,
    }

    try:
        if kind == "video":
            save_depth_frames = args.save_depth_frames
            if save_depth_frames is None:
                save_depth_frames = args.command == "phase1"
            result = run_video(
                phase=1 if args.command == "phase1" else 2,
                input_path=args.input,
                output_dir=args.output_dir,
                device=args.device,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                save_depth_frames=save_depth_frames,
                hand_model_path=args.hand_model if args.command == "phase2" else None,
                **camera,
            )
        elif args.command == "phase1":
            result = run_phase1_image(
                input_path=args.input,
                output_dir=args.output_dir,
                device=args.device,
                ground_truth_depth=args.ground_truth_depth,
                ground_truth_scale=args.ground_truth_scale,
                **camera,
            )
        else:
            result = run_phase2_image(
                input_path=args.input,
                output_dir=args.output_dir,
                hand_model_path=args.hand_model,
                device=args.device,
                **camera,
            )
    except (FileNotFoundError, IndexError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
