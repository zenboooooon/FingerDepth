"""Prepare verified RGB frames and hand landmarks for Depth Pro pseudo-labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingertip_depth.constants import DEFAULT_FEATURE_LANDMARK_INDICES
from fingertip_depth.hands import parse_hand_landmark_selection
from fingertip_depth.pseudo_label_inputs import prepare_pseudo_label_inputs
from fingertip_depth.video_cache import sha256_file


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _landmark_selection(value: str) -> tuple[int, ...]:
    try:
        return parse_hand_landmark_selection(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-video",
        type=Path,
        help="Decode this video once and store lossless PNG frames.",
    )
    source.add_argument(
        "--frame-cache-manifest",
        type=Path,
        help="Reuse an existing audited lossless frame-cache manifest.",
    )
    parser.add_argument(
        "--frame-cache-manifest-sha256",
        help="Expected SHA-256; required with --frame-cache-manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path("assets/hand_landmarker.task"),
    )
    parser.add_argument(
        "--landmarks",
        type=_landmark_selection,
        default=DEFAULT_FEATURE_LANDMARK_INDICES,
        help=(
            "Comma-separated MediaPipe indices/names, or 'all'. "
            "Default: 5,6,7,8 (index-finger MCP/PIP/DIP/TIP)."
        ),
    )
    parser.add_argument(
        "--focal-35mm-mm",
        type=_positive_float,
        default=36.0,
        help="Video focal length in 35mm-equivalent millimetres (default: 36).",
    )
    parser.add_argument(
        "--frame-transfer-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How audited cache PNGs are placed in the prepared dataset.",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        help="Optional smoke-test limit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frame_cache_manifest is not None and args.frame_cache_manifest_sha256 is None:
        parser.error("--frame-cache-manifest-sha256 is required with --frame-cache-manifest")
    if args.input_video is not None and args.frame_cache_manifest_sha256 is not None:
        parser.error("--frame-cache-manifest-sha256 is only valid with --frame-cache-manifest")

    try:
        manifest = prepare_pseudo_label_inputs(
            output_dir=args.output_dir,
            hand_model_path=args.hand_model,
            focal_35mm_equivalent_mm=args.focal_35mm_mm,
            feature_landmark_indices=args.landmarks,
            input_video_path=args.input_video,
            frame_cache_manifest_path=args.frame_cache_manifest,
            expected_frame_cache_manifest_sha256=args.frame_cache_manifest_sha256,
            frame_transfer_mode=args.frame_transfer_mode,
            max_frames=args.max_frames,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    manifest_path = args.output_dir / "manifest.json"
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "frame_count": manifest["frame_count"],
                "accepted_frame_count": manifest["accepted_frame_count"],
                "rejected_frame_count": manifest["rejected_frame_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
