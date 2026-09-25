"""Decode the iPhone MOV once into a lossless, hash-verified frame cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from fingertip_depth.video_cache import (
    VideoFrameCache,
    create_video_frame_cache,
    sha256_file,
)

_BASELINE_RECORDS_SHA256 = (
    "3c469db2b7be4f8b7d55721a690b4ff16130e8dc31fb8aa6b9256ec2f7cd5a10"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("phase1_2_sample/finger_movement.MOV"),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path("outputs/iphone_phase1_2/phase2_finger_movement/frames.jsonl"),
    )
    parser.add_argument(
        "--baseline-records-sha256",
        default=_BASELINE_RECORDS_SHA256,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/depth_model_comparison/shared_video_frames"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = create_video_frame_cache(
        input_path=args.input,
        baseline_records_path=args.baseline_records,
        baseline_records_sha256=args.baseline_records_sha256,
        output_dir=args.output_dir,
    )
    manifest_sha256 = sha256_file(manifest)
    cache = VideoFrameCache.load(
        manifest,
        input_path=args.input,
        baseline_records_sha256=args.baseline_records_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    cache.verify_all()
    print(f"Wrote and verified {manifest}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
