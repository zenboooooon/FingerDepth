"""Render the Phase 8 fingertip trajectory demo for finger_movement_2030."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fingertip_depth.student_trajectory_demo import create_student_trajectory_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path(
            "outputs/phase8_student_vit_landmarks_xy_spike_filtered_chronological_tail7/"
            "run_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase8_student_trajectory_demo/finger_movement_2030"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--panel-width", type=int, default=720)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _progress(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = create_student_trajectory_demo(
        run_manifest_path=args.run_manifest,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        panel_width=args.panel_width,
        overwrite=args.overwrite,
        progress=_progress,
    )
    output = {
        "demo_video": str(args.output_dir / "demo.mp4"),
        "trajectory_csv": str(args.output_dir / "trajectory.csv"),
        "manifest": str(args.output_dir / "manifest.json"),
        "counts": manifest["counts"],
        "prediction_depth_m": manifest["prediction_depth_m"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
