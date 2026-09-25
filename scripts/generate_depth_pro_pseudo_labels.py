"""Generate Phase 5-7 pseudo labels with fixed Depth Pro approximate-focal teacher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-manifest",
        type=Path,
        required=True,
        help="manifest.json produced by prepare_pseudo_label_inputs.py",
    )
    parser.add_argument(
        "--expected-prepared-manifest-sha256",
        help="Optional pinned SHA-256 for the prepared manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "unassigned"),
        default="train",
        help="Whole-sequence split; frames are never randomly split.",
    )
    parser.add_argument(
        "--frame-transfer-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
    )
    parser.add_argument(
        "--teacher-selection-report",
        type=Path,
        help=(
            "Optional audited model-comparison JSON documenting why "
            "depth_pro__approx_focal was selected."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=_positive_int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The isolated Depth Pro project deliberately does not install this root
    # package because its NumPy/OpenCV constraints differ. Expose only our source.
    project_src = Path(__file__).resolve().parents[1] / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

    from fingertip_depth.alternative_depth import DepthProEstimator
    from fingertip_depth.pseudo_labels import generate_depth_pro_pseudo_labels

    estimator = DepthProEstimator(device=args.device)
    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=args.prepared_manifest,
        output_dir=args.output_dir,
        estimator=estimator,
        sequence_id=args.sequence_id,
        split=args.split,
        frame_transfer_mode=args.frame_transfer_mode,
        expected_prepared_manifest_sha256=args.expected_prepared_manifest_sha256,
        teacher_selection_report_path=args.teacher_selection_report,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "dataset_manifest": str(args.output_dir / "dataset_manifest.json"),
                "teacher_condition_id": manifest["teacher"]["condition_id"],
                "feature_landmarks": manifest["feature_landmarks"],
                "target_landmark": manifest["target_landmark"],
                "counts": manifest["counts"],
                "trajectory": manifest["trajectory"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
