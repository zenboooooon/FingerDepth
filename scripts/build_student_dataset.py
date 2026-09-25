"""Build a multi-sequence student dataset with train-only HFlip augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingertip_depth.student_dataset import build_student_dataset
from fingertip_depth.video_cache import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        action="append",
        type=Path,
        required=True,
        help=(
            "Phase 7 dataset_manifest.json; repeat once per source video. "
            "The source manifest's sequence split is authoritative unless "
            "--validation-tail-fraction is supplied."
        ),
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        action="append",
        help=(
            "Pinned SHA-256 paired by input order with --source-manifest; when used, "
            "repeat once per source manifest."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-transfer-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize identity PNGs; HFlip PNGs are always newly encoded.",
    )
    parser.add_argument(
        "--validation-tail-fraction",
        type=float,
        help=(
            "Chronologically assign the final fraction of every source video's raw "
            "frame timeline to identity-only validation; e.g. 0.07. By default, "
            "the Phase 7 sequence-level splits remain authoritative."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_source_manifest_sha256 is not None and len(
        args.expected_source_manifest_sha256
    ) != len(args.source_manifest):
        parser.error(
            "--expected-source-manifest-sha256 must be repeated once per --source-manifest"
        )
    manifest = build_student_dataset(
        source_manifest_paths=args.source_manifest,
        output_dir=args.output_dir,
        frame_transfer_mode=args.frame_transfer_mode,
        expected_source_manifest_sha256s=args.expected_source_manifest_sha256,
        validation_tail_fraction=args.validation_tail_fraction,
    )
    manifest_path = args.output_dir / "dataset_manifest.json"
    print(
        json.dumps(
            {
                "dataset_manifest": str(manifest_path),
                "dataset_manifest_sha256": sha256_file(manifest_path),
                "splits": manifest["split_policy"],
                "counts": manifest["counts"],
                "selection_evidence_scopes": manifest["teacher"]["selection_evidence_scopes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
