"""Command-line entry point for the video-to-student training pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/training_pipeline.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingertip-train",
        description=(
            "Discover training videos, build verified pseudo labels and a student dataset, "
            "then train the fingertip-depth student model."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the incremental video-to-training pipeline.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Pipeline TOML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned work without writing artifacts or training.",
    )
    run_parser.add_argument(
        "--force-train",
        action="store_true",
        help="Run training even when a verified run for the same dataset and config exists.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Inspect discovered videos and cached pipeline artifacts without changing them.",
    )
    status_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Pipeline TOML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser


def _write_summary(summary: object) -> None:
    as_dict = getattr(summary, "as_dict", None)
    if not callable(as_dict):
        raise TypeError("pipeline summary must provide as_dict()")
    json.dump(as_dict(), sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .training_pipeline.config import load_pipeline_config
    from .training_pipeline.orchestrator import PipelineError, pipeline_status, run_pipeline

    try:
        config = load_pipeline_config(args.config, project_root=Path.cwd())
        if args.command == "run":
            summary = run_pipeline(
                config,
                dry_run=args.dry_run,
                force_train=args.force_train,
            )
        else:
            summary = pipeline_status(config)
        _write_summary(summary)
    except (OSError, TypeError, ValueError, PipelineError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
