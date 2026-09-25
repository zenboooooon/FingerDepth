"""Export the audited Phase 8 student as an iPhone-ready Core ML package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fingertip_depth.coreml_export import (
    StudentCoreMLWrapper,
    build_parity_fixture,
    convert_torchscript_to_coreml,
    load_parity_observations,
    load_student_export_source,
    make_coreml_wrapper,
    trace_student_for_coreml,
    write_export_manifest,
)

DEFAULT_RUN_MANIFEST = Path(
    "outputs/phase8_student_vit_landmarks_xy_spike_filtered_chronological_tail7/run_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/phase8_student_coreml_iphone15")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--parity-limit",
        type=int,
        default=0,
        help="number of chronological 2030 frames to package; 0 means all 604",
    )
    parser.add_argument("--parity-batch-size", type=int, default=16)
    return parser


def _progress(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parity_limit < 0:
        raise ValueError("--parity-limit must be non-negative")
    if args.parity_batch_size <= 0:
        raise ValueError("--parity-batch-size must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix an export with existing files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torchscript_path = output_dir / "StudentDepth.trace.pt"
    coreml_path = output_dir / "StudentDepth.mlpackage"
    parity_fixture_path = output_dir / "coreml_parity_inputs.npz"
    manifest_path = output_dir / "export_manifest.json"

    _progress("loading and verifying the selected Phase 8 checkpoint")
    source = load_student_export_source(args.run_manifest)
    wrapper = make_coreml_wrapper(source)
    reference_wrapper = StudentCoreMLWrapper(
        source.model,
        image_mean=source.training_config.image_mean,
        image_std=source.training_config.image_std,
    ).eval()

    _progress("tracing the fixed-shape deployment wrapper")
    traced, trace_validation = trace_student_for_coreml(
        wrapper,
        output_path=torchscript_path,
    )

    _progress("converting TorchScript to a float16 Core ML ML Program")
    coreml_conversion = convert_torchscript_to_coreml(
        traced,
        output_path=coreml_path,
        checkpoint_sha256=source.checkpoint_sha256,
    )

    _progress("building self-contained parity vectors from finger_movement_2030")
    observations = load_parity_observations(source.run_manifest_path)
    if args.parity_limit:
        observations = observations[: args.parity_limit]
    parity_fixture = build_parity_fixture(
        reference_wrapper,
        observations,
        output_path=parity_fixture_path,
        batch_size=args.parity_batch_size,
        progress=_progress,
    )

    manifest = write_export_manifest(
        output_path=manifest_path,
        source=source,
        torchscript_path=torchscript_path,
        coreml_path=coreml_path,
        parity_fixture_path=parity_fixture_path,
        trace_validation=trace_validation,
        coreml_conversion=coreml_conversion,
        parity_fixture=parity_fixture,
    )
    print(
        json.dumps(
            {
                "coreml_model": str(coreml_path),
                "parity_fixture": str(parity_fixture_path),
                "manifest": str(manifest_path),
                "sample_count": manifest["validation"]["parity_fixture"]["sample_count"],
                "runtime_parity": "pending macOS/iPhone",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
