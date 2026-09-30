"""Export the latest audited student checkpoint as an iPhone-ready Core ML package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fingertip_depth.coreml_export import (
    TORCH_EXPORT_BACKEND,
    convert_exported_program_to_coreml,
    export_student_for_coreml,
    load_student_export_source,
    make_export_wrapper,
    rebuild_parity_fixture,
    write_export_manifest,
)

DEFAULT_RUN_MANIFEST = Path("outputs/student_retrain_max_depth_080m_epochs40_patience12/run_manifest.json")
DEFAULT_OUTPUT_DIR = Path("outputs/student_depth_coreml_latest")
DEFAULT_PARITY_SOURCE_FIXTURE = Path(
    "outputs/phase8_student_coreml_iphone15/coreml_parity_inputs.npz"
)
DEFAULT_MODEL_ID = "latest_training_pipeline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--parity-source-fixture",
        type=Path,
        default=DEFAULT_PARITY_SOURCE_FIXTURE,
        help="existing audited fixture whose RGB/landmark inputs are reused",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
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
    if not args.model_id.strip():
        raise ValueError("--model-id must not be empty")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix an export with existing files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_program_path = output_dir / "StudentDepthLatest.pt2"
    coreml_path = output_dir / "StudentDepthLatest.mlpackage"
    parity_fixture_path = output_dir / "coreml_parity_inputs.npz"
    manifest_path = output_dir / "export_manifest.json"

    _progress("loading and verifying the selected latest checkpoint")
    source = load_student_export_source(args.run_manifest)
    wrapper = make_export_wrapper(source)

    _progress("capturing a strict fixed-shape ATEN ExportedProgram")
    exported, export_validation = export_student_for_coreml(
        wrapper,
        output_path=exported_program_path,
    )

    _progress("converting ExportedProgram to a float16 Core ML ML Program")
    coreml_conversion = convert_exported_program_to_coreml(
        exported,
        output_path=coreml_path,
        checkpoint_sha256=source.checkpoint_sha256,
        model_id=args.model_id,
        run_manifest_sha256=source.run_manifest_sha256,
    )

    _progress("reusing audited 604-frame inputs and recomputing latest PyTorch predictions")
    parity_fixture = rebuild_parity_fixture(
        wrapper,
        args.parity_source_fixture,
        output_path=parity_fixture_path,
        batch_size=args.parity_batch_size,
        limit=args.parity_limit,
        progress=_progress,
    )

    manifest = write_export_manifest(
        output_path=manifest_path,
        source=source,
        torchscript_path=None,
        coreml_path=coreml_path,
        parity_fixture_path=parity_fixture_path,
        trace_validation=None,
        coreml_conversion=coreml_conversion,
        parity_fixture=parity_fixture,
        exported_program_path=exported_program_path,
        export_validation=export_validation,
        model_id=args.model_id,
        export_backend=TORCH_EXPORT_BACKEND,
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
