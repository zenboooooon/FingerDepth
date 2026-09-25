"""Train the Phase 8 single-frame ViT and landmark Transformer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fingertip_depth.constants import DEFAULT_FEATURE_LANDMARK_INDICES
from fingertip_depth.hands import parse_hand_landmark_selection
from fingertip_depth.student_model import DEFAULT_IMAGE_ENCODER, StudentModelConfig
from fingertip_depth.student_training import (
    DEFAULT_DATASET_MANIFEST_SHA256,
    StudentTrainingConfig,
    TeacherSpikeFilterConfig,
    train_student_transformer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json"),
    )
    parser.add_argument(
        "--expected-dataset-manifest-sha256",
        default=DEFAULT_DATASET_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase8_student_vit_landmarks_xy"),
    )
    parser.add_argument("--landmarks", default=",".join(map(str, DEFAULT_FEATURE_LANDMARK_INDICES)))
    parser.add_argument("--image-encoder", default=DEFAULT_IMAGE_ENCODER)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=6)
    parser.add_argument("--fusion-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--no-preload-images", action="store_true")
    parser.add_argument("--skip-image-png-sha256", action="store_true")
    parser.add_argument(
        "--exclude-teacher-spikes",
        "--teacher-temporal-spike-filter",
        dest="exclude_teacher_spikes",
        action="store_true",
        help=(
            "exclude isolated teacher-depth excursions using identity-only temporal "
            "Hampel decisions; train HFlip views inherit the identity decision"
        ),
    )
    parser.add_argument("--teacher-spike-frame-radius", type=int, default=3)
    parser.add_argument("--teacher-spike-max-frame-gap", type=int, default=1)
    parser.add_argument("--teacher-spike-min-neighbors", type=int, default=3)
    parser.add_argument("--teacher-spike-absolute-floor-m", type=float, default=0.15)
    parser.add_argument("--teacher-spike-relative-floor-fraction", type=float, default=0.50)
    parser.add_argument("--teacher-spike-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--teacher-spike-mad-scale", type=float, default=1.4826)
    return parser


def _progress(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    landmark_indices = parse_hand_landmark_selection(args.landmarks)
    model_config = StudentModelConfig(
        image_encoder_name=args.image_encoder,
        pretrained_image_encoder=not args.no_pretrained,
        landmark_indices=landmark_indices,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        fusion_mlp_ratio=args.fusion_mlp_ratio,
        dropout=args.dropout,
    )
    training_config = StudentTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        precision=args.precision,
        freeze_image_encoder=args.freeze_image_encoder,
        preload_images=not args.no_preload_images,
        verify_image_png_sha256=not args.skip_image_png_sha256,
    )
    spike_filter_config = TeacherSpikeFilterConfig(
        enabled=args.exclude_teacher_spikes,
        frame_radius=args.teacher_spike_frame_radius,
        max_frame_gap=args.teacher_spike_max_frame_gap,
        min_neighbors=args.teacher_spike_min_neighbors,
        absolute_floor_m=args.teacher_spike_absolute_floor_m,
        relative_floor_fraction=args.teacher_spike_relative_floor_fraction,
        mad_multiplier=args.teacher_spike_mad_multiplier,
        mad_scale=args.teacher_spike_mad_scale,
    )
    result = train_student_transformer(
        dataset_manifest_path=args.dataset_manifest,
        expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
        output_dir=args.output_dir,
        model_config=model_config,
        training_config=training_config,
        spike_filter_config=spike_filter_config,
        device_name=args.device,
        progress=_progress,
    )
    print(
        json.dumps(
            {
                "run_manifest": str(args.output_dir / "run_manifest.json"),
                "best_epoch": result["results"]["best_epoch"],
                "epochs_completed": result["results"]["epochs_completed"],
                "best_validation": result["results"]["best_validation"],
                "validation_baselines": result["results"]["validation_baselines"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
