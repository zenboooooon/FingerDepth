"""Evaluate the fixed Metric3D adapter on the official three-image KITTI demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fingertip_depth.artifacts import depth_preview_bgr, write_json
from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.evaluation import evaluate_depth
from fingertip_depth.metric3d import Metric3Dv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Metric3D data/kitti_demo/test_annotations.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    repository_root = args.annotations.parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = Metric3Dv2(device=args.device)
    results: list[dict[str, object]] = []
    all_prediction: list[np.ndarray] = []
    all_target: list[np.ndarray] = []

    for sample in annotation["files"]:
        rgb_path = repository_root / sample["rgb"]
        target_path = repository_root / sample["depth"]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        target_raw = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or target_raw is None:
            raise ValueError(f"unable to read sample {rgb_path} / {target_path}")
        fx, fy, cx, cy = sample["cam_in"]
        camera = CameraIntrinsics(float(fx), float(fy), float(cx), float(cy))
        prediction = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), camera)
        target_m = target_raw.astype(np.float32) / float(sample["depth_scale"])
        metrics = evaluate_depth(prediction.depth_m, target_m)
        valid = target_m > 0
        all_prediction.append(prediction.depth_m[valid])
        all_target.append(target_m[valid])
        stem = rgb_path.stem
        np.save(args.output_dir / f"{stem}_depth_m.npy", prediction.depth_m)
        cv2.imwrite(
            str(args.output_dir / f"{stem}_depth_preview.png"),
            depth_preview_bgr(prediction.depth_m),
        )
        results.append(
            {
                "image": str(rgb_path.resolve()),
                "inference_ms": prediction.inference_ms,
                "metrics": metrics,
            }
        )

    combined_metrics = evaluate_depth(
        np.concatenate(all_prediction),
        np.concatenate(all_target),
    )
    summary = {
        "sample_count": len(results),
        "samples": results,
        "combined_metrics": combined_metrics,
        "mean_inference_ms": float(np.mean([item["inference_ms"] for item in results])),
        "median_inference_ms": float(np.median([item["inference_ms"] for item in results])),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
