'深度推定値の統計量を集計し、正解深度との誤差を評価します。'

from __future__ import annotations

import math

import numpy as np


# 有効な深度値の件数、範囲、代表値などの統計を計算します。
def depth_statistics(depth_m: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    values = depth_m[valid]
    if values.size == 0:
        return {
            "valid_pixel_count": 0,
            "min_m": None,
            "median_m": None,
            "mean_m": None,
            "max_m": None,
        }
    return {
        "valid_pixel_count": int(values.size),
        "min_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "mean_m": float(np.mean(values)),
        "max_m": float(np.max(values)),
    }


# 推定深度を正解深度と比較し、誤差と評価指標を返します。
def evaluate_depth(prediction_m: np.ndarray, target_m: np.ndarray) -> dict[str, float | int]:
    if prediction_m.shape != target_m.shape:
        raise ValueError(
            f"prediction/target shapes differ: {prediction_m.shape} vs {target_m.shape}"
        )
    valid = np.isfinite(prediction_m) & np.isfinite(target_m) & (prediction_m > 0) & (target_m > 0)
    if not np.any(valid):
        raise ValueError("ground-truth evaluation has no valid pixels")
    pred = prediction_m[valid].astype(np.float64)
    target = target_m[valid].astype(np.float64)
    error = pred - target
    return {
        "valid_pixel_count": int(pred.size),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(math.sqrt(np.mean(np.square(error)))),
        "abs_rel": float(np.mean(np.abs(error) / target)),
    }
