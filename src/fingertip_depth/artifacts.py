"""Output serialization and visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def depth_preview_bgr(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth_m[valid]
        low, high = np.percentile(values, [2.0, 98.0])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("depth map contains invalid percentile values")
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
        preview[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def annotate_fingertip(
    bgr: np.ndarray,
    *,
    u_px: int,
    v_px: int,
    depth_m: float,
    handedness: str | None,
) -> np.ndarray:
    annotated = bgr.copy()
    color = (0, 255, 255)
    cv2.circle(annotated, (u_px, v_px), 8, color, 2, lineType=cv2.LINE_AA)
    label = f"index tip: {depth_m:.3f} m"
    if handedness:
        label = f"{handedness} {label}"
    text_x = min(max(u_px + 10, 0), max(0, annotated.shape[1] - 280))
    text_y = min(max(v_px - 10, 24), max(24, annotated.shape[0] - 8))
    cv2.putText(
        annotated,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated
