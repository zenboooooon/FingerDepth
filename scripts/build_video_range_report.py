"""Build an audited report for a video with a known approximate depth range."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fingertip_depth.artifacts import write_json
from fingertip_depth.video_cache import VideoFrameCache, sha256_file

_CONDITIONS = (
    (
        "metric3d_v2_s__approx_k",
        "Metric3D v2-S / approximate K",
        "Metric3D v2-S / 近似K",
    ),
    (
        "unidepth_v2_l__approx_k",
        "UniDepth V2-L / approximate K",
        "UniDepth V2-L / 近似K",
    ),
    (
        "unidepth_v2_l__no_camera",
        "UniDepth V2-L / no camera",
        "UniDepth V2-L / カメラ指定なし",
    ),
    (
        "depth_pro__approx_focal",
        "Depth Pro / approximate focal",
        "Depth Pro / 近似焦点",
    ),
    (
        "depth_pro__estimated_focal",
        "Depth Pro / estimated focal",
        "Depth Pro / 焦点推定",
    ),
)

_COLORS = (
    (35, 35, 35),
    (204, 102, 0),
    (0, 153, 230),
    (0, 130, 70),
    (170, 70, 170),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not an object: {path}")
            if int(value.get("frame_index", -1)) != len(records):
                raise ValueError(f"frame indices are not contiguous from zero: {path}")
            records.append(value)
    if not records:
        raise ValueError(f"records are empty: {path}")
    return records


def _phase2_summary(value: dict[str, Any], *, path: Path) -> dict[str, Any]:
    if "phase2_finger_movement" in value:
        phase2 = value["phase2_finger_movement"]
    elif "phase2" in value:
        phase2 = value["phase2"]
    else:
        phase2 = value
    if not isinstance(phase2, dict) or "source_sha256" not in phase2:
        raise ValueError(f"unable to find a Phase 2 summary in {path}")
    return phase2


def _coordinate_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": record["frame_index"],
        "timestamp_ms": record["timestamp_ms"],
        "width": record["width"],
        "height": record["height"],
        "hand_detected": record.get("hand_detected", bool(record.get("fingertips"))),
        "fingertips": [
            {
                key: value
                for key, value in fingertip.items()
                if key not in {"depth_m", "depth_valid", "depth_error"}
            }
            for fingertip in record.get("fingertips", [])
        ],
    }


def _coordinate_digest(records: Iterable[dict[str, Any]]) -> str:
    canonical = json.dumps(
        [_coordinate_payload(record) for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sequence_digest(values: Iterable[str]) -> str:
    return hashlib.sha256("".join(values).encode("ascii")).hexdigest()


def _depth_series(records: Iterable[dict[str, Any]]) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in records:
        fingertips = record.get("fingertips") or []
        if not fingertips:
            continue
        fingertip = fingertips[0]
        value = fingertip.get("depth_m")
        if (
            fingertip.get("depth_valid")
            and value is not None
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            values[int(record["frame_index"])] = float(value)
    return values


def calculate_band_metrics(
    depths_m: Sequence[float] | np.ndarray,
    *,
    total_frames: int,
    expected_min_m: float,
    expected_max_m: float,
) -> dict[str, Any]:
    """Classify valid depths and measure their minimum error to a closed interval.

    The below/in-range/above rates use valid frames as their denominator. The
    unsigned violation is ``abs(z - clip(z, expected_min, expected_max))``;
    therefore it is a lower bound on absolute error when only a range is known.
    """

    if not math.isfinite(expected_min_m) or not math.isfinite(expected_max_m):
        raise ValueError("expected range bounds must be finite")
    if expected_min_m < 0.0 or expected_max_m <= expected_min_m:
        raise ValueError("expected_max_m must be greater than a non-negative expected_min_m")
    values = np.asarray(depths_m, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("at least one valid depth is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("valid depths must be finite")
    if total_frames < values.size or total_frames <= 0:
        raise ValueError("total_frames must be positive and at least the valid-depth count")

    below = values < expected_min_m
    above = values > expected_max_m
    in_range = ~(below | above)
    clipped = np.clip(values, expected_min_m, expected_max_m)
    signed_violation = values - clipped
    violation = np.abs(signed_violation)
    valid_count = int(values.size)

    return {
        "total_frames": int(total_frames),
        "valid_frames": valid_count,
        "valid_rate": valid_count / total_frames,
        "classification_rate_denominator": "valid_frames",
        "below_range_frames": int(np.count_nonzero(below)),
        "below_range_rate": float(np.mean(below)),
        "in_range_frames": int(np.count_nonzero(in_range)),
        "in_range_rate": float(np.mean(in_range)),
        "above_range_frames": int(np.count_nonzero(above)),
        "above_range_rate": float(np.mean(above)),
        "band_violation_m": {
            "mean": float(np.mean(violation)),
            "median": float(np.median(violation)),
            "p95": float(np.percentile(violation, 95)),
            "rmse": float(np.sqrt(np.mean(np.square(violation)))),
            "signed_mean": float(np.mean(signed_violation)),
        },
    }


def _depth_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("at least one valid depth is required")
    p05, median, p95 = np.percentile(array, [5, 50, 95])
    return {
        "min": float(np.min(array)),
        "p05": float(p05),
        "median": float(median),
        "p95": float(p95),
        "max": float(np.max(array)),
        "p95_minus_p05": float(p95 - p05),
        "max_minus_min": float(np.max(array) - np.min(array)),
    }


def _stable_delta_summary(
    records: list[dict[str, Any]], depths: dict[int, float]
) -> dict[str, float | int]:
    threshold_px = 0.10 * math.hypot(
        int(records[0]["width"]), int(records[0]["height"])
    )
    deltas: list[float] = []
    for previous, current in pairwise(records):
        first = int(previous["frame_index"])
        second = int(current["frame_index"])
        if second != first + 1 or first not in depths or second not in depths:
            continue
        previous_tips = previous.get("fingertips") or []
        current_tips = current.get("fingertips") or []
        if not previous_tips or not current_tips:
            continue
        pixel_delta = math.hypot(
            int(current_tips[0]["u_px"]) - int(previous_tips[0]["u_px"]),
            int(current_tips[0]["v_px"]) - int(previous_tips[0]["v_px"]),
        )
        if pixel_delta <= threshold_px:
            deltas.append(abs(depths[second] - depths[first]))
    if not deltas:
        raise ValueError("no stable consecutive valid-depth pairs are available")
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "coordinate_step_threshold_px": threshold_px,
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _inference_summary(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    values = np.asarray(
        [
            float(record["depth_inference_ms"])
            for record in records
            if record.get("depth_inference_ms") is not None
            and math.isfinite(float(record["depth_inference_ms"]))
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("no finite inference timings are available")
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _draw_legend(canvas: np.ndarray, labels: list[str], *, y: int) -> None:
    x = 100
    for label, color in zip(labels, _COLORS):
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        entry_width = 76 + width
        if x + entry_width > canvas.shape[1] - 50:
            x = 100
            y += 29
        cv2.line(canvas, (x, y), (x + 34, y), color, 4, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (x + 42, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        x += entry_width


def _draw_time_series(
    output_path: Path,
    *,
    records: list[dict[str, Any]],
    depth_series: list[dict[int, float]],
    labels: list[str],
    expected_min_m: float,
    expected_max_m: float,
) -> None:
    canvas = np.full((1020, 1800, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 115, 165, 1735, 900
    timestamps = {
        int(record["frame_index"]): int(record["timestamp_ms"]) / 1000.0
        for record in records
    }
    x_max = max(timestamps.values())
    observed_max = max(value for series in depth_series for value in series.values())
    y_max = max(expected_max_m * 1.25, observed_max * 1.05)
    y_max = math.ceil(y_max * 10.0) / 10.0
    y_tick = 0.1 if y_max <= 1.6 else 0.2
    x_map = lambda value: left + value * (right - left) / x_max
    y_map = lambda value: bottom - value * (bottom - top) / y_max

    band_top = round(y_map(expected_max_m))
    band_bottom = round(y_map(expected_min_m))
    cv2.rectangle(
        canvas,
        (left, band_top),
        (right, band_bottom),
        (226, 246, 226),
        cv2.FILLED,
    )
    for value in np.arange(0.0, y_max + y_tick / 2, y_tick):
        y = round(y_map(float(value)))
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:.1f}",
            (left - 76, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
    for value in np.arange(0.0, math.floor(x_max) + 1.0, 1.0):
        x = round(x_map(float(value)))
        cv2.line(canvas, (x, top), (x, bottom), (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:.0f}",
            (x - 12, bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
    for value in (expected_min_m, expected_max_m):
        cv2.line(
            canvas,
            (left, round(y_map(value))),
            (right, round(y_map(value))),
            (70, 155, 70),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)

    for series, color in zip(depth_series, _COLORS):
        previous_index: int | None = None
        previous_point: tuple[int, int] | None = None
        for frame_index in sorted(series):
            point = (
                round(x_map(timestamps[frame_index])),
                round(y_map(series[frame_index])),
            )
            if previous_index is not None and frame_index == previous_index + 1:
                assert previous_point is not None
                cv2.line(canvas, previous_point, point, color, 2, cv2.LINE_AA)
            previous_index = frame_index
            previous_point = point

    cv2.putText(
        canvas,
        "Fingertip depth vs. expected range",
        (115, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"expected band: {expected_min_m:.2f}-{expected_max_m:.2f} m",
        (115, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (45, 125, 45),
        2,
        cv2.LINE_AA,
    )
    _draw_legend(canvas, labels, y=130)
    cv2.putText(
        canvas,
        "Video time (s)",
        ((left + right) // 2 - 65, bottom + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "INDEX_FINGER_TIP depth (m)",
        (left + 12, top + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write chart: {output_path}")


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _markdown_report(result: dict[str, Any], *, chart_link: str) -> str:
    conditions = result["conditions"]
    lower = result["expected_range_m"]["min"]
    upper = result["expected_range_m"]["max"]
    best = min(conditions, key=lambda item: item["band"]["band_violation_m"]["rmse"])
    best_rate = max(conditions, key=lambda item: item["band"]["in_range_rate"])
    audit = result["audit"]
    lines = [
        "# 指先距離帯動画の5条件比較",
        "",
        "## 結論",
        "",
        (
            f"撮影時の指先距離を約 **{lower:.2f}–{upper:.2f} m** とする区間制約では、"
            f"帯違反RMSEが最小なのは **{best['label_ja']}** "
            f"({_f(best['band']['band_violation_m']['rmse'])} m)、帯内率が最大なのは "
            f"**{best_rate['label_ja']}** ({100.0 * best_rate['band']['in_range_rate']:.1f}%)でした。"
        ),
        (
            "これは「真値が各フレームで区間内」という情報に対する整合性の比較です。"
            "帯内の予測であっても、時刻ごとの距離、移動量、前後方向の正しさは証明しません。"
        ),
        "",
        f"![{lower:.2f}–{upper:.2f} m距離帯と指先深度]({chart_link})",
        "",
        "## 距離帯判定",
        "",
        "below / 帯内 / aboveの率は、深度が有効なフレームを分母としています。境界値は帯内です。",
        "",
        f"| 条件 | 有効 | {lower:.2f} m未満 | 帯内 | {upper:.2f} m超 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in conditions:
        band = item["band"]
        lines.append(
            f"| {item['label_ja']} | {band['valid_frames']}/{band['total_frames']} "
            f"({100.0 * band['valid_rate']:.1f}%) | "
            f"{band['below_range_frames']} ({100.0 * band['below_range_rate']:.1f}%) | "
            f"{band['in_range_frames']} ({100.0 * band['in_range_rate']:.1f}%) | "
            f"{band['above_range_frames']} ({100.0 * band['above_range_rate']:.1f}%) |"
        )
    lines.extend(
        [
            "",
            "## 帯違反距離",
            "",
            (
                f"`violation = |z - clip(z, {lower:.2f}, {upper:.2f})|` です。"
                "帯内は0、帯外は最密境界までの距離で、"
                "フレーム真値が未知なため絶対誤差の下限です。signed meanの正値は主に過大推定を示します。"
            ),
            "",
            "| 条件 | mean (m) | median (m) | p95 (m) | RMSE (m) | signed mean (m) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in conditions:
        violation = item["band"]["band_violation_m"]
        lines.append(
            f"| {item['label_ja']} | {_f(violation['mean'])} | {_f(violation['median'])} | "
            f"{_f(violation['p95'])} | {_f(violation['rmse'])} | "
            f"{violation['signed_mean']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 分布・連続性・速度",
            "",
            "rangeはp95−p05、安定追跡差は指先座標の移動が画像対角の10%以下の連続有効フレーム間で計算しています。",
            "",
            "| 条件 | depth p05 / median / p95 (m) | range (m) | 安定追跡差 median / p95 (m) | 推論時間 median (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in conditions:
        depth = item["depth_m"]
        stable = item["stable_tracking_adjacent_absolute_delta_m"]
        inference = item["inference_ms"]
        lines.append(
            f"| {item['label_ja']} | {_f(depth['p05'])} / {_f(depth['median'])} / "
            f"{_f(depth['p95'])} | {_f(depth['p95_minus_p05'])} | "
            f"{_f(stable['median'])} / {_f(stable['p95'])} | "
            f"{_f(inference['median'], 1)} |"
        )
    lines.extend(
        [
            "",
            "推論時間の計測範囲は完全に同一ではなく、Metric3Dはnetwork forward、他モデルは公式 `infer` の処理時間です。",
            "",
            "## 入力同一性の監査",
            "",
            f"- 動画SHA-256: `{audit['source']['sha256']}`",
            f"- baseline frames SHA-256: `{audit['coordinates']['baseline_records_sha256']}`",
            f"- 5条件共通の座標digest: `{audit['coordinates']['canonical_sha256']}`",
            f"- lossless PNG cache manifest SHA-256: `{audit['frame_cache']['manifest_sha256']}`",
            f"- 5条件共通のBGR画素hash列digest: `{audit['input_pixels']['sequence_sha256']}`",
            (
                f"- {audit['source']['frame_count']}フレームすべてで動画・時刻・画像サイズ・座標列・入力画素hashが一致。"
                f"cache PNGと復号後BGRも全件再検証: "
                f"`{audit['frame_cache']['all_png_and_bgr_verified']}`。"
            ),
            "",
            "## 解釈上の制約",
            "",
            (
                f"- {lower:.2f}–{upper:.2f} mは約値の区間GTで、"
                "各フレームの独立した測定値ではありません。"
            ),
            "- フレームは時系列上強く自己相関するため、独立な606標本として有意差検定はしていません。",
            "- 動画の36 mm相当値から得た焦点距離は近似値で、校正済みKではありません。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, conditions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "condition_id",
                "label_ja",
                "total_frames",
                "valid_frames",
                "valid_rate",
                "below_range_frames",
                "below_range_rate",
                "in_range_frames",
                "in_range_rate",
                "above_range_frames",
                "above_range_rate",
                "band_violation_mean_m",
                "band_violation_median_m",
                "band_violation_p95_m",
                "band_violation_rmse_m",
                "band_violation_signed_mean_m",
                "depth_p05_m",
                "depth_median_m",
                "depth_p95_m",
                "depth_p95_minus_p05_m",
                "stable_delta_count",
                "stable_delta_median_m",
                "stable_delta_p95_m",
                "stable_delta_max_m",
                "inference_median_ms",
                "inference_p95_ms",
            ],
        )
        writer.writeheader()
        for item in conditions:
            band = item["band"]
            violation = band["band_violation_m"]
            depth = item["depth_m"]
            stable = item["stable_tracking_adjacent_absolute_delta_m"]
            inference = item["inference_ms"]
            writer.writerow(
                {
                    "condition_id": item["id"],
                    "label_ja": item["label_ja"],
                    "total_frames": band["total_frames"],
                    "valid_frames": band["valid_frames"],
                    "valid_rate": band["valid_rate"],
                    "below_range_frames": band["below_range_frames"],
                    "below_range_rate": band["below_range_rate"],
                    "in_range_frames": band["in_range_frames"],
                    "in_range_rate": band["in_range_rate"],
                    "above_range_frames": band["above_range_frames"],
                    "above_range_rate": band["above_range_rate"],
                    "band_violation_mean_m": violation["mean"],
                    "band_violation_median_m": violation["median"],
                    "band_violation_p95_m": violation["p95"],
                    "band_violation_rmse_m": violation["rmse"],
                    "band_violation_signed_mean_m": violation["signed_mean"],
                    "depth_p05_m": depth["p05"],
                    "depth_median_m": depth["median"],
                    "depth_p95_m": depth["p95"],
                    "depth_p95_minus_p05_m": depth["p95_minus_p05"],
                    "stable_delta_count": stable["count"],
                    "stable_delta_median_m": stable["median"],
                    "stable_delta_p95_m": stable["p95"],
                    "stable_delta_max_m": stable["max"],
                    "inference_median_ms": inference["median"],
                    "inference_p95_ms": inference["p95"],
                }
            )


def build_video_range_report(
    *,
    baseline_summary_path: Path,
    baseline_records_path: Path,
    comparison_dir: Path,
    expected_min_m: float,
    expected_max_m: float,
    output_dir: Path,
    report_path: Path,
    verify_cache_files: bool = True,
) -> dict[str, Any]:
    baseline_summary = _phase2_summary(
        _load_json(baseline_summary_path), path=baseline_summary_path
    )
    baseline_records = _load_records(baseline_records_path)
    baseline_records_sha256 = sha256_file(baseline_records_path)
    source_path = Path(baseline_summary["source"])
    source_sha256 = sha256_file(source_path)
    if source_sha256 != baseline_summary["source_sha256"]:
        raise ValueError("baseline source video SHA-256 does not match the source file")
    if int(baseline_summary["processed_frames"]) != len(baseline_records):
        raise ValueError("baseline summary and records have different frame counts")

    record_sets = [baseline_records]
    summaries = [baseline_summary]
    manifest_paths: list[Path] = []
    manifest_hashes: list[str] = []
    expected_coordinate_digest = _coordinate_digest(baseline_records)

    for condition_id, _label, _label_ja in _CONDITIONS[1:]:
        summary_path = comparison_dir / condition_id / "summary.json"
        summary = _phase2_summary(_load_json(summary_path), path=summary_path)
        records_path = (
            comparison_dir / condition_id / "phase2_finger_movement" / "frames.jsonl"
        )
        records = _load_records(records_path)
        if summary.get("condition", {}).get("id") != condition_id:
            raise ValueError(f"condition ID differs for {condition_id}")
        if summary["source_sha256"] != source_sha256:
            raise ValueError(f"source video hash differs for {condition_id}")
        if int(summary["processed_frames"]) != len(baseline_records):
            raise ValueError(f"frame count differs for {condition_id}")
        coordinate_source = summary["baseline_coordinate_source"]
        for key in ("verified_sha256", "expected_sha256"):
            if coordinate_source.get(key) != baseline_records_sha256:
                raise ValueError(f"baseline records hash differs for {condition_id}: {key}")
        for key in (
            "coordinate_detection_sha256",
            "selected_coordinate_detection_sha256",
        ):
            if coordinate_source.get(key) != expected_coordinate_digest:
                raise ValueError(f"coordinate digest differs for {condition_id}: {key}")
        if coordinate_source.get("media_pipe_rerun") is not False:
            raise ValueError(f"MediaPipe coordinates were not reused for {condition_id}")

        decoder = summary["video_decoder"]
        if decoder.get("mode") != "lossless_png_frame_cache":
            raise ValueError(f"shared lossless frame cache was not used for {condition_id}")
        if decoder.get("png_file_sha256_verified") is not True:
            raise ValueError(f"cached PNG verification is not recorded for {condition_id}")
        if decoder.get("bgr_pixel_sha256_verified") is not True:
            raise ValueError(f"cached BGR verification is not recorded for {condition_id}")
        manifest_path = Path(decoder["manifest_path"]).resolve()
        manifest_hash = sha256_file(manifest_path)
        if manifest_hash != decoder.get("manifest_sha256"):
            raise ValueError(f"frame-cache manifest hash differs for {condition_id}")
        manifest_paths.append(manifest_path)
        manifest_hashes.append(manifest_hash)
        summaries.append(summary)
        record_sets.append(records)

    if len(set(manifest_paths)) != 1 or len(set(manifest_hashes)) != 1:
        raise ValueError("candidate conditions did not use one identical frame-cache manifest")
    manifest_path = manifest_paths[0]
    manifest_hash = manifest_hashes[0]
    cache = VideoFrameCache.load(
        manifest_path,
        input_path=source_path,
        baseline_records_sha256=baseline_records_sha256,
        expected_manifest_sha256=manifest_hash,
    )
    if cache.frame_count != len(baseline_records):
        raise ValueError("frame-cache and baseline records have different frame counts")
    if verify_cache_files:
        cache.verify_all()

    manifest_frames = cache.manifest["frames"]
    manifest_pixels = tuple(str(frame["bgr_pixel_sha256"]) for frame in manifest_frames)
    input_sequences: list[tuple[str, ...]] = []
    coordinate_digests: list[str] = []
    for condition, records in zip(_CONDITIONS, record_sets):
        condition_id = condition[0]
        sequence = tuple(str(record.get("input_bgr_pixel_sha256", "")) for record in records)
        if not all(sequence):
            raise ValueError(f"input BGR hash is missing for {condition_id}")
        if sequence != manifest_pixels:
            raise ValueError(f"input BGR pixels differ from the cache for {condition_id}")
        for record, manifest_frame in zip(records, manifest_frames):
            if (
                int(record["timestamp_ms"]) != int(manifest_frame["timestamp_ms"])
                or int(record["width"]) != int(manifest_frame["width"])
                or int(record["height"]) != int(manifest_frame["height"])
            ):
                raise ValueError(f"frame metadata differs from the cache for {condition_id}")
        input_sequences.append(sequence)
        coordinate_digests.append(_coordinate_digest(records))
    if len(set(input_sequences)) != 1:
        raise ValueError("condition input BGR pixel sequences differ")
    if len(set(coordinate_digests)) != 1:
        raise ValueError("condition fingertip coordinate sequences differ")

    depth_series = [_depth_series(records) for records in record_sets]
    valid_index_sets = [tuple(sorted(series)) for series in depth_series]
    if len(set(valid_index_sets)) != 1:
        raise ValueError("condition valid-depth frame indices differ")
    conditions: list[dict[str, Any]] = []
    for definition, summary, records, series in zip(
        _CONDITIONS, summaries, record_sets, depth_series
    ):
        condition_id, label, label_ja = definition
        ordered_depths = [series[index] for index in sorted(series)]
        band = calculate_band_metrics(
            ordered_depths,
            total_frames=len(records),
            expected_min_m=expected_min_m,
            expected_max_m=expected_max_m,
        )
        conditions.append(
            {
                "id": condition_id,
                "label": label,
                "label_ja": label_ja,
                "model": summary["depth_model"],
                "band": band,
                "depth_m": _depth_summary(ordered_depths),
                "stable_tracking_adjacent_absolute_delta_m": _stable_delta_summary(
                    records, series
                ),
                "inference_ms": _inference_summary(records),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "video_range_report.json"
    csv_path = output_dir / "video_range_report.csv"
    chart_path = output_dir / "video_range_timeseries.png"
    result: dict[str, Any] = {
        "experiment": "Five-condition fingertip video expected-range evaluation",
        "expected_range_m": {
            "min": expected_min_m,
            "max": expected_max_m,
            "bounds_inclusive": True,
            "ground_truth_type": "approximate interval; no per-frame trajectory",
        },
        "source": str(source_path.resolve()),
        "conditions": conditions,
        "audit": {
            "source": {
                "sha256": source_sha256,
                "frame_count": len(baseline_records),
                "all_condition_hashes_identical": True,
            },
            "coordinates": {
                "baseline_records_sha256": baseline_records_sha256,
                "canonical_sha256": coordinate_digests[0],
                "all_condition_hashes_identical": True,
            },
            "frame_cache": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "all_candidate_manifest_hashes_identical": True,
                "all_png_and_bgr_verified": verify_cache_files,
            },
            "input_pixels": {
                "sequence_sha256": _sequence_digest(input_sequences[0]),
                "all_condition_hashes_identical": True,
                "all_condition_hashes_match_manifest": True,
            },
            "valid_depth_frames": {
                "frame_count": len(valid_index_sets[0]),
                "all_condition_indices_identical": True,
            },
        },
        "artifacts": {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
            "timeseries_png": str(chart_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    write_json(json_path, result)
    _write_csv(csv_path, conditions)
    _draw_time_series(
        chart_path,
        records=baseline_records,
        depth_series=depth_series,
        labels=[item[1] for item in _CONDITIONS],
        expected_min_m=expected_min_m,
        expected_max_m=expected_max_m,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    chart_link = Path(os.path.relpath(chart_path, report_path.parent)).as_posix()
    report_path.write_text(
        _markdown_report(result, chart_link=chart_link),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("outputs/iphone_phase1_2_2030/summary.json"),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path(
            "outputs/iphone_phase1_2_2030/phase2_finger_movement/frames.jsonl"
        ),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("outputs/depth_model_comparison_2030"),
    )
    parser.add_argument("--expected-min-m", type=float, default=0.20)
    parser.add_argument("--expected-max-m", type=float, default=0.30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/depth_model_comparison_2030"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/finger_movement_2030_range_results.md"),
    )
    parser.add_argument(
        "--skip-cache-file-verification",
        action="store_true",
        help="Skip re-reading every cached PNG; manifest and recorded pixel hashes are still checked.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_video_range_report(
        baseline_summary_path=args.baseline_summary,
        baseline_records_path=args.baseline_records,
        comparison_dir=args.comparison_dir,
        expected_min_m=args.expected_min_m,
        expected_max_m=args.expected_max_m,
        output_dir=args.output_dir,
        report_path=args.report,
        verify_cache_files=not args.skip_cache_file_verification,
    )
    print(
        json.dumps(
            {
                "conditions": len(result["conditions"]),
                "expected_range_m": result["expected_range_m"],
                "audit": result["audit"],
                "artifacts": result["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
