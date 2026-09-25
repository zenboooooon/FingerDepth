"""Build the audited five-condition comparison tables, figures, and report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fingertip_depth.artifacts import write_json
from fingertip_depth.image_io import read_bgr
from fingertip_depth.video_cache import pixel_sha256

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


def _records(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not an object: {path}")
            values.append(value)
    if not values:
        raise ValueError(f"records are empty: {path}")
    return values


def _depth_series(records: Iterable[dict[str, Any]]) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in records:
        fingertips = record.get("fingertips") or []
        if not fingertips:
            continue
        fingertip = fingertips[0]
        depth = fingertip.get("depth_m")
        if fingertip.get("depth_valid") and depth is not None and math.isfinite(depth):
            values[int(record["frame_index"])] = float(depth)
    return values


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
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(
            _coordinate_payload(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation inputs must have the same length >= 2")
    return float(np.corrcoef(left, right)[0, 1])


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("numeric summary cannot be empty")
    mean = float(np.mean(array))
    return {
        "count": int(array.size),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": mean,
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "std": float(np.std(array)),
        "coefficient_of_variation": float(np.std(array) / abs(mean)),
    }


def _extra_series(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = (record.get("prediction_extras") or {}).get(key)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def _stable_direction_agreement(
    baseline_records: list[dict[str, Any]],
    baseline_depth: dict[int, float],
    candidate_depth: dict[int, float],
) -> dict[str, float | int]:
    threshold_px = 0.10 * math.hypot(
        int(baseline_records[0]["width"]), int(baseline_records[0]["height"])
    )
    comparisons: list[bool] = []
    for previous, current in pairwise(baseline_records):
        first = int(previous["frame_index"])
        second = int(current["frame_index"])
        if second != first + 1:
            continue
        if first not in baseline_depth or second not in baseline_depth:
            continue
        if first not in candidate_depth or second not in candidate_depth:
            continue
        previous_tip = previous["fingertips"][0]
        current_tip = current["fingertips"][0]
        pixel_delta = math.hypot(
            int(current_tip["u_px"]) - int(previous_tip["u_px"]),
            int(current_tip["v_px"]) - int(previous_tip["v_px"]),
        )
        if pixel_delta > threshold_px:
            continue
        baseline_delta = baseline_depth[second] - baseline_depth[first]
        candidate_delta = candidate_depth[second] - candidate_depth[first]
        if baseline_delta == 0.0 or candidate_delta == 0.0:
            continue
        comparisons.append((baseline_delta > 0) == (candidate_delta > 0))
    return {
        "pair_count": len(comparisons),
        "agreement_rate": float(np.mean(comparisons)),
    }


def _draw_axes(
    canvas: np.ndarray,
    *,
    bounds: tuple[int, int, int, int],
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    x_map: Any,
    y_map: Any,
    x_title: str,
    y_title: str,
) -> None:
    left, top, right, bottom = bounds
    for value, label in y_ticks:
        y = round(y_map(value))
        cv2.line(canvas, (left, y), (right, y), (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (left - 70, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (55, 55, 55), 1, cv2.LINE_AA,
        )
    for value, label in x_ticks:
        x = round(x_map(value))
        cv2.line(canvas, (x, top), (x, bottom), (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (x - 16, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (55, 55, 55), 1, cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
    cv2.putText(
        canvas, x_title, ((left + right) // 2 - 70, bottom + 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, y_title, (15, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (35, 35, 35), 1, cv2.LINE_AA,
    )


def _draw_legend(canvas: np.ndarray, labels: list[str], *, y: int) -> None:
    x = 95
    for label, color in zip(labels, _COLORS):
        label_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )[0][0]
        entry_width = 34 + 42 + label_width
        if x + entry_width > canvas.shape[1] - 40:
            x = 95
            y += 29
        cv2.line(canvas, (x, y), (x + 34, y), color, 4, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (x + 42, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (35, 35, 35), 1, cv2.LINE_AA,
        )
        x += entry_width


def _phase1_chart(
    output_path: Path,
    actual: list[float],
    predictions: list[list[float]],
    labels: list[str],
) -> None:
    canvas = np.full((900, 1600, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 105, 145, 1530, 790
    x_min, x_max = 0.2, 1.6
    all_y = [value for row in predictions for value in row] + actual
    y_max = math.ceil(max(all_y) * 2.0) / 2.0
    x_map = lambda value: left + (value - x_min) * (right - left) / (x_max - x_min)
    y_map = lambda value: bottom - value * (bottom - top) / y_max
    _draw_axes(
        canvas,
        bounds=(left, top, right, bottom),
        x_ticks=[(value, f"{value:.1f}") for value in np.arange(0.2, 1.61, 0.2)],
        y_ticks=[(value, f"{value:.1f}") for value in np.arange(0.0, y_max + 0.01, 0.5)],
        x_map=x_map,
        y_map=y_map,
        x_title="Known distance (m)",
        y_title="Green-box ROI median depth (m)",
    )
    cv2.putText(
        canvas, "Phase 1: known-distance response", (105, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (25, 25, 25), 2, cv2.LINE_AA,
    )
    ground_truth = [(round(x_map(value)), round(y_map(value))) for value in actual]
    for start, end in pairwise(ground_truth):
        cv2.line(canvas, start, end, (130, 130, 130), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, "ideal y=x", (round(x_map(1.35)), round(y_map(1.35)) - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA,
    )
    for row, color in zip(predictions, _COLORS):
        points = [(round(x_map(x)), round(y_map(y))) for x, y in zip(actual, row)]
        cv2.polylines(canvas, [np.asarray(points)], False, color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 6, color, cv2.FILLED, cv2.LINE_AA)
    _draw_legend(canvas, labels, y=88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write chart: {output_path}")


def _phase2_chart(
    output_path: Path,
    record_sets: list[list[dict[str, Any]]],
    series: list[dict[int, float]],
    labels: list[str],
) -> None:
    canvas = np.full((980, 1800, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 105, 145, 1735, 865
    timestamps = {
        int(record["frame_index"]): int(record["timestamp_ms"]) / 1000.0
        for record in record_sets[0]
    }
    x_max = max(timestamps.values())
    y_max = math.ceil(max(value for values in series for value in values.values()) * 2.0) / 2.0
    x_map = lambda value: left + value * (right - left) / x_max
    y_map = lambda value: bottom - value * (bottom - top) / y_max
    _draw_axes(
        canvas,
        bounds=(left, top, right, bottom),
        x_ticks=[(value, f"{value:.0f}") for value in np.arange(0.0, x_max + 0.01, 1.0)],
        y_ticks=[(value, f"{value:.1f}") for value in np.arange(0.0, y_max + 0.01, 0.5)],
        x_map=x_map,
        y_map=y_map,
        x_title="Video time (s)",
        y_title="INDEX_FINGER_TIP depth (m)",
    )
    cv2.putText(
        canvas, "Phase 2: shared fingertip coordinates", (105, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (25, 25, 25), 2, cv2.LINE_AA,
    )
    for values, color in zip(series, _COLORS):
        previous_index: int | None = None
        previous_point: tuple[int, int] | None = None
        for frame_index in sorted(values):
            point = (
                round(x_map(timestamps[frame_index])),
                round(y_map(values[frame_index])),
            )
            if previous_index is not None and frame_index == previous_index + 1:
                assert previous_point is not None
                cv2.line(canvas, previous_point, point, color, 2, cv2.LINE_AA)
            previous_index = frame_index
            previous_point = point
    _draw_legend(canvas, labels, y=92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"failed to write chart: {output_path}")


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _markdown_report(comparison: dict[str, Any]) -> str:
    conditions = comparison["conditions"]
    baseline_mae = conditions[0]["phase1"]["aggregate"]["mae_m"]
    best = min(conditions[1:], key=lambda item: item["phase1"]["aggregate"]["mae_m"])
    improvement = 100.0 * (baseline_mae - best["phase1"]["aggregate"]["mae_m"]) / baseline_mae
    lines = [
        "# 単眼metric depthモデル比較結果",
        "",
        "実行日: 2026-09-25",
        "",
        "## 結論",
        "",
        (
            f"この5点の既知距離試験では **{best['label_ja']}** が最小MAE "
            f"**{_f(best['phase1']['aggregate']['mae_m'])} m** で、Metric3D baseline "
            f"({_f(baseline_mae)} m)より **{improvement:.1f}%低い** 結果でした。"
        ),
        (
            "ただし、全5条件で距離に対する予測が厳密単調増加にならず、全条件に正のbiasがあります。"
            "したがって、どのモデルも今回の撮影条件で絶対距離計として合格とは言えません。"
        ),
        (
            "動画では4候補がMetric3Dより滑らかでしたが、真値軌跡がないため、"
            "この差は連続性の比較であって精度の証明ではありません。"
        ),
        "",
        "![Phase 1 comparison](../outputs/depth_model_comparison/phase1_known_distance_comparison.png)",
        "",
        "## Phase 1: 既知距離",
        "",
        "後付けのscale/offset補正は一切適用していません。値は緑箱ROI内の正の有限深度の中央値です。",
        "",
        "| 条件 | 0.3 m | 0.5 m | 0.7 m | 1.0 m | 1.5 m | MAE (m) | RMSE (m) | MARE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in conditions:
        aggregate = item["phase1"]["aggregate"]
        predictions = item["phase1"]["predictions_m"]
        lines.append(
            "| " + item["label_ja"] + " | "
            + " | ".join(_f(value) for value in predictions)
            + f" | {_f(aggregate['mae_m'])} | {_f(aggregate['rmse_m'])}"
            + f" | {100.0 * aggregate['mean_absolute_relative_error']:.1f}%"
            + f" | {_f(aggregate['pearson_r'])} | {_f(aggregate['spearman_r'])} |"
        )
    lines.extend(
        [
            "",
            "観察:",
            "",
            "- UniDepth V2-Lは2条件とも候補中で誤差が小さく、近似K入力がカメラ指定なしをわずかに上回りました。",
            "- Depth Proでは焦点推定条件が近似焦点条件より良好でした。",
            "- Metric3DはPearson相関が最高ですが、大きな正のoffsetによりMAEは最大です。相関の高さと絶対距離精度は別です。",
            "",
            "![Phase 2 comparison](../outputs/depth_model_comparison/phase2_fingertip_depth_comparison.png)",
            "",
            "## Phase 2: 人差し指の前後移動",
            "",
            (
                "MediaPipeは再実行せず、baselineの267フレームの座標JSONLをSHA-256で固定しました。"
                "全条件で同じ217フレーム（81.27%）が有効で、座標列hashも一致しています。"
            ),
            "",
            "| 条件 | depth p5 / median / p95 (m) | p95-p5 (m) | 安定追跡隣接差 median / p95 / max (m) | 推論時間中央値 (ms) | baselineとのPearson / Spearman | 方向符号一致 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in conditions:
        movement = item["phase2"]["movement"]
        depth = movement["depth_m"]
        delta = movement["stable_tracking_adjacent_frame_absolute_delta_m"]
        trajectory = item["phase2"]["trajectory_vs_baseline"]
        if trajectory is None:
            correlation = "reference"
            direction = "reference"
        else:
            correlation = (
                f"{_f(trajectory['pearson_r'])} / {_f(trajectory['spearman_r'])}"
            )
            direction = f"{100.0 * trajectory['stable_direction']['agreement_rate']:.1f}%"
        lines.append(
            f"| {item['label_ja']} | {_f(depth['p05'])} / {_f(depth['median'])} / {_f(depth['p95'])}"
            f" | {_f(depth['p95_minus_p05_movement_range'])}"
            f" | {_f(delta['median'])} / {_f(delta['p95'])} / {_f(delta['max'])}"
            f" | {_f(item['phase2']['inference_median_ms'], 1)}"
            f" | {correlation} | {direction} |"
        )
    lines.extend(
        [
            "",
            (
                "安定追跡隣接差は、座標移動が画像対角の10%以下の連続フレームだけで計算しています。"
                "大きなlandmark再局在を除外しても、Metric3Dのp95は約0.169 m、4候補は約0.015–0.019 mでした。"
            ),
            "",
            (
                "一方、絶対的な動画depth中央値はモデル間で約0.273–0.968 mと大きく異なります。"
                "中央値で正規化した安定追跡差でも、Metric3Dのmedian/p95は5.44%/17.47%、"
                "4候補は約2.01–2.57%/4.69–6.90%で、候補の方が滑らかです。"
                "それでもPhase 1が示すbiasを踏まえると、動画だけからどれが正しいとは決められません。"
            ),
            "",
            (
                "5回のlandmark再局在があり、特にframe 30と119付近の極値は指先運動量として解釈できません。"
                "表のモデル間相関と方向符号一致もbaselineとの一致であり、軌跡真値との一致ではありません。"
            ),
            "",
            "## カメラ推定の監査",
            "",
            "カメラ指定なし／焦点推定条件では値をモデルへ渡していません。比較用近似値は26 mm・36 mm相当から求めた値です。",
            "",
            "| 条件 | 写真または動画 | 比較用近似値 (px) | モデル推定 p5 / median / p95 (px) | CV |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparison["camera_estimates"]:
        estimated = row["estimated"]
        lines.append(
            f"| {row['label_ja']} | {row['scope']} | "
            f"{_f(row['reference_approx_px'], 1)} | "
            f"{_f(estimated['p05'], 1)} / {_f(estimated['median'], 1)} / {_f(estimated['p95'], 1)}"
            f" | {100.0 * estimated['coefficient_of_variation']:.2f}% |"
        )
    lines.extend(
        [
            "",
            (
                "UniDepthの返却Kは、近似K入力条件でもモデル予測Kです。入力したKとは別に記録しています。"
                "Depth Proの焦点推定条件では `f_px=None` を明示し、近似焦点を渡していません。"
            ),
            "",
            "## 実装・再現性",
            "",
            "- UniDepth V2-L: source `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`, weights revision `52b349b514bd8b47642f67ac78cb7b5dc5c51dd9`。",
            "- Depth Pro: source `9e65e4dbe9568d23c546fcec53302b10445e109e`, weights revision `ccd1350a774eb2248bcdfb3be430e38f1d3087ef`。",
            "- checkpointは推論前にSHA-256検証。UniDepthはCC BY-NC 4.0、Depth ProはAppleのモデルライセンスです。",
            "- Depth Pro公式依存の `numpy<2` とbaselineのOpenCV 5依存が衝突するため、`environments/depth_pro` の独立uv lockを使用。",
            "- OpenCV間のMOV decode差を排除するため、OpenCV 5で一度だけlossless PNG cacheを作成。manifest SHAと全267枚のPNG/raw BGR SHAを両環境で検証。",
            "- Phase 1も全5画像のdecode後raw BGR SHAが4候補とbaseline環境で一致。ROIとsource hashも一致。",
            "- 推論時間の計測範囲は完全同一ではありません。Metric3Dはnetwork forwardのみ、候補は各公式 `infer`（内部resize・後処理を含む）なので参考値です。",
            "",
            "実行コマンド:",
            "",
            "```bash",
            "uv run --locked --group unidepth python scripts/cache_comparison_video_frames.py",
            "uv run --locked --group unidepth python scripts/evaluate_depth_model_comparison.py --backend unidepth",
            "uv run --locked --project environments/depth_pro python scripts/evaluate_depth_model_comparison.py --backend depth-pro",
            "uv run --locked --group unidepth python scripts/build_depth_model_comparison_report.py",
            "```",
            "",
            "## 判断",
            "",
            (
                "今回のデータだけで次に進めるなら、**UniDepth V2-L + 近似K**を第一候補とします。"
                "ただし採用ではなく、既知距離MAEが最も小さいという暫定順位です。"
                "次の必須試験は、カメラキャリブレーション済みKと、指先軌跡の独立した距離真値を使う評価です。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_comparison(
    *,
    baseline_summary_path: Path,
    baseline_records_path: Path,
    comparison_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    baseline = _load_json(baseline_summary_path)
    baseline_records = _records(baseline_records_path)
    condition_data: list[dict[str, Any]] = []
    record_sets: list[list[dict[str, Any]]] = [baseline_records]
    labels = [item[1] for item in _CONDITIONS]

    baseline_phase1 = baseline["phase1_known_distance"]
    baseline_phase2 = baseline["phase2_finger_movement"]
    expected_phase1_inputs = [
        (
            row["source_sha256"],
            float(row["actual_distance_m"]),
            tuple(row["bbox_xywh"]),
        )
        for row in baseline_phase1["rows"]
    ]
    expected_video_sha256 = baseline_phase2["source_sha256"]
    expected_phase1_pixel_hashes: list[str] = []
    for row in baseline_phase1["rows"]:
        bgr = read_bgr(Path(row["source"]))
        if bgr is None:
            raise ValueError(f"unable to re-read baseline Phase 1 input: {row['source']}")
        expected_phase1_pixel_hashes.append(pixel_sha256(bgr))
    manifest_hashes: list[str] = []
    input_pixel_hash_sequences: list[tuple[str, ...]] = []
    condition_data.append(
        {
            "id": _CONDITIONS[0][0],
            "label": _CONDITIONS[0][1],
            "label_ja": _CONDITIONS[0][2],
            "model": baseline_phase1["depth_model"],
            "phase1": {
                "predictions_m": [row["predicted_depth_m"] for row in baseline_phase1["rows"]],
                "aggregate": baseline_phase1["aggregate"],
            },
            "phase2": {
                "movement": baseline_phase2["movement_evaluation"],
                "inference_median_ms": baseline_phase2["median_depth_inference_ms"],
                "trajectory_vs_baseline": None,
            },
        }
    )
    camera_estimates: list[dict[str, Any]] = []
    for condition_id, label, label_ja in _CONDITIONS[1:]:
        summary = _load_json(comparison_dir / condition_id / "summary.json")
        records = _records(
            comparison_dir / condition_id / "phase2_finger_movement" / "frames.jsonl"
        )
        observed_phase1_inputs = [
            (
                row["source_sha256"],
                float(row["actual_distance_m"]),
                tuple(row["bbox_xywh"]),
            )
            for row in summary["phase1"]["rows"]
        ]
        if observed_phase1_inputs != expected_phase1_inputs:
            raise ValueError(f"Phase 1 inputs/ROI differ for {condition_id}")
        if summary["phase1"]["roi_method"] != baseline_phase1["roi_method"]:
            raise ValueError(f"Phase 1 ROI method differs for {condition_id}")
        observed_phase1_pixel_hashes = [
            str(row["input_bgr_pixel_sha256"]) for row in summary["phase1"]["rows"]
        ]
        if observed_phase1_pixel_hashes != expected_phase1_pixel_hashes:
            raise ValueError(f"Phase 1 decoded BGR pixels differ for {condition_id}")
        if summary["phase2"]["source_sha256"] != expected_video_sha256:
            raise ValueError(f"Phase 2 source video differs for {condition_id}")
        decoder = summary["phase2"]["video_decoder"]
        if decoder.get("mode") != "lossless_png_frame_cache":
            raise ValueError(f"{condition_id} did not use the shared lossless frame cache")
        manifest_path = Path(decoder["manifest_path"])
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_hash != decoder.get("manifest_sha256"):
            raise ValueError(f"frame-cache manifest hash differs for {condition_id}")
        manifest = _load_json(manifest_path)
        if manifest.get("source_sha256") != expected_video_sha256:
            raise ValueError("frame-cache source does not match the baseline video")
        pixel_hashes = tuple(str(record["input_bgr_pixel_sha256"]) for record in records)
        manifest_pixel_hashes = tuple(
            str(frame["bgr_pixel_sha256"]) for frame in manifest["frames"]
        )
        if pixel_hashes != manifest_pixel_hashes:
            raise ValueError(f"cached input pixels differ for {condition_id}")
        manifest_hashes.append(manifest_hash)
        input_pixel_hash_sequences.append(pixel_hashes)
        record_sets.append(records)
        condition_data.append(
            {
                "id": condition_id,
                "label": label,
                "label_ja": label_ja,
                "model": summary["depth_model"],
                "phase1": {
                    "predictions_m": [row["predicted_depth_m"] for row in summary["phase1"]["rows"]],
                    "aggregate": summary["phase1"]["aggregate"],
                },
                "phase2": {
                    "movement": summary["phase2"]["movement_evaluation"],
                    "inference_median_ms": summary["phase2"]["inference_ms"]["median"],
                    "video_decoder": summary["phase2"]["video_decoder"],
                },
            }
        )

        if condition_id == "unidepth_v2_l__no_camera":
            photo_fx_values = [
                float(row["prediction_extras"]["predicted_intrinsics"]["fx_px"])
                for row in summary["phase1"]["rows"]
            ]
            fx_values = [
                float(record["prediction_extras"]["predicted_intrinsics"]["fx_px"])
                for record in records
            ]
            camera_estimates.extend(
                [
                    {
                        "condition_id": condition_id,
                        "label_ja": label_ja,
                        "scope": "写真",
                        "reference_approx_px": 4290.606017802147,
                        "estimated": _numeric_summary(photo_fx_values),
                    },
                    {
                        "condition_id": condition_id,
                        "label_ja": label_ja,
                        "scope": "動画",
                        "reference_approx_px": 1996.9207064108248,
                        "estimated": _numeric_summary(fx_values),
                    },
                ]
            )
        if condition_id == "depth_pro__estimated_focal":
            video_values = _extra_series(records, "output_focal_px")
            photo_values = [
                float(row["prediction_extras"]["output_focal_px"])
                for row in summary["phase1"]["rows"]
            ]
            camera_estimates.extend(
                [
                    {
                        "condition_id": condition_id,
                        "label_ja": label_ja,
                        "scope": "写真",
                        "reference_approx_px": 4290.606017802147,
                        "estimated": _numeric_summary(photo_values),
                    },
                    {
                        "condition_id": condition_id,
                        "label_ja": label_ja,
                        "scope": "動画",
                        "reference_approx_px": 1996.9207064108248,
                        "estimated": _numeric_summary(video_values),
                    },
                ]
            )

    if len(set(manifest_hashes)) != 1:
        raise ValueError(f"condition frame-cache manifest hashes differ: {manifest_hashes}")
    if len(set(input_pixel_hash_sequences)) != 1:
        raise ValueError("condition input BGR pixel hash sequences differ")

    coordinate_digests = [_coordinate_digest(records) for records in record_sets]
    if len(set(coordinate_digests)) != 1:
        raise ValueError(f"condition coordinate hashes differ: {coordinate_digests}")
    depths = [_depth_series(records) for records in record_sets]
    shared_indices = sorted(set.intersection(*(set(values) for values in depths)))
    if len(shared_indices) != 217:
        raise ValueError(f"expected 217 common valid frames, observed {len(shared_indices)}")
    baseline_values = np.asarray([depths[0][index] for index in shared_indices])
    for item, values in zip(condition_data[1:], depths[1:]):
        candidate_values = np.asarray([values[index] for index in shared_indices])
        item["phase2"]["trajectory_vs_baseline"] = {
            "shared_valid_frames": len(shared_indices),
            "pearson_r": _correlation(baseline_values, candidate_values),
            "spearman_r": _correlation(_rank(baseline_values), _rank(candidate_values)),
            "stable_direction": _stable_direction_agreement(
                baseline_records, depths[0], values
            ),
        }

    actual = [row["actual_distance_m"] for row in baseline_phase1["rows"]]
    comparison: dict[str, Any] = {
        "experiment": "Five-condition monocular metric-depth comparison",
        "conditions": condition_data,
        "known_distances_m": actual,
        "camera_estimates": camera_estimates,
        "phase2_coordinate_audit": {
            "records_sha256": hashlib.sha256(baseline_records_path.read_bytes()).hexdigest(),
            "canonical_coordinate_sha256": coordinate_digests[0],
            "all_condition_hashes_identical": True,
            "processed_frames": len(baseline_records),
            "common_valid_frames": len(shared_indices),
        },
        "input_pixel_audit": {
            "phase1_bgr_pixel_sha256": expected_phase1_pixel_hashes,
            "phase1_all_condition_hashes_identical": True,
            "phase2_frame_cache_manifest_sha256": manifest_hashes[0],
            "phase2_bgr_pixel_hash_sequence_sha256": hashlib.sha256(
                "".join(input_pixel_hash_sequences[0]).encode("ascii")
            ).hexdigest(),
            "phase2_all_condition_hashes_identical": True,
        },
        "calibration_fit_applied": False,
    }
    write_json(comparison_dir / "comparison.json", comparison)

    with (comparison_dir / "phase1_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(["actual_distance_m", *[item["id"] for item in condition_data]])
        for index, distance in enumerate(actual):
            writer.writerow(
                [distance, *[item["phase1"]["predictions_m"][index] for item in condition_data]]
            )

    with (comparison_dir / "phase2_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(["frame_index", "timestamp_ms", *[item["id"] for item in condition_data]])
        timestamps = {int(row["frame_index"]): row["timestamp_ms"] for row in baseline_records}
        for index in range(len(baseline_records)):
            writer.writerow([index, timestamps[index], *[values.get(index, "") for values in depths]])

    _phase1_chart(
        comparison_dir / "phase1_known_distance_comparison.png",
        actual,
        [item["phase1"]["predictions_m"] for item in condition_data],
        labels,
    )
    _phase2_chart(
        comparison_dir / "phase2_fingertip_depth_comparison.png",
        record_sets,
        depths,
        labels,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(comparison), encoding="utf-8")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("outputs/iphone_phase1_2/summary.json"),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path("outputs/iphone_phase1_2/phase2_finger_movement/frames.jsonl"),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("outputs/depth_model_comparison"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/depth_model_comparison_results.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = build_comparison(
        baseline_summary_path=args.baseline_summary,
        baseline_records_path=args.baseline_records,
        comparison_dir=args.comparison_dir,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "conditions": len(comparison["conditions"]),
                "coordinate_audit": comparison["phase2_coordinate_audit"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
