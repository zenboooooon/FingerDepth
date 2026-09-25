# 単眼metric depthモデル比較結果

実行日: 2026-09-25

## 結論

この5点の既知距離試験では **UniDepth V2-L / 近似K** が最小MAE **1.044 m** で、Metric3D baseline (1.709 m)より **38.9%低い** 結果でした。
ただし、全5条件で距離に対する予測が厳密単調増加にならず、全条件に正のbiasがあります。したがって、どのモデルも今回の撮影条件で絶対距離計として合格とは言えません。
動画では4候補がMetric3Dより滑らかでしたが、真値軌跡がないため、この差は連続性の比較であって精度の証明ではありません。

![Phase 1 comparison](../outputs/depth_model_comparison/phase1_known_distance_comparison.png)

## Phase 1: 既知距離

後付けのscale/offset補正は一切適用していません。値は緑箱ROI内の正の有限深度の中央値です。

| 条件 | 0.3 m | 0.5 m | 0.7 m | 1.0 m | 1.5 m | MAE (m) | RMSE (m) | MARE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Metric3D v2-S / 近似K | 1.695 | 2.698 | 2.525 | 2.643 | 2.982 | 1.709 | 1.732 | 285.7% | 0.777 | 0.700 |
| UniDepth V2-L / 近似K | 1.557 | 2.179 | 1.611 | 1.632 | 2.241 | 1.044 | 1.111 | 199.4% | 0.479 | 0.700 |
| UniDepth V2-L / カメラ指定なし | 1.630 | 2.271 | 1.671 | 1.706 | 2.309 | 1.117 | 1.183 | 212.1% | 0.459 | 0.700 |
| Depth Pro / 近似焦点 | 1.070 | 2.764 | 2.441 | 2.443 | 2.695 | 1.483 | 1.566 | 236.5% | 0.590 | 0.400 |
| Depth Pro / 焦点推定 | 1.208 | 2.644 | 2.212 | 2.104 | 2.658 | 1.365 | 1.433 | 227.0% | 0.598 | 0.600 |

観察:

- UniDepth V2-Lは2条件とも候補中で誤差が小さく、近似K入力がカメラ指定なしをわずかに上回りました。
- Depth Proでは焦点推定条件が近似焦点条件より良好でした。
- Metric3DはPearson相関が最高ですが、大きな正のoffsetによりMAEは最大です。相関の高さと絶対距離精度は別です。

![Phase 2 comparison](../outputs/depth_model_comparison/phase2_fingertip_depth_comparison.png)

## Phase 2: 人差し指の前後移動

MediaPipeは再実行せず、baselineの267フレームの座標JSONLをSHA-256で固定しました。全条件で同じ217フレーム（81.27%）が有効で、座標列hashも一致しています。

| 条件 | depth p5 / median / p95 (m) | p95-p5 (m) | 安定追跡隣接差 median / p95 / max (m) | 推論時間中央値 (ms) | baselineとのPearson / Spearman | 方向符号一致 |
|---|---:|---:|---:|---:|---:|---:|
| Metric3D v2-S / 近似K | 0.629 / 0.968 / 1.348 | 0.719 | 0.053 / 0.169 / 0.385 | 27.1 | reference | reference |
| UniDepth V2-L / 近似K | 0.243 / 0.361 / 0.471 | 0.228 | 0.007 / 0.017 / 0.020 | 32.2 | 0.829 / 0.888 | 63.8% |
| UniDepth V2-L / カメラ指定なし | 0.223 / 0.330 / 0.429 | 0.206 | 0.007 / 0.015 / 0.020 | 31.6 | 0.822 / 0.884 | 63.8% |
| Depth Pro / 近似焦点 | 0.199 / 0.317 / 0.441 | 0.241 | 0.008 / 0.017 / 0.021 | 114.7 | 0.831 / 0.873 | 63.6% |
| Depth Pro / 焦点推定 | 0.169 / 0.273 / 0.422 | 0.253 | 0.007 / 0.019 / 0.027 | 115.1 | 0.820 / 0.826 | 61.4% |

安定追跡隣接差は、座標移動が画像対角の10%以下の連続フレームだけで計算しています。大きなlandmark再局在を除外しても、Metric3Dのp95は約0.169 m、4候補は約0.015–0.019 mでした。

一方、絶対的な動画depth中央値はモデル間で約0.273–0.968 mと大きく異なります。中央値で正規化した安定追跡差でも、Metric3Dのmedian/p95は5.44%/17.47%、4候補は約2.01–2.57%/4.69–6.90%で、候補の方が滑らかです。それでもPhase 1が示すbiasを踏まえると、動画だけからどれが正しいとは決められません。

5回のlandmark再局在があり、特にframe 30と119付近の極値は指先運動量として解釈できません。表のモデル間相関と方向符号一致もbaselineとの一致であり、軌跡真値との一致ではありません。

## カメラ推定の監査

カメラ指定なし／焦点推定条件では値をモデルへ渡していません。比較用近似値は26 mm・36 mm相当から求めた値です。

| 条件 | 写真または動画 | 比較用近似値 (px) | モデル推定 p5 / median / p95 (px) | CV |
|---|---|---:|---:|---:|
| UniDepth V2-L / カメラ指定なし | 写真 | 4290.6 | 4608.1 / 4794.3 / 4841.6 | 2.01% |
| UniDepth V2-L / カメラ指定なし | 動画 | 1996.9 | 1559.6 / 1620.6 / 1702.9 | 2.57% |
| Depth Pro / 焦点推定 | 写真 | 4290.6 | 3730.5 / 4103.8 / 4720.9 | 9.44% |
| Depth Pro / 焦点推定 | 動画 | 1996.9 | 1638.7 / 1782.1 / 1949.5 | 5.82% |

UniDepthの返却Kは、近似K入力条件でもモデル予測Kです。入力したKとは別に記録しています。Depth Proの焦点推定条件では `f_px=None` を明示し、近似焦点を渡していません。

## 実装・再現性

- UniDepth V2-L: source `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`, weights revision `52b349b514bd8b47642f67ac78cb7b5dc5c51dd9`。
- Depth Pro: source `9e65e4dbe9568d23c546fcec53302b10445e109e`, weights revision `ccd1350a774eb2248bcdfb3be430e38f1d3087ef`。
- checkpointは推論前にSHA-256検証。UniDepthはCC BY-NC 4.0、Depth ProはAppleのモデルライセンスです。
- Depth Pro公式依存の `numpy<2` とbaselineのOpenCV 5依存が衝突するため、`environments/depth_pro` の独立uv lockを使用。
- OpenCV間のMOV decode差を排除するため、OpenCV 5で一度だけlossless PNG cacheを作成。manifest SHAと全267枚のPNG/raw BGR SHAを両環境で検証。
- Phase 1も全5画像のdecode後raw BGR SHAが4候補とbaseline環境で一致。ROIとsource hashも一致。
- 推論時間の計測範囲は完全同一ではありません。Metric3Dはnetwork forwardのみ、候補は各公式 `infer`（内部resize・後処理を含む）なので参考値です。

実行コマンド:

```bash
uv run --locked --group unidepth python scripts/cache_comparison_video_frames.py
uv run --locked --group unidepth python scripts/evaluate_depth_model_comparison.py --backend unidepth
uv run --locked --project environments/depth_pro python scripts/evaluate_depth_model_comparison.py --backend depth-pro
uv run --locked --group unidepth python scripts/build_depth_model_comparison_report.py
```

## 判断

今回のデータだけで次に進めるなら、**UniDepth V2-L + 近似K**を第一候補とします。ただし採用ではなく、既知距離MAEが最も小さいという暫定順位です。次の必須試験は、カメラキャリブレーション済みKと、指先軌跡の独立した距離真値を使う評価です。
