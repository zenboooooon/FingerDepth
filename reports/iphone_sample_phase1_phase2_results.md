# iPhone実データによるPhase 1 / Phase 2評価

実行日: 2026-09-24 (Asia/Tokyo)

## 結論

| 項目 | 判定 | 要点 |
|---|---|---|
| Phase 1パイプライン | 成功 | 5枚のHEICをdecodeし、固定Metric3D v2-Sのmeter Depth Mapを生成・保存できた |
| Phase 1既知距離精度 | **不合格** | MAE 1.709 m、RMSE 1.732 m、全5点で過大推定、0.5 mと0.7 mで順序逆転 |
| Phase 2統合 | 成功 | MOV全267 frameを処理し、MediaPipe landmark 8のsingle-pixel Depthを記録できた |
| Phase 2 Depth変動 | 定性的に確認 | 複数の山谷と0.719 mのp5–p95レンジを観測。方向を検証する独立GTはない |
| Phase 2連続性 | **現状不十分** | 検出率81.27%、最長30 frame欠損、追跡中の隣接変化p95 169 mm |

Metric3Dのraw出力へ、既知距離から求めたscale補正やoffset補正は適用していない。この5点を使って補正し、同じ5点で精度を再評価するデータ漏洩も行っていない。

元の計画には定量acceptance thresholdがないため、「不合格」「現状不十分」は3Dストローク用途を想定した実用上の事後判定である。

## 入力とカメラ内部パラメータ

### 静止画

- `03image.HEIC` / `05image.HEIC` / `07image.HEIC` / `10image.HEIC` / `15image.HEIC`
- nominal distance: 0.3 / 0.5 / 0.7 / 1.0 / 1.5 m
- iPhone 15 main wide、実焦点距離5.96 mm、35 mm換算26 mm
- EXIF orientation適用後: 4284×5712 px

### 動画

- `finger_movement.MOV`
- 1440×1920 px、267 frames、8.902 s、平均29.994 FPS
- ユーザー指定の撮影設定は35 mm換算36 mm。container metadataの25 mm相当値は今回の換算には使用しない

ファイル内に校正済み内部行列はない。CIPAの対角画角近似

\[
f_{px}=f_{35}\frac{\sqrt{W^2+H^2}}{\sqrt{36^2+24^2}}
\]

を用い、静止画は `fx=fy=4290.606 px`、動画は `fx=fy=1996.921 px` とした。主点は画像中心、skewは0と仮定した。これは校正値ではない。35 mm換算値が1 mm単位へ丸められたと仮定した下限だけでも静止画約±1.9%、動画約±1.4%で、動画にはEIS/cropによる追加誤差があり得る。Metric3Dの復元Depthは `fx` に線形なので、焦点誤差は同率のDepth scale誤差になる。

### Metric3D v2のメートル換算監査

固定した公式commitの[`hubconf.py`](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/hubconf.py#L145-L199)とローカル実装を照合した。公式経路は

\[
s=\min(616/H, 1064/W)
\]

\[
f_x^{resized}=s f_x^{original}
\]

\[
D_{metric}=D_{canonical}\frac{f_x^{resized}}{1000}
\]

であり、[`model.inference()`](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/mono/model/monodepth_model.py#L9-L12)自体はこの実カメラ換算を行わない。本実装の実入力に対する値は次のとおり。

| 入力 | 35 mm換算 | 元画像 `fx` [px] | resize倍率 `s` | resize後 `fx` [px] | metric係数 |
|---|---:|---:|---:|---:|---:|
| 静止画 4284×5712 | 26 mm | 4290.606 | 0.107843 | 462.712 | 0.462712 |
| 動画 1440×1920 | 36 mm | 1996.921 | 0.320833 | 640.679 | 0.640679 |

旧25 mm相当で算出した動画Depthに対し、36 mm相当では理論上`36/25 = 1.44`倍になる。成果物は36 mm条件で全frameを再推論・再集計して置き換えた。

ローカルでは35 mm換算値を直接`fx`へ渡さず、まず対角画角近似でpixel単位へ変換する。`prepare_metric3d_input`が公式と同じresize/paddingを行い、`restore_metric_depth`がunpad・元解像度への補間後に`fx_original × s / 1000`を**1回だけ**乗算する。その後のpipelineはmeter Depthをそのまま使うため、換算漏れ・二重適用・mm/px混同はない。比例性は同じcanonical Depthで`fx`を2倍にすると出力も2倍になる回帰テストで確認した。各summaryの`metric3d_scale_conversion`にも上表の実値と適用回数1を保存した。

画像・動画に含まれる位置情報は評価成果物や本レポートへ転記していない。

## Phase 1：既知距離試験

### 評価方法

Depth値を見ず、全画像に同じ処理を適用して緑箱ROIを抽出した。

1. HSV `H=35..100, S>=55, V>=20`
2. 中央50%幅かつ画像高35%より下へ空間制限
3. 画像短辺1% kernelでclosing
4. 最大componentの外輪郭をfill
5. 短辺10% erosionで境界を除外
6. ROI内の有限かつ正のDepth中央値を代表値とする

ROIは5枚すべてで箱に一致した。入力データ自体のsanity checkとして、bbox幅と距離逆数の相関は0.99909、`bbox幅×距離` の変動係数は3.90%だった。したがって、抽出対象と距離ラベルは透視投影の逆数則に強く整合している。

### Raw結果

| 実距離 [m] | 推定Depth [m] | signed error [m] | abs. relative error | ROI IQR [m] |
|---:|---:|---:|---:|---:|
| 0.3 | 1.695 | +1.395 | 465.1% | 1.674–1.719 |
| 0.5 | 2.698 | +2.198 | 439.6% | 2.692–2.705 |
| 0.7 | 2.525 | +1.825 | 260.8% | 2.518–2.532 |
| 1.0 | 2.643 | +1.643 | 164.3% | 2.635–2.657 |
| 1.5 | 2.982 | +1.482 | 98.8% | 2.970–2.999 |

集約値:

- MAE: **1.7087 m**
- RMSE: **1.7324 m**
- mean bias: **+1.7087 m**
- mean absolute relative error: **285.72%**
- Pearson \(r\): 0.7766
- Spearman \(r_s\): 0.7000
- strict monotonic increase: **false**
- `predicted = slope × actual + intercept` 診断: slope 0.8025、intercept +1.8667 m、\(R^2=0.6032\)

全点で大きく過大推定し、0.5 mの推定値が0.7 mより大きい。35 mm換算焦点距離の±2%不確かさでは説明できず、仮に±5%としても結論は変わらない。距離がカメラから箱中心への斜距離だった場合も、bbox中心のoff-axis補正は最大2.1%（0.3 m点）であり、誤差の主因にはならない。

箱bbox中心のsingle-pixel値も各ROI中央値との差が最大0.0103 mであり、代表値の選択を変えても失敗判定は変わらない。

ROI内IQRが狭いのは、各画像内でDepth Mapが空間的に一様だったことを示すだけで、絶対値が正しいことや撮影間再現性を示さない。結果は、近距離・小物体・屋内床面という条件でMetric3D v2-Sのscene priorが対象距離を十分に捉えられなかった可能性と整合するが、原因の確定には校正済みK、反復撮影、別モデル比較が必要である。

## Phase 2：人差し指の前後移動試験

### 評価方法

- MediaPipe Hand Landmarker VIDEO mode
- `INDEX_FINGER_TIP`（landmark 8）
- 元RGBと同じ1440×1920へ復元したDepth Mapのsingle pixel `D[v,u]`
- decoder PTSをtimestampに使用
- 欠損を補間せず、連続検出区間だけで隣接frame差を計算

動画は意図的な運動を含むため、隣接差を「静止jitter」とは呼ばずcontinuity診断として扱う。

### 結果

| 指標 | 値 |
|---|---:|
| 処理frame | 267 / 267 |
| hand / valid fingertip depth | 217 / 267 (81.27%) |
| 欠損区間 | frame 0–29、98–117 |
| 最長欠損 | 30 frames（約1.0 s） |
| 連続検出区間 | 30–97、118–266 |
| Depth p5 / median / p95 | 0.629 / 0.968 / 1.348 m |
| p95−p5 movement range | 0.719 m |
| Metric3D forward中央値 | 27.09 ms |

ユーザー説明では人差し指を前後移動した動画であり、Depth系列にも滑らかな大域的上昇・下降が複数回見える。ただし方向labelや真値軌跡がないため、前進・後退との符号一致は独立には検証できない。再捕捉時の誤検出を含むraw single-pixel系列には大きな不連続もある。

| 隣接有効frameの \(|ΔZ|\) | raw全pair | 2D追跡継続pair |
|---|---:|---:|
| pair数 | 215 | 210 |
| median | 53.0 mm | 52.6 mm |
| p95 | 187.4 mm | 169.0 mm |
| max | 2832.3 mm | 385.1 mm |
| `>100 mm`率 | 19.53% | 18.57% |
| `>200 mm`率 | 3.72% | 2.38% |

「2D追跡継続pair」は、隣接frameのlandmark 8移動が画像対角の10%（240 px）以下というheuristicで分類したpairである。除外対象5 pairもraw結果から削除せず、別カテゴリとしてJSONへ全件保存した。最大2.832 m jumpはframe 119→120で、landmark自体も265.6 px移動しており、再ローカライズ不連続と整合する。

5-frame rolling medianからのabsolute residualはmedian 26.7 mm、p95 114.2 mmだった。申告された前後運動と整合するDepth変化は観測できる一方、方向応答の正しさは未検証で、欠損と大きなframe間変動もある。single-pixelのまま3Dストロークへ使用するには不十分である。

## 成果物

- Phase 1集計: `outputs/iphone_phase1_2/phase1_known_distance/summary.json`
- Phase 1表: `outputs/iphone_phase1_2/phase1_known_distance/results.csv`
- 各画像: raw `depth_m.npy`、`depth_preview.png`、`box_mask.png`、`roi_overlay.jpg`
- Phase 2集計: `outputs/iphone_phase1_2/phase2_finger_movement/summary.json`
- 全frame record: `outputs/iphone_phase1_2/phase2_finger_movement/frames.jsonl`
- 時系列図: `outputs/iphone_phase1_2/phase2_finger_movement/fingertip_depth_timeseries.png`
- landmark 8 annotated動画: `outputs/iphone_phase1_2/phase2_finger_movement/annotated.mp4`
- 両Phase統合JSON: `outputs/iphone_phase1_2/summary.json`

再現コマンド:

```bash
uv sync --locked
uv run python scripts/evaluate_iphone_samples.py --device cuda:0 \
  --photo-focal-35mm-mm 26 \
  --video-focal-35mm-mm 36
```

品質確認:

```text
uv run pytest       # 51 passed
uv run ruff check . # All checks passed
```

## 次の判断

現在の固定構成は「処理が動く」というPhase 1/2の実装要件は満たすが、この撮影条件における絶対距離精度とsingle-pixel時間連続性は満たさない。次へ進む場合は、順序として以下が妥当である。

1. 同じ撮影モードでChArUco等によりカメラを校正し、EXIF近似を置換する。
2. 近距離屋内向けの別metric-depthモデルを、今回の5点とは別のholdoutを含めて比較する。
3. Phase 3の5×5 / 11×11 medianとhand maskを比較する。
4. 欠損・再ローカライズを品質flagとして保持し、Phase 4の時間filterを評価する。
5. 各距離で複数回撮影し、撮影間分散と信頼区間を求める。
