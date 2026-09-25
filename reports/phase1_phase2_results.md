# Phase 1 / Phase 2 実装・実験結果

実行日: 2026-09-24 (Asia/Tokyo)

## 結論

- Phase 1: 固定したMetric3D v2-Sで、静止画・動画から元RGB解像度のmeter単位Depth Mapを生成・保存する処理を実装した。Metric3D upstream同梱KITTI 3枚デモの有効272,068 pixelでは、合算AbsRel 0.0610、MAE 1.313 m、RMSE 2.571 mだった。
- Phase 2: MediaPipe Hand LandmarkerのINDEX_FINGER_TIP（landmark 8）を検出し、元RGB座標系へ復元したDepth Mapのsingle pixel `D[v,u]` を取得する処理を実装した。公式手画像で静止画・VIDEO modeともend-to-endで完走した。
- 自動テストは51件すべて成功した。
- 提供されたiPhoneデータでも両試験を実施した。既知距離はMAE 1.709 mで不合格、
  前後移動は217/267 frameを検出したが連続性は不十分だった。詳細は [iPhone実データ評価](iphone_sample_phase1_phase2_results.md)。

## 固定構成

### Depth

- Model: Metric3D v2-S (`metric3d_vit_small`)
- Official repository commit: `eb5b6fac0dc155e4e52f576e304fbf11655ff339`
- Checkpoint: `metric_depth_vit_small_800k.pth`
- Checkpoint SHA-256: `b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6`（実ファイルと一致）
- Input: RGB、アスペクト比維持resize、616×1064、公式平均値で中央padding
- Inference: FP32、cuda:0
- Metric conversion: `depth_m = canonical_depth * (resized_fx_px / 1000)`

checkpointはraw byteを先にSHA-256検証し、その後PyTorchのweights-only modeで読み込む。Metric3Dはcanonical focal 1000 px上で推論するため、meter scaleには撮影解像度に対応する校正済み `fx` が必須である。

iPhone実データでは公式`hubconf.py`と換算経路を照合し、写真26 mm相当から`fx=4290.606 px`、動画36 mm相当から`fx=1996.921 px`へ変換した。resize後のmetric係数はそれぞれ0.462712と0.640679で、後処理に1回だけ適用される。実値は各summaryの`metric3d_scale_conversion`へ保存した。

### Hand pose

- Model: MediaPipe Hand Landmarker `float16/1`
- Model SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- Landmark: `INDEX_FINGER_TIP`, index 8
- Coordinate conversion: 計画書の `round(xW), round(yH)` をhalf-upの `floor(xW + 0.5)` で実装し、画像範囲へclip
- MediaPipeの相対 `z` / world landmarksは不使用

### Runtime

- GPU: NVIDIA GeForce RTX 5090, 32 GB
- Driver: 595.84
- Python: 3.12.13
- uv: 0.11.27
- PyTorch: 2.14.0+cu130
- CUDA runtime: 13.0
- MediaPipe: 0.10.35
- OpenCV: 5.0.0
- xFormers: 未使用（native attention fallback）
- Package lock: `uv.lock`

Metric3D upstream commitのv2環境定義は `torch==2.0.1` / `torchvision==0.15.2` だが、これはRTX 5090のsm_120およびPython 3.12に対応しない。architecture・checkpoint・公式前後処理は固定し、実行runtimeのみBlackwell対応版へ更新した。

## Phase 1 結果

### Metric3D upstream同梱KITTI 3枚デモ

3画像とも内部パラメータ `[707.0493, 707.0493, 604.0814, 180.5066]`、GT scale 256 units/mを使用した。

| Image | Valid pixels | MAE [m] | RMSE [m] | AbsRel | Forward [ms] |
|---|---:|---:|---:|---:|---:|
| 0000000050 | 96,131 | 1.1457 | 2.3207 | 0.05135 | 193.47 |
| 0000000100 | 85,098 | 1.2987 | 2.5160 | 0.06816 | 26.96 |
| 0000000005 | 90,839 | 1.5037 | 2.8569 | 0.06458 | 27.04 |
| **pixel合算** | **272,068** | **1.3131** | **2.5706** | **0.06102** | — |

初回193.47 msにはCUDA kernel初回実行コストが含まれる。同一process内のforward中央値は27.04 ms（約37.0 FPS相当）だった。これは `model.inference` の同期時間のみで、前後処理、動画decode、MediaPipe、可視化、CPU転送、I/Oを含まない。

これはupstreamが同梱する選択済み3画像でのscale経路sanity checkであり、KITTI公式benchmarkへの提出、標準split/crop/cap評価、モデル一般精度の評価ではない。近距離室内・手の精度も保証しない。

成果物: `outputs/phase1_kitti_demo_final/summary.json`、各画像のfloat32 `*_depth_m.npy` とpreview。

Phase 1動画も2 frameで実行し、CLIの既定動作として各frameのraw `depth/*.npy`、preview MP4、JSONL、summaryを生成した（`outputs/phase1_video_smoke_final/`）。

### 計画書の既知距離試験

| 実距離 | raw推定Depth | signed error |
|---:|---:|---:|
| 0.3 m | 1.695 m | +1.395 m |
| 0.5 m | 2.698 m | +2.198 m |
| 0.7 m | 2.525 m | +1.825 m |
| 1.0 m | 2.643 m | +1.643 m |
| 1.5 m | 2.982 m | +1.482 m |

iPhone 15の35 mm換算metadataからpixel焦点距離を近似し、緑箱のeroded ROI中央値を使用した。
scale/offset補正は適用していない。MAE 1.709 m、RMSE 1.732 mであり、この条件の絶対距離精度は不合格だった。
詳細な手法・制約・成果物は [iPhone実データ評価](iphone_sample_phase1_phase2_results.md) に記載した。

## Phase 2 結果

### 静止画integration

MediaPipe公式 `woman_hands.jpg`（640×960）で1手を検出した。

| Field | Result |
|---|---:|
| handedness / score | Left / 0.9360 |
| normalized `(x, y)` | (0.064333, 0.774692) |
| fingertip pixel `(u, v)` | (41, 744) |
| single-pixel depth | 0.808995 m |
| model forward | 192.53 ms（cold） |

この画像の実カメラ内部パラメータは非公開なので、integration確認用に `fx=fy=1000 px` を仮定した。0.808995 mは座標統合の完走値であり、実距離精度の結果ではない。JSONにはhand model path/hash/versionと `depth_valid: true` も記録した。

成果物: `outputs/phase2_official_hand_final/result.json`、`annotated.png`、`depth_m.npy`、`depth_preview.png`。

### VIDEO mode smoke test

同じ公式画像から決定的な微小zoom/panを加えた12 frame・6 FPS動画を生成し、VIDEO API、PTS timestamp、動画出力を確認した。

- Processed: 12 / 12 frames
- Hand detected: 12 / 12 frames (100%)
- Valid fingertip depth: 12 / 12 frames
- Timestamps: 0, 167, 333, ..., 1833 ms（厳密単調増加）
- Metric3D forward: mean 40.66 ms、median 26.92 ms
- `frames.jsonl`、summary、annotated MP4を正常生成

これは静止画由来の合成smoke fixtureであり、このtest単独では実指の前後運動を評価していない。

成果物: `outputs/phase2_video_smoke_final/`。

### iPhone実指の前後移動試験

`finger_movement.MOV`全267 frameを処理し、217 frame（81.27%）でlandmark 8と有効Depthを得た。
ユーザー指定の36 mm相当で再評価し、Depthのp5–p95レンジは0.719 mで前後運動の山谷は観測できた。一方、最長30 frameの欠損、
raw最大2.832 mの再捕捉jump、2D追跡継続時でも隣接差p95 169 mmがあり、連続性は不十分だった。

成果物: `outputs/iphone_phase1_2/phase2_finger_movement/`。詳細は
[iPhone実データ評価](iphone_sample_phase1_phase2_results.md)。

## 実装・検証範囲

- Phase 1: 静止画/動画、meter Depth Map、raw保存、可視化、GT評価、JSON/JSONL
- Phase 2: MediaPipe IMAGE/VIDEO mode、landmark 8、single-pixel depth、annotated画像/動画
- 座標整合: Metric3D paddingを除去し、元 `(H,W)` へbilinear resize後に `D[v,u]`
- 動画時刻: decoder PTS優先、利用不能時index/FPS fallback、厳密単調化
- 欠損指先Depth: frameを中断せずnull/invalidとして記録
- 再現性: uv lock、commit固定、checkpoint事前SHA検証、hand asset hashを結果へ記録
- Tests: 前後処理、portrait/奇数padding、座標、D[v,u]、GT評価、no-hand、0/NaN depth、PTS fallback、fake pipeline integration

## 残る検証・改善

提供iPhoneデータにより実験計画のPhase 1既知距離試験とPhase 2前後移動試験まで実施した。
ただし、内部パラメータはmetadata近似であり、既知距離は各1枚、動画の真値軌跡はない。

1. 対象撮影モードで`fx, fy, cx, cy`を校正し、metadata近似を置換する。
2. 各距離を複数回撮影し、撮影間分散を評価する。
3. 近距離向け別metric-depthモデルを独立holdoutで比較する。
4. Phase 3/4でROI samplingと時間filterを評価する。

現構成は実装統合には成功したが、対象条件で十分な絶対精度・連続性があるとは言えない。
