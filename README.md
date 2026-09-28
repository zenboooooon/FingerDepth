# Monocular Fingertip Metric Depth

単眼RGB画像・動画に対して、次の固定構成を実行するPoCです。

- Depth: **Metric3D v2-S** (`metric3d_vit_small`)
- Hand pose: **MediaPipe Hand Landmarker float16/1**
- Fingertip: **INDEX_FINGER_TIP (landmark 8)**
- Depth sampling: 元画像解像度へ復元したDepth Mapのsingle pixel

Metric3Dコードは公式リポジトリcommit
`eb5b6fac0dc155e4e52f576e304fbf11655ff339`、checkpointはSHA-256で固定しています。

## Setup (`uv`)

```bash
uv sync
uv run python scripts/download_hand_landmarker.py
uv run pytest
```

初回のMetric3D実行時には固定commitのソースとv2-S checkpoint（約150 MB）を取得します。
checkpointはデシリアライズ前にSHA-256を照合し、weights-onlyで読み込みます。

## 動画を追加してStudentを学習

root環境と依存関係を分離したDepth Pro環境を、lock済みの構成で準備します。

```bash
uv sync --locked
uv sync --locked --project environments/depth_pro
uv run --locked python scripts/download_hand_landmarker.py
```

新しい学習動画は `data/training_videos/train/` へ追加します。`validation/` は
sequence-held-out評価用の固定集合です。学習動画を増やすたびに追加・再分割せず、同じ撮影sessionの
clipや再encodeをtrain/validationへ跨がせないでください。

```bash
cp /path/to/new_capture.MOV data/training_videos/train/new_capture.MOV
uv run --locked fingertip-train run
```

既定では `.mov` / `.mp4` / `.m4v` を検出し、検証済み中間成果物をcacheして変更のない動画を
再処理しません。実行前の確認には `fingertip-train status` または
`fingertip-train run --dry-run`、同一datasetを意図的に再学習するときだけ `--force-train` を
使います。別設定は `--config path/to/config.toml` で指定できます。

ラベル生成が中断された場合も、同じ `fingertip-train run` を再実行してください。完全性を
検証した準備済みframeと、チェックポイント済みのDepth Pro結果を自動的に再利用し、最後の
チェックポイントから再開します。保存間隔は
`configs/training_pipeline.toml` の `[teacher] checkpoint_interval_frames` で指定でき、既定は
100枚です。正常完了時は100枚未満の端数も保存します。動画、焦点距離、モデル、設定、または
関連実装が変わった場合は、以前の進捗を混在させず再利用を拒否します。最終cacheは全成果物の
検証が完了するまで公開されません。

既定の動画焦点距離は既存iPhone実験に合わせた **36 mm相当の近似値**であり、校正値では
ありません。別のカメラやzoomでは `configs/training_pipeline.toml` の
`video_overrides` にproject-relative動画pathと正しい35 mm相当値を設定してから実行してください。
詳細は [training video guide](data/training_videos/README.md) を参照してください。

後段のPhaseごとの個別commandは、監査済み旧成果物の再現や段階別debug用のlegacy手順として
残しています。通常の動画追加・学習には上記の `fingertip-train` を使用してください。

### Runtime requirement

固定したupstream decoderが内部tensorを`cuda:0`へ生成するため、現実装はNVIDIA CUDA GPUの
`cuda:0`専用です。`--device auto`または`--device cuda:0`を使用してください。

upstreamのv2環境定義は古いPyTorch 2.0.1ですが、RTX 5090では動かないため、本プロジェクトは
architecture・checkpoint・前後処理を変えずにBlackwell対応PyTorchを`uv.lock`へ固定しています。

## Metric scale

Metric3D v2はcanonical camera（焦点距離1000 px）上のdepthを予測します。本実装は公式例と
同じく、入力resize後の焦点距離を使って次を適用します。

```text
depth_m = canonical_depth * (resized_fx_px / 1000)
```

したがって `--fx-px` には、**撮影解像度に対応する校正済み焦点距離**が必要です。
`--fx-px`へ35 mm換算値をそのまま渡すことはできません。本サンプル実験では対角画角から
pixel焦点距離へ換算した近似値を使いますが、正式な精度評価では校正値へ置換してください。
歪みの強いカメラは事前にundistortしてください。
`fy`, `cx`, `cy` は未指定時に `fy=fx`、主点=画像中心となります。

iPhoneサンプルのsummaryには、元解像度の`fx`、resize倍率、resize後`fx`、
canonical-to-metric係数、および換算適用回数を`metric3d_scale_conversion`として記録します。
換算は`restore_metric_depth`内で1回だけ適用されます。

## Phase 1

静止画:

```bash
uv run fingertip-depth phase1 \
  --input path/to/image.jpg \
  --output-dir outputs/phase1_image \
  --fx-px 1420.3 --fy-px 1418.7 --cx-px 959.2 --cy-px 539.8
```

GT depthがある場合:

```bash
uv run fingertip-depth phase1 \
  --input image.png \
  --output-dir outputs/phase1_eval \
  --fx-px 707.0493 --fy-px 707.0493 \
  --cx-px 604.0814 --cy-px 180.5066 \
  --ground-truth-depth depth.png --ground-truth-scale 256
```

動画:

```bash
uv run fingertip-depth phase1 \
  --input path/to/video.mp4 \
  --output-dir outputs/phase1_video \
  --fx-px 1420.3 --max-frames 100
```

静止画は `depth_m.npy` / `depth_preview.png` / `result.json` を生成します。Phase 1動画は
デフォルトで全処理frameのfloat32 Depth Map、`depth_preview.mp4`、`frames.jsonl`、
`summary.json`を保存します。容量を抑える場合は `--no-save-depth-frames` を指定してください。

## Phase 2

```bash
uv run fingertip-depth phase2 \
  --input path/to/hand.jpg \
  --output-dir outputs/phase2_image \
  --fx-px 1420.3 \
  --hand-model assets/hand_landmarker.task
```

動画も同じコマンドで自動判定されます。必要なら `--input-type image|video` で上書きできます。
動画timestampはdecoderのPTSを優先し、利用不能時にframe index/FPSへfallbackします。MediaPipe
VIDEO modeの要件に合わせ、丸め後も必ず厳密単調増加にします。

MediaPipeの正規化座標 `(x, y)` は実験計画に従って
`floor(x*W + 0.5), floor(y*H + 0.5)`（数学的round）へ変換し、画像端へclipします。
Depth Mapは必ず元RGBと同じ `(H, W)` へ戻してから `D[v, u]` を参照します。MediaPipeの
world landmarkや相対`z`はmetric depthとして使用しません。指先pixelのDepthが0/NaNなら
frameを中断せず、`depth_m: null, depth_valid: false`として記録します。

## iPhoneサンプル実験

HEICはOpenCVでdecodeできない環境でも、`pillow-heif` fallbackとEXIF orientation適用で
静止画として処理できます。提供サンプルの既知距離試験と前後移動試験は次で再実行できます。

```bash
uv run python scripts/evaluate_iphone_samples.py --device cuda:0 \
  --photo-focal-35mm-mm 26 \
  --video-focal-35mm-mm 36
```

`evaluate_iphone_samples.py`と`evaluate_depth_model_comparison.py`は、Phase 2の入力動画を
`--video PATH`で明示できます。省略時は後方互換のため、従来どおり
`<input-dir>/finger_movement.MOV`を使用します。カメラから約20–30 cmで動かした動画だけを
Metric3Dで評価する場合は次を実行します。

```bash
uv run --locked python scripts/evaluate_iphone_samples.py --device cuda:0 \
  --phase 2 \
  --video phase1_2_sample/finger_movement_2030.MOV \
  --video-focal-35mm-mm 36 \
  --output-dir outputs/iphone_phase1_2_2030
```

この実行の出力ルートは`outputs/iphone_phase1_2_2030/`です。

標準実行の出力先は `outputs/iphone_phase1_2/` です。緑箱ROIは固定HSV規則で抽出し、Depth中央値を
代表値にします。動画はlandmark 8のsingle-pixel Depth、欠損区間、frame間変化、
再ローカライズstepを保存します。

実測では、パイプラインは完走したものの既知距離精度はMAE 1.709 mで不合格、動画は
217/267 frame（81.27%）で指先Depthを取得しました。詳細は
[iPhone実データ評価](reports/iphone_sample_phase1_phase2_results.md)を参照してください。

## 代替Depthモデル比較

Metric3D v2-S baselineを変更せず、同じ5枚の既知距離画像と指先前後移動動画で次の4条件を
比較します。

- UniDepth V2-L (`lpiccinelli/unidepth-v2-vitl14`): 近似K入力 / カメラ指定なし
- Depth Pro: 近似焦点入力 / モデルによる焦点推定

依存関係をlocked状態で準備します。Depth Proは公式依存の`numpy<2`がroot環境のOpenCV 5と
衝突するため、`environments/depth_pro`の独立したuv projectを使用します。

```bash
uv sync --locked --group unidepth
uv sync --locked --project environments/depth_pro
```

両backendへ同一の動画画素を入力するため、最初にroot環境でlossless PNGキャッシュを1回だけ
生成します。その後、各backendの2条件を評価し、最後に統合レポートを生成します。

```bash
uv run --locked --group unidepth python scripts/cache_comparison_video_frames.py
uv run --locked --group unidepth python scripts/evaluate_depth_model_comparison.py --backend unidepth
uv run --locked --project environments/depth_pro python scripts/evaluate_depth_model_comparison.py --backend depth-pro
uv run --locked --group unidepth python scripts/build_depth_model_comparison_report.py
```

中間結果、監査用manifest、CSV、JSON、比較図は`outputs/depth_model_comparison/`、読みやすい
結果は[代替Depthモデル比較レポート](reports/depth_model_comparison_results.md)へ出力されます。
初回実行ではcommit/revision固定済みのsourceとcheckpointを取得し、推論前にcheckpointの
SHA-256を検証します。

UniDepth V2-LはCC BY-NC 4.0、Depth ProはApple Machine Learning Research Model Licenseの
対象です。特に商用利用・再配布の可否は各ライセンス原文を確認してください。また本比較の
焦点距離/Kは35 mm換算値から求めた近似であり、正式な精度評価には校正済みKと独立した指先
距離真値が必要です。

## Phase 5–7: Depth Pro教師データ生成

Phase 3（ROI比較）とPhase 4（時系列filter）は今回スキップし、Depth Proの
`approx_focal`条件からlandmark 8の**単一pixel**だけを教師値として取得します。`2.90 cm`は
焦点距離ではなく、20–30 cm動画でこの条件を選んだ際のrange-band violation平均です。
焦点距離入力は動画の36 mm相当から求めた`fx=fy=1832.9295592659223 px`です。

パイプラインは依存関係が衝突するため2段階です。

1. rootのuv環境でフレームとMediaPipe landmarksを監査付きで準備する。
2. 独立したDepth Pro uv環境で教師深度、camera XYZ、3D軌跡、学習用JSONL/CSVを生成する。

`--landmarks`はindex、公式名、または`all`を受け付けます。既定値は人差し指chainの
`5,6,7,8`（MCP/PIP/DIP/TIP）で、教師targetは指定に関係なくlandmark 8です。

```bash
uv run --locked python scripts/prepare_pseudo_label_inputs.py \
  --frame-cache-manifest outputs/depth_model_comparison_2030/shared_video_frames/manifest.json \
  --frame-cache-manifest-sha256 654866c63dce9985ff7e9d41f61db94a838f7ee29eb76b242dc8abd277dbff50 \
  --output-dir outputs/depth_pro_teacher_2030/prepared \
  --hand-model assets/hand_landmarker.task \
  --landmarks INDEX_FINGER_MCP,INDEX_FINGER_PIP,INDEX_FINGER_DIP,INDEX_FINGER_TIP \
  --focal-35mm-mm 36 \
  --frame-transfer-mode hardlink

uv run --locked --project environments/depth_pro \
  python scripts/generate_depth_pro_pseudo_labels.py \
  --prepared-manifest outputs/depth_pro_teacher_2030/prepared/manifest.json \
  --expected-prepared-manifest-sha256 4dc88eee8415d66437a467a9fe55a98e61e3a9f6e03ce57684e6f1c9412b53d7 \
  --output-dir outputs/depth_pro_teacher_2030/dataset \
  --sequence-id finger_movement_2030 \
  --split train \
  --frame-transfer-mode hardlink \
  --teacher-selection-report outputs/depth_model_comparison_2030/video_range_report.json
```

主な成果物は`dataset_manifest.json`、`samples.jsonl`、`targets.csv`、sequence単位の
`splits/train.txt`、3D軌跡のCSV/PLY/PNGです。各sampleには選択landmarks、画像hash、K、
timestamp、教師`Z`、camera座標`(X,Y,Z)`が含まれます。動画frameをランダムに分割すると
temporal leakageになるため、splitは必ずsequence単位で指定します。

実データの件数・深度分布・監査結果は
[Depth Pro教師データ結果](reports/depth_pro_teacher_dataset_2030.md)を参照してください。
このデータはpseudo-labelでありground truthではありません。

## 3動画のStudent用統合データセット

Phase 7の3つの監査済みdatasetを、動画sequence単位の固定splitで統合します。

- train: `finger_movement_2030`（604 identity）+ `finger_movement_3`（2,533 identity）
- validation: `finger_movement`（217 identity）
- augmentation: trainだけをHFlipし、validationはidentity-only

```bash
uv run --locked python scripts/build_student_dataset.py \
  --source-manifest outputs/depth_pro_teacher_three_sequences/finger_movement/dataset/dataset_manifest.json \
  --source-manifest outputs/depth_pro_teacher_2030/dataset/dataset_manifest.json \
  --source-manifest outputs/depth_pro_teacher_three_sequences/finger_movement_3/dataset/dataset_manifest.json \
  --expected-source-manifest-sha256 953cb2e70fda2d05b9af2e59127bca974a172abd09418185ec52d752537f593e \
  --expected-source-manifest-sha256 20a8490c38da962f424fe8ea53e6c11859110fe5b234ec461be64b6db8b2c022 \
  --expected-source-manifest-sha256 2252ab029993e36717e0697c7a456e4a831dd4e1df1ea763028c71effda7f6a0 \
  --output-dir outputs/depth_pro_student_dataset_three_sequences \
  --frame-transfer-mode hardlink
```

完成datasetはtrain **6,274**（identity 3,137 + HFlip 3,137）、validation **217**、
合計 **6,491 sample**です。ただし独立したidentity pseudo-labelは合計3,354件であり、HFlipは
教師値の独立観測を増やしません。

HFlipでは`u_h=W-1-u`、`cx_h=W-1-cx`、`X_h=-X`とし、`v/Y/Z`は維持します。
検出時handednessを保持した上で、view上の`Left/Right`を交換します。Depth ProをHFlip画像へ
再実行せず、光軸方向`Z`をidentityから再利用します。

統合manifestは
`outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json`、SHA-256は
`45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc`です。
source/prepared/dataset hash、reject内訳、Depth分布、leakage監査、制約を含む詳細は
[3動画Student用統合データセット結果](reports/depth_pro_student_dataset_three_sequences.md)を参照してください。
このdatasetもpseudo-labelであり、validationは実距離ground truthではありません。

## Phase 8: Single-frame Student Transformer

RGBをDINO ViTへ入力したimage tokenと、人差し指chain（landmark 5/6/7/8）の
画像平面XY埋め込みを2層Transformerで融合し、landmark 8の光軸方向Depthを1 scalarで
回帰します。MediaPipeのrelative `z`はDepth shortcut/leakageを避けるため、loaderから
モデルまで一切入力しません。

MediaPipeの21部位すべてに独立したtype tokenを用意し、今回入力した5/6/7/8だけを
学習対象としています。未選択17 tokenはoptimizerから除外され、学習後もbit-exactで
初期値と一致することを確認済みです。

`nohup`で20 epochを完走したsequence-held-out validation（217 sample）では、
MAE **3.532 cm**、median absolute error **1.831 cm**でした。全件RMSE **16.748 cm**は
2件の孤立したDepth Pro pseudo-label spikeに強く支配されます。正式指標、constant baseline、
外れ値の事後診断、実行command、checkpoint/hashは
[Phase 8 Single-frame Student Transformer結果](reports/phase8_student_transformer_results.md)を参照してください。

## Scope

実装済み:

- Phase 1: 静止画・動画のMetric3D v2 metric depth、raw保存、可視化、GT評価
- Phase 2: IMAGE/VIDEO mode、landmark 8、single-pixel depth、annotated画像/動画
- Phase 5: `(u,v,Z)`からcamera座標`(X,Y,Z)`を復元
- Phase 6: 欠損区間を跨がない3D軌跡CSV/PLY/PNG
- Phase 7: configurable hand landmarks付きDepth Pro pseudo-label dataset生成
- 3動画の固定sequence splitとtrain-only HFlipによるStudent用dataset統合
- Phase 8: DINO ViT image tokenとXY-only typed landmark tokenを融合するSingle-frame Student Transformer
- iPhone 15 / iOS 26.6.1固定のCore ML変換とMediaPipeリアルタイム3D軌跡描画MVP
- code commit/checkpoint/hand asset hashと実行metadataの記録
- 単体・fake統合・実モデル試験

今回スキップ／後続Phase:

- ROI median / hand mask（Phase 3、今回スキップ）
- temporal filter（Phase 4、今回スキップ）
- Temporal Transformerとtrajectory-aware loss（Phase 9）

実装smoke testは [Phase 1 / Phase 2 report](reports/phase1_phase2_results.md)、
提供iPhoneデータの実測値は
[iPhone実データ評価](reports/iphone_sample_phase1_phase2_results.md)、iOS実装と未完了の実機ゲートは
[iPhone 15固定iOS MVP実装報告](reports/iphone15_ios_mvp_implementation.md)を参照してください。

## Upstream licenses

- Metric3D code: BSD 2-Clause。checkpointの利用条件は公開元を別途確認してください。
- MediaPipe: Apache License 2.0。
