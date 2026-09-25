# iPhone 15固定・リアルタイム指先3D描画 MVP 実装報告

## 対象条件

- 端末: iPhone 15 (`iPhone15,4`)
- OS: iOS 26.6.1
- カメラ: 背面wide、縦向き1080×1920、30 fps、手ぶれ補正off
- 画角: wide cameraの26 mm相当に `36 / 26 = 1.384615...` のzoomを設定
- 学習時K: 36 mm相当対角画角近似 `fx=fy=1832.9296 px`。実測Kは±10%以内を要求
- 手: 1 hand、MediaPipe Hand Landmarkerのlandmark 5, 6, 7, 8
- student入力: 224×224 RGBと選択landmarkの正規化XYだけ
- 出力: 人差し指先の光軸方向深さZ [m]
- 座標系: x-right / y-down / z-forward

## 実装済み

1. 学習済みPhase 8 checkpointとrun manifestのSHA-256検証
2. 画像正規化とlandmarkの `[0, 1] -> [-1, 1]` を内包した固定shape wrapper
3. `coremltools 9.0`によるfloat16 ML Program変換
4. iPhone 15 / iOS 26.6.1を実行時に検証するUIKitアプリ
5. AVFoundationの同一フレームをMediaPipeとstudentへ渡すlatest-frame-wins pipeline
6. AVFoundation Kのフレーム単位取得と、取得不能時の36 mm相当近似K
7. `(u, v, Z)`から`(X, Y, Z)`への逆投影
8. カメラ上の軌跡、X-Zパネル、深さ・fps・遅延HUD
9. 描画・停止・消去操作
10. CSV trajectoryとJSON session metadataの保存
11. Core ML比較用の604フレーム自己完結fixtureとmacOS検証CLI
12. Core MLのcheckpoint SHA、対象端末metadata、入出力契約を起動時に検証
13. 1080×1920および実測FOVを推論前に検証し、不一致時は描画操作ごと停止
14. MediaPipeのportrait座標をpreview layer用の未回転capture座標へ明示変換

MediaPipeのrelative-zとworld landmarksはstudent入力にも3D位置算出にも使用していない。

## Core ML変換結果

- source checkpoint SHA-256: `e2e8941d2187e20dc716580fbbafb294cc9809db54b467a2d09fb52b39492252`
- Core ML形式: ML Program / specification version 9 / float16
- Core ML package容量: 51,027,727 bytes
- 入力:
  - `image`: RGB 224×224。Core MLで `1/255`、model内でImageNet mean/std正規化
  - `landmarks_xy`: float32 `[1, 4, 2]`
- 出力: `depth_m`: float32 `[1, 1]`
- 変換用TorchScriptに `aten::_transformer_encoder_layer_fwd` と `aten::Int` が残っていないことを確認

PyTorch 2.7の評価時Transformer融合演算はcoremltools 9.0が未対応だった。このため、再学習や重み変更は行わず、同一重みをQKV線形変換、softmax、行列積、residual、MLP、LayerNormへ明示的に展開した。確認入力における元PyTorch出力、展開wrapper、TorchScript出力の差は0.0 mだった。

## 604フレーム照合fixture

- sequence: `finger_movement_2030`
- sample count: 604
- image: uint8 RGB `[604, 224, 224, 3]`
- landmark: float32 `[604, 4, 2]`
- PyTorch FP32予測範囲: 0.2237596–0.4262169 m
- fixture容量: 60,295,161 bytes
- Core ML runtime合格条件:
  - 平均絶対差 <= 0.001 m
  - 最大絶対差 <= 0.003 m

このfixtureは既にOpenCVで224×224 RGBへ変換済みであり、Core MLモデル単体の変換誤差を測る。
iOS側の回転、BGRA→RGB、Core Image resize、MediaPipe座標との同期は含まない。

## 検証結果

- Ruff: pass
- Python tests: 209 passed
- Core ML package static inspection: pass
  - `image`: RGB image 224×224
  - `landmarks_xy`: float32 `[1, 4, 2]`
  - `depth_m`: float32 `[1, 1]`
- Artifact provenance preflight: pass
  - model `84938b…d161b` / fixture `2ae8de…eb7c` / checkpoint `e2e894…2252` / manifest `5acb0e…ba83`
- Core ML runtime parity: 未実施（LinuxではCore ML runtimeを実行できない）
- Xcode build / iPhone実機試験: 未実施（Mac/Xcode/iPhoneが必要）

## Mac/iPhoneで次に行うゲート

1. `scripts/verify_student_coreml.py`で成果物hashを確認し、604フレームのモデルparityを測定
2. raw 1080×1920フレームからCore Image resizeまでを通すgolden試験を作成・実行
3. XcodeGenとCocoaPodsからworkspaceを作成しbuild
4. iPhone 15 / iOS 26.6.1でcamera orientation、軌跡重畳、K、36 mm相当FOVを確認
5. 5分連続試験で出力25 Hz以上、median 33 ms以下、p95 50 ms以下、backlogなしを確認
6. 20–40 cmの実距離試験でMAE 1 cm以下、静止Z標準偏差0.5 cm以下を確認

Core ML parityまたは実機速度が不合格の場合だけ、float32変換、Core ML最適化、encoder軽量化の順で切り分ける。
