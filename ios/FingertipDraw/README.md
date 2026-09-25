# Fingertip Draw iOS MVP

固定した iPhone 15（`iPhone15,4`）/ iOS 26.6.1 の背面広角カメラで、MediaPipe の
landmark 5–8 と Phase 8 student model から人差し指先の `(X, Y, Z)` を推定し、カメラ映像と
X–Z パネルへ軌跡を描画する最小アプリです。

## 固定条件

- 縦向き、1080×1920、30 fps
- 背面 wide camera、手ぶれ補正なし
- 26 mm 相当のwide cameraへ `36 / 26` 倍のzoomを設定し、学習動画の36 mm相当FOVへ合わせる
- 実測Kが得られた場合は学習時の `f=1832.93 px` から±10%以内であることを要求する
- MediaPipe入力は同じカメラフレーム、`.liveStream`、1 hand
- student入力はRGB全体をcropなしで224×224へbicubic resize
- 入力landmarkは 5, 6, 7, 8 の正規化 `(x, y)` のみ。MediaPipe relative-zは使用しない
- Core ML出力は加工していない光軸方向深さ `Z [m]`
- 逆投影座標系は x-right / y-down / z-forward
- 起動時にCore MLのcheckpoint SHA、対象端末metadata、入出力名・型・shapeを検証する

端末、OS、解像度、実測FOV、モデルのいずれかが固定条件と異なる場合は推論を開始しません。

## Macでの準備

Python依存関係とCore ML変換は `uv` で固定されています。MediaPipeのiOSバイナリだけは公式の
配布方法に従いCocoaPodsを使います。プロジェクト生成にはXcodeGen 2.38.0以上が必要です。

```bash
cd /path/to/depth_for_task
uv sync --project environments/coreml_export
uv pip install --python environments/coreml_export/.venv/bin/python --no-deps -e .

# Linuxで生成済みのモデルを再生成する場合だけ実行
environments/coreml_export/.venv/bin/python scripts/export_student_coreml.py

# macOS Core ML runtimeで604フレームをPyTorch基準値と比較
environments/coreml_export/.venv/bin/python scripts/verify_student_coreml.py

cd ios/FingertipDraw
xcodegen generate
pod install
open FingertipDraw.xcworkspace
```

XcodeではSigning Teamを選び、接続したiPhone 15を実行先にします。Core ML一致試験の合格条件は
平均絶対差1 mm以下、最大絶対差3 mm以下です。この604フレーム試験は変換後モデル単体を照合する
もので、iOSのカメラ回転・BGRA→RGB・Core Image resizeは実機側で別に確認します。

## 操作

- `描画`: 新しい推定点を軌跡へ追加
- `停止`: 推定表示は続けたまま軌跡への追加を停止
- `消去`: 現在の軌跡を消去

画面上部には深さ、実効fps、MediaPipe/student/全体の遅延、カメラ内部パラメータが実測か近似かを
表示します。各推定値はアプリのDocuments内に `trajectory.csv`、条件は `session.json` として保存
されます。

## 現在の検証境界

- Core MLパッケージ生成と入出力仕様監査はLinuxで完了
- portrait座標からpreview用未回転座標への変換、画素丸め、逆投影、FOVゲートの単体テストを用意
- Core ML runtimeの数値一致、Core Image前処理golden試験の作成・実行、Xcode buildはMacで行う
- カメラ向き、軌跡重畳、実測K、速度、発熱、実距離精度は固定した実機で確認する
- 平滑化、ARKit、移動カメラ座標への変換はまだ含めない
