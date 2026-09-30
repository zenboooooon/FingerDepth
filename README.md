# Monocular Fingertip Depth and Student Training

## Metric3Dの退避

Metric3Dによる推論経路は廃止しました。Metric3Dを呼び出すCLI、画像・動画パイプライン、サンプル評価スクリプトはありません。旧ソースは実行・importできないMarkdownとして [`archive/metric3d_retired.md`](archive/metric3d_retired.md) にまとめています。

`fingertip-depth phase1/phase2` と `scripts/evaluate_iphone_samples.py`、`scripts/evaluate_kitti_demo.py` は削除済みです。過去のMetric3D結果レポートは履歴資料として残っていますが、以後の学習経路からは呼び出されません。

## 動画を追加してStudentを学習

root環境と依存関係を分離したDepth Pro環境を準備します。

```bash
uv sync --locked
uv sync --locked --project environments/depth_pro
uv run --locked python scripts/download_hand_landmarker.py
```

学習動画を `data/training_videos/train/` に置きます。`validation/` はsequence-held-out評価用の固定集合として扱い、同じ撮影sessionのclipや再encodeをtrain/validationへ分けないでください。

```bash
cp /path/to/new_capture.MOV data/training_videos/train/new_capture.MOV
uv run --locked fingertip-train status
uv run --locked fingertip-train run
```

`fingertip-train run --dry-run` は実行予定を確認します。同じdatasetを明示的に再学習するときは `--force-train`、別設定を使うときは `--config path/to/config.toml` を指定します。既定では `.mov` / `.mp4` / `.m4v` を検出し、ハッシュを検証した成果物を再利用します。

処理は次の順です。

1. train/validation動画を検出し、動画内容からsequence IDを確定する。
2. root環境でPNGフレームとMediaPipe手指ランドマークを準備する。
3. 分離したDepth Pro環境で教師深度を生成する。
4. 動画sequence単位のsplitを保ってStudent用datasetを構築する。
5. Studentモデルを学習し、run manifestとcheckpointを保存する。

ラベル生成が中断された場合は同じ `fingertip-train run` を再実行します。検証済みの準備フレームとDepth Proのチェックポイント結果から再開します。動画、焦点距離、モデル、設定、実装が変わると、異なる進捗は再利用しません。最終cacheは成果物の検証後に公開されます。

既定焦点距離はiPhone実験に基づく36 mm相当の近似で、校正値ではありません。別のカメラやzoomでは `configs/training_pipeline.toml` の `video_overrides` に正しい35 mm相当値を指定してください。詳細は [training video guide](data/training_videos/README.md) を参照してください。

## 教師ラベルとStudent

Depth Proの `approx_focal` 条件から、人差し指先端（MediaPipe landmark 8）の単一画素深度を教師値として使います。これらは実測ground truthではなくpseudo-labelです。カメラ座標 `(X,Y,Z)` と軌跡成果物も作成します。

デフォルトのStudent入力はRGB画像と人差し指chain（landmark 5/6/7/8）の画像平面XYです。MediaPipeの相対 `z` はモデルに渡しません。学習datasetではtrain側だけを水平反転し、validationは反転しません。現在の学習設定では、教師深度の孤立した時系列スパイクと、教師深度が0.8 m以上の標本をtrain/validationの両方から除外します。

Phaseごとの手動コマンドは監査済みデータの再現や個別debug向けです。新しい動画の通常処理には `fingertip-train` を使います。

## 学習後

- `scripts/export_student_coreml.py` は検証済みcheckpointをCore MLへ変換します。
- `scripts/render_student_trajectory_demo.py` はStudentの予測深度から3D軌跡の可視化を作ります。
- `scripts/analyze_student_validation.py` は保存済み検証結果を分析します。

軌跡デモは可視化であり、ground-truth精度評価ではありません。現在のカメラ内部パラメータは近似値です。

## 実装範囲

- Depth Pro疑似ラベルの準備、検証、再開
- sequence単位splitとtrain-only水平反転によるdataset統合
- ViT画像特徴とXYランドマーク特徴を融合するsingle-frame Student
- Core ML変換とオフライン軌跡デモ
- 保存済みの別モデル比較・旧評価レポートの読み取り

ROI median / hand mask（Phase 3）、時系列ラベルfilter（Phase 4）、Temporal Transformer（Phase 9）は現在のStudent学習経路では使いません。

## Upstream licenses

- MediaPipe: Apache License 2.0。
- Depth ProとUniDepthの利用条件は各モデルのライセンスに従います。
