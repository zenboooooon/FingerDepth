# Depth Pro教師による3動画Student用統合データセット

実行日: 2026-09-25 (Asia/Tokyo)

## 結論

3本の動画をsequence単位で分離し、Depth Proのpseudo-labelを使うSingle-frame Student向け
データセットを生成した。

- train: `finger_movement_2030` + `finger_movement_3`
- validation: `finger_movement`
- train identity: 3,137 sample
- train HFlip: 3,137 sample
- train合計: **6,274 sample**
- validation: **217 sample**（identityのみ）
- 全体: **6,491 sample / 6,491 materialized image files**

確定manifestは
`outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json`、SHA-256は
`45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc`である。

ただし、train 6,274件は3,137個のidentity pseudo-labelをidentity/HFlipの2 viewにした値であり、
独立した教師観測が6,274件あるわけではない。全体も3,354個のidentity pseudo-labelと3,137個の
augmentation viewから成る。labelは実距離ground truthではなく、同じDepth Pro教師を再現するための
pseudo-labelである。

## 構成と件数

| source sequence | split | source frames | rejected | identity | HFlip | materialized views | image size | identity採用率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `finger_movement_2030` | train | 661 | 57 | 604 | 604 | 1,208 | 1080×1920 | 91.38% |
| `finger_movement_3` | train | 3,206 | 673 | 2,533 | 2,533 | 5,066 | 1080×1920 | 79.01% |
| `finger_movement` | validation | 267 | 50 | 217 | 0 | 217 | 1440×1920 | 81.27% |
| **合計** | — | **4,134** | **780** | **3,354** | **3,137** | **6,491** | mixed | **81.13%** |

trainではidentity 3,137件をHFlipでちょうど2倍にし、validationにはaugmentationを適用していない。
split名と成果物名は`valid`ではなく`validation`であり、明示的なsplit fileは
`splits/train.txt`と`splits/validation.txt`である。test splitはまだない。

`finger_movement_3`はtrain identityの2,533/3,137件、すなわち80.75%を占める。HFlip後も比率は
変わらないため、sampleを一様に読む学習ではこのsequenceがtrain lossを支配しやすい。

## 教師、feature、target

- Teacher condition: Depth Pro `depth_pro__approx_focal`
- Depth Pro source commit: `9e65e4dbe9568d23c546fcec53302b10445e109e`
- checkpoint revision: `ccd1350a774eb2248bcdfb3be430e38f1d3087ef`
- checkpoint SHA-256:
  `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`
- precision: `float16`
- feature landmarks: 5/6/7/8
  (`INDEX_FINGER_MCP/PIP/DIP/TIP`)
- target: landmark 8 (`INDEX_FINGER_TIP`)の単一pixel Depth
- target semantics: 光軸方向の`Z`、単位metre
- confidence: なし
- fitted scale/offset: なし
- Phase 3 ROI、Phase 4 temporal filter、label clipping: すべて未適用

3 sequenceとも35 mm換算36 mmから得たcentered pinhole近似を使う。校正済みKではない。

| sequence | `fx=fy` [px] | `cx` [px] | `cy` [px] |
|---|---:|---:|---:|
| `finger_movement_2030` | 1832.929559 | 539.5 | 959.5 |
| `finger_movement_3` | 1832.929559 | 539.5 | 959.5 |
| `finger_movement` | 1996.920706 | 719.5 | 959.5 |

camera座標は`x-right, y-down, z-forward`で、identity sampleは次で復元している。

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = z_teacher_m
```

trainとvalidationで横解像度とKが異なる。統合datasetは元解像度を保持しているため、Student側で
resize/cropを追加する場合は、画像だけでなくlandmark座標とKを同じ変換で更新する必要がある。

## Train-only HFlip

HFlipは画像だけでなく、pixel、normalized landmark、K、camera XYZ、handednessを一貫して変換する。

```text
I_h[v, W - 1 - u] = I[v, u]
u_h = W - 1 - u
v_h = v
fx_h = fx
fy_h = fy
cx_h = W - 1 - cx
cy_h = cy
X_h = -X
Y_h = Y
Z_h = Z
```

整数pixelをauthorityとし、normalized `x`は変換後に本実装のhalf-up roundingで必ず`u_h`へ戻る
値を保存する。landmark indexとMediaPipe relative `z`は維持する。検出時のhandednessは
`handedness_detected_source`へ保持し、view上の`Left/Right`だけを交換する。

光軸方向`Z`は左右反転で不変とし、HFlip画像に対するDepth Pro推論は再実行していない。したがって
HFlipは画像・2D/3D幾何viewのaugmentationであり、新しい教師測定ではない。また、source Phase 7の
軌跡はidentity-onlyの物理軌跡として残し、HFlip点を別の実軌跡として結合していない。

## 教師Depth分布

次の統計はaugmentationで重複させる前のidentity pseudo-labelについて計算した。HFlipは`Z`を再利用
するため、train view全体の分布もtrain identityと同じである。

| subset | n | min | p05 | median | p95 | max | `>0.5 m` | `>1 m` | `>2 m` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `finger_movement_2030` | 604 | 0.216553 | 0.244543 | 0.292847 | 0.407922 | 0.429443 | 0 | 0 | 0 |
| `finger_movement_3` | 2,533 | 0.128540 | 0.196167 | 0.260010 | 0.353955 | 2.324219 | 6 | 3 | 2 |
| **train identity合計** | **3,137** | **0.128540** | **0.201294** | **0.268799** | **0.374902** | **2.324219** | **6** | **3** | **2** |
| `finger_movement` validation | 217 | 0.173950 | 0.199414 | 0.317139 | 0.440869 | 2.587891 | 2 | 1 | 1 |

大部分は0.2–0.4 m付近だが、trainとvalidationの双方に2 m超のlong-tail値がある。今回のfilterは
構造チェックだけであり、これらをclipまたは除外していない。Student評価では全sampleの通常指標を
隠さず示した上で、median/percentileなどoutlierに頑健な指標も併記する必要がある。

## Reject内訳

次はframe単位で相互排他的なreason組合せを集計した結果である。

| sequence | `no_hand` | feature 5 out | feature 5+6 out | target/feature 8 out | 合計 |
|---|---:|---:|---:|---:|---:|
| `finger_movement_2030` | 55 | 2 | 0 | 0 | 57 |
| `finger_movement_3` | 536 | 131 | 4 | 2 | 673 |
| `finger_movement` | 50 | 0 | 0 | 0 | 50 |
| **合計** | **641** | **133** | **4** | **2** | **780** |

feature 5+6 outの4件は、個別reasonで数えるとfeature 5にもfeature 6にも含まれる。target/feature 8
outの2件も同じframeに2 reasonを持つため、reasonを単純加算してframe数と比較してはいけない。
準備段階を通過した3,354件ではDepth Pro推論失敗は0件だった。

この採用規則は「MediaPipeが手を検出し、feature 5/6/7/8がすべて画面内にあるframe」へデータを
条件付ける。したがって、検出失敗、遮蔽、画面端に弱いcaseは学習分布から過少代表になる。

## Splitとleakage監査

- split unitはsource video sequenceであり、同一動画内のframeをランダム分割していない。
- 3本のsource video SHA-256はすべて異なる。
- augmented viewはsource splitを継承する。
- validationはidentity-onlyである。
- source manifestの宣言artifact hash、各sample PNG hash、復号後BGR pixel hashを統合前に全件検証した。
- source video SHA、sequence ID、teacher/checkpoint、feature/target、filter、camera座標規約の整合性を検証した。
- builderはtrain/validation identity間のBGR pixel hash重複を拒否する。
- 完成後にtrain全6,274 viewとvalidation全217 viewを追加監査し、各split内のhashはすべてunique、
  cross-split BGR pixel hashの共通要素は**0件**だった。これによりtrain HFlipとvalidation identityの
  完全一致も今回の成果物にはないことを確認した。

最後のall-view cross-split検査は今回のレポート作成時に追加実施した監査であり、現行builderが
manifestへ記録するproduction invariantはidentity間の検査までである。

sequence holdoutはframe-level leakageを防ぐが、異なる被験者、撮影session、device、背景、照明を
保証するmetadataはない。また、validationの`finger_movement`は既存のDepthモデル比較に使用済みで、
完全未使用のtest setではない。今後validationでhyperparameterやcheckpointを選べば、そのsequenceも
model selection dataになる。最終評価用には別の未使用test sequenceが必要である。

## Provenance

| sequence | source video SHA-256 | prepared manifest SHA-256 | Phase 7 dataset manifest SHA-256 |
|---|---|---|---|
| `finger_movement` | `f052d06f5b8e625fef79ee087064271ac05503458ae3525f9d8b4524168f5802` | `6fda82f014632b32d1719f74fd75ca7f56897efbc66061dc3c3360217228589d` | `953cb2e70fda2d05b9af2e59127bca974a172abd09418185ec52d752537f593e` |
| `finger_movement_2030` | `1a9bbadaa0af3dde2b651ca18b4619685d0950208d9b45e01e334d58450a6ba3` | `4dc88eee8415d66437a467a9fe55a98e61e3a9f6e03ce57684e6f1c9412b53d7` | `20a8490c38da962f424fe8ea53e6c11859110fe5b234ec461be64b6db8b2c022` |
| `finger_movement_3` | `6050106a4dc284fcd08eedfc56f2b0b65d70365090a72910ae6d8ecbf3f2e370` | `1e0fbb1438421e051430d0952f83303d71164f42f9e4bc20eded30a4d116ad9a` | `2252ab029993e36717e0697c7a456e4a831dd4e1df1ea763028c71effda7f6a0` |

`finger_movement`と`finger_movement_2030`は監査済みlossless frame cacheを再利用し、
`finger_movement_3`は動画を直接1回decodeしてlossless PNG化した。すべてのprepared manifestでPNGと
復号後BGRを検証している。

Depth Pro `approx_focal`の選定根拠である20–30 cm帯違反平均`0.0290375058 m`（2.90 cm）は
`finger_movement_2030`だけにscopeされる。これは焦点距離でもper-frame真値誤差でもなく、
`finger_movement_3`やvalidationの実距離精度を裏付ける値ではない。

統合成果物の主要hashは次のとおりである。

| artifact | SHA-256 |
|---|---|
| `dataset_manifest.json` | `45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc` |
| `samples.jsonl` | `4edede3b0330c773f44dc9d5d7882857f901a226bd812206182b9f97157ab241` |
| `targets.csv` | `ff1b6b4b53511a306808aea1fb523912b6bbf86b69fbf18da0dfddd5cc1faf21` |
| `splits/train.txt` | `4e1fb42d99dd125ea634cc4dcef335d68e6dae33f4580f22f064a8af5b0d047d` |
| `splits/validation.txt` | `c263767a45d77973b86753330947c44523ed987e3c026246ecab053f543fd8c6` |
| ordered output PNG hashes | `59c41c18f0aaa177d33662a6ae7e8dee7fd7edc1d64beb28f36ae134036141fd` |
| ordered output BGR pixel hashes | `3ba63e63874a7268a0aee9001ee5412bd0ba2fd7cd0d67fe69768f1afcbe7661` |

identity PNGは可能な限りsourceへのhardlink、HFlipはPNG compression level 3で新規にlossless encodeした。
完成datasetの測定上のdisk footprintは約11 GBだった。実装hash、OpenCV/NumPy version、全source
descriptorは統合manifestに保存している。

## 再現手順

各output directoryは新規または空である必要がある。既存の監査済みframe cacheを再利用する2 sequence、
直接decodeする1 sequenceの順にPhase 7 sourceを生成し、その後統合する。

### 1. `finger_movement`（validation）

```bash
uv run --locked python scripts/prepare_pseudo_label_inputs.py \
  --frame-cache-manifest outputs/depth_model_comparison/shared_video_frames/manifest.json \
  --frame-cache-manifest-sha256 e86ae1ab0793499a05255d79942e357af26306d2fa9a9b0c8d030703794a9068 \
  --output-dir outputs/depth_pro_teacher_three_sequences/finger_movement/prepared \
  --hand-model assets/hand_landmarker.task \
  --landmarks INDEX_FINGER_MCP,INDEX_FINGER_PIP,INDEX_FINGER_DIP,INDEX_FINGER_TIP \
  --focal-35mm-mm 36 \
  --frame-transfer-mode hardlink

uv run --locked --project environments/depth_pro \
  python scripts/generate_depth_pro_pseudo_labels.py \
  --prepared-manifest outputs/depth_pro_teacher_three_sequences/finger_movement/prepared/manifest.json \
  --expected-prepared-manifest-sha256 6fda82f014632b32d1719f74fd75ca7f56897efbc66061dc3c3360217228589d \
  --output-dir outputs/depth_pro_teacher_three_sequences/finger_movement/dataset \
  --sequence-id finger_movement \
  --split validation \
  --frame-transfer-mode hardlink \
  --device cuda:0
```

### 2. `finger_movement_2030`（train）

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
  --teacher-selection-report outputs/depth_model_comparison_2030/video_range_report.json \
  --device cuda:0
```

### 3. `finger_movement_3`（train）

```bash
uv run --locked python scripts/prepare_pseudo_label_inputs.py \
  --input-video phase1_2_sample/finger_movement_3.MOV \
  --output-dir outputs/depth_pro_teacher_three_sequences/finger_movement_3/prepared \
  --hand-model assets/hand_landmarker.task \
  --landmarks INDEX_FINGER_MCP,INDEX_FINGER_PIP,INDEX_FINGER_DIP,INDEX_FINGER_TIP \
  --focal-35mm-mm 36 \
  --frame-transfer-mode hardlink

uv run --locked --project environments/depth_pro \
  python scripts/generate_depth_pro_pseudo_labels.py \
  --prepared-manifest outputs/depth_pro_teacher_three_sequences/finger_movement_3/prepared/manifest.json \
  --expected-prepared-manifest-sha256 1e0fbb1438421e051430d0952f83303d71164f42f9e4bc20eded30a4d116ad9a \
  --output-dir outputs/depth_pro_teacher_three_sequences/finger_movement_3/dataset \
  --sequence-id finger_movement_3 \
  --split train \
  --frame-transfer-mode hardlink \
  --device cuda:0
```

### 4. 3 sequenceを統合

`--expected-source-manifest-sha256`は`--source-manifest`と同じ入力順で対応させる。

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

## 成果物

- manifest:
  `outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json`
- full samples:
  `outputs/depth_pro_student_dataset_three_sequences/samples.jsonl`
- compact target table:
  `outputs/depth_pro_student_dataset_three_sequences/targets.csv`
- train split:
  `outputs/depth_pro_student_dataset_three_sequences/splits/train.txt`
- validation split:
  `outputs/depth_pro_student_dataset_three_sequences/splits/validation.txt`
- images:
  `outputs/depth_pro_student_dataset_three_sequences/frames/<sequence>/<identity|hflip>/`

行数を実測し、`samples.jsonl=6491`、`targets.csv=6492`（headerを含む）、
`train.txt=6274`、`validation.txt=217`、画像file=6491でmanifestと一致した。

## 品質確認

```text
uv run --locked pytest                                # 126 passed
uv run --locked ruff check .                          # All checks passed
uv run --locked ruff format --check \
  src/fingertip_depth/student_dataset.py \
  scripts/build_student_dataset.py \
  tests/test_student_dataset.py                       # 3 files already formatted
uv lock --check                                       # OK
uv lock --check --project environments/depth_pro      # OK
```

pytestではmodel comparisonのconstant fake inputに対するPearson計算由来のNumPy RuntimeWarningが2件
出るが、test failureはない。

repository全体への`ruff format --check .`は、この統合dataset追加3ファイル以外の17ファイルに
既存の整形差分を検出した。上記の対象3ファイルはformat checkを通過しており、未整形ファイルへの
一括変更は今回行っていない。

## 解釈上の制約と次の作業

1. validationはStudent trainからsequence単位でhold outされているが、targetは同じDepth Pro教師の
   pseudo-labelである。測れるのはteacherへのdistillation fidelityであり、実距離精度ではない。
2. 3 sequenceは時系列frameの集合で強く自己相関する。3,354 identity sampleや217 validation sampleを
   IID観測として扱った有意差検定・信頼区間は不適切である。
3. 異なるvideo hashは確認したが、別被験者・別session・別device・別背景を保証するmetadataはない。
4. validationは既存のモデル比較に使われており、完全未使用testではない。校正済みK、独立した指先
   実距離GT、未使用test sequenceを別途用意する必要がある。
5. HFlipはtrain viewを増やすが、教師Depthの多様性や撮影domainを増やさない。
6. `finger_movement_3`への偏り、MediaPipe検出/画面内filterによるselection bias、単一pixel labelの
   long-tailを、sampling、loss、評価指標の設計時に考慮する必要がある。
7. 現成果物はdataset構築までであり、Studentモデルの学習、checkpoint選択、validation結果はまだない。
8. Depth Pro checkpointと生成物の利用・再配布はApple Machine Learning Research Model Licenseの
   原文を確認する必要がある。
