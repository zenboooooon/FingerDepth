# Phase 8 Single-frame Student Transformer 学習結果

実行日: 2026-09-25 (Asia/Tokyo)

## 結論

3動画をsequence単位でtrain/validationへ分離したDepth Pro pseudo-label datasetを使い、
RGB画像と人差し指landmark 5/6/7/8の画像平面座標だけを入力するSingle-frame Student
Transformerを学習した。20 epochを完走し、validation MSEで選んだbest checkpointはepoch 20だった。

- validation: 217 sample、MAE **3.531995 cm**、median absolute error **1.831055 cm**
- validation RMSE: **16.747969 cm**、p95 absolute error: **5.185547 cm**
- constant train-mean baseline比: MSE **17.69%**減、MAE **58.33%**減
- negative prediction: **0件**
- 学習時間: **119.681 s**（RTX 5090、bfloat16）

RMSEとmax errorは、validation中の2つの不連続なDepth Pro targetに強く支配される。これは後述する
事後診断であり、正式な評価値から除外していない。また、targetは実距離ground truthではなくDepth Proの
単一pixel pseudo-labelであるため、この結果は「教師の再現性能」であって実距離精度の証明ではない。

確定run manifestは
`outputs/phase8_student_vit_landmarks_xy/run_manifest.json`、SHA-256は
`607faa5b5e25a9743f1b7bcc18d2ff6d0c51cb20441ffd0d542b028332ae66f8`である。

## Taskと入力

予測対象は人差し指先端（landmark 8）の光軸方向深度`z_teacher_m`、単位metreである。モデルは
sigmoid、softplus、clippingを適用しないraw scalarを1つ出力し、標準化していないtargetとのMSE
（m²）で学習する。これはcamera中心からのradial distanceではなくcameraの`Z`である。

入力は次だけに限定した。

- RGB: 224×224へbicubicで直接resize（crop/letterboxなし）、ImageNet mean/stdで正規化
- landmark: 5/6/7/8（MCP/PIP/DIP/TIP）の`x_normalized`、`y_normalized`
- 座標変換: `x'=2*x_normalized-1`、`y'=2*y_normalized-1`

datasetにはMediaPipeのrelative `z`も保存されるが、モデル入力には明示的に含めていない。relative `z`が
target深度と相関するshortcut/leakageになり得るためである。camera intrinsics、camera XYZ、教師深度、
handednessも入力していない。したがって、landmark数値featureは各点2次元のXYのみである。

## Dataset、split、HFlip

使用したmanifestは
`outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json`、SHA-256は
`45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc`である。

| split | source sequence | identity | HFlip | 学習時sample |
|---|---|---:|---:|---:|
| train | `finger_movement_2030` | 604 | 604 | 1,208 |
| train | `finger_movement_3` | 2,533 | 2,533 | 5,066 |
| validation | `finger_movement` | 217 | 0 | 217 |
| **合計** | 3 sequence | **3,354** | **3,137** | **6,491** |

trainは6,274 viewだが、独立したtrain pseudo-labelは3,137件である。HFlipはtrain identityだけに
適用し、教師`Z`を再利用するため独立観測を増やさない。validationはidentity-onlyである。
split unitはsource video sequenceであり、同じ動画内のframeをランダム分割していない。

HFlip datasetは画像と同時に`u_h=W-1-u`、`cx_h=W-1-cx`、`X_h=-X`を適用し、`v/Y/Z`を維持する。
ただしPhase 8が数値入力として読むのは変換後のnormalized XYだけである。dataset生成・hash・分布の詳細は
[3動画Student用統合データセット結果](depth_pro_student_dataset_three_sequences.md)を参照すること。

## Model architecture

画像encoderはDINOで事前学習された`timm`の`vit_small_patch16_224.dino`で、embedding dimensionは
384である。`forward_features`が返すimage token列を使い、encoderもfreezeせずfine-tuneした。

融合部は次の構成である。

1. 各landmarkのXYを`Linear(2,384) -> GELU -> Linear(384,384) -> LayerNorm`で埋め込む。
2. 座標埋め込みにlandmark type tokenとlandmark modality tokenを加える。
3. learned depth query、ViT image tokens、4個のlandmark tokensを連結する。
4. 2層のpre-norm Transformer Encoder（6 heads、FFN 1,536、GELU、dropout 0.1）で融合する。
5. depth query位置を`LayerNorm -> Linear(384,192) -> GELU -> Dropout -> Linear(192,1)`へ渡す。

最終Linearのweightは0、biasはtrain target meanの`0.27726958168134364 m`で初期化した。総parameter数は
25,449,601、trainableは25,443,073、うちViT encoderは21,665,664 parameterすべてを更新した。

### 21個のlandmark type token

MediaPipe hand landmark 0〜20すべてについて独立した384次元type tokenを実体化した（合計8,064
parameter）。今回入力した5/6/7/8の4 tokenだけを`requires_grad=true`としてoptimizerへ入れ、残り17個は
freezeした。best checkpointの監査結果は次のとおりである。

| token | 選択 | trainable parameter | 初期値からのL2変化 |
|---|---|---:|---:|
| 5 / INDEX_FINGER_MCP | yes | 384 | 0.039555326104164124 |
| 6 / INDEX_FINGER_PIP | yes | 384 | 0.03970351442694664 |
| 7 / INDEX_FINGER_DIP | yes | 384 | 0.03988010436296463 |
| 8 / INDEX_FINGER_TIP | yes | 384 | 0.03545846417546272 |
| 0–4, 9–20 | no | 0 | すべて0、bit-exact unchanged |

この監査により「21種類を定義したこと」と「未使用tokenまで誤って更新していないこと」の両方を確認した。

## 学習条件

| 項目 | 値 |
|---|---|
| epoch / batch size | 20 / 32 |
| optimizer | AdamW |
| encoder LR / new-module LR | `1e-5` / `1e-4` |
| weight decay | 0.05（biasと1次元parameterは0） |
| scheduler | step単位linear warmup 5% + cosine decay |
| gradient clipping | global norm 1.0 |
| early stopping | validation MSE、patience 6（今回は未発火） |
| precision / device | bfloat16 / `cuda:0` |
| seed | 20260925 |
| DataLoader workers | 4 |
| image preload | 有効、全source PNG SHA-256検証済み |
| best checkpoint基準 | 最小validation MSE |

runtimeはPython 3.12.13、PyTorch 2.14.0+cu130、timm 1.0.30、CUDA 13.0、NVIDIA
GeForce RTX 5090だった。事前学習済みDINO weightはローカルcacheからofflineで読み込んだ。

## 結果

### Best checkpoint（epoch 20）

次は`run_manifest.json`に記録された全sampleでの正式な指標である。

| 指標 | train (n=6,274) | validation (n=217) |
|---|---:|---:|
| MSE [m²] | 0.00005937761701061576 | 0.028049445166016505 |
| RMSE [m] | 0.007705687315912563 | 0.16747968583089862 |
| MAE [m] | 0.004850281170306025 | 0.035319947976670506 |
| median absolute error [m] | 0.00390625 | 0.018310546875 |
| p95 absolute error [m] | 0.01171875 | 0.05185546874999997 |
| bias [m] | 0.0019307888956407396 | -0.0024937220982142855 |
| max absolute error [m] | 0.19921875 | 2.322265625 |
| Pearson r | 0.995114912676446 | 0.33984508807835756 |
| negative prediction | 0 | 0 |
| consecutive-frame pair | 6,228 | 215 |
| consecutive-frame delta MAE [m] | 0.00499449468378693 | 0.03246899981831396 |

trainはHFlipを含むview単位の評価である。trainとvalidationの大きな差、特にPearson rの
`0.9951`対`0.3398`は、sequence間domain shift、少数sequenceへのoverfit、教師outlierの影響を
区別できないため、未知動画への一般化性能とは解釈しない。

### Constant baselineとの比較

| validation指標 | Student | train mean定数 | train median定数 |
|---|---:|---:|---:|
| MSE [m²] | 0.028049445166016505 | 0.0340796038688054 | 0.03497755836506593 |
| RMSE [cm] | 16.74796858308986 | 18.460661924428766 | 18.70228819290996 |
| MAE [cm] | 3.5319947976670507 | 8.475415898485531 | 8.714188958093319 |
| median absolute error [cm] | 1.8310546875 | 6.865141761884363 | 6.6650390625 |
| p95 absolute error [cm] | 5.1855468749999964 | 16.359955894365637 | 17.20703125 |
| bias [cm] | -0.24937220982142855 | -4.876784721410568 | -5.723860077044931 |

train-mean定数比でStudentはMSEを17.6943%、RMSEを9.2775%、MAEを58.3266%、median absolute
errorを73.3282%、p95 absolute errorを68.3034%削減した。train-median定数比ではMSEを
19.8073%、MAEを59.4685%削減した。

## 教師値とoutlierに関する事後診断

validation predictionをframe順に調べると、最大2件は近傍から孤立したtargetだった。

| frame | target [m] | prediction [m] | absolute error [m] | 近傍target |
|---:|---:|---:|---:|---|
| 119 | 2.587890625 | 0.265625 | 2.322265625 | frame 118: 0.217285、frame 120: 0.276855 |
| 30 | 0.9990234375 | 0.26953125 | 0.7294921875 | frame 31: 0.222168 |

frame 119だけでvalidation squared-error総和の**88.60%**、この2件で**97.34%**を占める。
参考として最大1件を除くとn=216でRMSE 5.6675 cm / MAE 2.4732 cm、2件を除くとn=215で
RMSE 2.7420 cm / MAE 2.1454 cmとなる。ただしこれはthresholdを事後選択したdiagnosticであり、正式指標や
baseline比較の代用にはしない。ground truthがないため、実際の高速な前後移動かDepth Proの単一pixel
spikeかを確定できない。

その他の評価上の制約は次のとおりである。

- Depth Pro targetにconfidence、ROI median、hand mask、temporal filter、outlier clippingを使っていない。
- validation sequenceは以前のdepth model比較にも使われ、今回もcheckpoint選択に使ったため未使用testではない。
- 3動画だけで、被験者・撮影session・背景・deviceが独立であることを保証するmetadataはない。
- HFlipはviewを倍増するがtargetの独立観測を増やさず、train loss上は各identityを2回重み付けする。
- 最終的な実距離性能には、教師から独立した既知距離または計測器付きの未使用test sequenceが必要である。

## nohup実行記録

成功runのnohup部分は次のとおりである。

```bash
nohup env HF_HUB_OFFLINE=1 uv run --locked python -u scripts/train_student_transformer.py \
  --dataset-manifest outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json \
  --expected-dataset-manifest-sha256 45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc \
  --output-dir outputs/phase8_student_vit_landmarks_xy \
  --landmarks 5,6,7,8 \
  --image-encoder vit_small_patch16_224.dino \
  --epochs 20 \
  --batch-size 32 \
  --encoder-learning-rate 1e-5 \
  --head-learning-rate 1e-4 \
  --weight-decay 0.05 \
  --warmup-fraction 0.05 \
  --gradient-clip-norm 1.0 \
  --early-stopping-patience 6 \
  --precision bfloat16 \
  --device cuda:0 \
  --num-workers 4 \
  > outputs/phase8_student_vit_landmarks_xy.nohup.log 2>&1 &
```

外側の監視shellがPIDを`outputs/phase8_student_vit_landmarks_xy.pid`へ記録して`wait`し、PID
**980158**、exit status **0**だった。logは
`outputs/phase8_student_vit_landmarks_xy.nohup.log`で、20 epochと最終JSON出力を保持する。

初回attempt（PID **978742**）はepoch 1開始前、train image preload開始後にtracebackを残さず終了した。
`outputs/phase8_student_vit_landmarks_xy.attempt1.log`と`.attempt1.pid`を監査用に保存している。
cgroupの`memory.events`では`oom=0`、`oom_kill=0`であり、原因は断定できない。同一nohup commandを
監視shellが`wait`する形で再起動した上記runは正常終了した。初回attemptを成功runの性能値へ混在させていない。

## 成果物とhash

| artifact | SHA-256 |
|---|---|
| `run_manifest.json` | `607faa5b5e25a9743f1b7bcc18d2ff6d0c51cb20441ffd0d542b028332ae66f8` |
| `best_checkpoint.pt` | `48094b82e5ee8072ae76c6c4f05f817481539c215833f8a0ccf8015cd58d8f8d` |
| `last_checkpoint.pt` | `f3c53031503b953dbcfec845331b0524535d5e9afeb89bfec5a95ce742e60fd8` |
| `history.jsonl` | `432bce9803566c7e25d06be4a3222e65648f76403e14e044dce41b92b4ed30bd` |
| `validation_predictions.csv` | `284d612fc4241384f953350b4ad92294d75d71f8335a06b1b2a2c75b9f84a3dc` |

初期DINO encoder state SHA-256は
`44f5825bbfb299cfabe074f6f4080e5472dac26a27be1b706ec482e9179e130f`、best checkpointのencoder stateは
`b1e8828a04dcc0410d89a0803b0a4701597aa33ddcade04e289d50b34016e177`であり、fine-tuneで変化した。
実装、`pyproject.toml`、`uv.lock`、dataset artifactのhashは`run_manifest.json`に記録している。

## 再現時の注意

上のcommandは実行時のものだが、trainerは非empty output directoryを拒否する。現在の成果物を保持したまま
再実行する場合は、`--output-dir`、nohup log、PIDの各pathを同じ新規run名へそろえて変更すること。
DINO weightがローカルcacheにない環境では、最初にonlineでweightを取得してから
`HF_HUB_OFFLINE=1`を使う。dataset manifest hashが一致しない場合は学習を開始せず、まずdataset provenanceを
確認すること。
