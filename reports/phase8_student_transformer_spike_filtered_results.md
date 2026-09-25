# Phase 8 教師スパイク除外学習結果

実行日: 2026-09-25 (Asia/Tokyo)

## 結論

trainとvalidationの両方から、時系列の局所近傍に対して孤立したDepth Pro pseudo-labelを除外し、
Phase 8 Single-frame Student Transformerを同じ設定で再学習した。フィルタは8個のidentity観測を
スパイクと判定した。trainのHFlipを含めると除外viewは14件で、学習対象はtrain 6,274件から
6,262件、checkpoint選択対象はvalidation 217件から215件になった。

除外後validation 215件では、best checkpoint（epoch 3）はMAE **4.2589 cm**、RMSE
**5.2588 cm**だった。定数baselineには勝ったが、旧Phase 8 checkpointを同じ215件だけで
再集計したMAE **2.1454 cm**、RMSE **2.7420 cm**より悪い。MSEは旧checkpointの
**3.68倍**である。一方、連続フレーム間のDepth差のMAEは **0.7089 cm**から
**0.6323 cm**へ10.8%改善した。

したがって、明らかな教師推定失敗を除外するデータ方針は維持するが、今回のcheckpointを旧baselineの
改善版としては採用しない。除外後データではtrain誤差だけが急速に低下し、予測範囲が狭くなって遠い側を
過小推定した。ラベルスパイクを戻すのではなく、full ViT fine-tuningの学習率・凍結範囲・正則化・
early stoppingを除外後の分布に合わせて再調整する必要がある。

確定run manifestは
`outputs/phase8_student_vit_landmarks_xy_spike_filtered/run_manifest.json`、SHA-256は
`a7f17f7c5de2ecdbf5132fa67727fe041598566c7e37afa9e88c033c12b9fe07`である。

## スパイク除外規則

除外判定はofflineのデータ整備として、identity viewだけを対象に、split・source sequence・handごとに
frame順で行った。HFlipは判定用の時系列へ重複投入せず、対応するidentityの判定を継承する。

各候補frameについて、自身を除く前後3 frameの連続区間から局所中央値とMADを計算した。
3近傍以上ある場合に、次を満たす観測を除外した。

```text
absolute_deviation > max(
  0.15 m,
  0.50 * local_median,
  6.0 * 1.4826 * local_MAD
)
```

frame gapが1を超える箇所は別区間とし、境界では利用可能な片側近傍も使う。全candidateを未除外の元系列から
同時判定するため、判定順序には依存しない。この処理にRGB、landmark XY、MediaPipe relative `z`、
モデル予測、handednessは使っていない。元datasetも変更していない。

判定の全3,354 identity観測は
`teacher_spike_filter_decisions.jsonl`、規則・集計・除外根拠は`teacher_spike_filter.json`へ保存した。

## 除外された観測

| split | sequence | frame | target [m] | 局所中央値 [m] | 閾値 [m] | 除外view |
|---|---|---:|---:|---:|---:|---:|
| train | `finger_movement_3` | 509 | 0.833984 | 0.374268 | 0.187134 | identity + HFlip |
| train | `finger_movement_3` | 696 | 2.076172 | 0.371582 | 0.185791 | identity + HFlip |
| train | `finger_movement_3` | 1295 | 1.329102 | 0.383667 | 0.191833 | identity + HFlip |
| train | `finger_movement_3` | 1296 | 2.324219 | 0.381226 | 0.190613 | identity + HFlip |
| train | `finger_movement_3` | 2506 | 0.732910 | 0.228577 | 0.150000 | identity + HFlip |
| train | `finger_movement_3` | 2782 | 0.590332 | 0.357788 | 0.178894 | identity + HFlip |
| validation | `finger_movement` | 30 | 0.999023 | 0.231445 | 0.150000 | identity |
| validation | `finger_movement` | 119 | 2.587891 | 0.237305 | 0.150000 | identity |

`finger_movement_2030`では除外はなかった。除外後のidentity target最大値はtrain
0.429443 m、validation 0.462891 mとなった。0.128540 m付近の滑らかに持続する谷などは、絶対距離が
小さいことだけを理由に除外していない。

| split | 除外前view | identity除外 | HFlip除外 | 除外後view |
|---|---:|---:|---:|---:|
| train | 6,274 | 6 | 6 | 6,262 |
| validation | 217 | 2 | 0 | 215 |
| **合計** | **6,491** | **8** | **6** | **6,477** |

## 学習条件

モデル入力・architectureは旧Phase 8 runと同じである。

- RGBをDINO `vit_small_patch16_224.dino`へ入力し、ViT image tokenを使用
- landmark 5/6/7/8の画像平面XYだけを埋め込み、各部位のtype tokenを加算
- learned depth query、image tokens、landmark tokensを2層Transformerで融合
- 出力は光軸方向Depthのraw scalar、lossは標準化していないmetre単位targetのMSE
- relative `z`、教師Depth、camera XYZ、intrinsics、handednessは入力しない
- 21部位のtype tokenを保持し、選択した5/6/7/8だけを更新

主なoptimizer設定も旧runから変更していない。

| 項目 | 値 |
|---|---|
| 最大epoch / batch size | 20 / 32 |
| optimizer | AdamW |
| encoder LR / new-module LR | `1e-5` / `1e-4` |
| weight decay | 0.05 |
| scheduler | step単位linear warmup 5% + cosine decay |
| gradient clipping | global norm 1.0 |
| early stopping | filtered validation MSE、patience 6 |
| precision / device | bfloat16 / `cuda:0` |
| seed | 20260925 |

epoch 3が最良で、その後6 epoch改善しなかったためepoch 9で正常に早期停止した。学習時間は
62.595秒だった。runtimeはPython 3.12.13、PyTorch 2.14.0+cu130、timm 1.0.30、CUDA 13.0、
NVIDIA GeForce RTX 5090である。

## 除外後コホートでの結果

主要指標は、除外後のtrainとvalidationだけで計算した。

| 指標 | train (n=6,262) | validation (n=215) |
|---|---:|---:|
| MSE [m²] | 0.0000304583 | 0.0027654762 |
| RMSE [cm] | 0.551891 | 5.258780 |
| MAE [cm] | 0.428603 | 4.258891 |
| median absolute error [cm] | 0.341797 | 3.442383 |
| p95 absolute error [cm] | 1.123047 | 10.451660 |
| bias [cm] | +0.081929 | -2.193916 |
| max absolute error [cm] | 2.270508 | 12.890625 |
| Pearson r | 0.996075 | 0.952633 |
| consecutive-frame delta MAE [cm] | 0.355735 | 0.632290 |

filtered validationのconstant train-mean baselineはMSE 0.00728483 m²、RMSE 8.5351 cm、
MAE 7.1944 cmだった。今回のStudentはこのbaselineよりMSEを62.0%削減した。ただしtrainとvalidationの
差が大きく、3本だけのsequence splitでは未知動画への一般化を示せない。

## 旧Phase 8 checkpointとの公平比較

旧runのvalidation予測217件から、今回の規則で除外したframe 30と119だけを取り除き、両モデルを同一の
215 sampleで比較した。これにより、旧runの全217件指標を、外れ値を除いた新runの指標と直接比較する誤りを
避けた。

| validation指標（同じ215件） | 旧baseline | スパイク除外後 | 変化 |
|---|---:|---:|---:|
| MSE [m²] | 0.000751875 | 0.002765476 | +267.8%（3.68倍） |
| RMSE [cm] | 2.742034 | 5.258780 | +91.8% |
| MAE [cm] | 2.145428 | 4.258891 | +98.5% |
| median absolute error [cm] | 1.831055 | 3.442383 | +88.0% |
| p95 absolute error [cm] | 5.019531 | 10.451660 | +108.2% |
| max absolute error [cm] | 9.863281 | 12.890625 | +30.7% |
| Pearson r | 0.968259 | 0.952633 | -0.0156 |
| consecutive-frame delta MAE [cm] | 0.708872 | 0.632290 | **-10.8%** |

absolute errorは新runが小さいframe 81件、同じframe 3件、大きいframe 131件だった。予測範囲は旧runの
0.22559–0.44922 mに対し、新runは0.22754–0.33984 mへ圧縮された。target範囲は
0.17395–0.46289 mである。

target distance別のMAEも、このrange compressionを示す。

| target範囲 | 旧baseline MAE [cm] | スパイク除外後 MAE [cm] |
|---|---:|---:|
| `< 0.22 m` | 4.090 | 3.934 |
| `0.22–0.28 m` | 2.967 | 1.829 |
| `0.28–0.34 m` | 1.301 | 1.472 |
| `0.34–0.40 m` | 1.159 | 5.736 |
| `0.40–1.00 m` | 1.853 | 9.819 |

近い側では一部改善したが、0.34 m以上を強く過小推定した。train MSEはepoch 1の0.0008288から
epoch 3の0.0000532へ急減し、その後もepoch 9の0.0000155まで下がった一方、validationはepoch 3以降
改善しなかった。これは、2本のtrain sequenceへのfull ViT fine-tuningが除外後の小規模データで
過適合した可能性と整合する。

ただし、この単一seedの比較だけからスパイク除外そのものが性能を悪化させたとは断定できない。旧runは
20 epoch完走、新runはearly stoppingで9 epoch終了しており、20 epochを前提にしたcosine scheduleの
停止時期も同時に変わっている。除外効果、最適化条件、seed変動を分離した追加実験が必要である。

## 除外前validationの事後診断

best checkpointをfiltered validationだけで選択した後、監査目的で除外前217件も評価した。この値は
学習にもcheckpoint選択にも使っていない。

| 指標 | 旧baseline raw 217 | スパイク除外後モデル raw 217 |
|---|---:|---:|
| MSE [m²] | 0.02804945 | 0.03013759 |
| RMSE [cm] | 16.74797 | 17.36018 |
| MAE [cm] | 3.53199 | 5.62913 |
| median absolute error [cm] | 1.83105 | 3.51563 |
| p95 absolute error [cm] | 5.18555 | 10.66895 |
| bias [cm] | -0.24937 | -3.58409 |
| Pearson r | 0.33985 | 0.34228 |
| consecutive-frame delta MAE [cm] | 3.24690 | 3.16230 |

raw MSEとRMSEは2個のpseudo-label spikeに支配されるため、モデル選択値として解釈しない。

## 次の実験

データ方針としてはスパイク除外を維持し、次の順で最適化条件を切り分ける。

1. 同じfiltered cohortでearly stoppingを無効にし、20 epochを完走して停止時期の影響を分離する。
2. 複数seedで旧recipeと比較し、単一runのばらつきを測る。
3. ViTをfreezeする、またはencoder/head LRを下げ、range compressionとsequence overfitを抑える。
4. sequence数を増やし、未使用の実距離test sequenceを別途確保する。

比較時は常に同じfiltered validation IDを使う。validationが1 sequenceだけであること、targetがDepth Proの
単一pixel pseudo-labelで実距離ground truthではないことも維持すべき制約である。

## nohup実行記録

```bash
nohup env HF_HUB_OFFLINE=1 uv run --locked python -u scripts/train_student_transformer.py \
  --dataset-manifest outputs/depth_pro_student_dataset_three_sequences/dataset_manifest.json \
  --expected-dataset-manifest-sha256 45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc \
  --output-dir outputs/phase8_student_vit_landmarks_xy_spike_filtered \
  --landmarks 5,6,7,8 \
  --image-encoder vit_small_patch16_224.dino \
  --epochs 20 --batch-size 32 \
  --encoder-learning-rate 1e-5 --head-learning-rate 1e-4 \
  --weight-decay 0.05 --warmup-fraction 0.05 \
  --gradient-clip-norm 1.0 --early-stopping-patience 6 \
  --precision bfloat16 --device cuda:0 --num-workers 4 \
  --exclude-teacher-spikes \
  --teacher-spike-frame-radius 3 \
  --teacher-spike-max-frame-gap 1 \
  --teacher-spike-min-neighbors 3 \
  --teacher-spike-absolute-floor-m 0.15 \
  --teacher-spike-relative-floor-fraction 0.50 \
  --teacher-spike-mad-multiplier 6.0 \
  --teacher-spike-mad-scale 1.4826 \
  > outputs/phase8_student_vit_landmarks_xy_spike_filtered.nohup.log 2>&1 &
```

監視shellはPID **1025203**を`.pid`へ記録して`wait`し、`.exit`へ終了コード **0**を保存した。

## 成果物とhash

| artifact | SHA-256 |
|---|---|
| `run_manifest.json` | `a7f17f7c5de2ecdbf5132fa67727fe041598566c7e37afa9e88c033c12b9fe07` |
| `best_checkpoint.pt` | `92c753c4bab242870590fe8ab2682e374cd427f87615fb96077a72a4458358c5` |
| `last_checkpoint.pt` | `18ebec5b5829cc4f134e79a1883169d889323064ce94c525958067e026ac7f80` |
| `history.jsonl` | `22cc3bcf3318b2c07be1bdaeea5b6a590297da9c94c514af11cd506b2985d221` |
| `validation_predictions_filtered.csv` | `9fc622b88097196f9363ffa9530336613fd035bd23c6dee7f6bc42aedff6e63b` |
| `validation_predictions_raw.csv` | `8bd08108da6199e26e4d4684143cb3a1806b12917960a3e9a197fbb72dd36a5d` |
| `teacher_spike_filter.json` | `938f37afd8b6a3e76230f926141f27ad62d6a404e15c462689b351b9a267afa5` |
| `teacher_spike_filter_decisions.jsonl` | `072676751e649fe6dc2daefa59dbfcb45cba7b3f3e79640f6047f951b392aa58` |
| `included_train.txt` | `0b81d16e2bd27668c395ca3c7c08c70aa6bb73a8f698b4d87bc6fc9cacc10c48` |
| `included_validation.txt` | `198bb4819da7ec635c334e5c9cb96e1a3103fa0b95e3c621acf9e8da42fa3f9d` |

checkpointにはフィルタ設定、source dataset manifest hash、filter report hash、全判定hash、採用ID hashを
格納した。run manifestには実装・`pyproject.toml`・`uv.lock`・dataset artifactのhashも記録している。
