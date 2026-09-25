# Phase 8 同一動画・後半7% validation 学習結果

実行日: 2026-09-25 (Asia/Tokyo)

## 結論

3本の動画それぞれについて、raw動画時間軸の後半7%をvalidation、それ以前をtrainとし、
教師スパイクを除外したPhase 8 Single-frame Student Transformerを再学習した。

best checkpointはepoch 6で、validation 234件に対してMAE **0.4798 cm**、
RMSE **0.6367 cm**、bias **+0.0548 cm**、Pearson r **0.9783**だった。
train全体平均を出すだけのbaselineに対し、MAEを86.1%、RMSEを83.7%削減した。
この条件ではRGBとlandmark XYからDepth Pro疑似ラベルの絶対Z変化を学習できている。

ただし、validationの86.3%は22–28 cmに集中し、同じ動画の直前フレームがtrainに含まれる。
この結果は「既知撮影系列の時間的に近い後半に対する教師再現性能」であり、未知動画への汎化性能や、
実測値に対する物理的な絶対深度精度を示すものではない。

確定run manifestは
`outputs/phase8_student_vit_landmarks_xy_spike_filtered_chronological_tail7/run_manifest.json`
（SHA-256 `b35ce59b73ab01b1e6789f2a203bcccad59cd5eef081be2181865d695c195e3a`）、
拡張評価は
`validation_analysis_filtered.json`
（SHA-256 `ad1b5cf2942fb904cadd114c1ace7e5e55ea759830991d7279b75126f81ef0f9`）
に保存した。

## 画像解像度の扱い

保存データの事前縮小は行っていない。

| 動画 | 保存PNG | raw frame数 |
|---|---:|---:|
| `finger_movement` | 1440×1920 | 267 |
| `finger_movement_2030` | 1080×1920 | 661 |
| `finger_movement_3` | 1080×1920 | 3,206 |

identity 3,354枚は教師データのPNGと同じ解像度・画素hashで、hardlinkされている。
HFlipも同じ解像度のlossless PNGである。

一方、学習・validationではViTへ入れる直前に全画像を **224×224へbicubicで直接リサイズ**した。
cropやletterboxは使っていない。そのため、1440×1920画像は横方向が相対1.333倍、
1080×1920画像は相対1.778倍に伸びる。landmarkはnormalized XYなので位置対応は保たれるが、
RGB内の手形状・見かけのスケール・遠近手掛かりは変形する。

旧sequence-held-out条件では、train 2本が1080×1920、validationの
`finger_movement`が1440×1920だったため、この異なる変形量もdomain shiftの一因になり得る。
今回の同一動画内分割では各解像度がtrainとvalidationの両方に存在するので、その影響は大幅に弱まる。

## 分割とデータ監査

「後半7%」は、受理済みサンプル数ではなく、手検出・教師ラベル棄却前のraw動画frame総数を分母とした。

```text
validation_frame_count = ceil(source_frames_total * 0.07)
validation_start = source_frames_total - validation_frame_count
validation = frame_index >= validation_start
```

| 動画 | raw frames | valid開始 | train identity | train HFlip | valid identity |
|---|---:|---:|---:|---:|---:|
| `finger_movement` | 267 | 248 | 198 | 198 | 19 |
| `finger_movement_2030` | 661 | 614 | 557 | 557 | 47 |
| `finger_movement_3` | 3,206 | 2,981 | 2,365 | 2,365 | 168 |
| **合計** | **4,134** | — | **3,120** | **3,120** | **234** |

`finger_movement_3`はvalidation領域がframe 2,981から始まるが、最初に受理されたvalid sampleは
frame 3,038である。splitは教師深度を見る前に割り当て、HFlipはtrainだけへ適用した。

指定済みHampel型規則で8 identity spikeを除外した。すべてtrain側であり、HFlipを含む16 viewを除外した。

- `finger_movement`: frame 30, 119
- `finger_movement_3`: frame 509, 696, 1295, 1296, 2506, 2782
- filtered train: 6,224 views（3,112 identity相当）
- filtered validation: 234 identity、除外0

全6,474 PNGを復号して監査し、PNG hash不一致0、BGR画素hash不一致0、
train–validationの画素hash重複0、source sample ID重複0だった。
HFlip 3,120対の画像・landmark・指先座標・教師深度変換も全件一致した。

## モデルと学習条件

- RGB: DINO `vit_small_patch16_224.dino`のimage tokens
- landmark: INDEX_FINGER MCP/PIP/DIP/TIP（5/6/7/8）のnormalized XY
- MediaPipe relative `z`: 漏洩候補として不使用
- landmark type token: 21部位分を保持し、入力した5/6/7/8だけ学習
- fusion: 2層Transformer、6 heads、MLP ratio 4.0、dropout 0.1
- 出力: 光軸方向Depthのraw scalar [m]
- loss: 非標準化Depthに対するMSE [m²]
- trainable parameters: 25,443,073
- optimizer: AdamW、encoder LR `1e-5`、new-module LR `1e-4`
- batch size 32、weight decay 0.05、warmup 5%、gradient clip 1.0
- precision: bfloat16、device: NVIDIA GeForce RTX 5090
- seed: 20260925
- checkpoint選択: filtered validation MSE
- early stopping: patience 6

最大20 epochの設定で、epoch 6が最良となり、epoch 12で早期停止した。
学習部の所要時間は44.07秒だった。最終実行は`nohup`で起動し、
PID 1094695、終了コード0を記録した。

## 全体結果

| 指標 | train at best (n=6,224) | validation (n=234) |
|---|---:|---:|
| MSE [m²] | 0.0000135070 | 0.0000405374 |
| RMSE [cm] | 0.3675 | **0.6367** |
| MAE [cm] | 0.2875 | **0.4798** |
| median absolute error [cm] | 0.2319 | 0.3601 |
| p95 absolute error [cm] | 0.7324 | 1.3757 |
| bias [cm] | +0.0239 | +0.0548 |
| max absolute error [cm] | 1.8799 | 2.5879 |
| Pearson r | 0.9980 | 0.9783 |
| 負の予測 | 0 | 0 |

validationの連続231 frame pairでは、予測Depth差と教師Depth差のMAEは **0.3535 cm**、
RMSEは0.4522 cm、相関は0.4910だった。絶対レベルはよく追従した一方、
細かなframe間変化の再現は絶対Depthほど強くない。

## 定数baselineとの比較

| 方法 | RMSE [cm] | MAE [cm] | bias [cm] |
|---|---:|---:|---:|
| 今回のStudent | **0.6367** | **0.4798** | +0.0548 |
| filtered train平均を常に出力 | 3.9028 | 3.4548 | +2.4278 |
| filtered train中央値を常に出力 | 3.5396 | 2.9795 | +1.7863 |

train平均baselineに対して、RMSEは83.69%、MAEは86.11%低下した。
倍率ではそれぞれ6.13倍、7.20倍良い。したがって、全体平均だけを覚えた結果ではない。

## 動画別結果

| 動画 | n | 教師範囲 [m] | RMSE [cm] | MAE [cm] | bias [cm] | r |
|---|---:|---:|---:|---:|---:|---:|
| `finger_movement` | 19 | 0.2244–0.3970 | 1.0937 | 0.8249 | -0.7838 | 0.9940 |
| `finger_movement_2030` | 47 | 0.2625–0.3328 | 0.7102 | 0.5215 | +0.4176 | 0.9753 |
| `finger_movement_3` | 168 | 0.2180–0.2703 | 0.5369 | 0.4291 | +0.0481 | 0.8957 |
| **動画macro平均** | 3動画 | — | **0.7803** | **0.5919** | **-0.1060** | 0.9550 |

micro MAE 0.4798 cmよりmacro MAE 0.5919 cmが悪いのは、最も低誤差だった
`finger_movement_3`が168/234件、71.8%を占めるためである。

## 深度帯と較正

validation教師値は21.80–39.70 cm、平均25.51 cm、標準偏差3.06 cmである。
202/234件（86.3%）が22–28 cmに集中し、40 cm以上は0件だった。

| 教師Depth | n | RMSE [cm] | MAE [cm] | bias [cm] |
|---|---:|---:|---:|---:|
| <22 cm | 3 | 0.8167 | 0.8097 | +0.8097 |
| 22–28 cm | 202 | 0.5034 | 0.3998 | +0.0459 |
| 28–34 cm | 22 | 1.0207 | 0.8589 | +0.5149 |
| 34–40 cm | 7 | 1.6367 | 1.4579 | -1.4579 |
| >=40 cm | 0 | — | — | — |

診断用の線形回帰は次だった。予測値への事後較正は適用していない。

```text
prediction_m = 0.942764 * target_m + 0.015148
R² = 0.957125
```

予測range / 教師rangeは0.824で、遠い側を含むrangeが圧縮されている。
22–28 cmでは高精度だが、34–40 cmは7件しかなく、MAEも1.46 cmまで悪化した。

## 旧sequence-held-out条件との関係

旧runは`finger_movement_2030`と`finger_movement_3`で学習し、
`finger_movement`全体をvalidationとした。旧runのMAEは4.2589 cm、RMSEは5.2588 cmであり、
今回の値はそれぞれ8.88倍、8.26倍小さい。

ただし評価対象と難易度が違うため、この全体値をモデル改善として直接比較してはいけない。
同じ`finger_movement` frame 248–266の19件だけへ揃えると次になる。

| 条件 | RMSE [cm] | MAE [cm] | bias [cm] |
|---|---:|---:|---:|
| 旧: `finger_movement`完全held-out | 3.9084 | 3.2072 | -2.3926 |
| 今回: 同じ動画の前半93%を学習 | 1.0937 | 0.8249 | -0.7838 |

同一コホートでも、同じ動画の前半を学習できるとRMSEは3.57倍、MAEは3.89倍良くなった。
これは撮影系列固有の外観・カメラ・Depthスケール校正を利用できたことと整合するが、
未知動画への汎化を示さない。

また、時間的な境界は非常に近い。

- `finger_movement`: train frame 247 → valid frame 248
- `finger_movement_2030`: train frame 613 → valid frame 614
- `finger_movement_3`: 最後の受理train frame 2962 → 最初の受理valid frame 3038（約1.27秒）

validationをcheckpoint選択にも用いたため、これは独立test setの指標でもない。

## 解釈

今回の実験から、次は支持される。

1. RGBとlandmark XYだけでも、既知系列内ではDepth Pro教師の絶対Z変化を定数baselineより大幅に良く再現できる。
2. 選択landmarkのrelative `z`を使わずにMAE 0.48 cmへ到達したため、relative `z`漏洩に依存した結果ではない。
3. スパイク除外後も、単一frameモデルは主分布である22–28 cmを追従できる。

一方、次はまだ支持されない。

1. 未知動画・人物・背景・照明・カメラへの汎化。
2. 40 cm以上を含む広い距離域での精度。
3. 実測ground truthに対するmetric accuracy。評価対象はDepth Pro疑似ラベルである。
4. 時間的に離れた同一動画区間への汎化。
5. frame-to-frameの細かな運動を高忠実度で再現できること。

## 次の比較実験

優先順位は次のとおり。

1. **縦横比維持入力**: 長辺を224へ縮小して左右paddingし、landmark XYも同じpadding座標へ変換する。
   現行の直接224×224との比較は同じsplit・seedで行う。
2. **時間gap付き評価**: train終端とvalidation開始の間に未使用bufferを設け、1-frame隣接を避ける。
3. **独立test動画**: validationとは別に、checkpoint選択へ使わない動画を確保する。
4. **距離分布の拡張**: 20–50 cmを均等に含む撮影を増やし、距離binごとの件数を確保する。
5. **複数seed**: split効果と最適化ばらつきを分離する。

## 実装・再現性

追加・強化した内容は次である。

- raw動画時間軸の後半fractionでsplitするdataset builder
- binary float境界ずれを避ける厳密なceil計算
- source frame上限、accepted identity件数、valid先頭・末尾frameの再検証
- train全viewとvalidationの画素hash交差監査
- 動画別・macro・距離bin・較正・gap-safe時系列評価CLI
- 予測CSV hash、行数、error列、sample ID membership、時系列境界の再検証

全回帰テストは成功し、`ruff check .`も成功した。変更対象ファイルはRuff format済みである。
生成済みdatasetとrun artifactはhash固定のため、学習後の堅牢化修正で書き換えていない。
dataset manifest内の旧表現`augmented_views_inherit_source_split: true`は実sample splitを変更しないが、
将来生成分では元Phase 7 splitを上書きすることが分かる表現へ修正した。

## 主な成果物

| artifact | SHA-256 |
|---|---|
| dataset manifest | `795112948ba8e43d4913316369940320df23e63f7e06d333bafa892e2192f8f4` |
| run manifest | `b35ce59b73ab01b1e6789f2a203bcccad59cd5eef081be2181865d695c195e3a` |
| best checkpoint | `e2e8941d2187e20dc716580fbbafb294cc9809db54b467a2d09fb52b39492252` |
| validation predictions | `79541f0ba2730d11b1339832e5c2e2f04f4359b6aacd5a54db90ad4c0d21d14f` |
| extended validation analysis | `ad1b5cf2942fb904cadd114c1ace7e5e55ea759830991d7279b75126f81ef0f9` |
