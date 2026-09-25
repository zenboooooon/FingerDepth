# Depth Pro教師データ生成 — `finger_movement_2030`

## 結論

Phase 3/4をスキップし、Phase 5〜7を実装した。Depth Pro `approx_focal`を教師、MediaPipe
landmark 8を単一pixel targetとして、661 frameから604 sampleのpseudo-label datasetを生成した。
全604 sampleは既存のDepth Pro比較実行とpixel座標・深度がbit-exactで一致した
（最大絶対差`0 m`）。

`2.90 cm`は焦点距離ではない。既存606件に対する20–30 cm range-band violation平均
`0.0290375058 m`であり、教師条件を選ぶための近似的な指標である。per-frame ground truthに
対する誤差でもない。今回の604件だけで同じ指標を再計算すると`0.0291336565 m`となる。

## 構成

- Teacher: Depth Pro、`depth_pro__approx_focal`
- Depth target: `INDEX_FINGER_TIP`（landmark 8）の単一pixel
- Feature landmarks既定値: 5/6/7/8
  - `INDEX_FINGER_MCP`
  - `INDEX_FINGER_PIP`
  - `INDEX_FINGER_DIP`
  - `INDEX_FINGER_TIP`
- Phase 3 ROI: 未適用
- Phase 4 temporal filtering: 未適用
- label clipping / scale fitting: 未適用
- split: `finger_movement_2030` sequence全体を`train`

`--landmarks`では0〜20のindex、公式名、または`all`を任意順で指定できる。feature指定を
変更しても、現時点の教師targetはlandmark 8のままである。

## Camera modelとPhase 5

撮影情報36 mm相当、表示frame 1080×1920から対角画角近似のcentered pinhole Kを使用した。
これは校正済みKではない。

| parameter | value |
| --- | ---: |
| `fx`, `fy` | 1832.9295592659223 px |
| `cx` | 539.5 px |
| `cy` | 959.5 px |

各sampleに対し、Depth Proの光軸方向深度`Z`を用いて次を保存した。

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = z_teacher_m
```

camera座標規約は`x-right, y-down, z-forward`である。

## Dataset件数

| item | count |
| --- | ---: |
| source frames | 661 |
| hand detected | 606 |
| accepted samples | 604 |
| rejected frames | 57 |
| teacher inference failure | 0 |

拒否内訳はframe 0〜54の`no_hand` 55件と、frame 420/421の
`feature_landmark_out_of_frame:5` 2件である。後者ではlandmark 8自体は有効だが、既定featureの
MCP（landmark 5）が画面右外に出たため、欠損featureを持つ学習sampleにしなかった。旧処理の
landmark 8座標とはhand検出606件すべてで一致している。

## 教師深度の実測値

| statistic | value |
| --- | ---: |
| min | 0.216553 m |
| p05 | 0.244543 m |
| median | 0.292847 m |
| p95 | 0.407922 m |
| max | 0.429443 m |
| 20–30 cm帯内 | 323/604（53.48%） |
| range-band violation mean | 0.029134 m |
| range-band violation RMSE | 0.049083 m |

20–30 cmは動画全体の概算範囲でありper-frame真値ではないため、帯外値を削除・clipしていない。
Studentを学習した場合も、teacher一致は実距離精度の証明にはならない。

Depth Proの推論時間はCUDA上でmedian `114.91 ms/frame`、mean `115.00 ms/frame`、
p95 `115.10 ms/frame`だった。前処理、MediaPipe、I/Oはこの値に含まれない。

## Phase 6の軌跡

604点をCSV/PLY/3-view PNGへ出力した。

| axis | min | max |
| --- | ---: | ---: |
| X | -0.065707 m | 0.058081 m |
| Y | -0.026081 m | 0.024940 m |
| Z | 0.216553 m | 0.429443 m |

連続区間はframe 55〜419（365点）と422〜660（239点）の2本である。PLYは604 vertex、
602 edgeで、欠損したframe 420/421を跨ぐedgeは作っていない。

## 監査と再現性

- source video SHA-256:
  `1a9bbadaa0af3dde2b651ca18b4619685d0950208d9b45e01e334d58450a6ba3`
- frame-cache manifest SHA-256:
  `654866c63dce9985ff7e9d41f61db94a838f7ee29eb76b242dc8abd277dbff50`
- prepared manifest SHA-256:
  `4dc88eee8415d66437a467a9fe55a98e61e3a9f6e03ce57684e6f1c9412b53d7`
- dataset manifest SHA-256:
  `20a8490c38da962f424fe8ea53e6c11859110fe5b234ec461be64b6db8b2c022`
- BGR pixel sequence SHA-256:
  `e79f86e59d668621ac12c27ad4ac807904d8c3996e297891a4aeb46d6b478909`
- Depth Pro checkpoint SHA-256:
  `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`

661 prepared PNGと604 dataset PNGのfile/BGR hashを全件照合した。前者は監査済みframe cache、
後者はprepared PNGへのhardlinkであり、再encodeしていない。画素hash列の符号化は既存比較と
同じ「lowercase hexadecimal digestをASCII連結してSHA-256」に統一した。教師選定レポートは
source video SHA、frame-cache manifest SHA、pixel sequence SHAをprepared入力と照合済みである。
全JSONL/CSV/split/PLY/PNGと、実装・uv lockfileのSHAはdataset manifestに記録した。

検証は116 tests、Ruff、root/Depth Pro両`uv.lock` checkを通過した。

## 成果物

- prepared manifest: `outputs/depth_pro_teacher_2030/prepared/manifest.json`
- dataset manifest: `outputs/depth_pro_teacher_2030/dataset/dataset_manifest.json`
- full samples: `outputs/depth_pro_teacher_2030/dataset/samples.jsonl`
- compact target table: `outputs/depth_pro_teacher_2030/dataset/targets.csv`
- explicit split: `outputs/depth_pro_teacher_2030/dataset/splits/train.txt`
- trajectory CSV/PLY: `outputs/depth_pro_teacher_2030/dataset/trajectories/`
- trajectory plot: `outputs/depth_pro_teacher_2030/dataset/visualizations/finger_movement_2030_views.png`

## Phase 8への制約

この個別成果物でSingle-frame Student用の入力・target schemaは準備できたが、収録内容は
単一sequenceの604 sampleだけである。frameをランダムにtrain/validationへ分けるとtemporal
leakageになる。
したがってPhase 8の学習・汎化評価は、別撮影session、可能なら別背景・照明・距離・被験者の
sequenceを収集し、sequence単位のvalidation/testを確保してから行う。独立した実距離GTもないため、
現時点で評価できるのはteacherへのdistillation fidelityまでである。

後続で3動画をsequence単位に分け、train-only HFlipを適用した統合成果物を作成した。確定件数と
監査結果は[3動画Student用統合データセット結果](depth_pro_student_dataset_three_sequences.md)を参照。
