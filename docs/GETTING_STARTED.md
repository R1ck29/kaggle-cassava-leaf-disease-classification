## Directory Structure
```
.
├── bin                     : bashファイル
├── configs                 : configファイル
├── data                    : データ
│   └── (DATA_ID)           : データID
│       ├── raw             : 生データ(画像、アノテーションファイル)
│       ├── interim         : 中間生成データ(前処理後の画像、特徴量等)
│       └── split           : 学習に使用するCSVファイル置き場
├── models                  : 事前学習/学習済みモデル、学習時ログ、評価結果等の保存場所
├── notebooks                : チュートリアル用のnotebook
├── requirements            : requirementsファイル置き場
├── src                     : ソースファイル
│   ├── data                : データ処理関連(generator, transoform等)
│   ├── models              : モデル全般(backbone, loss, predictor等)
│   ├── tools 　　　         : 学習/評価用実行ファイル
│   ├── utils               : ユーティリティ
│   └── visualize           : 可視化関連
└── tests                   : テストコード
```

##  Configuration
See [Config.md](Config.md)

## Object Detection
See [Object_Detection_README.md](Object_Detection_README.md)

## Segmentation
See [seg_pytorch_README.md](seg_pytorch_README.md)

## Keypoint Detection
See [Keypoint_Detection_README.md](Keypoint_Detection_README.md)
