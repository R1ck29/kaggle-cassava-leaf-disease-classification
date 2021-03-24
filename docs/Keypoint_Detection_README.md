# **Pytorch Keypoint Detection**

## **Directory Structure**
Keypoint推定に関わるsrcコード以下ディレクトリは下記。
```
.
└─── src
    ├── data
    │   ├── generator
    │   │   └── keypoint
    │   └── transforms
    ├── models
    │   ├── backbone
    │   │   └── pytorch
    │   ├── loss
    │   │   └── detection
    │   ├── modeling
    │   │   └── keypoint
    │   ├── predictor
    │   │   └── keypoint
    │   └── utils
    │       └── keypoint
    ├── tools
    ├── utils
    │   └── pytorch
    └── visualize
```

## **Models**
Higher HRNet

[Arxiv](https://arxiv.org/abs/1908.10357)

[GitHub](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
- 学習済みの重み(ImageNet, COCO, MPII)のダウンロード
1. 学習済みの重みファイル(pth拡張子)を、[Google Driveのページ](https://drive.google.com/drive/folders/1jBpiVLULgZEWTzrnsho2aDQP4QBc1LAi)からダウンロードしてください
2. **./models/pretrained/** 配下に格納してください


## **Data Preparation**

1. CVATで"Points"を選択し、アノテーションを行う。

    複数対象についてアノテーションする場合は、Groupingを行うこと。("g"を押して、グループを選択、その後もう一度"g"を押す)
<div align="center">
  <img src="cvat_keypoint.png" width="600"/>
</div>
    
2. アノテーションファイルをdumpし、data/{DATA_ID}/raw/annotation以下に配置。また、画像ファイルについてはdata/{DATA_ID}/raw/img以下に配置。
```yaml
# configs/data/keypoint.yaml
DATA:
  DATA_ID: my_data1          #データID
  CSV_PATH:                  #train/valデータの情報を含んだCSVのpath
  N_FOLD: 1                  #FOLD数
  FORMAT: mpii               #データフォーマット、mpii, cocoについては現状カバー
  
  BASE_SIZE: 256.0           #ヒートマップのサイズ
  BASE_SIGMA: 2.0            #ガウシアンにおけるσ(GT用ヒートマップ生成時のパラメータ)
  SCALE_AWARE_SIGMA: false   #σを固定値でなく、対象のスケールに合わせて変える場合
  SCALE_TYPE: short          #スケールの手法
  SIGMA: 2                   #現状未使用
  WITH_CENTER: false         #予測keypointに、対象の中心を追加するかどうか
  
  INT_SIGMA: false           #現状未使用
  MAX_NUM_PEOPLE: 30         #最大の対象の数
```

3. DataFrame形式でアノテーションファイルの保存を行う。　(00_make_dataset.ipynb)
3. train/valのデータ分割を行い、data/{DATA_ID}/split/{CSV_PATH}に保存する。　(00_make_dataset.ipynb)
4. Generatorを作成する。Generatorの作成例に関しては01_check_dataset.ipynbを確認のこと。
5. Augmentationのパラメータを設定し、データを確認。(01_check_dataset.ipynb)

```yaml
# configs/augmentation/custom.yaml
AUGMENTATION:
  FRAMEWORK: custom         #変更不要(albumentationを使用したい場合は"albumentation"に書き換える)
  FLIP: 0.5                 #Flipの確率
  MAX_ROTATION: 30          #Rotateの最大角
  MAX_SCALE: 1.5            #Crop/Expandさせる際の最大倍率
  MAX_TRANSLATE: 40         #Translateの最大ピクセル数
  MIN_SCALE: 0.75           #Crop/Expandさせる際の最小倍率
```

## **Training**

Config設定について
1. configを設定。(下記は主要なconfigの設定例、その他configの設定については[Config.md](Config.md)を参照のこと)
```yaml
# configs/train.yaml
defaults:
   - data: keypoint
   - augmentation: custom
   - model: keypoint/hrnet
   - loss: keypoint
   - train: train
   - pytorch_lightning: trainer/lightning_trainer
   - optimizer: adam
   - scheduler: multi_step
   - callback: pytorch_lightning
   - private: default
   - system: system

FRAMEWORK: pytorch
TASK: keypoint
MODEL_ID: model_id         #任意のIDを設定

hydra:
  run:
    dir: ./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
  sweep:
    dir: ./models/
    subdir: ${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
```
```yaml
# configs/model/keypoint/hrnet.yaml
MODEL:
  MODEL_NAME: kpt_higher_hrnet                                    #モデル名(現状未使用)
  BACKBONE: hrnet                                                 #bacobone(現状はhrnetのみ)
  WEIGHT_PATH: pretrained/imagenet/hrnet_w32-imagenet.pth         #学習済み重みのpath
  WO_HEAD: false                                                  #重みのload時にHead部分を省くかどうか
  INPUT_SIZE: 512                                                 #入力画像サイズ
  OUTPUT_SIZE:                                                    #出力画像サイズ
  - 128
  - 256
  NUM_JOINTS: 16                                                  #Keypoint数
  TAG_PER_JOINT: true                                             #keypointのtag付けを行うかどうか
```
```yaml
# configs/system/system.yaml
SYSTEM:
  GPUS:                                                 #使用するGPUのID
  - 1
  - 2
```
```yaml
# configs/train/train.yaml
TRAIN:
  EPOCHS: 100                                          #Epoch数
  LR: 0.02                                             #学習率
  BATCH_SIZE: 4                                        #バッチサイズ
  MONITOR: val_loss                                    #重み保存時のモニタリング指標
  MODE: "min"                                          #重み保存の条件(val_lossが最小だった場合に保存。)
```

2. 下記コマンドを実行。結果はmodels/{MODEL_ID}/{%Y%m%d_%H_%M_%S}に保存される。
```
python src/tools/train.py
```
* BaseModelディレクトリ直下で実行しないと結果の出力先が変わるので注意。

## **Evaluation**

1. configを設定。

```yaml
# configs/test.yaml
defaults:
  - data: keypoint
  - test: keypoint
  - system: system

FRAMEWORK: pytorch
MODEL_PATH: /path/to/model/                       #テストを行うモデルのディレクトリ。models/{MODEL_ID}/{%Y%m%d_%H_%M_%S}を記入。
TEST_ID: test                                     #テストID
  
hydra:
  run:
    dir: ${MODEL_PATH}/result/${TEST_ID}
  sweep:
    dir: ${MODEL_PATH}/result/
    subdir: ${TEST_ID}
```
```yaml
# configs/test/keypoint.yaml
TEST:
  BATCH_SIZE: 8                                       #テスト時のバッチサイズ
  DETECTION_THRESHOLD: 0.1                            #ヒートマップの予測を採用するconfidenceの閾値
  PCK_FACTOR: 0.5                                     #PCKhを計算する場合の、headの大きさに対する係数
  PCK_THRESHOLD: 100                                  #PCK算出時に固定値(ピクセル間のユークリッド距離)を使う場合の閾値
  OKS_FACTOR: 0.1                                     #OKSを算出する場合の係数。COCOではKeypoint毎に設定しているが、簡単のため一律で設定。
  OKS_THRESHOLD: 0.5                                  #OKSを正解として扱う閾値。この場合予測とGTでOKS0.5以上なら正解として扱う。
```
2. 下記コマンドを実行。結果はmodel/{MODEL_PATH}/result/{TEST_ID}に保存される。
```
python src/tools/predict.py
```
