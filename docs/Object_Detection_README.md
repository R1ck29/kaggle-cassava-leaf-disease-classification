# [DL Pipeline] Object Detection README

## **Directory Structure**

```shell
.
├── bin # 前処理・訓練・推論のコードを実行するシェルスクリプト
├── configs # コンフィグ
│   ├── augmentation
│   ├── callback
│   ├── data
│   ├── experiment
│   ├── loss
│   ├── model
│   │   ├── detection
│   ├── optimizer
│   ├── private
│   ├── pytorch_lightning
│   │   └── trainer
│   ├── scheduler
│   ├── system
│   ├── test
│   └── train
├── data # 画像＆アノテーションデータ
├── docs # README
├── models # 学習済み重みやログ、推論結果
├── notebooks # Jupyter Notebook(チュートリアル等)
├── requirements # タスクごとの必要なパッケージ
├── src # コード
│   ├── data # 訓練・推論用データ作成
│   │   ├── generator
│   │   │   ├── detection
│   │   └── transforms
│   ├── models # モデル関連
│   │   ├── backbone # バックボーン部分
│   │   │   └── pytorch
│   │   ├── loss # 損失関数
│   │   │   └── detection
│   │   ├── modeling # アーキテクチャとPytorch Lightningの訓練コード
│   │   │   └── detection
│   │   ├── predictor # 推論コード
│   │   │   └── detection
│   │   └── utils # モデルに関連する各タスクで使用できる共通コード
│   │       └── detection
│   ├── preprocess # 前処理
│   │   └── detection
│   ├── tools # 訓練・推論実行コード
│   ├── utils # 各タスクで使用できる共通コード
│   │   └── pytorch
│   └── visualize # 可視化
└── tests # テストコード
```

## **Requirements and Installation**

- Python version >= 3.8
- PyTorch version >= 1.6.0
- **To install BaseModel** and develop locally:

  ```shell
  git clone https://github.com/arayabrain/BaseModel.git
  cd BaseModel

  pip install --upgrade pip
  pip install -r ./requirements/requirements_pytorch_detection.txt
  ```

## **Models**

- Faster-RCNN (ResNet50)
  - コンフィグ設定

    モデルに関する設定は以下の通りである。

      ```yaml
      # ./configs/model/detection/fasterrcnn.yaml
      MODEL:
        INPUT_SIZE: 1024 # 入力画像サイズ
        OUTPUT_SIZE: 1024 # 出力画像サイズ
        NUM_CLASSES: 2 # クラス数(背景含む)
        MODEL_NAME: fasterrcnn_resnet50_fpn # モデル名
        BACKBONE:
          CLASS_NAME: src.models.modeling.detection.faster_rcnn.${MODEL.MODEL_NAME} # モデル定義クラスパス
          PARAMS:
            pretrained: True # COCO2017で学習済み重みを使用するか
            num_classes: 91 # 学習済み重みのクラス数
            pretrained_backbone: True # バックボーンにおいてImagenet学習済み重みを使用するか
        HEAD:
          CLASS_NAME: src.models.modeling.detection.faster_rcnn.faster_rcnn.FastRCNNPredictor # HEAD定義クラスパス
      ```

- EfficientDet (D1-7)
  - 学習済みの重み(COCO)のダウンロード
    1. COCOの学習済みの重みファイル(pth拡張子)を、[Google Driveのページ](https://drive.google.com/drive/u/1/folders/1z-0psw36vkhcCjOL3koeYPhVkRblP9f8)からダウンロードしてください
    2. `./models/pretrained_models/efficientdet` 配下に格納してください
  - 人検知モデルの重みのダウンロード
    1. 人検知モデルの重みファイル(pth拡張子)とコンフィグフォルダ(`.hydra`)を、[Google Driveのページ](https://drive.google.com/drive/u/1/folders/1ox0lXCQ17WNXLLAWRZy61YRgrgzt7EES)からダウンロードしてください
    2. チューとリアリルを実行の場合は、`./models/pretrained_models/person ` 配下に格納してください。(`predict.py`で推論を実行する場合は、`./models/effdet_person_ca_v5/20201010_00_40_05`配下に格納してください)

  - コンフィグ設定

      モデルに関する設定は以下の通りである。

      ```yaml
      # ./configs/model/detection/efficientdet.yaml
      MODEL:
        INPUT_SIZE: 512 # 入力画像サイズ
        OUTPUT_SIZE: 1024 # 出力画像サイズ
        NUM_CLASSES: 1 # クラス数
        BASE_NAME: efficientdet_d4 # モデル名
        MODEL_NAME: tf_${MODEL.BASE_NAME} # モデル名(コード内の名称)
        CONFIG:
          CLASS_NAME: src.models.modeling.detection.effdet.get_efficientdet_config # モデル設定定義クラスパス
          PARAMS:
        BACKBONE:
          CLASS_NAME: src.models.modeling.detection.effdet.EfficientDet # モデル定義クラスパス
          PARAMS:
            WEIGHT_PATH: models/pretrained_models/efficientdet/ # COCO学習済み重み定義クラスパス
            PRETRAINED_BACKBONE: False # バックボーンにおいてImagenet学習済み重みを使用するか
        HEAD:
          CLASS_NAME: src.models.modeling.detection.effdet.efficientdet.HeadNet # HEAD定義クラスパス
          PARAMS:
            NUM_CLASSES: ${MODEL.NUM_CLASSES} # クラス数

        TRAIN:
          CLASS_NAME: src.models.modeling.detection.effdet.DetBenchTrain # 訓練用モデル定義クラスパス

        TEST:
          CLASS_NAME: src.models.modeling.detection.effdet.DetBenchPredict # 推論用モデル定義クラスパス
      ```

## **Dataset**

画像とアノテーションが格納されているパスはyamlで管理しているため、設定されているパスと一致している必要がある。

基本的には以下のように配置する。

前処理前のrawデータ(CVATから出力されたxml等)は、以下のパスに格納する

```shell
./data/${DATA.DATA_ID}/raw
```

### **アノテーションデータの前処理**

以下の手順で前処理を実行し、rawデータを加工し、訓練・検証・テストデータを準備する

#### **CVATから出力されたXML形式のアノテーションの場合**

  1. XMLファイルをCSVに変換する

      `ケース1: 全XMLファイルをCSVに変換し、train, testに分割する`

      1. 全xmlファイル群を同じディレクトリ配下に格納する

          ```shell
          # input
          ./data/${DATA.DATA_ID}/raw/cvat
          ```

      2. XMLをCSVに変換する。

          ```shell
          python -m src.preprocess.detection.xml_to_csv +experiment=cvat_test
          ```

          ```shell
          # output
          ./data/${DATA.DATA_ID}/interim/all_cvat.csv
          ```

      `ケース2: マニュアルでtrainとtestのフォルダに予めxmlファイルを分けて、CSV変換を実行する。`

      *手順2の特徴量作成と手順3の訓練とテストデータの分割は、ユーザ側で実施済みとする。

        1. マニュアルでtrainとtestのフォルダに予めxmlファイルを格納する。

            ```shell
            # input
            # 訓練に使用するxmlファイル群
            ./data/${DATA.DATA_ID}/raw/cvat_annotations/train
            # テストに使用するxmlファイル群
            ./data/${DATA.DATA_ID}/raw/cvat_annotations/test
            ```

        2. XMLをCSVに変換する。

            ```shell
            python -m src.preprocess.detection.xml_to_csv +experiment=cvat_test
            ```

            ```shell
            # output
            # 前処理後 訓練用、テスト用のCSVに分かれる
            ./data/${DATA.DATA_ID}/interim/train.csv
            ./data/${DATA.DATA_ID}/interim/test.csv
            ```

  2. 訓練・検証・テストに分割に必要な特徴量を作成する。(特徴量作成は個人でカスタマイズしても良い)

      ```shell
      python -m src.preprocess.detection.buid_features +experiment=cvat_test
      ```

      ```shell
      # input
      /data/${DATA.DATA_ID}/interim/all_cvat.csv
      # output
      # 特徴量の列が追加される
      ./data/${DATA.DATA_ID}/interim/all_features.csv
      ```

  3. train, testに分割する(分割方法は個人でカスタマイズしても良い)

      ```shell
      python -m src.preprocess.detection.train_test_split +experiment=cvat_test
      ```

      ```shell
      # input
      /data/${DATA.DATA_ID}/interim/all_features.csv
      # output
      # 訓練用、テスト用のCSVに分かれる
      ./data/${DATA.DATA_ID}/split/train.csv
      ./data/${DATA.DATA_ID}/split/test.csv
      ```

  4. train.csvから訓練用と検証用データに分割する。(分割方法は個人でカスタマイズしても良い)

      ```shell
      python -m src.preprocess.detection.make_folds +experiment=cvat_test
      ```

      ```shell
      # input
      /data/${DATA.DATA_ID}/split/train.csv
      # output
      # fold列が追加され、どのデータが訓練用か検証用かが決まる
      ./data/${DATA.DATA_ID}/split/split01.csv
      ```

      訓練用のCSV(例：./data/${DATA.DATA_ID}/split/split01.csv)に必須のカラム

      *COCOフォーマットの場合は、`xmax`と`ymax`カラムがなくても訓練可能。

      また、推論結果の出力をCOCOフォーマット(xmin,ymin,width,height)で希望の場合は、設定ファイル(`./configs/data/detection.yaml`)の`DATA.FORMAT`を`coco`に設定して下さい。
      - image_id(str) : 画像名(一意であること)
      - class_id(str): クラスID。1クラスの場合は1
      - xmin(float) : Bottom-left X座標
      - ymin(float) : Bottom-left Y座標
      - xmax(float) : Top-right X座標(pascal_vocフォーマットの場合)
      - ymax(float) : Top-right Y座標(pascal_vocフォーマットの場合)
      - width(float): bboxの幅 (xmax - xmin)
      - height(float) : bboxの高さ (ymax - ymin)
      - area : bboxのサイズ (width * height)
      - fold(int) : 訓練か検証かを判別する番号。5foldの交差検証では、0は検証1-4を訓練とする。
                    Leave-One-Outでは、全体の20％を検証とし0をセットし、残りを1とする。

      特徴量作成で追加されるカラム(*訓練に必須ではない)
        - count(int) : bboxの数
        - source(str) : データの取得元

#### **その他のアノテーションデータの場合**

上記で説明したCVAT以外のアノテーションデータで前処理の動作が確認できているのは、以下のデータである。

- [Open Images](https://storage.googleapis.com/openimages/web/download_v5.html)(訓練・検証・テストデータが予め決まっている)
  - 以下の手順を実施する。
    1. 訓練・検証・テストに分割に必要な特徴量を作成する。(特徴量作成は個人でカスタマイズしても良い)

        ```shell
        python -m src.preprocess.detection.buid_features +experiment=cvat_test
        ```

        ```shell
        # input
        ./data/${DATA.DATA_ID}/interim/train.csv
        ./data/${DATA.DATA_ID}/interim/validation.csv
        ./data/${DATA.DATA_ID}/interim/test.csv
        # output
        # 特徴量の列が追加される
        ./data/${DATA.DATA_ID}/interim/train_features.csv
        ./data/${DATA.DATA_ID}/interim/validation_features.csv
        ./data/${DATA.DATA_ID}/interim/test_features.csv
        ```

    2. 訓練用と検証用データに分割する(ここでは、既にあるtrain/valデータを連結、fold番号を割り振る)

        ```shell
        python -m src.preprocess.detection.make_folds +experiment=cvat_test
        ```

        ```shell
        # input
        ./data/${DATA.DATA_ID}/interim/train_features.csv
        ./data/${DATA.DATA_ID}/interim/validation_features.csv
        # output
        # 上記ファイルを連結し、fold列が追加され、検証用0、訓練用に1が割り振られる
        ./data/${DATA.DATA_ID}/split/split01.csv
        ```

- [小麦コンペ](https://www.kaggle.com/c/global-wheat-detection/overview)(Kaggle)(テストデータが予め決まっている)

  - 以下の手順を実施する。
    1. 訓練・検証・テストに分割に必要な特徴量を作成する

        ```shell
        python -m src.preprocess.detection.buid_features +experiment=cvat_test
        ```

        ```shell
        # input
        ./data/${DATA.DATA_ID}/interim/train.csv
        ./data/${DATA.DATA_ID}/interim/test.csv
        # output
        ./data/${DATA.DATA_ID}/interim/train_features.csv
        ./data/${DATA.DATA_ID}/interim/test_features.csv
        ```

    2. train.csvから訓練用と検証用データに分割する。(分割方法は個人でカスタマイズしても良い)

        ```shell
        python -m src.preprocess.detection.make_folds +experiment=cvat_test
        ```

        ```shell
        # input
        ./data/${DATA.DATA_ID}/interim/train_features.csv
        # output
        # fold列が追加され、どのデータが訓練用か検証用かが決まる
        ./data/${DATA.DATA_ID}/split/split01.csv
        ```

### **コンフィグ設定(画像&アノテーションデータ)**

- `訓練用`

  訓練用画像群の入っているフォルダのパスを *TRAIN_IMAGE_DIR* に記載する。*DATA_ID*も設定する。

  前処理後に生成された訓練用CSV(例：train_folds.csv)のパスは、*CSV_PATH* に記載する。

    ```yaml
    # 例：./data/configs/data/detection.yaml
    DATA:
      DATA_ID: my_data1
      TRAIN_IMAGE_DIR: data/${DATA.DATA_ID}/raw/train # 訓練用画像の格納先
      CSV_PATH: data/${DATA.DATA_ID}/split/train_folds.csv # 訓練用アノテーションデータの格納先
    ```

- `テスト用`

  テスト用画像群の入っているフォルダのパスは、 *TEST_IMAGE_DIR* に記載する。 ※**訓練用と同じフォルダにあっても構わない。**

  テスト用CSV(例: test.csv)のパスは、*TEST_CSV_PATH* に記載する。

    ```yaml
    # 例：./data/configs/test/detection.yaml
    TEST:  
      TEST_IMAGE_DIR: data/${DATA.DATA_ID}/raw/test # テスト用画像の格納先
      TEST_CSV_PATH: data/${DATA.DATA_ID}/raw/sample_submission.csv # テスト用アノテーションデータの格納先
    ```

## **Configuration**

物体検出タスク特有の設定は以下の通りである。

### Augmentation

Augmentationは`Albumentations`というライブラリを使用する。(Albumentationsの詳細については、[公式のGitHub](https://github.com/albumentations-team/albumentations)をご参照ください)

1. 訓練・検証・テストで使用するAugmentationを以下のyamlファイルに定義する。各オーグメンテーションについては[こちら](https://github.com/albumentations-team/albumentations_examples#augmentations-examples)も参考になります。

    ```yaml
    # ./configs/augmentation/albumentations.yaml
    ALBUMENTATIONS:
      TRAIN: # 訓練用データ
        AUGS:
        - CLASS_NAME: albumentations.OneOf # この中で定義された内一つを実行する
          p: 0.9 # pは適用する確率
          params:
            - CLASS_NAME: albumentations.HueSaturationValue
              params:
                always_apply: false
                hue_shift_limit:
                - -0.2
                - 0.2
                p: 0.9
                sat_shift_limit: 
                - -0.2
                - 0.2
                val_shift_limit: 
                - -0.2
                - 0.2
            - CLASS_NAME: albumentations.RandomBrightnessContrast
              params:
                always_apply: false
                brightness_by_max: true
                brightness_limit:
                - -0.2
                - 0.2
                contrast_limit:
                - -0.2
                - 0.2
                p: 0.9
        - CLASS_NAME: albumentations.Resize
          params:
            always_apply: false
            height: ${MODEL.INPUT_SIZE}
            interpolation: 1
            width: ${MODEL.INPUT_SIZE}
            p: 1
        - CLASS_NAME: albumentations.ToGray
          params:
            always_apply: false
            p: 0.01
        - CLASS_NAME: albumentations.HorizontalFlip
          params:
            always_apply: false
            p: 0.5
        - CLASS_NAME: albumentations.VerticalFlip
          params:
            always_apply: false
            p: 0.5
        - CLASS_NAME: albumentations.Cutout
          params: 
            always_apply: false
            num_holes: 6
            max_h_size: 44
            max_w_size: 44
            fill_value: 0
            p: 0.5
        - CLASS_NAME: albumentations.pytorch.transforms.ToTensorV2
          params:
            always_apply: true
            p: 1.0
        BBOX_PARAMS: # 物体検出タスクでは必須のパラメタ
          _target_: albumentations.core.composition.BboxParams
          format: pascal_voc
          label_fields:
            - labels

      VALID: # 検証用データ
        AUGS:
        - CLASS_NAME: albumentations.Resize
          params:
            always_apply: false
            height: ${MODEL.INPUT_SIZE}
            interpolation: 1
            width: ${MODEL.INPUT_SIZE}
            p: 1
        - CLASS_NAME: albumentations.pytorch.transforms.ToTensorV2
          params:
            always_apply: true
            p: 1.0
        BBOX_PARAMS:
          _target_: albumentations.core.composition.BboxParams
          format: pascal_voc
          label_fields:
            - labels

      TEST: # テスト用データ
        AUGS:
          - CLASS_NAME: albumentations.Resize
            params:
              always_apply: false
              height: ${MODEL.INPUT_SIZE}
              interpolation: 1
              width: ${MODEL.INPUT_SIZE}
              p: 1
          - CLASS_NAME: albumentations.pytorch.transforms.ToTensorV2
            params:
              always_apply: true
              p: 1.0
        BBOX_PARAMS: # テストデータにGTが存在しない場合は、このBBOX_PARAMSは削除すること
          _target_: albumentations.core.composition.BboxParams
          format: pascal_voc
          label_fields:
            - labels
    ```

2. 以下のyamlファイルの`defaults`にある`augmentation`を`albumentations`に設定する

    ```yaml
    # ./configs/train.yaml
      defaults:
         - data: detection
         - augmentation: albumentations # <- この項目を`albumentations`に設定する
         - model: detection/efficientdet
         - loss: wrmsse
         - train: train
         - pytorch_lightning: trainer/lightning_trainer
         - optimizer: adamw
         - scheduler: cosinewarm
         - callback: pytorch_lightning
         - private: custom
         - system: system

      FRAMEWORK: pytorch
      TASK: detection
      MODEL_ID: model_id

      hydra:
        run:
          dir: ./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
        sweep:
          dir: ./models/
          subdir: ${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
    ```

3. 設定上書き用ファイル（例：`./configs/experiment/effdet_person_ca_v5.yaml`）に以下の内容を記載する。

    `AUGMENTATION.FRAMEWORK`を`albumentations`に設定する

    ```yaml
    AUGMENTATION:
      FRAMEWORK: albumentations # 'albumentations'に設定
    ```

    例：以下のように、訓練実行時に設定上書き用ファイルを指定する。（設定の上書きについての説明は、`./docs/Config.md`をご参照ください）

    ```shell
    python -m src.tools.train +experiment=effdet_person_ca_v5
    ```

### Data

  アノテーションデータの前処理、使用するデータの情報の設定は以下で行う。

  ```yaml
  # ./configs/data/detection.yaml
  DATA:
    DATA_ID: my_data1 # データID(任意のわかりやすい名前に)
    TRAIN_IMAGE_DIR: data/${DATA.DATA_ID}/raw/train # 訓練画像ディレクトリ
    EXTRACT_ONE_CLASS: false # 複数クラスから1クラスのみにするか(OpenImages用)
    PROCESSED_CSV_NAME: '_features.csv' # 特徴量追加後(build_features.py)のファイル名
    CSV_PATH: data/${DATA.DATA_ID}/split/train_folds.csv # 訓練用アノテーションファイルパス
    RANDOM_KFOLD: False # ランダムで交差検証するかどうかのフラグ
    TEST_SIZE: 0.2 # train/test分割時のテストの割合
    N_FOLD: 5 # 交差検証の分割数(Leave-one-outでは1に設定する)
    FORMAT: pascal_voc # bboxのフォーマット。推論結果の出力形式が変わります。pascal_voc(xmin,ymin,xmax,ymax) or coco(xmin,ymin,width,height)
    IMAGE_ID_COL_NAME: image_id # 画像を一意に特定するID(画像名)のカラム名
    FOLD_TARGET_COL: count # 交差検証時の分割対象のカラム名
    REMOVE_LARGE_BBOXES: false # 大きすぎるbboxを削除するかのフラグ
    LARGE_BBOX_AREA_THRESHOLD: 200000 # REMOVE_LARGE_BBOXESがTrueの場合のbboxサイズ

  DATASET:
    CLASS_NAME: src.data.generator.detection.dataset.DatasetRetriever # Generatorクラスのパス
    CUTMIX: false # Generatorクラス内でcutmixを実施するかのフラグ
  ```

### Test

  推論に関するの設定は以下で行う。

  ```yaml
  # ./configs/test/detection.yaml
  TEST:
    BATCH_SIZE: 2 # 推論時バッチサイズ
    TEST_IMAGE_DIR: data/${DATA.DATA_ID}/raw/test # テスト用画像ディレクトリ
    TEST_CSV_PATH: data/${DATA.DATA_ID}/raw/test_submission.csv # テスト用アノテーションファイルパス
    VISUALIZE_RESULTS: false # 結果を可視化するかのフラグ
    VAL_PRED_IMG_DIR: pred_images/validation/ # 検証用データに対する結果の可視化画像の保存先
    TEST_PRED_IMG_DIR: pred_images/test/ # テスト用データに対する結果の可視化画像の保存先
    SAVE_ALL_IMAGES: false # 全テスト画像に対して可視化するかどうかのフラグ。Falseであれば、10枚を可視化し保存する
    DETECTION_THRESHOLD: 0.2 # 信頼度スコアの閾値
    FIND_BEST_THR: false # 検証用データを用いて、mAPが最大になる信頼度スコアの探索機能を実行するかどうかのフラグ
    ENSEMBLE_BOXES:
      NAME: WBF #アンサンブルの手法の名前 (WBF, NMW, SoftNMS, NMSから選択する)
      WEIGHTS: # 各モデルのアンサンブルの比重リスト。デフォルトはNoneで同じ比重に設定
      SIGMA: 0.5 # SoftNMSで使用する値
      THRESH: 0.001 # 残すボックスを決めるための閾値(SoftNMSで使用)
      IOU_THR: 0.4 # ボックスがマッチしているかを判定するIoUの閾値
      SKIP_BOX_THR: 0.4 # この値以下のボックスは除外する
    BEST_THR_CSV_NAME: best_score_threshold.csv # 信頼度スコアの探索機能の結果保存ファイル名
    TEST_SCORE_CSV_NAME: test_score.csv # テストデータに対するmAP算出結果が記載されたファイル名
  ```

## **Comet ML**

機械学習の実験管理ツール([公式ドキュメント](https://www.comet.ml/docs/))

詳細な使用方法については[こちら](https://qiita.com/TeraBytes/items/cf7a746330c887df844f)をご参照ください。

#### ***個人利用は無料ですが、ダッシュボードがパブリックになりますので、注意して下さい**

#### **主要機能**

- 学習曲線やデータの入出力の可視化
- 使用したハイパーパラメータとその値の記録
- 学習コードの記録
- 複数実験の比較

使用手順

1. comet ml のアカウントを作成し、APIキーを取得する
2. 以下のyamlファイルに取得したAPIキーを定義する

    ```yaml
    # ./configs/private/default.yaml
    PRIVATE:
      COMET_API: Your_API_Key
    ```

3. 以下のyamlファイルのcomet用の設定を行う(./configs/experimet 配下のyamlファイル内で以下の内容を上書く)

    ```shell
    # ./configs/callback/pytorch_lightning.yaml
    CALLBACK:
      LOGGER:
        TENSORBOARD: true
        COMMET:
          FLAG: false # 使用する場合はtrueにする
          SAVE_DIR: logs/
          WORKSPACE: r1ck29 # comet_ml登録時のUsername
          PROJECT_NAME: wheat_v2 # プロジェクト名 案件ごとに変更
          DEVICE: cuda
    ```

4. 訓練実行時に以下のようにyamlファイルを指定していることを確認する(以下の例ではcustom.yamlを指定)

    ```shell
    python -m src.tools.train private=custom
    ```

## **How to Run Programs**

### **Preprocess**

1. ./bin/preprocess_detection.sh 内の処理・引数は必要に応じて変更する

    各データセットごとの実行プログラムは以下の通りである。

    `CVAT`

    ```shell
    python -m src.preprocess.detection.xml_to_csv +experiment=effdet_cvat
    python -m src.preprocess.detection.build_features +experiment=effdet_cvat
    python -m src.preprocess.detection.train_test_split +experiment=effdet_cvat
    python -m src.preprocess.detection.make_folds +experiment=effdet_cvat
    ```

    `Open Images`

    ```shell
    python -m src.preprocess.detection.build_features +experiment=effdet_open_images_person
    python -m src.preprocess.detection.make_folds +experiment=effdet_open_images_person
    ```

    `Global Wheat Detection dataset (Kaggle)`

    ```shell
    python -m src.preprocess.detection.build_features +experiment=effdet_wheat
    python -m src.preprocess.detection.make_folds +experiment=effdet_wheat
    ```

2. 以下のコマンドを実行

```shell
sh ./bin/preprocess_detection.sh
```

### **Train**

1. ./bin/train_detection.sh 内の引数は必要に応じて変更する
2. 以下のコマンドを実行

```shell
sh ./bin/train_detection.sh
```

### **Test**

1. ./bin/predict_detection.sh 内の引数は必要に応じて変更する
2. BaseModelディレクトリ配下で、以下のコマンドを実行

```shell
sh ./bin/predict_detection.sh
```
