# Config System

以下のタスクで共通のコンフィグの設定について説明する。

- detection
- segmentation
- keypoint

## About Hydra

Hydraは、Config管理のライブラリです。(詳細は[公式ドキュメント](https://hydra.cc/)をご参照ください)

### Process Steps

  1. 前処理・訓練時は`./configs/train.yaml`、推論時は`./configs/test.yaml`が読み込まれる。
  2. 1のyamlファイル内の`defaults`で指定されたフォルダのyamlファイルの内容が読みこまれる。
  3. 実行時のコマンドライン引数で指定されたyamlファイル内の設定、または直接指定された値に、2で読み込んだ各設定が上書かれる。

### Set Parameters

1. yamlファイル単位の設定変更

    以下の`train.yaml`の`defaults`部分を変更する。

    `data: detection` は、`./configs/data/detection.yaml` を参照させる。

    例：optimizerを変更する場合、`./configs/optimizer`配下のyamlファイルを切り替える際に変更する。

    `タスク別`と書かれている項目は、特定のタスクでしか使わない設定も含み、タスク別にyamlファイルの切り替えが必要。

    特定のタスクでしか使わない設定については、各タスクのドキュメントをご参照ください。

    訓練時の`defaults`

    ```yaml
    # train.yaml
    defaults:
      - data: detection # データ(タスク別)
      - augmentation: custom # データ拡張(タスク別)
      - model: detection/efficientdet # モデル(タスク別)
      - loss: wrmsse # ロス(タスク別)
      - train: train # 訓練時に使用する(タスク共通)
      - pytorch_lightning: trainer/lightning_trainer # Pytorch LightniingのTrainerクラスの引数(タスク共通)
      - optimizer: adamw # オプティマイザ(タスク共通)
      - scheduler: plateau # オプティマイザ(タスク共通)
      - callback: pytorch_lightning # コールバック(タスク共通)
      - private: custom # APIキー等の個人情報(タスク別)
      - system: system # GPU関連(タスク共通)

    FRAMEWORK: pytorch # pytorch, tensorflowから設定(タスク別)
    TASK: keypoint # detection, segmentation, keypointから設定(タスク別)
    MODEL_ID: model_id # モデルの重みやログの出力先ディレクトリ名(タスク共通)
    ```

    推論時の`defaults`

    ```yaml
    # ./configs/test.yaml
    defaults:
      - data: detection # データ(タスク別)
      - test: detection # 推論時に使用するパラメタ。例: テストデータのパスや閾値等(タスク別)
      - system: system # GPU関連(タスク共通)

    FRAMEWORK: pytorch # pytorch, tensorflowから設定(タスク別)
    TASK: keypoint # detection, segmentation, keypointから設定(タスク別)
    MODEL_PATH: /path/to/model/ # 訓練済みモデルまでのパス(タスク共通)
    TEST_ID: tutorial # テスト結果保存フォルダ名(タスク共通)
    ```

    上記で指定されているyamlファイルをディレクトリで表すと以下の通りになる。

    ```shell
    ./configs/
    ├── augmentation
    │   ├── albumentations.yaml
    │   ├── custom.yaml　# - augmentation: custom
    │   └── hardnet.yaml
    ├── callback
    │   └── pytorch_lightning.yaml # - callback: pytorch_lightning
    ├── data
    │   ├── detection.yaml # - data: detection <-yamlファイル名を指定する。
    │   ├── keypoint.yaml
    │   └── seg_pytorch.yaml
    ├── experiment
    │   ├── effdet_person_cosine_annealing.yaml
    ├── loss
    │   ├── keypoint.yaml
    │   └── wrmsse.yaml # - loss: wrmsse
    ├── model
    │   ├── detection
    │   │   ├── efficientdet.yaml # - model: detection/efficientdet
    │   │   └── fasterrcnn.yaml
    │   └── keypoint
    │       └── hrnet.yaml
    ├── optimizer
    │   ├── adam.yaml
    │   ├── adamw.yaml # - optimizer: adamw
    │   └── sgd.yaml
    ├── private
    │   ├── custom.yaml # - private: custom
    │   └── default.yaml
    ├── pytorch_lightning
    │   └── trainer
    │       └── lightning_trainer.yaml # - pytorch_lightning: trainer/lightning_trainer
    ├── scheduler
    │   ├── cosine.yaml
    │   ├── cosinewarm.yaml
    │   ├── cyclic.yaml
    │   ├── multi_step copy.yaml
    │   ├── multi_step.yaml
    │   ├── multi_step_reg.yaml
    │   ├── onecycle.yaml
    │   ├── plateau.yaml # - scheduler: plateau
    │   └── step.yaml
    ├── segmentation
    │   └── config.yaml
    ├── system
    │   └── system.yaml # - system: system
    ├── test
    │   ├── detection.yaml # - test: detection
    │   ├── keypoint.yaml
    │   └── seg_pytorch.yaml
    ├── test.yaml
    ├── train
    │   └── train.yaml # - train: train
    └── train.yaml
    ```

2. 各パラメタの上書き

    `./configs/experiment` フォルダ配下にyamlを作成し、上書く設定を記載し、`+experiment=` の後にyamlファイル名を指定する。

    指定したファイルでデフォルトの各パラメタが上書かれる。

    以下のコマンドでは、`./configs/experiment` 配下の`open_images_test.yaml`ファイルでデフォルトの設定を上書いている。

    ```shell
    # ./bin/train_detection.sh
    python -m src.tools.train +experiment=open_images_test # 細かいパラメタの変更が記載されたyaml(ここに記載された値でデフォルトの値を上書く)
    ```

    例：上書く内容を記載したyaml

    ```yaml
    # ./configs/experiment/open_images_test.yaml
    AUGMENTATION:
      FRAMEWORK: albumentations # albumentations使用時は記載する

    SYSTEM:
      GPUS: '1'
      SEED: 666
      CUDNN:
        ENABLED: true
        BENCHMARK: false
        DETERMINISTIC: true

    DATA:
      DATA_ID: open_images_person
      TRAIN_IMAGE_DIR: data/${DATA.DATA_ID}/raw/person/images
      PROCESSED_CSV_NAME: '_run_test_v2.csv'
      EXTRACT_ONE_CLASS: true
      CSV_PATH: data/${DATA.DATA_ID}/split/train_folds_run_test_v2.csv
      N_FOLD: 1
      FORMAT: pascal_voc
      IMAGE_ID_COL_NAME: ImageID

    MODEL:
      INPUT_SIZE: 512
      OUTPUT_SIZE: 512
      NUM_CLASSES: 1
      BASE_NAME: efficientdet_d4

    TRAIN:
      EPOCHS: 100
      LR: 0.0002
      BATCH_SIZE: 4
      DEBUG: false

    CALLBACK:
      LOGGER:
        COMMET:
          FLAG: false
          PROJECT_NAME: open_images_person
      EARLY_STOPPING:
        FLAG: true

    FRAMEWORK: pytorch
    TASK: detection
    MODEL_ID: open_images_test

    TEST:
      SAVE_ALL_IMAGES: false
      FIND_BEST_THR: true
      BATCH_SIZE: 2
      DETECTION_THRESHOLD: 0.2
      TEST_IMAGE_DIR: data/${DATA.DATA_ID}/raw/person/images
      TEST_CSV_PATH: data/${DATA.DATA_ID}/processed/test_person_run_test_v2.csv # for train 2_1, use v3
    ```

### Description of Parameters

各パラメタについて説明する。

#### Main Config

Hydraがメインで実行するyamlファイルはtrainとtestがある。

どちらも`訓練時または推論時に必ず確認・変更すること。`(例：`TASK: keypoint`など)

#### Train

訓練時に読み込まれるyamlファイル

```yaml
# ./confifs/train.yaml
defaults:
  - data: detection
  - augmentation: custom
  - model: detection/efficientdet
  - loss: wrmsse
  - train: train
  - pytorch_lightning: trainer/lightning_trainer
  - optimizer: adamw
  - scheduler: cosinewarm
  - callback: pytorch_lightning
  - private: custom
  - system: system

FRAMEWORK: pytorch # pytorch, tensorflowから設定(タスク別)
TASK: keypoint # detection, segmentation, keypointから設定(タスク別)
MODEL_ID: model_id # モデルの重みやログの出力先ディレクトリ名(タスク共通)

hydra:
  run:
    dir: ./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
  sweep:
    dir: ./models/
    subdir: ${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
```

- `MODEL_ID`: 訓練モデルを識別するID。`./models`配下のディレクトリ名がここで設定した文字列と一致すること。

  - config file

      ```yaml
      # ./configs/train.yaml
      MODEL_ID: model_id # モデルの重みやログの出力先フォルダ名
      ```

- `hydra.run.dir`: `MODEL_ID`を含むパス。`MODEL_ID`配下に`timestamp`形式のディレクトリが作成される。ここにモデルの重みやログ、推論結果が保存される。

  - config file

      ```yaml
      # ./configs/train.yaml
      hydra:
        run:
            dir: ./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S} # 上記モデルIDを含む、重みやログの出力パス
      ```

#### Test

推論時に読みこまれるyamlファイル

```yaml
# ./configs/test.yaml
defaults:
  - data: detection # データ(タスク別)
  - test: detection # テスト関連(タスク別)
  - system: system # GPU関連(タスク共通)

FRAMEWORK: pytorch # pytorch, tensorflowから設定(タスク別)
TASK: keypoint # detection, segmentation, keypointから設定(タスク別)
MODEL_PATH: /path/to/model/ # 訓練済みモデルまでのパス(タスク共通)
TEST_ID: tutorial # テスト結果保存フォルダ名(タスク共通)
  
hydra:
  run:
    dir: ${MODEL_PATH}/result/${TEST_ID}
  sweep:
    dir: ${MODEL_PATH}/result/
    subdir: ${TEST_ID}
```

- `MODEL_PATH`: 訓練済みモデルのパス。`./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}`を指定する。どの`timestamp`のフォルダが推論で使用するモデルかを確認すること。

  - config file

      ```yaml
      # ./configs/test.yaml
      MODEL_PATH: /path/to/model/
      ```

  - 推論時のモデル指定方法

      訓練時は、以下のように`timestamp`のフォルダが作成され、そこに重みやログが保存される。

      ```shell
      # ./configs/train.yaml
      ./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
      ```

      よって、以下のように推論コード実行時に使用したいモデルが格納されている`timestamp`のフォルダ名までのパスを記載する必要がある。

      ```shell
      # ./bin/predict_detection.sh
      python -m src.tools.predict +experiment=hydra_test \　                          <- 上書く設定が記載されたyamlファイルの指定
                                  MODEL_PATH=models/$\{MODEL_ID\}/20200909_14_34_46/  <- モデルの指定
      ```

- `TEST_ID`: 推論結果を識別するID。`./result`配下のディレクトリ名がここで設定した文字列と一致すること。

  - config file

      ```yaml
      # ./configs/test.yaml
      TEST_ID: tutorial # 推論結果保存先のフォルダ名
      ```

#### Models Directory Structure

上記で説明したパラメタは`./models`ディレクトリでは以下に該当する。

また、モデルの訓練・推論実行時に出力されるディレクトリやファイルは以下の通りである。

```shell
./models/
├── effdet_person_cosine_annealing # <- MODEL_ID
│   └── 20200914_20_41_11 # <- hydra.run.dir (重みやログの保存先)
│       ├──.hydra
│       │    ├── config.yaml # <- 訓練実行時の全config
│       │    ├── hydra.yaml # <- Hydraのconfig内容(hydra.run.dir等)
│       │    └── overrides.yaml # <- pythonファイル実行時の引数に与えられた、上書き内容(例： - +experiment=effdet_person_cosine_annealing)
│       ├── augmentation
│       │   └── albumentations # <- albumentations使用時のaugmentation内容保存先
│       ├── code # <- 訓練実行時コード保存先
│       │   └── src
│       │       ├── data
│       │       ├── models
│       │       ├── preprocess
│       │       ├── tools
│       │       ├── utils
│       │       └── visualize
│       ├── fold0_epoch=27_val_loss=0.452.ckpt # <- PytorchLightningが出力するチェックポイント
│       ├── best_loss_fold0.pth # <- 上記チェックポイントの重みのみ(推論で使用)
│       ├── requirements.txt # <- 訓練実行時環境内容保存先
│       ├── train.log # <- 訓練時の出力ログ
│       └── result # 推論結果
│           └── tutorial # <- TEST_ID
│               ├── augmentation # <- albumentations使用時のaugmentation保存先
│               └── pred_images　# <- 推論結果可視化画像
└──pretrained_models # <- COCO等のデータで学習済みの重み
    └── efficientdet
```

#### Data

- `{DATA_ID}`: データの種類を識別するID。./data配下のディレクトリ名がここで設定した文字列と一致すること。

  - config file

      ```yaml
      # ./configs/data/detection.yaml
      DATA:
      DATA_ID: my_data1
      ```

  - directory

      ```shell
      ./data/
      ├── my_data1 # <- DATA_ID
      │   ├── interim
      │   ├── raw
      │   │   ├── test
      │   │   └── train
      │   └── split
      ```

#### System

```yaml
# ./configs/system/system.yaml
SYSTEM:
  GPUS:
  - 0 # <- GPU番号 (GPUS: '1'もOK。CPU使用の場合は、空欄にする)
  - 1 # <- GPU番号 (GPU複数使用時)
  SEED: false # <- 使用時はseed固定の値を入れる (例: SEED: 42)
  NUM_WORKERS: 4 # データ読み込みのサブプロセス数
  CUDNN:
    ENABLED: true # CUDNN使用フラグ
    BENCHMARK: true # Benchmark使用フラグ(結果再現性担保のフラグ(DETERMINISTIC)をtrueにする時はfalseに設定)
    DETERMINISTIC: false # trueで結果再現性を担保
```

#### PyTorch Lightning

   - Callback

        ```yaml
        # ./configs/callback/pytorch_lightning.yaml
        CALLBACK:
            MODEL_CHECKPOINT:
                CLASS_NAME: pl.callbacks.ModelCheckpoint # モデルのチェックポイントクラスのパス。
                PARAMS:
                    monitor: ${TRAIN.MONITOR}
                    save_top_k: 1
                    mode: ${TRAIN.MODE}
                    save_weights_only: false
                    verbose: true

            EARLY_STOPPING:
                FLAG: false # EARLY_STOPPINGを使用する場合は、true
                CLASS_NAME: pl.callbacks.EarlyStopping # EARLY_STOPPINGのクラスのパス。Pytorch Lightningの実装を用いる。
                PARAMS:
                    monitor: ${TRAIN.MONITOR}
                    patience: 20
                    mode: ${TRAIN.MODE}
                    verbose: true

            LOGGER:
                TENSORBOARD: true # Tensor Boardを使用する場合は、true
                COMMET:
                    FLAG: false # CometMLを使用する場合は、true
                    SAVE_DIR: logs/ # CometML
                    WORKSPACE: r1ck29 # CometMLのアカウント名
                    PROJECT_NAME: wheat_v2 # CometMLのプロジェクト名
                    DEVICE: cuda # cudaまたはcpu
                JSON: false # Jsonロガーを使用する場合は、true
        ```

   - Trainer

     ```yaml
     # ./configs/pytorch_lightning/trainer/lightning_trainer.yaml
     TRAINER:
        gpus: ${SYSTEM.GPUS} # 使用するGPU
        distributed_backend: dp # 分散学習の種類('dp'はDataParallel)
        benchmark: ${SYSTEM.CUDNN.BENCHMARK} # ベンチマークの使用有無
        deterministic: ${SYSTEM.CUDNN.DETERMINISTIC} # 結果再現性の担保の有無
        accumulate_grad_batches: 1 # 毎Kバッチの勾配の蓄積数
        profiler: False # コードのプロファイルの出力有無(コードのボトルネック調査に使用)
        max_epochs: ${TRAIN.EPOCHS} # エポック数
        log_save_interval: 100 # ログの保存間隔
        gradient_clip_val: 0 # 勾配クリッピングの値
        num_sanity_val_steps: 2 # 訓練前の検証用データのサニティーチェック
        weights_summary: # モデルの重みの出力有無。デフォルトは空白に設定し、表示しない。
     ```