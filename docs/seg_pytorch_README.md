# [DL Pipeline] Segmentation for PyTorch README

## **Directory Structure**

```bash
.
├── configs 
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
├── data # dataset
├── docs # README
├── models # trained weight, log, evaluation
├── notebooks # Jupyter Notebook(including tutorial)
├── requirements
├── serving
├── src # codes
│   ├── data
│   │   ├── generator
│   │   │   ├── seg_pytorch # augmentations and custom dataset functions
│   │   └── transforms
│   ├── models
│   │   ├── backbone # backbone architecture or layers for architecture
│   │   │   └── pytorch
│   │   ├── loss # loss functions
│   │   │   └── seg_pytorch
│   │   ├── modeling # architecture and training codes for Pytorch Lightning
│   │   │   └── seg_pytorch
│   │   ├── predictor # codes for evaluation
│   │   │   └── seg_pytorch
│   │   └── utils # codes used across multiple model
│   │       └── seg_pytorch
│   ├── preprocess
│   │   └── seg_pytorch
│   ├── tools # codes for train, prediction
│   ├── utils # codes used across multiple tasks
│   │   └── pytorch
│   └── visualize 
└── tests
```

## **Requirements and Installation**

- Python version : 3.7.2
- library : requirements/requirements_pytorch_segmentation.txt


## **Models**

- HarDNet

    - Download pretrained weight
    
        Download the HarDNet trained weight from [Google Drive](https://drive.google.com/drive/folders/1zK8A0oit8sKzEdPBU-RTEtF0d_gwxXlw) under Segmentation Dir
        
        Allocate trained weight under models dir(models/cityscapes/20201013_10_41_53/state.pth)

## **Dataset**
    
Preprocessing of the dataset.

**Preprocessing Cityscapes**.

```bash
 src/preprocess_segmentation_pytorch/cityscapes/preprocess_seg_pytorch.py
```

In Cityscapes pre-processing, following operations will be executed.

1.Get RGB images and GroundTruth(GT) path names. And create a flag whether each data is training data or evaluation data.

2.Save information created in 1 as a data frame.


after running script above, each file will be allocated as follows under data dir
    
```bash
cityscapes 
   ├── raw
   │   └── Cityscapes dataset
   └── split
       └── split01.pkl : DataFrame

```
    
    
**Preprocessing CVAT annotated dataset**.

```bash
 src/preprocess/seg_pytorch/preprocess_cvat/read_cvat.py
```

In CVAT pre-processing, following operations will be executed.

1.Create GroundTruth(GT) from the xml file, which is cvat's output

2.Get RGB images and GroundTruth(GT) path names. And create a flag whether each data is training data or evaluation data.

3.Save information created in 2 as a data frame.


before running script above, allocate image file and xml file under the data dir like below
    
```bash

 cvat
   └── raw
      ├── cvat : xml files
      └── images : RGB images

```
    
after running script above, each file will be allocated as follows 
    
```bash
cvat
   ├── interim
   │   ├── color : gt(rgb)
   │   └── gt : gt(1channel)
   ├── raw
   │   ├── cvat : xml files
   │   └── images : RGB images
   └── split
       ├── colormap.npy : colormap
       └── split01.pkl : DataFrame

```
    
    
    

## **How to use Tools**

### **Hydra**

Library of Config management ([Official Documentation](https://hydra.cc/))
    
- The basic structure of this system

    During training, train.yaml under /configs is loaded. train.yaml under /configs is read during training and test.yaml is read during prediction.
    
1. Change yaml file configuration.
    
    The following is the setup of an experiment using Cityscapes.


    ```yaml
    # train.yaml
    defaults:
      - data: seg_pytorch
      - augmentation: hardnet
      - model: seg_pytorch/hardnet
      - loss: hardnet
      - train: train
      - pytorch_lightning: trainer/lightning_trainer
      - optimizer: sgd
      - scheduler: poly_lr
      - callback: pytorch_lightning
      - private: default
      - test: seg_pytorch
      - system: system

    FRAMEWORK: pytorch
    TASK: seg_pytorch
    MODEL_ID: cityscapes

    hydra:
      run:
        dir: ./models/${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
      sweep:
        dir: ./models/
        subdir: ${MODEL_ID}/${now:%Y%m%d_%H_%M_%S}
    ```
    
    ```yaml
    # test.yaml
    defaults:
      - data: seg_pytorch
      - test: seg_pytorch
      - system: system

    FRAMEWORK: pytorch
    MODEL_PATH: {path for trial name}
    DATA_PATH: {path for cityscapes}
    TEST_ID: tutorial
    GPUS: 0
    TASK: seg_pytorch

    hydra:
      run:
        dir: ${MODEL_PATH}/result/${TEST_ID} #${now:%Y%m%d_%H_%M_%S}
      sweep:
        dir: ${MODEL_PATH}/result/
        subdir: ${TEST_ID} #${now:%Y%m%d_%H_%M_%S}
    ```
    
    ```yaml
    # other setting
    RAMEWORK: pytorch
    TASK: seg_pytorch
    MODEL_ID: cityscapes
    DATA:
      DATA_ID: cityscapes
      SPLIT_ID: split01
      DATASET: Cityscapes
      CLASS_NUM: 19
      N_FOLD: 1
      path: {path for cityscapes dataset}
    AUGMENTATION:
      hflip: 0.5
      rscale_crop:
      - 1024
      - 1024
    MODEL:
      arch: hardnet
      n_classes: 19
      pretrain: ../../../src/models/modeling/seg_pytorch/hardnet_petite_base.pth
      fuse_bn: true
    LOSS:
      name: bootstrapped_cross_entropy
      min_K: 4096
      loss_th: 0.3
      size_average: true
    TRAIN:
      EPOCHS: 100
      LR: 0.02
      BATCH_SIZE: 4
      MONITOR: val_loss
      MODE: min
      DEBUG: false
    TRAINER:
      gpus: ${SYSTEM.GPUS}
      distributed_backend: dp
      benchmark: ${SYSTEM.CUDNN.BENCHMARK}
      deterministic: ${SYSTEM.CUDNN.DETERMINISTIC}
      accumulate_grad_batches: 1
      profiler: false
      max_epochs: ${TRAIN.EPOCHS}
      log_save_interval: 100
      gradient_clip_val: 0
      num_sanity_val_steps: 2
      weights_summary: null
    OPTIMIZER:
      NAME: torch.optim.SGD
      PARAMS:
        lr: ${TRAIN.LR}
        momentum: 0.9
        weight_decay: 0.0005
        nesterov: false
    SCHEDULER:
      NAME: poly_lr
      PARAMS:
        max_iter: ${TRAIN.EPOCHS}
    CALLBACK:
      MODEL_CHECKPOINT:
        CLASS_NAME: pl.callbacks.ModelCheckpoint
        PARAMS:
          monitor: ${TRAIN.MONITOR}
          save_top_k: 1
          mode: ${TRAIN.MODE}
          save_weights_only: false
          verbose: true
      EARLY_STOPPING:
        FLAG: false
        CLASS_NAME: pl.callbacks.EarlyStopping
        PARAMS:
          monitor: ${TRAIN.MONITOR}
          patience: 20
          mode: ${TRAIN.MODE}
          verbose: true
      LOGGER:
        TENSORBOARD: true
        COMMET:
          FLAG: false
          SAVE_DIR: logs/
          WORKSPACE: r1ck29
          PROJECT_NAME: wheat_v2DEVICE: cuda
        JSON: false
    PRIVATE:
      COMET_API: Your_API_Key
    TEST:
      BATCH_SIZE: 1
    SYSTEM:
      GPUS:
      - 1
      SEED: false
      NUM_WORKERS: 4
      CUDNN:
        ENABLED: true
        BENCHMARK: true
        DETERMINISTIC: false

    
    ```
    

2. Run training script

training script is src/tools/train.py

```bash
python src/tools/train.py
```

 3. Run predection script

predection script is src/tools/predict.py

```bash
python src/tools/predict.py
```

