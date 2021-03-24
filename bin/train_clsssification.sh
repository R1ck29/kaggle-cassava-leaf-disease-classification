# python -m src.tools.train +experiment=person_attribute_pa100k

# batch 1
# python -m src.tools.train +experiment=person_attribute \
#                           AUGMENTATION.FRAMEWORK=albumentations_classification

# python -m src.tools.train +experiment=person_attribute


# python -m src.tools.train +experiment=person_attribute \
#                             CALLBACK.EARLY_STOPPING.FLAG=false

# python -m src.tools.train +experiment=person_attribute \
#                             AUGMENTATION.FRAMEWORK=albumentations_classification
#                             CALLBACK.EARLY_STOPPING.FLAG=false


# batch 2 test
# python -m src.tools.train optimizer=adamw \
#                           +experiment=person_attribute_adamw \
#                           AUGMENTATION.FRAMEWORK=albumentations_classification

# python -m src.tools.train optimizer=adamw \
#                           +experiment=person_attribute_adamw \
#                           AUGMENTATION.FRAMEWORK=albumentations_classification \
#                           TRAIN.MONITOR=val_loss \
#                           TRAIN.MODE="min"

# python -m src.tools.train +experiment=person_attribute \
#                           AUGMENTATION.FRAMEWORK=albumentations_classification \
#                           TRAIN.MONITOR=val_loss \
#                           TRAIN.MODE="min"

# cosine val_score
# python -m src.tools.train scheduler=cosine \
#                           +experiment=person_attribute_cosine

# # cosine warm  val_score
# python -m src.tools.train scheduler=cosinewarm \
#                             +experiment=person_attribute_cosinewarm


# # reduceLR val_score
# python -m src.tools.train +experiment=person_attribute

# # reduceLR val_loss
# python -m src.tools.train +experiment=person_attribute \
#                           TRAIN.MONITOR=val_loss \
#                           TRAIN.MODE="min"

# try no early epoch 50 val_loss

# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=64 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.INPUT_SIZE.HEIGHT=256 \
#                             MODEL.INPUT_SIZE.WIDTH=256 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=256 \
#                             MODEL.OUTPUT_SIZE.WIDTH=256 \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=64 \
#                             TRAIN.LR=1e-5 \
#                             MODEL.INPUT_SIZE.HEIGHT=256 \
#                             MODEL.INPUT_SIZE.WIDTH=256 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=256 \
#                             MODEL.OUTPUT_SIZE.WIDTH=256 \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=64 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.INPUT_SIZE.HEIGHT=256 \
#                             MODEL.INPUT_SIZE.WIDTH=256 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=256 \
#                             MODEL.OUTPUT_SIZE.WIDTH=256 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16

                            #                            MODEL.BACKBONE.PARAMS.WEIGHT_PATH=models/pretrained_models/efficientnet_1.0/ \

# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet \
#                             TRAIN.MODE=max


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# lr 1e-4 is better than 5

# batch16
# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.BATCH_SIZE=16 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet \
#                             TRAIN.MODE=max \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=16 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.INPUT_SIZE.HEIGHT=384 \
#                             MODEL.INPUT_SIZE.WIDTH=384 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=384 \
#                             MODEL.OUTPUT_SIZE.WIDTH=384 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet
                            # +TRAINER.amp_level='O1' \
                            # +TRAINER.precision=16

# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=16 \
#                             TRAIN.LR=5e-4 \
#                             MODEL.INPUT_SIZE.HEIGHT=384 \
#                             MODEL.INPUT_SIZE.WIDTH=384 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=384 \
#                             MODEL.OUTPUT_SIZE.WIDTH=384 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet



# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=5e-4 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-5 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.MODEL_NAME=efficientnet-b4 \
#                             MODEL_ID=efficientnetb4_defaults_adam_cosinew \
#                             MODEL.INPUT_SIZE.HEIGHT=448 \
#                             MODEL.INPUT_SIZE.WIDTH=448 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=448 \
#                             MODEL.OUTPUT_SIZE.WIDTH=448 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet
                    

# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet


# python -m src.tools.train +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=2e-4 \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet


############################################# Loss experiments ####################################
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.focal_loss.FocalLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.focal_cosine_loss.FocalCosineLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet


# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.symmetric_ce_loss.SymmetricCrossEntropy \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet
   
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.taylor_ce_loss.TaylorCrossEntropyLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adatruem_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG= \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# tf_efficientnetb3_ns_adam_cosinew 5folds LabelSmoothingLoss
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# # LabelSmoothingLoss val_loss
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                            +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_loss \
#                             TRAIN.MODE=min \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet


# smoothing 0.0
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.bi_tempered_logistic_loss.BiTemperedLogisticLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# smoothing 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=1 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.bi_tempered_logistic_loss.BiTemperedLogisticLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet


# # tf_efficientnetb3_ns_adam_cosinew 5folds TaylorCrossEntropyLoss should be the best score
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.taylor_ce_loss.TaylorCrossEntropyLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# b6 batch 8
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb6_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b6_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.taylor_ce_loss.TaylorCrossEntropyLoss \
#                             MODEL.BACKBONE.CLASS_NAME=src.models.backbone.pytorch.efficientnet_pytorch.EfficientNet-PyTorch-v1.efficientnet_pytorch.model.EfficientNet

# run fold0 to 2
# tf_efficientnetb3_ns_adam_cosinew 5folds TaylorCrossEntropyLoss and label smoothing 0.2
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.taylor_ce_loss.TaylorCrossEntropyLossv2

# # run fold0 to 2
# # b3 batch 8 to 16
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=16 \
#                             TRAIN.LR=1e-4 \
#                             +TRAINER.amp_level='O1' \
#                             +TRAINER.precision=16
#                             LOSS.CLASS_NAME=src.models.loss.classification.taylor_ce_loss.TaylorCrossEntropyLoss \

# tf_efficientnetb3_ns_adam_cosinew 5folds TaylorCrossEntropyLoss and label smoothing 0.2
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             TRAINER.accumulate_grad_batches=4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.taylor_ce_loss.TaylorCrossEntropyLossv2 \

# label smoothing 0.2
# tf_efficientnetb3_ns_adam_cosinew 5folds LabelSmoothingLoss
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss \

# # smoothing 0.2
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.bi_tempered_logistic_loss.BiTemperedLogisticLoss \

# seresnext101_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3 accumulate_grad_batches=4
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=seresnext101_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=seresnext101_32x4d \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             TRAINER.accumulate_grad_batches=4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss
                            
# seresnext50_32x4d
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=seresnext50_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=seresnext50_32x4d \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             TRAINER.accumulate_grad_batches=4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss

# tf_efficientnetb3_ns_adam_cosinew 5folds LabelSmoothingLoss 0.3 accumulate_grad_batches=4
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             TRAINER.accumulate_grad_batches=4 \
#                             LOSS.CLASS_NAME=src.models.loss.classification.label_smoothing_loss.LabelSmoothingLoss

# AdamW LabelSmoothingLoss 0.2
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             optimizer=adamw \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adamw_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             OPTIMIZER.PARAMS.weight_decay=1e-6 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \


# tf_efficientnetb3_ns_adam_cosinew 5folds LabelSmoothingLoss 0.2 accumulate_grad_batches=2
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             TRAINER.accumulate_grad_batches=2
                            # LOSS.PARAMS.smoothing=0.2 \



# vit_base_patch32_384_adam_cosinew 5folds LabelSmoothingLoss 0.3
# 'vit_base_patch32_384',
#  'vit_base_resnet26d_224',
#  'vit_base_resnet50d_224',
# 'ViT-B_16'

# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_vit_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=vit_base_patch32_384_adam_cosinew \
#                             MODEL.MODEL_NAME=vit_base_patch32_384 \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=32 \
#                             TRAIN.LR=1e-4

# tf_efficientnetb4_ns_adam_cosinew 5folds LabelSmoothingLoss 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb4_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b4_ns \
#                             MODEL.INPUT_SIZE.HEIGHT=456 \
#                             MODEL.INPUT_SIZE.WIDTH=456 \
#                             MODEL.OUTPUT_SIZE.HEIGHT=456 \
#                             MODEL.OUTPUT_SIZE.WIDTH=456 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4 \
#                             TRAINER.accumulate_grad_batches=1 \
#                             LOSS.PARAMS.smoothing=0.3

# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_vit_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=vit_base_patch16_384_adam_cosinew \
#                             MODEL.MODEL_NAME=vit_base_patch16_384 \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4

# deit_base_patch16_224 deit_base_patch16_384
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_vit_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=deit_base_patch16_384_adam_cosinew \
#                             MODEL.MODEL_NAME=deit_base_patch16_384 \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4

# later 
# seresnext101_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3 accumulate_grad_batches=4
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=seresnext101_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=seresnext101_32x4d \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4

# tf_efficientnetb3_ns_adam_cosinew 5folds LabelSmoothingLoss 0.3 
#LR 1e-4 to 5e-5
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                           loss=label_smoothing_loss \
#                            +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
#                             MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=5e-5


# resnext101_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=resnext101_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=resnext101_32x4d \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-3


######################################################### New Data #######################################
# epoch10
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_vit_adam_cosinew_data_v2 \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=vit_base_patch16_384_adam_cosinew \
#                             MODEL.MODEL_NAME=vit_base_patch16_384 \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4


# # deit_base_patch16_224 deit_base_patch16_384 10 epochs
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_vit_adam_cosinew_data_v2 \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=deit_base_patch16_384_adam_cosinew \
#                             MODEL.MODEL_NAME=deit_base_patch16_384 \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
                            # TRAIN.LR=1e-4


# resnext50_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew_data_v2 \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=resnext50_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=resnext50_32x4d \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=16 \
#                             TRAIN.LR=1e-4


# tf_efficientnetb3_ns_adam_cosinew 5folds LabelSmoothingLoss 0.3 
python -m src.tools.train augmentation=albumentations_classification_rrc \
                          loss=label_smoothing_loss \
                           +experiment=classification_adam_cosinew_data_v2 \
                            CALLBACK.LOGGER.COMMET.FLAG=false \
                            DATA.N_FOLD=5 \
                            MODEL_ID=tf_efficientnetb3_ns_adam_cosinew \
                            MODEL.MODEL_NAME=tf_efficientnet_b3_ns \
                            LOSS.PARAMS.smoothing=0.3 \
                            TRAIN.MONITOR=val_score \
                            TRAIN.MODE=max \
                            TRAIN.BATCH_SIZE=8 \
                            TRAIN.LR=1e-4


# later from fold1
# resnext101_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew_data_v2 \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=resnext101_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=resnext101_32x4d \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-3


# seresnext101_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew_data_v2 \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=seresnext101_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=seresnext101_32x4d \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=8 \
#                             TRAIN.LR=1e-4

# seresnext50_32x4d_adam_cosinew 5folds LabelSmoothingLoss 0.3
# python -m src.tools.train augmentation=albumentations_classification_rrc \
#                             loss=label_smoothing_loss \
#                             +experiment=classification_adam_cosinew_data_v2 \
#                             CALLBACK.LOGGER.COMMET.FLAG=true \
#                             DATA.N_FOLD=5 \
#                             MODEL_ID=seresnext50_32x4d_adam_cosinew \
#                             MODEL.MODEL_NAME=seresnext50_32x4d \
#                             LOSS.PARAMS.smoothing=0.3 \
#                             TRAIN.MONITOR=val_score \
#                             TRAIN.MODE=max \
#                             TRAIN.BATCH_SIZE=16 \
#                             TRAIN.LR=1e-4