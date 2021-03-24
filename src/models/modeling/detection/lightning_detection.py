import os
from typing import Any, Dict, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.generator.detection.build import get_training_datasets
from src.models.modeling.detection.get_model import get_model
from src.models.utils.detection.coco_eval import CocoEvaluator
from src.models.utils.detection.coco_utils import (_get_iou_types,
                                                   get_coco_api_from_dataset)
from src.utils.common import collate_fn, load_obj
from src.utils.pytorch.utils import AverageMeter
from hydra.core.hydra_config import HydraConfig

class LightningDetection(pl.LightningModule):
    #TODO: fix args for load_from_checkpoint
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig, fold: int, pretrained_weights: Optional[bool] = True):
    # def __init__(self, cfg: Optional[DictConfig]=None, fold: Optional[int]=0, pretrained_weights: Optional[bool]=None):
    # def __init__(self, cfg: DictConfig, fold: int, pretrained_weights: bool):
        super(LightningDetection, self).__init__()
        self.hparams = hparams
        self.cfg = cfg
        self.pretrained_weights = pretrained_weights
        self.fold = fold
        self.mode = 'train'
        self.model = get_model(self.cfg, mode=self.mode, pretrained_weights=self.pretrained_weights)
        self.summary_loss = AverageMeter()
        self.target_res = {}
        self.hydra_cwd = HydraConfig.get().run.dir
        self.weight_dir = hydra.utils.to_absolute_path(self.hydra_cwd)
        self.best_loss = 10**5
        self.best_score = 0.0

    def forward(self, x):#, *args, **kwargs):
        return self.model(x)#, *args, **kwargs)

    def prepare_data(self):
        datasets = get_training_datasets(self.cfg, self.fold)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            num_workers=self.cfg.SYSTEM.NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            num_workers=self.cfg.SYSTEM.NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # prepare coco evaluator
        coco = get_coco_api_from_dataset(valid_loader.dataset)
        iou_types = _get_iou_types(self.model)
        self.coco_evaluator = CocoEvaluator(coco, iou_types)

        return valid_loader

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.OPTIMIZER.PARAMS.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.OPTIMIZER.PARAMS.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.OPTIMIZER.PARAMS.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.OPTIMIZER.CLASS_NAME)(params)

        else:
            optimizer = load_obj(self.cfg.OPTIMIZER.CLASS_NAME)(self.model.parameters(), **self.cfg.OPTIMIZER.PARAMS)
        scheduler = load_obj(self.cfg.SCHEDULER.CLASS_NAME)(optimizer, **self.cfg.SCHEDULER.PARAMS)

        return [optimizer], [{'scheduler': scheduler,
                              'interval': self.cfg.SCHEDULER.STEP,
                              'monitor': self.cfg.SCHEDULER.MONITOR}]

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        if 'EfficientDet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            images = torch.stack(images)
            images = images.to('cuda').float()
            boxes = [target['boxes'].to('cuda').float() for target in targets]
            labels = [target['labels'].to('cuda').float() for target in targets]
            
            self.target_res['bbox'] = boxes
            self.target_res['cls'] = labels
            self.target_res["img_scale"] = torch.tensor([1.0] * self.cfg.TRAIN.BATCH_SIZE, dtype=torch.float).to('cuda')
            # TODO: change x for model
            self.target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * self.cfg.TRAIN.BATCH_SIZE, dtype=torch.float).to('cuda')

            outputs  = self.model(images, self.target_res)
            loss = outputs['loss']

            self.summary_loss.update(loss.detach().item(), self.cfg.TRAIN.BATCH_SIZE)
            summary_loss = torch.tensor(self.summary_loss.avg, dtype=torch.float).to('cuda')
            
            log = {'train_loss': loss, 'custom_loss': summary_loss}
            return {'loss': loss, 'custom_loss': summary_loss, 'log': log}

        else:
            loss_dict = self.model(images, targets)
            # total loss
            losses = sum(loss for loss in loss_dict.values())

            return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch

        if 'EfficientDet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            images = torch.stack(images)
            images = images.to('cuda').float()

            boxes = [target['boxes'].to('cuda').float() for target in targets]
            labels = [target['labels'].to('cuda').float() for target in targets]
            
            self.target_res['bbox'] = boxes
            self.target_res['cls'] = labels
            self.target_res["img_scale"] = torch.tensor([1.0] * self.cfg.TRAIN.BATCH_SIZE, dtype=torch.float).to('cuda')
            self.target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * self.cfg.TRAIN.BATCH_SIZE, dtype=torch.float).to('cuda')

            outputs = self.model(images, self.target_res)
            loss = outputs['loss']

            self.summary_loss.update(loss.detach().item(), self.cfg.TRAIN.BATCH_SIZE)
            summary_loss = torch.tensor(self.summary_loss.avg, dtype=torch.float).to('cuda')

            return {'val_loss': loss, 'val_custom_loss': summary_loss}
        else:
            targets = [{k: v for k, v in t.items()} for t in targets]
            outputs = self.model(images, targets)
            res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
            self.coco_evaluator.update(res)
            return {}


    def validation_epoch_end(self, outputs):
        if 'EfficientDet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            #TODO: calc COCO score
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_custom_loss = torch.stack([x['val_custom_loss'] for x in outputs]).mean()

            if val_loss_mean < self.best_loss:
                self.best_loss = val_loss_mean
                ckpt_model_name = self.weight_dir + f'/best_loss_fold{self.fold}.pth'
                torch.save(self.model.model.state_dict(), ckpt_model_name)
                print(f'Best Loss found: {self.best_loss}')
                print(f'Best Loss weight saved to: {ckpt_model_name}')

            tensorboard_logs = {'val_loss': val_loss_mean, 'val_custom_loss': val_custom_loss}
            return {'val_loss': val_loss_mean, 'val_custom_loss': val_custom_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        else:
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            # coco metric
            metric = self.coco_evaluator.coco_eval['bbox'].stats[0]
            metric = torch.as_tensor(metric)

            if metric > self.best_score:
                self.best_score = metric
                ckpt_model_name = self.weight_dir + f'/best_score_fold{self.fold}.pth'
                torch.save(self.model.state_dict(), ckpt_model_name)
                print(f'Best mAP found: {self.best_score}')
                print(f'Best mAP weight saved to: {ckpt_model_name}')

            tensorboard_logs = {'val_score': metric}
            return {'val_score': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
