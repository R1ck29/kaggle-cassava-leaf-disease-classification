import logging
import os
from typing import Any, Dict, Optional

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from src.data.generator.classification.build import get_training_datasets
from src.models.modeling.classification.get_model import get_model
from src.models.utils.classification.metrics import get_pedestrian_metrics
from src.utils.common import load_obj
from src.utils.pytorch.utils import AverageMeter, to_scalar
from torch.nn.utils import clip_grad_norm_


class LightningClassification(pl.LightningModule):
    #TODO: fix args for load_from_checkpoint
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig, fold: int, pretrained_weights: Optional[bool] = True):
    # def __init__(self, cfg: Optional[DictConfig]=None, fold: Optional[int]=0, pretrained_weights: Optional[bool]=None):
    # def __init__(self, cfg: DictConfig, fold: int, pretrained_weights: bool):
        super(LightningClassification, self).__init__()
        self.hparams = hparams
        self.cfg = cfg
        self.pretrained_weights = pretrained_weights
        self.fold = fold
        self.mode = 'train'
        self.model = get_model(self.cfg, mode=self.mode, pretrained_weights=self.pretrained_weights)
        self.summary_loss = AverageMeter()
        self.metric = accuracy #pl.metrics.Accuracy()
        self.gt_all = []
        self.preds_all = []
        self.hydra_cwd = HydraConfig.get().run.dir
        self.weight_dir = hydra.utils.to_absolute_path(self.hydra_cwd)
        self.best_loss = 10**5
        self.best_score = float(-np.inf) #0.0
        self.running_loss = None
        self.loss_sum = 0
        self.sample_num = 0
        self.log = logging.getLogger(__name__)


    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)


    def prepare_data(self):
        datasets = get_training_datasets(self.cfg, self.fold)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']
        if 'CEL_Sigmoid' in self.cfg.LOSS.CLASS_NAME:
            self.sample_weight =  self.get_sample_weight(self.train_dataset.n_classes)

    
    def get_sample_weight(self, num_classes):
        train = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.DATA.CSV_PATH), dtype={'image_id': str, 'class_id': str})
        labels = []
        for idx in range(len(train)):
            item = train.iloc[idx]
            target = np.zeros(num_classes)
            for cls in item.class_id.split():
                target[int(cls)] = 1
            labels.append(target)
        labels = np.array(labels)
        # labels = train_set.label
        sample_weight = labels.mean(0)
        return sample_weight


    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            num_workers=self.cfg.SYSTEM.NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            num_workers=self.cfg.SYSTEM.NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
        )
        return valid_loader

    def configure_optimizers(self):
        if 'SGD' in self.cfg.OPTIMIZER.CLASS_NAME and 'resnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            lr_ft = 0.01
            lr_new = 0.1
            self.param_groups = [{'params': self.model.module.finetune_params(), 'lr': lr_ft},
                {'params': self.model.module.fresh_params(), 'lr': lr_new}]
            optimizer = load_obj(self.cfg.OPTIMIZER.CLASS_NAME)(self.param_groups, momentum=self.cfg.OPTIMIZER.PARAMS.momentum, weight_decay=self.cfg.OPTIMIZER.PARAMS.weight_decay, nesterov=self.cfg.OPTIMIZER.PARAMS.nesterov)
        else:
            optimizer = load_obj(self.cfg.OPTIMIZER.CLASS_NAME)(self.model.parameters(), **self.cfg.OPTIMIZER.PARAMS)
        scheduler = load_obj(self.cfg.SCHEDULER.CLASS_NAME)(optimizer, **self.cfg.SCHEDULER.PARAMS)

        return [optimizer], [{'scheduler': scheduler,
                              'interval': self.cfg.SCHEDULER.STEP,
                              'monitor': self.cfg.SCHEDULER.MONITOR}]

    # def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
    #                                second_order_closure, on_tpu, using_native_amp, using_lbfgs):
    #                 optimizer.step()

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        if 'resnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME or 'efficientnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            if 'CEL_Sigmoid' in self.cfg.LOSS.CLASS_NAME:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)(self.sample_weight)
            elif 'TaylorCrossEntropyLossv2' in self.cfg.LOSS.CLASS_NAME:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)(n=2, smoothing=0.2, num_classes=self.cfg.DATA.NUM_CLASSES)
            elif 'LabelSmoothingLoss' in self.cfg.LOSS.CLASS_NAME:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)(classes=self.cfg.DATA.NUM_CLASSES, smoothing=self.cfg.LOSS.PARAMS.smoothing)
            else:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)()

            images = images.cuda()
            targets = targets.cuda()

            if 'efficientnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
                logits = self.model(images) 
            else:
                logits = self.model(images, targets)
                
            loss = criterion(logits, targets)

            if self.cfg.MODEL.CLIP_GRAD_NORM:
                clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # make larger learning rate works
            
            if 'AttrDataset' in self.cfg.DATASET.CLASS_NAME:
                self.summary_loss.update(to_scalar(loss))
                summary_loss = torch.tensor(self.summary_loss.avg, dtype=torch.float).to('cuda')

                self.gt_all.append(targets.cpu().numpy())
                probs = torch.sigmoid(logits)
                self.preds_all.append(probs.detach().cpu().numpy())
                log = {'train_loss': loss}#, 'loss_meter': self.summary_loss.val}

                return {'loss': loss, 'train_loss_avg': summary_loss, 'log': log}#, 'progress_bar': float(self.summary_loss.val)}

            elif 'CustomDataset' in self.cfg.DATASET.CLASS_NAME:
                if self.running_loss is None:
                    self.running_loss = loss.item()
                else:
                    self.running_loss = self.running_loss * .99 + loss.item() * .01

                score = self.metric(logits.argmax(1), targets)

                log = {'train_loss': loss, 'running_loss': self.running_loss, 'train_accuracy': score}
                
                return {'loss': loss, 'train_running_loss': self.running_loss, 'logits': logits, 'target': targets, 'log': log, 'train_accuracy': score}
            else:
                raise NotImplementedError(f'Dataset class: {self.cfg.DATASET.CLASS_NAME} is not defined.')

            # self.summary_loss.update(loss.detach().item(), self.cfg.TRAIN.BATCH_SIZE)
            # summary_loss = torch.tensor(self.summary_loss.avg, dtype=torch.float).to('cuda')

            # train_loss = loss_meter.avg
            # gt_label = np.concatenate(gt_all, axis=0)
            # preds_all = np.concatenate(preds_all, axis=0)

            # self.log.info(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')
            # return {'loss': loss, 'train_loss_avg': summary_loss, 'log': log}#, 'progress_bar': float(self.summary_loss.val)}

        else:
            raise NotImplementedError(f'Model class: {self.cfg.MODEL.BACKBONE.CLASS_NAME} is not defined.')


    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        if 'resnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME or 'efficientnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            if 'CEL_Sigmoid' in self.cfg.LOSS.CLASS_NAME:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)(self.sample_weight)
            elif 'TaylorCrossEntropyLossv2' in self.cfg.LOSS.CLASS_NAME:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)(n=2, smoothing=0.2, num_classes=self.cfg.DATA.NUM_CLASSES)
            elif 'LabelSmoothingLoss' in self.cfg.LOSS.CLASS_NAME:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)(classes=self.cfg.DATA.NUM_CLASSES, smoothing=self.cfg.LOSS.PARAMS.smoothing)
            else:
                criterion = load_obj(self.cfg.LOSS.CLASS_NAME)()
            
            images = images.cuda()
            targets = targets.cuda()
            
            logits = self.model(images)

            if 'AttrDataset' in self.cfg.DATASET.CLASS_NAME:
                self.gt_all.append(targets.cpu().numpy())
                targets[targets == -1] = 0
            elif 'CustomDataset' in self.cfg.DATASET.CLASS_NAME:
                self.preds_all += [torch.argmax(logits, 1).detach().cpu().numpy()]
                self.gt_all += [targets.detach().cpu().numpy()]
            else:
                raise NotImplementedError(f'Dataset class: {self.cfg.DATASET.CLASS_NAME } is not defined.')

            loss = criterion(logits, targets)

            if 'AttrDataset' in self.cfg.DATASET.CLASS_NAME:
                probs = torch.sigmoid(logits)
                self.preds_all.append(probs.cpu().numpy())            
                self.summary_loss.update(to_scalar(loss))
                summary_loss = torch.tensor(self.summary_loss.val, dtype=torch.float).to('cuda')
                summary_loss_avg = torch.tensor(self.summary_loss.avg, dtype=torch.float).to('cuda')
                #TODO: self.summary_loss.avg -> loss?
                return {'val_loss': loss, 'val_loss_avg': summary_loss_avg, 'val_summary_loss': summary_loss}
            elif 'CustomDataset' in self.cfg.DATASET.CLASS_NAME:
                self.loss_sum += loss.item()*targets.shape[0]
                self.sample_num += targets.shape[0]
                val_loss = self.loss_sum / self.sample_num
                score = self.metric(logits.argmax(1), targets)
                logs = {'val_loss': loss, 'val_accuracy': score}
                return {'val_loss': loss, 'val_summary_loss': torch.tensor(val_loss), 'logits': logits, 'target': targets, 'val_accuracy': score, 'log': logs}
            else:
                raise NotImplementedError(f'Dataset class: {self.cfg.DATASET.CLASS_NAME } is not defined.')

        else:
            raise NotImplementedError(f'Model class: {self.cfg.MODEL.BACKBONE.CLASS_NAME} is not defined.')


    def validation_epoch_end(self, outputs):
        if 'resnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME or 'efficientnet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            #TODO: not same as original
            val_summary_loss = torch.stack([x['val_summary_loss'] for x in outputs]).mean()
            self.log.info(f'val_loss_mean: {val_loss_mean:.4f}')
            self.log.info(f'val_summary_loss: {val_summary_loss:.4f}')

            if 'AttrDataset' in self.cfg.DATASET.CLASS_NAME:
                val_summary_loss_avg = torch.stack([x['val_loss_avg'] for x in outputs]).mean()

            valid_gt_old = np.concatenate(self.gt_all, axis=0)
            valid_pred_old = np.concatenate(self.preds_all, axis=0)
            valid_gt = torch.cat([x['target'] for x in outputs])
            valid_pred = torch.cat([x['logits'] for x in outputs])

            if 'AttrDataset' in self.cfg.DATASET.CLASS_NAME:
                valid_result = get_pedestrian_metrics(valid_gt, valid_pred)

                print(f'Evaluation on test set, \n',
                    'mA: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                        valid_result.instance_f1))
                cur_metric = valid_result.ma
                metric_name = 'mA'
            elif 'CustomDataset' in self.cfg.DATASET.CLASS_NAME:
                cur_metric = (valid_pred_old==valid_gt_old).mean()
                score = self.metric(valid_pred.argmax(1), valid_gt)
                metric_name = 'Accuracy'
                self.log.info('validation multi-class accuracy = {:.4f}'.format(cur_metric))
                self.log.info('validation lightning metrics accuracy = {:.4f}'.format(score))
            else:
                raise NotImplementedError(f'Dataset class: {self.cfg.DATASET.CLASS_NAME } is not defined.')

            if score > self.best_score:
                self.best_score = score
                ckpt_model_name = self.weight_dir + f'/best_val_score_fold{self.fold}.pth'
                torch.save(self.model.state_dict(), ckpt_model_name)
                self.log.info(f'Best {metric_name} found: {self.best_score}')
                self.log.info(f'Best {metric_name} weight saved to: {ckpt_model_name}')

            if val_loss_mean < self.best_loss:
                self.best_loss = val_loss_mean
                ckpt_model_name_loss = self.weight_dir + f'/best_val_loss_fold{self.fold}.pth'
                torch.save(self.model.state_dict(), ckpt_model_name_loss)
                self.log.info(f'Best Loss found: {self.best_loss}')
                self.log.info(f'Best Loss weight saved to: {ckpt_model_name_loss}')
            
            best_score = torch.as_tensor(self.best_score, dtype=torch.float).to('cuda')

            if 'AttrDataset' in self.cfg.DATASET.CLASS_NAME:
                tensorboard_logs = {'val_loss': val_loss_mean, 'val_summary_loss_avg': val_summary_loss_avg, 'val_summary_loss': val_summary_loss_val, 'val_score': best_score}
                return {'val_loss': val_loss_mean, 'val_summary_loss_avg': val_summary_loss_avg, 'val_summary_loss': val_summary_loss_val, 'val_score': best_score, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
            elif 'CustomDataset' in self.cfg.DATASET.CLASS_NAME:
                tensorboard_logs = {'val_loss': val_loss_mean,  'val_summary_loss': val_summary_loss, 'val_score': best_score, 'val_accuracy': cur_metric}
                return {'val_loss': val_loss_mean, 'val_summary_loss': val_summary_loss, 'val_score': best_score, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        else:
            raise NotImplementedError(f'Model class: {self.cfg.MODEL.BACKBONE.CLASS_NAME} is not defined.')
