import os
from os.path import join, dirname
import sys
import pandas as pd

import torch
from pytorch_lightning.core.lightning import LightningModule

sys.path.append(join(dirname(__file__), "../../../.."))
from src.models.modeling.keypoint.higher_hrnet import get_pose_net
from src.models.loss.keypoints import MultiLossFactory
from src.data.generator.keypoint import make_dataloader
from src.utils.common import load_obj
from src.utils.pytorch import AverageMeter

# TODO:モデルによらないラッパー作成
class LightningKeypoint(LightningModule):
    def __init__(self, cfg, data_path, save_dir, fold):
        super().__init__()
        self.cfg = cfg
        self.data_path = data_path
        self.csv_path = os.path.join(data_path, cfg.DATA.DATA_ID, 'split', cfg.DATA.CSV_PATH)
        self.fold = fold
        self.save_dir = save_dir
        
        #Higher HRNet固有
        self.model = get_pose_net(self.cfg, dir_path=dirname(save_dir)+'/../')
        self.loss = MultiLossFactory(self.cfg)
        
        self.df = []
        # initialize loss
        self.val_loss = 0
        self.heatmaps_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.push_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.pull_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        
        self.val_heatmaps_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.val_push_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.val_pull_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        
        self.epoch = 0
        
    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        return make_dataloader(self.cfg, self.csv_path, self.cfg.TRAIN.BATCH_SIZE, is_train=True, fold=self.fold)
    
    def val_dataloader(self):
        return make_dataloader(self.cfg, self.csv_path, self.cfg.TRAIN.BATCH_SIZE, is_train=False, fold=self.fold)
    
    def training_step(self, batch, batch_idx):
        images, heatmaps, masks, joints = batch
        outputs = self.model(images)
        
        heatmaps_losses, push_losses, pull_losses = self.loss(outputs, heatmaps, masks, joints)

        loss = 0
        for idx in range(self.cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                self.heatmaps_loss_meter[idx].update(heatmaps_loss.item(), images.size(0))
                loss = loss + heatmaps_loss
                
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                self.push_loss_meter[idx].update(push_loss.item(), images.size(0))
                loss = loss + push_loss

            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                self.pull_loss_meter[idx].update(pull_loss.item(), images.size(0))
                loss = loss + pull_loss
        
        return {'loss':loss}
    
    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        
        # CSV Logger
        if not torch.cuda.is_available():
            loss_mean = loss_mean.cpu().numpy()
            self.val_loss = self.val_loss.cpu().numpy()
        self.df.append([self.epoch + 1,
                        loss_mean, self.heatmaps_loss_meter[0].avg, self.heatmaps_loss_meter[1].avg,
                        self.push_loss_meter[0].avg, self.pull_loss_meter[0].avg,
                        self.val_loss, self.val_heatmaps_loss_meter[0].avg, self.val_heatmaps_loss_meter[1].avg,
                        self.val_push_loss_meter[0].avg, self.val_pull_loss_meter[0].avg])
        
        pd.DataFrame(self.df,
                     columns=(['Epoch', 'loss', 'hm1_loss', 'hm2_loss',
                               'push_loss', 'pull_loss', 'val_loss', 'val_hm1_loss',
                               'val_hm2_loss', 'val_push_loss', 'val_pull_loss'])).to_csv(
            os.path.join(self.save_dir, 'train_log.csv'), index=False)
        
        self.epoch += 1
        
        # initialize loss
        self.heatmaps_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.push_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.pull_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        
        self.val_heatmaps_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.val_push_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        self.val_pull_loss_meter = [AverageMeter() for _ in range(self.cfg.LOSS.NUM_STAGES)]
        
        return {'loss':loss_mean}
    
    def validation_step(self, batch, batch_idx):
        images, heatmaps, masks, joints = batch
        outputs = self.model(images)

        heatmaps_losses, push_losses, pull_losses = self.loss(outputs, heatmaps, masks, joints)

        loss = 0
        for idx in range(self.cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                self.val_heatmaps_loss_meter[idx].update(heatmaps_loss.item(), images.size(0))
                loss = loss + heatmaps_loss
                
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                self.val_push_loss_meter[idx].update(push_loss.item(), images.size(0))
                loss = loss + push_loss

            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                self.val_pull_loss_meter[idx].update(pull_loss.item(), images.size(0))
                loss = loss + pull_loss
        
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_loss  = val_loss_mean
        
        return {'val_loss':val_loss_mean}
    
    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.OPTIMIZER.CLASS_NAME)(self.model.parameters(), **self.cfg.OPTIMIZER.PARAMS)
        
        scheduler = load_obj(self.cfg.SCHEDULER.CLASS_NAME)(optimizer, **self.cfg.SCHEDULER.PARAMS)
        return [optimizer], [scheduler]