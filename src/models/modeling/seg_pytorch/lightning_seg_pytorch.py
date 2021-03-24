import os
from os.path import join, dirname
import sys
import pandas as pd

import torch
from pytorch_lightning.core.lightning import LightningModule

sys.path.append(join(dirname(__file__), "../../../.."))
from src.models.modeling.seg_pytorch.hardnet import hardnet
from src.models.loss.seg_pytorch import bootstrapped_cross_entropy2d
from src.data.generator.seg_pytorch import make_dataloader
from src.utils.common import load_obj
from src.models.modeling.seg_pytorch import get_model
from src.models.loss.seg_pytorch import get_loss_function
from src.data.generator.seg_pytorch import get_loader
from src.utils.pytorch.utils import RunningScore, AverageMeter
from src.data.transforms.build import get_composed_augmentations
from src.utils.pytorch.schedulers import *


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

        
class LightningSegPytorch(LightningModule):
    def __init__(self, cfg, data_path, save_dir, fold):
        super().__init__()
        self.cfg = cfg
        self.data_path = data_path
        self.save_dir = save_dir
        
        # hardnetの呼び出し
        self.model = get_model(cfg.MODEL.arch, cfg.MODEL.n_classes).apply(weights_init)
        total_params = sum(p.numel() for p in self.model.parameters())
        print( 'Parameters:',total_params )
        
        # 重みの呼び出し
        weight = torch.load(cfg.MODEL.pretrain)
        self.model.base.load_state_dict(weight)
        
        self.loss_func = get_loss_function(cfg)
        self.val_loss_meter = AverageMeter()
        self.running_metrics_val = RunningScore(cfg.MODEL.n_classes)
        
        self.best_iou, self.epoch_iou, self.epoch_acc = -100, -100, -100
        self.train_loss, self.val_loss = -100, -100
        self.epoch, self.df = 0, []
        
    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        return make_dataloader(self.cfg, 'train', self.cfg.DATA.DATA_ID)
    
    def val_dataloader(self):
        return make_dataloader(self.cfg, 'val', self.cfg.DATA.DATA_ID)
    
    def training_step(self, batch, batch_idx):
        images, labels, _, _ = batch
        outputs = self.model(images)
        loss = self.loss_func(input=outputs, target=labels)
        
        return {'loss':loss}
    
    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_loss = loss_mean.item()
        
        return {'loss':loss_mean}
    
    def validation_step(self, batch, batch_idx):
        images, labels, _, _ = batch
        outputs = self.model(images)
        loss = self.loss_func(input=outputs, target=labels)
        
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        
        self.val_loss_meter.update(loss.detach().item(), self.cfg.TEST.BATCH_SIZE)
        self.running_metrics_val.update(gt, pred)
        
        return {'val_loss':loss}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        score, class_iou = self.running_metrics_val.get_scores()
        
        mean_IoU, mean_Acc = score['Mean IoU : \t'], score['Mean Acc : \t']
        
        if mean_IoU >= self.best_iou:
            self.best_iou = mean_IoU
            
            state = {"epoch": self.epoch,
                     "model_state": self.model.state_dict()}
            
            save_path = os.path.join(self.save_dir,
                                     "state.pkl")
            
            save_pt_path = os.path.join(self.save_dir,
                                     "state.pth")
            
            torch.save(self.model.state_dict(), save_pt_path)
            torch.save(state, save_path)
            
        self.val_loss = val_loss_mean.item()
        self.epoch_iou, self.epoch_acc = mean_IoU, mean_Acc
        
        self.df.append([self.epoch,
                        self.best_iou, self.epoch_iou, self.epoch_acc,
                        self.train_loss, self.val_loss])
        
        pd.DataFrame(self.df,
                     columns=(['Epoch', 'BestMeanIoU', 'EpochMeanIoU', 'EpochMeanACC', 'TrainMeanLoss', 'ValMeanLoss'])).to_csv(os.path.join(self.save_dir, 'train_log.csv'), index=False)
        
        for k, v in score.items():
            print(k, v)
        
        for k, v in class_iou.items():
            print("Class {} IoU: {}".format(k, v))
                
        self.epoch += 1
        self.val_loss_meter.reset()
        self.running_metrics_val.reset()
        
        return {'val_loss':val_loss_mean, 'mean_IoU':mean_IoU, 'mean_Acc':mean_Acc, 'best_IoU':self.best_iou}
    
    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.OPTIMIZER.CLASS_NAME)(self.model.parameters(), **self.cfg.OPTIMIZER.PARAMS)
        
        if self.cfg.SCHEDULER.CLASS_NAME == 'poly_lr':
            train_dataloader = make_dataloader(self.cfg, 'train', self.cfg.DATA.DATA_ID)
            td_len = len(train_dataloader.dataset)
            epochs = self.cfg.TRAIN.EPOCHS
            bs = self.cfg.TRAIN.BATCH_SIZE
            self.cfg.SCHEDULER.PARAMS.max_iter = int(td_len / bs) * epochs
            scheduler = get_scheduler(optimizer, self.cfg)
        
        else:
            scheduler = load_obj(self.cfg.SCHEDULER.CLASS_NAME)(optimizer, **self.cfg.SCHEDULER.PARAMS)
            
        
        return [optimizer], [scheduler]