import os
import random
from glob import glob
from typing import Any, Dict, Generator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateLogger,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from src.utils.pytorch.loggers import JsonLogger
from src.utils.common import flatten_omegaconf
from src.utils.common import collate_fn
from src.models.modeling.classification.get_model import get_model


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def set_seed(cfg: DictConfig) -> None:
    """set all seed for deterministic
    Args:
        cfg (DictConfig): config values 
    
    Raises:
        ValueError: if the cuDNN settings not properly set
    """
    seed = cfg.SYSTEM.SEED

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    if not cfg.SYSTEM.CUDNN.BENCHMARK:
        torch.backends.cudnn.benchmark = cfg.SYSTEM.CUDNN.BENCHMARK
    else:
        raise ValueError(f'To reproduce results, "cfg.SYSTEM.CUDNN.BENCHMARK" must be False')
    if cfg.SYSTEM.CUDNN.DETERMINISTIC:
        torch.backends.cudnn.deterministic = cfg.SYSTEM.CUDNN.DETERMINISTIC
    else:
        raise ValueError(f'To reproduce results, "cfg.SYSTEM.CUDNN.DETERMINISTIC" must be True')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_model(cfg: DictConfig, fold: int, data_path: Optional[str] = None, output_path: Optional[str] = None) -> Any:
    """実験用モデル(src/tools/train.pyで使用)を返す。

    Args:
        cfg (CfgNode): config
        dire_path (str): 出力先のパス
        
    Returns:
        model (LightningModule): lightningでラップされたPytorchモデル
    """
    print('-'*30, 'Task :', cfg.TASK, '-'*30)

    if cfg.TASK == 'classification':
        from src.models.modeling.classification.lightning_classification import LightningClassification
        hparams = flatten_omegaconf(cfg)
        model = LightningClassification(hparams=hparams, cfg=cfg, fold=fold)
        
    elif cfg.TASK == 'detection':
        from src.models.modeling.detection.lightning_detection import LightningDetection
        hparams = flatten_omegaconf(cfg)
        model = LightningDetection(hparams=hparams, cfg=cfg, fold=fold)
        
    elif cfg.TASK == 'segmentation':
        raise NotImplementedError
        
    elif cfg.TASK == 'keypoint':
        from src.models.modeling.keypoint import LightningKeypoint
        model = LightningKeypoint(cfg, data_path, output_path, fold)
        
    elif cfg.TASK == 'seg_pytorch':
        from src.models.modeling.seg_pytorch import LightningSegPytorch
        model = LightningSegPytorch(cfg, data_path, output_path, fold)
        
    return model

def load_predictor(cfg: DictConfig, model_cfg: DictConfig) -> Any:
    """実験用モデル(src/tools/train.pyで使用)を返す。

    Args:
        cfg (DictConfig): テスト時のconfig
        mdel_cfg (DictConfig): モデル学習時のconfig
        
    Returns:
        model (LightningModule): predictor
    """
    if cfg.TASK == 'classification':
        from src.models.predictor.classification.predictor import Predictor
    if cfg.TASK == 'detection':
        from src.models.predictor.detection.detector import Predictor
    elif cfg.TASK == 'keypoint':
        from src.models.predictor.keypoint import Predictor        
    elif cfg.TASK == 'seg_pytorch':
        from src.models.predictor.seg_pytorch import Predictor
        
    predictor = Predictor(cfg, model_cfg)
    return predictor


def save_model(cfg: DictConfig, ckpt_dir:str, fold:int) -> None:
    """実験用モデル(訓練済み)を重みのみ保存する。

    Args:
        cfg (CfgNode): config
        ckpt_dir (str): 訓練済みモデルが保存されているディレクトリ
        fold: 対象のfold
        
    Returns:
        None
    """        
    ckpt_path = glob(ckpt_dir + f'/fold{fold}*.ckpt')

    if len(ckpt_path) == 1:
        ckpt_path = ckpt_path[0]
        print(f'Found Pretrained Weight : {ckpt_path}')
    elif len(ckpt_path) > 1:
        print(f'There are more than one weight file found : {ckpt_path}')
    else:
        print(f'Weight file not found : {ckpt_path}')
        
    if cfg.TASK == 'classification':
        from src.models.modeling.classification.lightning_classification import LightningClassification
        model = LightningClassification.load_from_checkpoint(checkpoint_path=str(ckpt_path), cfg=cfg, fold=fold, pretrained_weights=False)

        # save as a simple torch model
        ckpt_model_name = ckpt_path.replace('ckpt','pth')

        print(f'Ckpt Model saved to : {ckpt_model_name}')
        torch.save(model.model.state_dict(), ckpt_model_name)

    elif cfg.TASK == 'detection':
        from src.models.modeling.detection.lightning_detection import LightningDetection
        model = LightningDetection.load_from_checkpoint(checkpoint_path=str(ckpt_path), cfg=cfg, fold=fold, pretrained_weights=False)

        # save as a simple torch model
        ckpt_model_name = ckpt_path.replace('ckpt','pth')

        print(f'Ckpt Model saved to : {ckpt_model_name}')
        torch.save(model.model.state_dict(), ckpt_model_name)

    elif cfg.TASK == 'keypoint':
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        
        for key in list(state_dict.keys()):
            state_dict[key.split('model.')[1]] = state_dict[key]
            del state_dict[key]
        # save as a simple torch model
        ckpt_model_name = ckpt_path.replace('ckpt','pth')
        torch.save(state_dict, ckpt_model_name)

    
def get_callback(cfg:DictConfig, output_path:str, fold:int) -> Any:
    """実験用コールバック関数(src/tools/train.pyで使用)を返す。

    Args:
        cfg (CfgNode): config
        output_path (str): 出力先のパス
        fold (int): fold番号
        
    Returns:
        logger (pytorch_lightning.loggers): pytorch lightningで用意されているLogging関数(自作も可)
        checkpoint (ModelCheckpoint): pytorch lightningにおけるモデルの保存設定
    """
    
    loggers = []
    if cfg.CALLBACK.LOGGER.COMMET.FLAG:
        print(f'Comet Logger: {cfg.CALLBACK.LOGGER.COMMET.FLAG}')
        comet_logger = CometLogger(save_dir=cfg.CALLBACK.LOGGER.COMMET.SAVE_DIR,
                                workspace=cfg.CALLBACK.LOGGER.COMMET.WORKSPACE,
                                project_name=cfg.CALLBACK.LOGGER.COMMET.PROJECT_NAME,
                                api_key=cfg.PRIVATE.COMET_API,
                                experiment_name=os.getcwd().split('\\')[-1])
        loggers.append(comet_logger)
    
    if cfg.CALLBACK.LOGGER.JSON:
        print(f'Json Logger : {cfg.CALLBACK.LOGGER.JSON}')
        json_logger = JsonLogger()
        loggers.append(json_logger)

    tb_logger = TensorBoardLogger(save_dir=output_path)
    loggers.append(tb_logger)

    lr_logger = LearningRateLogger()

    monitor_name = cfg.CALLBACK.MODEL_CHECKPOINT.PARAMS.monitor
    if monitor_name == 'val_loss':
        model_checkpoint = ModelCheckpoint(filepath=output_path + f'/fold{fold}' + '_{epoch}_{val_loss:.3f}',
                                       **cfg.CALLBACK.MODEL_CHECKPOINT.PARAMS)
    elif monitor_name == 'val_score':
        model_checkpoint = ModelCheckpoint(filepath=output_path + f'/fold{fold}' + '_{epoch}_{val_score:.3f}',
                                       **cfg.CALLBACK.MODEL_CHECKPOINT.PARAMS)
    else:
        model_checkpoint = ModelCheckpoint(filepath=output_path + f'/fold{fold}' + '_{epoch}_{other_metric:.3f}',
                                        **cfg.CALLBACK.MODEL_CHECKPOINT.PARAMS)
    
    print(f'Early stopping: {cfg.CALLBACK.EARLY_STOPPING.FLAG}')                                   
    if cfg.CALLBACK.EARLY_STOPPING.FLAG:
        early_stopping = EarlyStopping(**cfg.CALLBACK.EARLY_STOPPING.PARAMS)
    else:
        early_stopping = False

    return loggers, lr_logger, model_checkpoint, early_stopping

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0



def get_test_data_loader(cfg, model_cfg, test_df=None):
    if cfg.TASK == 'detection':
        from src.data.generator.detection.build import get_test_dataset
        test_dataset = get_test_dataset(model_cfg, test_df=test_df, test_cfg=cfg)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=model_cfg.SYSTEM.NUM_WORKERS,
            shuffle=False,
            collate_fn=collate_fn
        )

    elif cfg.TASK == 'classification':
        from src.data.generator.classification.build import get_test_dataset
        test_dataset = get_test_dataset(model_cfg, test_df=test_df, test_cfg=cfg)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=model_cfg.SYSTEM.NUM_WORKERS,
            shuffle=False,
            collate_fn=collate_fn
        )
        
    elif cfg.TASK == 'keypoint':
        from src.data.generator.keypoint import make_test_dataloader
        df = pd.read_pickle(os.path.join('../../../../../data', model_cfg.DATA.DATA_ID, 'split', model_cfg.DATA.CSV_PATH))
        test_dataloader = make_test_dataloader(df, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.SYSTEM.NUM_WORKERS)
        
    elif cfg.TASK == 'seg_pytorch':
        from src.data.generator.seg_pytorch import make_dataloader
        model_cfg.TEST.BATCH_SIZE = 1
        test_dataloader = make_dataloader(model_cfg, 'test', cfg.DATA.DATA_ID, pred_flg=True)
    
    return test_dataloader


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


    
def fuse_bn_sequential(block):
    """
    This function takes a sequential block and fuses the batch normalization with convolution
    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    if not isinstance(block, nn.Sequential):
        return block
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

    return model


def to_scalar(vt):
    """
    preprocess a 1-length pytorch Variable or Tensor to scalar
    """
    # if isinstance(vt, Variable):
    #     return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        if vt.dim() == 0:
            return vt.detach().cpu().numpy().flatten().item()
        else:
            return vt.detach().cpu().numpy()
    elif isinstance(vt, np.ndarray):
        return vt
    else:
        raise TypeError('Input should be a ndarray or tensor')