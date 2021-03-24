import os 
from .seg_Cityscapes import cityscapesDataset
from .seg_Cvat import cvatDataset

from torch.utils import data
from ...transforms import get_composed_augmentations


def build_dataset(cfg, split, data_aug, data_flg='cvat', pred_flg=False, data_path=None):
    
    """
    for building Cityscapes dataloader
        :param cfg is config file
        :split is train or val
    """
    
    if data_path:
        data_path = data_path
        
    else:
        if pred_flg:
            data_path = os.path.join('../../../../../data', cfg.DATA.DATA_ID, 'split', '{}.pkl'.format(cfg.DATA.CSV_PATH))
        else:
            data_path = os.path.join('../../../data', cfg.DATA.DATA_ID, 'split', '{}.pkl'.format(cfg.DATA.CSV_PATH))
            
    if data_flg == 'cvat':
        dataset = cvatDataset(
            cfg=cfg,
            path=data_path,
            split=split,
            augmentations=data_aug
        )
        
    if data_flg == 'cityscapes':
        dataset = cityscapesDataset(
            cfg=cfg,
            path=data_path,
            split=split,
            augmentations=data_aug
        )
        
    return dataset


def make_dataloader(cfg, split, data_flg, pred_flg=False, data_path=None):
    
    """
    for building Cityscapes dataloader
        :param cfg is config file
        :split is train or val
    """
    
    if split == 'train':
        shuffle = True
        batch_size = cfg.TRAIN.BATCH_SIZE
        data_flg = cfg.DATA.DATASET
        data_aug = get_composed_augmentations(cfg)
        
    else:
        shuffle = False
        batch_size = cfg.TEST.BATCH_SIZE
        data_flg = cfg.DATA.DATASET
        data_aug = None
        
    dataset = build_dataset(cfg, split, data_aug, data_flg, pred_flg=pred_flg, data_path=data_path)
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        shuffle=shuffle,
    )
    
    return loader
    
    
    