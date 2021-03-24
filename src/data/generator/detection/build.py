import os
import shutil
from typing import Dict, Optional, Tuple

import albumentations as A
import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from src.utils.common import load_obj

from hydra.utils import instantiate
import omegaconf


def load_augs(cfg: DictConfig, bbox_params: DictConfig,) -> A.Compose:
    """
    Load albumentations
    Args:
        cfg: model config
        bbox_params: bbox parameters
    Returns:
        composed object
    """
    augs = []
    for a in cfg:
        if a['CLASS_NAME'] == 'albumentations.OneOf':
            small_augs = []
            for small_aug in a['params']:
                # yaml can't contain tuples, so we need to convert manually
                params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                          small_aug['params'].items()}
                aug = load_obj(small_aug['CLASS_NAME'])(**params)
                small_augs.append(aug)
            aug = load_obj(a['CLASS_NAME'])(small_augs)
            aug.p=a['p']
            augs.append(aug)
        else:
            params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                      a['params'].items()}
            aug = load_obj(a['CLASS_NAME'])(**params)
            augs.append(aug)
    if bbox_params is not None:
        transforms = A.Compose(augs, bbox_params=instantiate(bbox_params))
    else:
        transforms = A.Compose(augs)
    return transforms


def get_training_datasets(cfg: DictConfig, val_fold: int, eval_oof: Optional[bool] = False) -> Tuple:
    """ Get datases for modelling

    Args:
        cfg (DictConfig): model config for traiing/prediction
        val_fold (int): fold number for validation dataset
        eval_oof (Optional[bool], optional): for switching box coordinates order. 
                                            should be True when evaluating the model with validation dataset. Defaults to False.

    Returns:
        datasets for training and validation
    """

    train = pd.read_csv(hydra.utils.to_absolute_path(cfg.DATA.CSV_PATH), dtype={'image_id': str})

    train_df = train[train['fold'] != val_fold]
    valid_df = train[train['fold'] == val_fold]

    # for debug training
    if cfg.TRAIN.DEBUG:
        train_df = train_df.head()
        valid_df = valid_df.head()

    print(f'Train df length: {len(train_df)}')
    print(f'Val df length: {len(valid_df)}')

    train_img_dir = f'{hydra.utils.to_absolute_path(cfg.DATA.TRAIN_IMAGE_DIR)}'
    
    # train dataset
    dataset_class = load_obj(cfg.DATASET.CLASS_NAME)

    # initialize augmentations
    print(f'Framework: {(cfg.AUGMENTATION.FRAMEWORK)}')
    if 'albumentations_detection' in cfg.AUGMENTATION.FRAMEWORK:
        train_augs = load_augs(cfg['ALBUMENTATIONS']['TRAIN']['AUGS'], cfg['ALBUMENTATIONS']['TRAIN']['BBOX_PARAMS'])
        valid_augs = load_augs(cfg['ALBUMENTATIONS']['VALID']['AUGS'], cfg['ALBUMENTATIONS']['VALID']['BBOX_PARAMS'])
    elif cfg.AUGMENTATION.FRAMEWORK == 'custom':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if 'FasterRCNNDataset' in cfg.DATASET.CLASS_NAME:        
        train_dataset = dataset_class(dataframe=train_df, mode='train', image_dir=train_img_dir, cfg=cfg, transforms=train_augs)
        valid_dataset = dataset_class(dataframe=valid_df, mode='valid', image_dir=train_img_dir, cfg=cfg, transforms=valid_augs)
    elif'DatasetRetriever' in cfg.DATASET.CLASS_NAME:
        train_dataset = dataset_class(dataframe=train, mode='train', image_dir=train_img_dir, cfg=cfg, image_ids=train_df.image_id.unique(), transforms=train_augs, test=False) # train_df.index.values
        if eval_oof:
            valid_dataset = dataset_class(dataframe=train, mode='eval_oof', image_dir=train_img_dir, cfg=cfg, image_ids=valid_df.image_id.unique(), transforms=valid_augs, test=True) # valid_df.index.values
        else:
            valid_dataset = dataset_class(dataframe=train, mode='valid', image_dir=train_img_dir, cfg=cfg, image_ids=valid_df.image_id.unique(), transforms=valid_augs, test=True) # valid_df.index.values
    else:
        print('Dataset Class name is not Defined.')
        raise NotImplementedError

    return {'train': train_dataset, 'valid': valid_dataset}


def get_test_dataset(cfg: DictConfig, test_df: Optional[pd.DataFrame] = None, test_cfg: Optional[DictConfig] = None) -> object:
    """
    Get test dataset

    Args:
        cfg (DictConfig): model config for prediction
        test_df (Optional[pd.DataFrame], optional): dataframe for prediction. If the dataframe is set, use it first. Defaults to None.
        test_cfg (DictConfig): test config for prediction

    Returns:
        test_dataset (Dataset): Pytorch dataset for prediction
    """
    if test_cfg is not None:
        test_img_dir = test_cfg.TEST.TEST_IMAGE_DIR
    else:
        test_img_dir = cfg.TEST.TEST_IMAGE_DIR
    test_img_dir = hydra.utils.to_absolute_path(test_img_dir)

    if test_df is None:
        if test_cfg is not None:
            test_csv_path = test_cfg.TEST.TEST_CSV_PATH
        else:
            test_csv_path = cfg.TEST.TEST_CSV_PATH
        test_df = pd.read_csv(hydra.utils.to_absolute_path(test_csv_path), dtype={'image_id': str})
    else:
        print('Using test dataset from args.')

    print(f'Framework: {(cfg.AUGMENTATION.FRAMEWORK)}')
    if 'albumentations_detection' in cfg.AUGMENTATION.FRAMEWORK:
        print('Using transforms defined in yaml config file.')
        if 'xmin' in test_df.columns and 'ymin' in test_df.columns and 'xmax' in test_df.columns and 'ymax' in test_df.columns:
            test_augs = load_augs(cfg['ALBUMENTATIONS']['TEST']['AUGS'], cfg['ALBUMENTATIONS']['TEST']['BBOX_PARAMS'])
        else:
            test_augs = load_augs(cfg['ALBUMENTATIONS']['TEST']['AUGS'], bbox_params=None)
            print('Applying augmentation for No GT.')
    elif cfg.AUGMENTATION.FRAMEWORK == 'custom':
        raise NotImplementedError
    else:
        raise NotImplementedError
        
    dataset_class = load_obj(cfg.DATASET.CLASS_NAME)

    if 'FasterRCNNDataset' in cfg.DATASET.CLASS_NAME:
        if 'xmin' in test_df.columns and 'ymin' in test_df.columns and 'xmax' in test_df.columns and 'ymax' in test_df.columns:
            mode = 'valid'
        else:
            mode = 'test'
        test_dataset = dataset_class(dataframe=test_df, mode=mode, image_dir=test_img_dir, cfg=cfg, transforms=test_augs)
    elif'DatasetRetriever' in cfg.DATASET.CLASS_NAME:
        if 'xmin' in test_df.columns and 'ymin' in test_df.columns and 'xmax' in test_df.columns and 'ymax' in test_df.columns:
            mode = 'eval_oof'
        else:
            mode = 'test'
        test_dataset = dataset_class(dataframe=test_df, mode=mode, image_dir=test_img_dir, cfg=cfg, image_ids=test_df.image_id.unique(), transforms=test_augs, test=False) # valid_df.index.values
    else:
        raise NotImplementedError
    return test_dataset
