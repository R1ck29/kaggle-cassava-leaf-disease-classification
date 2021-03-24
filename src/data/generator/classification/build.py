import os
import shutil
from typing import Dict, Optional, Tuple

import albumentations as A
import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from src.utils.common import load_obj
from src.data.generator.classification.dataset import get_transform

from hydra.utils import instantiate
import omegaconf


def load_augs(cfg: DictConfig, bbox_params: Optional[DictConfig] = None) -> A.Compose:
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


def get_training_datasets(cfg: DictConfig, val_fold: int, eval_oof: Optional[bool] = False, 
                        train_augs: Optional[A.core.composition.Compose] = None, valid_augs: Optional[A.core.composition.Compose] = None) -> Tuple:
    """ Get datases for modelling

    Args:
        cfg (DictConfig): model config for traiing/prediction
        val_fold (int): fold number for validation dataset
        eval_oof (Optional[bool], optional): for switching box coordinates order. 
                                            should be True when evaluating the model with validation dataset. Defaults to False.

    Returns:
        datasets for training and validation
    """

    train = pd.read_csv(hydra.utils.to_absolute_path(cfg.DATA.CSV_PATH), dtype={'image_id': str, 'class_id': str})

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
    if 'albumentations_classification' in cfg.AUGMENTATION.FRAMEWORK:
        if train_augs is not None:
            train_augs = train_augs
        else:
            train_augs = load_augs(cfg['ALBUMENTATIONS']['TRAIN']['AUGS'])
        
        if valid_augs is not None:
            valid_augs = valid_augs
        else:
            valid_augs = load_augs(cfg['ALBUMENTATIONS']['VALID']['AUGS'])
    elif cfg.AUGMENTATION.FRAMEWORK == 'custom':
        train_augs, valid_augs = get_transform(cfg)
    else:
        raise NotImplementedError

    print('Train \n', train_augs)
    print('Valid \n', valid_augs)

    if'AttrDataset' in cfg.DATASET.CLASS_NAME:
        train_dataset = dataset_class(dataframe=train, mode='train', image_dir=train_img_dir, cfg=cfg, image_ids=train_df.image_id.unique(), transforms=train_augs)
        valid_dataset = dataset_class(dataframe=train, mode='valid', image_dir=train_img_dir, cfg=cfg, image_ids=valid_df.image_id.unique(), transforms=valid_augs)
    elif 'CustomDataset' in cfg.DATASET.CLASS_NAME:
        train_dataset = dataset_class(train_df, train_img_dir, transforms=train_augs, output_label=True)
        valid_dataset = dataset_class(valid_df, train_img_dir, transforms=valid_augs, output_label=True)
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
    
    print(f'Framework: {(cfg.AUGMENTATION.FRAMEWORK)}')
    if 'albumentations_classification' in cfg.AUGMENTATION.FRAMEWORK:
        print('Using transforms defined in yaml config file.')
        test_augs = load_augs(cfg['ALBUMENTATIONS']['TEST']['AUGS'])
    elif cfg.AUGMENTATION.FRAMEWORK == 'custom':
        _, test_augs = get_transform(cfg)
    else:
        raise NotImplementedError

    if test_df is None:
        if test_cfg is not None:
            test_csv_path = test_cfg.TEST.TEST_CSV_PATH
        else:
            test_csv_path = cfg.TEST.TEST_CSV_PATH
        test_df = pd.read_csv(hydra.utils.to_absolute_path(test_csv_path), dtype={'image_id': str})
    else:
        print('Using test dataset from args.')
        
    dataset_class = load_obj(cfg.DATASET.CLASS_NAME)

    if 'AttrDataset' in cfg.DATASET.CLASS_NAME:
        if 'class_id' in test_df.columns and 'attr_name' in test_df.columns:
            mode = 'eval_oof'
        else:
            mode = 'test'
        test_dataset = dataset_class(dataframe=test_df, mode=mode, image_dir=test_img_dir, cfg=cfg, image_ids=test_df.image_id.unique(), transforms=test_augs)
    elif 'CustomDataset' in cfg.DATASET.CLASS_NAME:
        test_dataset = dataset_class(test_df, test_img_dir, transforms=test_augs, output_label=False)
    else:
        raise NotImplementedError
    return test_dataset
