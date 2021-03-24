import os
import torch.utils.data

from .Keypoint_COCO import COCOKeypoints as coco
from .Keypoint_MPII import MPIIKeypoints as mpii
from .Keypoint import EvalDataset
from ...transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator


def build_dataset(cfg, csv_path, is_train, fold=None, eval_mode=False):
    if cfg.DATA.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.MODEL.NUM_JOINTS, cfg.DATA.SIGMA
        ) for output_size in cfg.MODEL.OUTPUT_SIZE
    ]
    
    joints_generator = [
        JointsGenerator(
            cfg.DATA.MAX_NUM_PEOPLE,
            cfg.MODEL.NUM_JOINTS,
            output_size,
            cfg.MODEL.TAG_PER_JOINT
        ) for output_size in cfg.MODEL.OUTPUT_SIZE
    ]
    
    if not eval_mode:
        transforms = build_transforms(cfg, is_train)
    else:
        transforms = None
    
    dataset = eval(cfg.DATA.FORMAT)(cfg, csv_path, is_train, heatmap_generator, joints_generator, transforms, fold)    
    return dataset


def make_dataloader(cfg, csv_path, batch_size, is_train=True, fold=None, eval_mode=False):
    if is_train:
        shuffle = True
    else:
        shuffle = False
    
    dataset = build_dataset(cfg, csv_path, is_train, fold=fold, eval_mode=eval_mode)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=True
    )

    return data_loader

def make_test_dataloader(df, batch_size, num_workers, pin_memory=False):
    dataset = EvalDataset(df)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return data_loader