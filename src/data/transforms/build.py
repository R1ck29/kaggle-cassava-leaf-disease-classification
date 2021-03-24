from . import transforms as T
from .transforms import *

#hydra使用時
import omegaconf

FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'MPII':[
        5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10
    ],
    'MPII_WITH_CENTER': [
        5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10, 16
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}


def build_transforms(cfg, is_train=True):
    #assert is_train is True, 'Please only use build_transforms for training.'
    assert isinstance(cfg.MODEL.OUTPUT_SIZE, (list, tuple, omegaconf.listconfig.ListConfig)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    input_size = cfg.MODEL.INPUT_SIZE
    output_size = cfg.MODEL.OUTPUT_SIZE
    scale_type = cfg.DATA.SCALE_TYPE
    
    if is_train:
        max_rotation = cfg.AUGMENTATION.MAX_ROTATION
        min_scale = cfg.AUGMENTATION.MIN_SCALE
        max_scale = cfg.AUGMENTATION.MAX_SCALE
        max_translate = cfg.AUGMENTATION.MAX_TRANSLATE
        flip = cfg.AUGMENTATION.FLIP
    else:
        max_rotation = 0
        min_scale = 1
        max_scale = 1
        max_translate = 0
        flip = 0
        
    #TODO; カスタムのflip_configを受け入れる
    if 'coco' in cfg.DATA.FORMAT:
        dataset_name = 'COCO'
    elif 'mpii' in cfg.DATA.FORMAT:
        dataset_name = 'MPII'
    elif 'crowd_pose' in cfg.DATA.FORMAT:
        dataset_name = 'CROWDPOSE'
    else:
        raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATA.FORMAT)
    if cfg.DATA.WITH_CENTER:
        coco_flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']
    else:
        coco_flip_index = FLIP_CONFIG[dataset_name]
        
    transforms = T.Compose(
        [
            T.RandomPhotometricAugmentation(output_size),
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate,
                scale_aware_sigma=cfg.DATA.SCALE_AWARE_SIGMA
            ),
            T.RandomHorizontalFlip(coco_flip_index, output_size, flip),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    return transforms


def get_composed_augmentations(cfg):
    
    augmentation = [RandomHorizontallyFlip(cfg.AUGMENTATION.hflip), 
                    RandomScaleCrop(cfg.AUGMENTATION.rscale_crop)
                   ]
    
    return AugCompose(augmentation)