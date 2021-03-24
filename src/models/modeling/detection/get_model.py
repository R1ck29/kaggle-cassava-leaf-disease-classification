import glob
import sys
import gc
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig

from src.utils.common import load_obj


def get_model(cfg: DictConfig, mode: str, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = '', device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get model

    Args:
        cfg: config
        mode: test or train
        pretrained_weights (bool): flag to load coco or other pretrtained weight 
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """

    if 'fasterrcnn' in cfg.MODEL.BACKBONE.CLASS_NAME:
        model = get_faster_rcnn(cfg, finetuned_weights, device)
    elif 'EfficientDet' in cfg.MODEL.BACKBONE.CLASS_NAME:
        model = get_efficient_det(cfg, mode, pretrained_weights, finetuned_weights, device)
    else:
        print(f'Model is not defined. {cfg.MODEL.BACKBONE.CLASS_NAME}')
        raise NotImplementedError
    return model


def get_faster_rcnn(cfg: DictConfig, finetuned_weights: Optional[str] = '', device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get Faster-RCNN model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    model = load_obj(cfg.MODEL.BACKBONE.CLASS_NAME)
    model = model(**cfg.MODEL.BACKBONE.PARAMS)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(cfg.MODEL.HEAD.CLASS_NAME)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(in_features, cfg.MODEL.NUM_CLASSES)

    if finetuned_weights:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
          
    return model


def get_efficient_det(cfg: DictConfig, mode: str, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = '', device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """Get EfficientDet model

    Args:
        cfg (DictConfig): config for the model
        mode (str): train or test
        pretrained_weights (bool): flag to load coco or other pretrtained weight 
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Raises:
        NotImplementedError: mode should be train or test

    Returns:
        Any: Efficient Det Model
    """
 
    config = load_obj(cfg.MODEL.CONFIG.CLASS_NAME)
    config = config(cfg.MODEL.MODEL_NAME)

    model = load_obj(cfg.MODEL.BACKBONE.CLASS_NAME)
    model = model(config, pretrained_backbone=cfg.MODEL.BACKBONE.PARAMS.PRETRAINED_BACKBONE)

    if pretrained_weights:
        pretrained_weight_path = glob.glob(f'{hydra.utils.to_absolute_path(cfg.MODEL.BACKBONE.PARAMS.WEIGHT_PATH)}/{cfg.MODEL.BASE_NAME}*')
        if len(pretrained_weight_path) > 1:
            print(f'Found too many weight path: {pretrained_weight_path}')
            sys.exit()
        elif len(pretrained_weight_path)==0:
            print(f'Found no weight path: {pretrained_weight_path}')
            sys.exit()
        print(f'Loading pretrained Pretrained model : {pretrained_weight_path[0]}')
        if device == torch.device('cpu'):
            checkpoint = torch.load(pretrained_weight_path[0], map_location='cpu')
        else:
            checkpoint = torch.load(pretrained_weight_path[0])
        model.load_state_dict(checkpoint)
        del checkpoint
        gc.collect()

    config.num_classes = cfg.MODEL.HEAD.PARAMS.NUM_CLASSES
    config.image_size = cfg.MODEL.INPUT_SIZE

    head = load_obj(cfg.MODEL.HEAD.CLASS_NAME)
    model.class_net = head(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    if finetuned_weights:
        if device == torch.device('cpu'):
            checkpoint = torch.load(finetuned_weights, map_location='cpu')
        else:
            checkpoint = torch.load(finetuned_weights)
        # checkpoint = checkpoint['state_dict']
        # handle the case when I forgot to unwrap the bench ...
        if 'backbone.conv_stem.weight' not in checkpoint:
            model_state_dict = {}
            for k,v in checkpoint.items():
                if 'model.' == k[:6]:
                    k = k[6:]
                    model_state_dict[k] = v
        else:
            model_state_dict = checkpoint
        model.load_state_dict(model_state_dict)
        print(f'Loaded Finetuned model: {finetuned_weights}')
        del checkpoint
        gc.collect()
        
    if mode=='train':
        bench = load_obj(cfg.MODEL.TRAIN.CLASS_NAME)
    elif mode=='test':
        bench = load_obj(cfg.MODEL.TEST.CLASS_NAME)
    else:
        print('mode should be set as train or test.')
        raise NotImplementedError

    print(f'Loaded cofig for {cfg.MODEL.BASE_NAME}')
    return bench(model, config)