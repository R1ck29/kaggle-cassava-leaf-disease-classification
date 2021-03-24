import glob
import sys
import gc
from typing import Any, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.common import load_obj
from src.models.backbone.pytorch.timm_0_1_30.timm.models.factory import create_model
import timm
from vision_transformer_pytorch import VisionTransformer


def get_model(cfg: DictConfig, mode: str, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get model

    Args:
        cfg: config
        mode: test or train
        pretrained_weights (bool): flag to load ImageNet or other pretrtained weight 
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    if 'seresnext' in cfg.MODEL.MODEL_NAME:
        model = get_seresnext(cfg, pretrained_weights, finetuned_weights, device)
    elif 'resnext' in cfg.MODEL.MODEL_NAME:
        model = get_resnext(cfg, pretrained_weights, finetuned_weights, device)
    elif 'vit' in cfg.MODEL.MODEL_NAME:
        model = get_vit(cfg, pretrained_weights, finetuned_weights, device)
    elif 'deit' in cfg.MODEL.MODEL_NAME:
        model = get_deit(cfg, pretrained_weights, finetuned_weights, device)
    elif 'resnet' in cfg.MODEL.BACKBONE.CLASS_NAME:
        model = get_resnet(cfg, finetuned_weights, device)
    elif 'efficientnet' in cfg.MODEL.BACKBONE.CLASS_NAME:
        model = get_efficientnet(cfg, pretrained_weights, finetuned_weights, device)
    else:
        print(f'Model is not defined. {cfg.MODEL.BACKBONE.CLASS_NAME}')
        raise NotImplementedError
    return model


def get_resnet(cfg: DictConfig, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get ResNet model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    backbone = load_obj(cfg.MODEL.BACKBONE.CLASS_NAME) #resnet50()
    backbone = backbone()
    classifier = load_obj(cfg.MODEL.BASE_CLASSIFIER.CLASS_NAME) #BaseClassifier(nattr=train_set.attr_num)
    classifier = classifier(**cfg.MODEL.BASE_CLASSIFIER.PARAMS)
    model = load_obj(cfg.MODEL.CUSTOM_CLASSIFIER.CLASS_NAME) #FeatClassifier(backbone, classifier)
    model = model(backbone, classifier)

    if torch.cuda.is_available() and device == torch.device('cuda'):
        model = torch.nn.DataParallel(model).cuda()

    if finetuned_weights is not None:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
          
    return model


def get_efficientnet(cfg: DictConfig, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get EfficientNet model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    if 'tf_' in cfg.MODEL.MODEL_NAME and 'ns' in cfg.MODEL.MODEL_NAME:
        print(f'Creating model with timm: {cfg.MODEL.MODEL_NAME}')
        print(f'Pretrained: {pretrained_weights}')
        model = create_model(cfg.MODEL.MODEL_NAME, pretrained=pretrained_weights)
        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)
    else:
        model = load_obj(cfg.MODEL.BACKBONE.CLASS_NAME)
        model = model.from_name(cfg.MODEL.MODEL_NAME)

        if pretrained_weights:
            pretrained_weight_path = glob.glob(f'{hydra.utils.to_absolute_path(cfg.MODEL.BACKBONE.PARAMS.WEIGHT_PATH)}/{cfg.MODEL.MODEL_NAME}*')
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
    #         self.model=EfficientNet.from_pretrained('efficientnet-b3',num_classes=CLASSES)
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, cfg.MODEL.NUM_CLASSES)

    # if torch.cuda.is_available() and device == torch.device('cuda'):
    #     model = torch.nn.DataParallel(model).cuda()

    if finetuned_weights is not None:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
          
    return model


def get_seresnext(cfg: DictConfig, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get SE-ResNext model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    print(f'Creating model with timm: {cfg.MODEL.MODEL_NAME}')
    print(f'Pretrained: {pretrained_weights}')
    model = create_model(cfg.MODEL.MODEL_NAME, pretrained=pretrained_weights)

    n_features = model.last_linear.in_features
    model.last_linear = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)

    # n_features = model.fc.in_features
    # model.fc = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)
    # n_features = model.head.in_features
    # model.head = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)

    # if torch.cuda.is_available() and device == torch.device('cuda'):
    #     model = torch.nn.DataParallel(model).cuda()

    if finetuned_weights is not None:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
    return model


def get_resnext(cfg: DictConfig, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get SE-ResNext model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    print(f'Creating model with timm: {cfg.MODEL.MODEL_NAME}')
    print(f'Pretrained: {pretrained_weights}')
    model = create_model(cfg.MODEL.MODEL_NAME, pretrained=pretrained_weights)

    # n_features = model.last_linear.in_features
    # model.last_linear = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)
    # n_features = model.head.in_features
    # model.head = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)

    # if torch.cuda.is_available() and device == torch.device('cuda'):
    #     model = torch.nn.DataParallel(model).cuda()

    if finetuned_weights is not None:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
    return model


def get_vit(cfg: DictConfig, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get ViT model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    print(f'Creating model with timm: {cfg.MODEL.MODEL_NAME}')
    print(f'Pretrained: {pretrained_weights}')
    model = timm.create_model(cfg.MODEL.MODEL_NAME, pretrained=pretrained_weights)
    model.head = nn.Linear(model.head.in_features, cfg.MODEL.NUM_CLASSES)

    # model = VisionTransformer.from_name(cfg.MODEL.MODEL_NAME, num_classes=cfg.MODEL.NUM_CLASSES) 


    # if torch.cuda.is_available() and device == torch.device('cuda'):
    #     model = torch.nn.DataParallel(model).cuda()

    if finetuned_weights is not None:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
    return model


def get_deit(cfg: DictConfig, pretrained_weights: Optional[bool] = False, finetuned_weights: Optional[str] = None, device: Optional[torch.device] = torch.device('cuda')) -> Any:
    """
    Get DeiT model

    Args:
        cfg: config
        finetuned_weights (str): a pretrained weight path to load
        device (torch.device): cuda or cpu

    Returns:
        initialized model
    """
    print(f'Pretrained: {pretrained_weights}')
    model = torch.hub.load('facebookresearch/deit:main', cfg.MODEL.MODEL_NAME, pretrained=pretrained_weights)
    n_features = model.head.in_features
    model.head = nn.Linear(n_features, cfg.MODEL.NUM_CLASSES)

    # if torch.cuda.is_available() and device == torch.device('cuda'):
    #     model = torch.nn.DataParallel(model).cuda()

    if finetuned_weights is not None:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(finetuned_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(finetuned_weights))
        print(f'Loaded Finetuned weights: {finetuned_weights}')
    return model