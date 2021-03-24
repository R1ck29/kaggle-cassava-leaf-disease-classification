import logging

from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR

from .schedulers import *

key2scheduler = {
    "constant_lr": ConstantLR,
    "poly_lr": PolynomialLR,
    "multi_step": MultiStepLR,
    "cosine_annealing": CosineAnnealingLR,
    "exp_lr": ExponentialLR,
}

def get_scheduler(optimizer, cfg):
    
    if cfg is None:
        return ConstantLR(optimizer)

    s_type = cfg.SCHEDULER.CLASS_NAME
    
    return key2scheduler[s_type](optimizer, **cfg.SCHEDULER.PARAMS)