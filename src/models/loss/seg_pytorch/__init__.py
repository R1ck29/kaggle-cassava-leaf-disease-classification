import logging
import functools

from .loss import *

key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
}


def get_loss_function(cfg):
    if cfg.LOSS.name is None:
        return cross_entropy2d

    else:
        loss_dict = cfg.LOSS
        loss_name = cfg.LOSS.name
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        return functools.partial(key2loss[loss_name], **loss_params)