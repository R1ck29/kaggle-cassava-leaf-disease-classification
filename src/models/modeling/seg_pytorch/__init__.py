import copy
import torchvision.models as models


def get_model(name, n_classes, version=None):
    name = str(name)
    model = _get_model_instance(name)
    model = model(n_classes=n_classes)
    
    return model


def _get_model_instance(name):
    try:
        return {
            "hardnet": hardnet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
        


from .hardnet import *
from .lightning_seg_pytorch import *