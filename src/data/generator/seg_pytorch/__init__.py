from .build import build_dataset, make_dataloader
from .seg_Cityscapes import cityscapesDataset
from .seg_Cvat import cvatDataset

def get_loader(name):
    """get_loader
    :param name:
    """
    return {

        "cityscapes": cityscapesDataset,
        "cvat": cvatDataset,
        

    }[name]