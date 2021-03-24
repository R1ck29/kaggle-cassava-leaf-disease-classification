import os

from omegaconf import OmegaConf
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN


class FRCNNObjectDetector(FasterRCNN):
    """Faster-RCNN model for TorchServe

    Change Attributes describing below.

    Attributes:
        model_id: A model identifier
        base_dir: A path to the BaseModel repository
        model_path: A path to model config store
    """
    def __init__(self, **kwargs):
        base_dir = '/data1/r_suzuki/pipeline'
        model_id = 'fasterrcnn_person'
        model_path = f'BaseModel/models/{model_id}/20201026_12_42_08_2/'
        cfg_path = os.path.join(base_dir, model_path,'.hydra/config.yaml')
        print(f'Loaded config file : {cfg_path}')
        cfg = OmegaConf.load(cfg_path)
        num_classes = cfg.MODEL.NUM_CLASSES
        backbone = resnet_fpn_backbone(cfg.MODEL.BACKBONE.NAME, True)
        super(FRCNNObjectDetector, self).__init__(backbone, num_classes, **kwargs)