import os
import json
import pandas as pd
import numpy as np
from imageio import imread
import pycocotools
from torch.utils.data import Dataset

from .Keypoint import _KeypointDataset

class COCOKeypoints(_KeypointDataset):
    """
    """
    def __init__(self, cfg, path_split, is_train, heatmap_generator, joints_generator, transforms=None, fold=None):
        super().__init__(cfg, path_split, is_train, heatmap_generator, joints_generator, transforms, fold)
        
        if cfg.DATA.WITH_CENTER:
            assert cfg.MODEL.NUM_JOINTS == 18, 'Number of joint with center for COCO is 18'
        else:
            assert cfg.MODEL.NUM_JOINTS == 17, 'Number of joint for COCO is 17'
            
        
    def __len__(self):
        return len(self.df)
    
    def get_data(self, index):
        img = imread(self.df.img_paths.iloc[index])
        anno = self.df.annotations.iloc[index]
        
        mask = self.get_mask(anno, index, img.shape[0], img.shape[1])
        
        #Keypointの存在するGTのみ抽出。
        anno = [obj for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0]

        # TODO(bowen): to generate scale-aware sigma, modify `get_joints` to associate a sigma to each joint
        joints = self.get_joints(anno)
        
        return img, joints, mask
    
    def get_joints(self, anno):
        num_people = len(anno)

        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))
        else:
            joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])
            if self.with_center:
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                if num_vis_joints > 0:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                    joints[i, -1, 2] = 1
            if self.scale_aware_sigma:
                # get person box
                box = obj['bbox']
                size = max(box[2], box[3])
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.round(sigma + 0.5))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma

        return joints
    
    def get_mask(self, anno, idx, height, width):
        #HeatMapLoss算出時用のマスクを生成する。
        #iscrowd、およびnum_keypointsが0のobjectに対してはGTが存在しないので、Loss算出時に省く。
        m = np.zeros((height, width))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(obj['segmentation'], height, width)
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(obj['segmentation'], height, width)
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5