import os
import json
import pandas as pd
import numpy as np
from imageio import imread
from torch.utils.data import Dataset

from .Keypoint import _KeypointDataset

class MPIIKeypoints(_KeypointDataset):
    """
    """
    def __init__(self, cfg, path_split, is_train, heatmap_generator, joints_generator, transforms=None, fold=None):
        super().__init__(cfg, path_split, is_train, heatmap_generator, joints_generator, transforms, fold)
        
        if cfg.DATA.WITH_CENTER:
            assert cfg.MODEL.NUM_JOINTS == 17, 'Number of joint with center for MPII is 18'
        else:
            assert cfg.MODEL.NUM_JOINTS == 16, 'Number of joint for MPII is 17'
        
    def __len__(self):
        return len(self.df)
    
    def get_data(self, index):
        img = imread(self.df.image_path.iloc[index])
        joints = self.get_joints(index)
        mask = np.ones((img.shape[0], img.shape[1]))
        return img, joints, mask
    
    def get_joints(self, idx):
        
        #parse
        """
        iter_num = int(self.df.numOtherPeople.iloc[idx])
        joint_self = self.df.joint_self.iloc[idx]
        joint_others = self.df.joint_others.iloc[idx]

        if iter_num == 1:
            joints  = np.array([joint_self, joint_others])
        elif iter_num > 1:
            joints  = np.array([joint_self, *joint_others])
        else:
            joints  = np.array([joint_self])
            
        joints[np.logical_not(np.all(joints==0, axis=2)),2] += 1
        """
        #parse
        joints = self.df.joints.iloc[idx].copy()
        #学習時にはx,yのオーダーで渡す必要あり、要fix
        joints[...,:2] = joints[...,:2][...,::-1]
        return joints