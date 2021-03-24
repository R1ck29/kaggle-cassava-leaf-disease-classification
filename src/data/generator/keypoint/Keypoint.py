import os
import numpy as np
import pandas as pd
from imageio import imread
from torch.utils.data import Dataset

class _KeypointDataset(Dataset):
    """キーポイント推定用の雛形データセット
    """
    
    def __init__(self, cfg, pkl_path, is_train, heatmap_generator, joints_generator, transforms=None, fold=None):
        """
        
        Args:
            cfg (CfgNode): config
            path_split (str): train/valが格納されたpickleのpath
            is_train (bool): train/valの切り替え
            heatmap_generator (HeatmapGenerator): GT(ヒートマップ)生成用のobject
            joints_generator (JointsGenerator): GT(joint)生成用のobject
            transforms (torchvision.transforms.Compose): Augmentation
            
        """
        df = pd.read_pickle(pkl_path)
        self.transforms = transforms
        
        if fold:
            self.train_df = df[df.fold != fold]
            self.val_df = df[df.fold == fold]
        
            if is_train:
                self.df = self.train_df
            else:
                self.df = self.val_df
        
        else:
            self.df = df
        
        self.num_scales = self._init_check(heatmap_generator, joints_generator)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.with_center = cfg.DATA.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 if self.with_center else self.num_joints
        self.scale_aware_sigma = cfg.DATA.SCALE_AWARE_SIGMA
        self.base_sigma = cfg.DATA.BASE_SIGMA
        self.base_size = cfg.DATA.BASE_SIZE
        self.int_sigma = cfg.DATA.INT_SIGMA
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joints_generator
        
    def __len__(self):
        raise NotImplementedError

    def get_data(self, index):
        '''
        expected to return four arguments:
            * img (ndarray)
            * joints (list of lists of arrays)
            * mask (a scalar)
        '''
        raise NotImplementedError
        
    def __getitem__(self, index):
        img, joints, mask = self.get_data(index)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()
        
        # Transform(Augmentation + Normalization)
        if self.transforms:
            img, mask_list, joints_list = self.transforms(img, mask_list, joints_list)

        # Generate GT
        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        return img, target_list, mask_list, joints_list
    
    def _init_check(self, heatmap_generator, joints_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(joints_generator, (list, tuple)), 'joints_generator should be a list or tuple'
        assert len(heatmap_generator) == len(joints_generator), \
            'heatmap_generator and joints_generator should have same length,'\
            'got {} vs {}.'.format(len(heatmap_generator), len(joints_generator))
        return len(heatmap_generator)
    
    
class EvalDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        img, joints, fname = self.get_data(index)
        return img, joints, fname
    
    def get_data(self, index):
        fname = self.df.image_path.iloc[index]
        img = imread(fname)
        joints = self.get_joints(index)
        return np.array(img).astype(np.uint8), joints, fname
    
    def get_joints(self, idx):
        #parse
        joints = self.df.joints.iloc[idx].copy()
        #学習時にはx,yのオーダーで渡す必要あり、要fix
        return joints