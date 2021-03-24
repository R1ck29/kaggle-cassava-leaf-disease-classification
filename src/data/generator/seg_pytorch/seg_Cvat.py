import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
import torch
from torch.utils import data

import sys
sys.path.append('../../')


class cvatDataset(data.Dataset):
    
    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
        "cvat": [0.0, 0.0, 0.0],
    }  

    def __init__(
        self,
        path,
        cfg,
        split='train',
        img_size=(720, 1280),
        augmentations=None,
        version="cvat",
        is_transform=True,
        img_norm=True,
        
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.split = split
        self.path = path
        self.is_transform = is_transform
        
        self.augmentations = augmentations
        
        self.img_norm = img_norm
        
        self.n_classes = cfg.MODEL.n_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}
        
        df = pd.read_pickle(self.path)
        
        self.images = sorted(list(df.groupby('split_name').get_group(self.split)['img_path']))
        self.annotations = sorted(list(df.groupby('split_name').get_group(self.split)['gt_path']))

        self.files[self.split] = self.images

        self.void_classes = [0]
        self.valid_classes = [1, 2, 3]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = self.annotations[index]
        
        name = img_path.split(os.sep)[-1][:-4] + ".png"
        
        if self.split == 'train':
            img_path = os.path.join('../../../', img_path)
            lbl_path = os.path.join('../../../', lbl_path)
        
        if self.split == 'val':
            img_path = os.path.join('../../../', img_path)
            lbl_path = os.path.join('../../../', lbl_path)
        
        if self.split == 'test':
            img_path = os.path.join('../../../../../', img_path)
            lbl_path = os.path.join('../../../../../', lbl_path)
        
        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, name, (img_path, lbl_path)

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = np.array(Image.fromarray(img).resize(
                (self.img_size[1], self.img_size[0])))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)

        value_scale = 255
        mean = [0.406, 0.456, 0.485]
        mean = [item * value_scale for item in mean]
        std = [0.225, 0.224, 0.229]
        std = [item * value_scale for item in std]

        if self.img_norm:
            img = (img - mean) / std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = np.array(Image.fromarray(lbl).resize(
                (self.img_size[1], self.img_size[0]), resample=Image.NEAREST))
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
    
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def decode_segmap_id(self, temp):
        ids = np.zeros((temp.shape[0], temp.shape[1]),dtype=np.uint8)
        for l in range(0, self.n_classes):
            ids[temp == l] = self.valid_classes[l]
        return ids

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask