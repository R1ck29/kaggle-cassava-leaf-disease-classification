import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from albumentations.core.composition import Compose
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms as T
import torch
import cv2


class AttrDataset(Dataset):
    # def __init__(self, df, args, transform=None, target_transform=None):
    def __init__(self, dataframe: pd.DataFrame, cfg: DictConfig, image_ids, transforms: Compose, mode: str = 'train', image_dir: str = None):

        self.image_dir = image_dir
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.image_ids = image_ids #os.listdir(self.image_dir) if self.df is None else self.df['image_id'].unique()
        if 'image_path' not in self.df.columns:
            self.image_file_format = os.listdir(self.image_dir)[0].split('.')[-1]
        else:
            self.image_file_format = None
        self.transforms = transforms
        # self.cutmix = self.cfg.DATASET.CUTMIX
        self.df = dataframe
        self.dataset = self.cfg.DATA.DATA_ID
        self.n_classes = self.cfg.DATA.NUM_CLASSES

    
    def __getitem__(self, index: int):
        image, labels, image_name = self.load_image_and_labels(index)

        if self.transforms is not None:
            if 'albumentations_classification' in self.cfg.AUGMENTATION.FRAMEWORK:
                albu_dict = {'image': image}
                transorm = self.transforms(**albu_dict)
                image = transorm['image']
            elif self.cfg.AUGMENTATION.FRAMEWORK == 'custom':
                image = self.transforms(image)
        if labels is not None:
            labels = labels.to('cpu').detach().numpy().copy().astype(np.float32)

        return image, labels, image_name

    def __len__(self) -> int:
        return self.image_ids.shape[0]


    def load_image_and_labels(self, index):
        image_id = self.image_ids[index]

        if self.image_file_format is not None:
            if self.image_file_format in image_id:
                image_id = image_id.replace('.' + self.image_file_format, '')
            image_name = f'{image_id}.{self.image_file_format}'
        else:
            assert len(self.df[self.df['image_id']==image_id].image_path.values) == 1
            image_path = self.df[self.df['image_id']==image_id].image_path.values[0]
            image_name = image_path.split('/')[-1]
        
        if 'albumentations_classification' in self.cfg.AUGMENTATION.FRAMEWORK:
            image = cv2.imread(f'{self.image_dir}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        elif self.cfg.AUGMENTATION.FRAMEWORK == 'custom':
            image = Image.open(f'{self.image_dir}/{image_name}')

        # image /= 255.0
        records = self.df[self.df['image_id'] == image_name]
        item = records.T.squeeze() #records.iloc[0]
        
        # labels = torch.tensor(records['class_id'].astype(int).values)
        if 'class_id' in records.columns and isinstance(item.class_id,str):
            labels = torch.zeros(self.n_classes)
            for cls in item.class_id.split():
                labels[int(cls)] = 1
            assert len(labels) == self.n_classes
        else:
            labels = None

        return image, labels, image_name


    def check_classes(self):
        if self.cfg.DATA.DATA_ID == 'PA100k':
            assert self.n_classes == 5 #4 #26
        elif self.cfg.DATA.DATA_ID == 'PETA':
            assert self.n_classes == 6 #5 #35
        elif self.cfg.DATA.DATA_ID == 'RAP':
            assert self.n_classes == 5 #4 #51
        elif self.cfg.DATA.DATA_ID == 'RAP2':
            assert self.n_classes == 6 #5 #54
        else:
            raise NotImplementedError


class CustomDataset(Dataset):
    def __init__(
        self, df, image_dir, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.image_dir = image_dir
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['class_id']
            target = torch.tensor(int(target))

        image_id = self.df.iloc[index]['image_id']
          
        path = "{}/{}".format(self.image_dir, self.df.iloc[index]['image_id'])
        
        img  = self.get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label:
            return img, target, image_id
        else:
            return img, image_id

    def get_img(self, path):
        im_bgr = cv2.imread(path)
        im_rgb = im_bgr[:, :, ::-1]
        #print(im_rgb)
        return im_rgb


def get_transform(cfg):
    height = cfg.MODEL.INPUT_SIZE.HEIGHT
    width = cfg.MODEL.INPUT_SIZE.WIDTH
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
