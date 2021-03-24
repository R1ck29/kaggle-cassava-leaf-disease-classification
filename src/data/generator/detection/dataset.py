import os
import random
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import Compose
from omegaconf import DictConfig
from torch.utils.data import Dataset


class FasterRCNNDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, cfg: DictConfig, transforms: Compose, mode: str = 'train', image_dir: str = ''
    ):
        """
        Prepare data for wheat competition.

        Args:
            dataframe: dataframe with image id and bboxes
            mode: train/val/test
            cfg: config with parameters
            image_dir: path to images
            transforms: albumentations
        """
        self.image_dir = image_dir
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.yxyx = True
        self.image_ids = os.listdir(self.image_dir) if self.df is None else self.df['image_id'].unique()
        self.image_file_format = os.listdir(self.image_dir)[0].split('.')[-1]
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[np.array, Dict[str, Union[torch.Tensor, np.array]], str]:
        image_id = self.image_ids[idx].split('.')[0]
        image = cv2.imread(f'{self.image_dir}/{image_id}.{self.image_file_format}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # normalization.
        # TO DO: refactor preprocessing
        image /= 255.0

        # test dataset must have some values so that transforms work.
        target = {
            'labels': torch.as_tensor([[0]], dtype=torch.float32),
            'boxes': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)
        }

        # for train and valid test create target dict.
        if self.mode != 'test':
            image_data = self.df.loc[self.df['image_id'] == image_id]

            if not 'xmin' in image_data.columns and not 'ymin' in image_data.columns:
                boxes = None
            elif not 'xmax' in image_data.columns and not 'ymax' in image_data.columns:
                boxes = image_data[['xmin', 'ymin', 'width', 'height']].values
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            else:
                boxes = image_data[['xmin', 'ymin', 'xmax', 'ymax']].values

            areas = image_data['area'].values
            areas = torch.as_tensor(areas, dtype=torch.float32)

            # there is only one class
            # labels_1 = torch.ones((image_data.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(image_data[['class_id']].values, dtype=torch.int64)[0]
            iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)

            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor([idx])
            target['area'] = areas
            target['iscrowd'] = iscrowd

            if self.transforms:
                if 'EfficientDet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
                    for i in range(10):
                        sample = self.transforms(**{
                            'image': image,
                            'bboxes': target['boxes'],
                            'labels': labels
                        })
                        # assert len(sample['bboxes']) == labels.shape[0], 'not equal!'
                        if len(sample['bboxes']) > 0:
                            image = sample['image']
                            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                            target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                            target['labels'] = torch.stack(tuple(sample['labels'])) # <--- add this!
                            break
                else:
                    image_dict = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
                    image_dict = self.transforms(**image_dict)
                    image = image_dict['image']
                    target['boxes'] = torch.as_tensor(image_dict['bboxes'], dtype=torch.float32)
                
        elif self.mode == 'valid':
            image_data = self.df.loc[self.df['image_id'] == image_id]

            if not 'xmin' in image_data.columns and not 'ymin' in image_data.columns:
                boxes = None
            elif not 'xmax' in image_data.columns and not 'ymax' in image_data.columns:
                boxes = image_data[['xmin', 'ymin', 'width', 'height']].values
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            else:
                boxes = image_data[['xmin', 'ymin', 'xmax', 'ymax']].values
                
            # labels_1 = torch.ones((image_data.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(image_data[['class_id']].values, dtype=torch.int64)[0]
            target['boxes'] = boxes
            target['labels'] = labels

            image_dict = {'image': image,'bboxes': target['boxes'], 'labels': target['labels']}
            image_dict = self.transforms(**image_dict)
            image = image_dict['image']
            target['boxes'] = torch.as_tensor(image_dict['bboxes'], dtype=torch.float32)
            
        elif self.mode == 'test':
            if self.transforms:
                sample = {'image': image}
                sample = self.transforms(**sample)
                image = sample['image']
            return image, target, image_id
        else:
            image_dict = {'image': image, 'bboxes': target['boxes'], 'labels': target['labels']}
            image = self.transforms(**image_dict)['image']
        
        # yxyx for efficientdet
        if 'EfficientDet' in self.cfg.MODEL.BACKBONE.CLASS_NAME:
            if self.yxyx and len(target['boxes'])>0:
                target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)


class DatasetRetriever(Dataset):
    # def __init__(self, marking, image_ids, transforms=None, test=False):
    def __init__(self, dataframe: pd.DataFrame, cfg: DictConfig, image_ids, transforms: Compose, mode: str = 'train', image_dir: str = '', test=False):
        super().__init__()
        self.image_dir = image_dir
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.image_ids = image_ids #os.listdir(self.image_dir) if self.df is None else self.df['image_id'].unique()
        self.image_file_format = os.listdir(self.image_dir)[0].split('.')[-1]
        self.transforms = transforms
        self.cutmix = self.cfg.DATASET.CUTMIX
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.cutmix:
            if self.test or random.random() > 0.5:
                image, boxes, labels = self.load_image_and_boxes(index)
            else:
                image, boxes, labels = self.load_cutmix_image_and_boxes(index)
        else:
            image, boxes, labels = self.load_image_and_boxes(index)

        if boxes is None and self.mode =='test':
            if self.transforms:
                sample = {'image': image}
                sample = self.transforms(**sample)
                image = sample['image']
            target = None
            return image, target, image_id        
        # there is only one class
        #TODO: make option for multiple classes
        # labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32) #boxes
                    # 'boxes': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        image_data = self.df[self.df['image_id'] == image_id]
        areas = image_data['area'].values
        areas = torch.as_tensor(areas, dtype=torch.float32)
        target['area'] = areas
        iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)
        target['iscrowd'] = iscrowd

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                # assert len(sample['bboxes']) == labels.shape[0], 'not equal!'
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    if self.mode != 'eval_oof':
                        target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                        target['labels'] = torch.stack(tuple(sample['labels']))
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        if self.image_file_format not in image_id:
            image = cv2.imread(f'{self.image_dir}/{image_id}.{self.image_file_format}', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['image_id'] == image_id]
        if not 'xmin' in records.columns and not 'ymin' in records.columns:
            boxes = None
        elif not 'xmax' in records.columns and not 'ymax' in records.columns:
            boxes = records[['xmin', 'ymin', 'width', 'height']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = torch.tensor(records['class_id'].astype(int).values)
        return image, boxes, labels


    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = self.cfg.MODEL.INPUT_SIZE
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        if s == (imsize // 2):
            result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
            
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            if s != (imsize // 2):
                result_image = np.full((s * 2, s * 2, image.shape[2]), 114, dtype=np.float32)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes, result_labels
