from __future__ import print_function

import gc
import glob
import os
import random
import sys

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch._six
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.generator.detection.build import get_training_datasets
from src.models.modeling.detection.get_model import get_model
from src.models.modeling.detection.lightning_detection import \
    LightningDetection
from src.models.utils.detection.metrics import calculate_final_score, run_wbf
from src.utils.common import collate_fn
from src.utils.pytorch.utils import set_seed

    
def save_pred_images(test_image_dir, image_id, boxes, scores, pred_images_dir, gt_boxes=None, extension='jpg'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    img_ = cv2.imread(f'{test_image_dir}/{image_id}.{extension}')  # BGR
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    for box, score in zip(boxes,scores):
        # cv2.rectangle(img_, (box[0], box[1]), (box[2]+box[0], box[3]+box[1]), (220, 0, 0), 2)
        cv2.rectangle(img_, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 2)
        cv2.putText(img_, '%.2f'%(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        if gt_boxes is not None:
            for gt_box in gt_boxes:
                cv2.rectangle(img_, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 220), 2)
                
    ax.set_axis_off()
    if gt_boxes is None:
        ax.set_title(f"{image_id} \n RED: Predicted")
    else:
        ax.set_title(f"{image_id} \n RED: Predicted | BLUE - Ground-truth")
    ax.imshow(img_)
    fig.savefig(f'{pred_images_dir}/pred_{image_id}.{extension}', bbox_inches='tight')
    plt.close(fig)
    

class Evaluator:
    def __init__(self, cfg, model_cfg):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.image_size = self.model_cfg.MODEL.INPUT_SIZE
        self.image_size_ratio = self.model_cfg.MODEL.OUTPUT_SIZE / self.image_size
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.weight_dir = os.path.join(hydra.utils.get_original_cwd(), self.cfg.MODEL_PATH)
        
        
    def predict_validation(self, weight_dir, device):
        all_predictions = []
        image_size = self.model_cfg.MODEL.INPUT_SIZE
        image_size_ratio = self.model_cfg.MODEL.OUTPUT_SIZE / image_size
        for fold_number in range(self.model_cfg.DATA.N_FOLD):

            print('-'*30, f'Prediction on Validation: Fold {fold_number}', '-'*30)
            #TODO: test ckpt and pth path
            weight_path = glob.glob(f'{weight_dir}/fold{fold_number}*.pth')

            if len(weight_path) > 1:
                print(f'Found more than one weight path: {weight_path}')
                sys.exit()
            elif len(weight_path)==0:
                print(f' No weight path found for Fold {fold_number}: {weight_path}')
                return all_predictions
                
            # model = LightningDetection.load_from_checkpoint(checkpoint_path=str(weight_path), cfg=cfg, fold=fold_number, coco_weights=None)
            model = get_model(self.model_cfg, mode='test', finetuned_weights=weight_path[0], device=self.device)
            model.eval()
            model.cuda()
            
            datasets = get_training_datasets(self.model_cfg, fold_number, eval_oof=True)
            valid_dataset = datasets['valid']
            print(f'Fold{fold_number} {len(valid_dataset)}')
                
            val_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=self.cfg.TEST.BATCH_SIZE,
                                                num_workers=self.model_cfg.SYSTEM.NUM_WORKERS,
                                                shuffle=False,
                                                collate_fn=collate_fn)

            for images, targets, image_ids in val_loader:
                # images = list(image.to(device) for image in images)
                images = torch.stack(images).to(device).float()

                if 'EfficientDet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    img_size = torch.tensor([images[0].shape[-2:]] * self.cfg.TEST.BATCH_SIZE, dtype=torch.float).to(device)
                    outputs = model(images, torch.tensor([1]*images.shape[0], dtype=torch.float).to(device), img_size) #, img_size
                elif 'fasterrcnn' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    outputs = model(images)
                else:
                    print(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
                    raise NotImplementedError

                for i, image in enumerate(images):
                    if 'EfficientDet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                        boxes = outputs[i].detach().cpu().numpy()[:,:4]    
                        scores = outputs[i].detach().cpu().numpy()[:,4]

                        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                    elif 'fasterrcnn' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                        boxes = outputs[i]['boxes'].data.cpu().numpy() #preds
                        scores = outputs[i]['scores'].data.cpu().numpy()
                    else:
                        print(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
                        raise NotImplementedError

                    # boxes, scores, labels = run_wbf(outputs, image_index=i, image_size=cfg.MODEL.INPUT_SIZE) # added

                    boxes = boxes[scores >= self.cfg.TEST.DETECTION_THRESHOLD].astype(np.int32)
                    scores = scores[scores >= self.cfg.TEST.DETECTION_THRESHOLD]
                    target = targets[i]['boxes'].cpu().numpy()

            #         preds_sorted_idx = np.argsort(scores)[::-1]
            #         boxes = boxes[preds_sorted_idx]
                    all_predictions.append({
                        'pred_boxes': (boxes*image_size_ratio).clip(min=0, max=self.model_cfg.MODEL.OUTPUT_SIZE-1).astype(int),
                        'scores': scores,
                        'gt_boxes': (target*image_size_ratio).clip(min=0, max=self.model_cfg.MODEL.OUTPUT_SIZE-1).astype(int),
                        'image_id': image_ids[i],
                    })

        return all_predictions


    def find_best_threshold(self, all_predictions):
        # Search best threshold for best score:
        best_final_score, best_score_threshold = 0, 0
        print('-'*20,'Finding best threshold for best score', '-'*20)
        for score_threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0]):
            final_score, _ = calculate_final_score(all_predictions, score_threshold)
            if final_score > best_final_score:
                best_final_score = final_score
                best_score_threshold = score_threshold

        return best_final_score, best_score_threshold


    def evaluate(self):
        set_seed(self.cfg)

        weight_dir = os.path.join(hydra.utils.get_original_cwd(), self.cfg.MODEL_PATH)
        train_image_dir = hydra.utils.to_absolute_path(self.model_cfg.DATA.TRAIN_IMAGE_DIR)

        os.makedirs(self.cfg.TEST.VAL_PRED_IMG_DIR, exist_ok=True)
    
        # Prediction on Validation Dataset
        all_predictions = self.predict_validation(weight_dir, device=self.device)
        
        best_final_score, best_score_threshold = self.find_best_threshold(all_predictions)

        print('-'*30)
        print(f'[Model Name]: {self.cfg.MODEL_ID}')
        print(f'[Best Score Threshold]: {best_score_threshold}')
        print(f'[OOF Score]: {best_final_score:.4f}')
        print('-'*30)

        result = [(self.cfg.MODEL_ID, best_score_threshold, round(best_final_score, 4))]
        val_score_df = pd.DataFrame(result, columns=['model_name', 'best_score_threshold', 'oof_score'])
        val_score_df.to_csv(self.cfg.TEST.BEST_THR_CSV_NAME, index=False)
        print(f'Best Score Threshold csv saved to: {self.cfg.TEST.BEST_THR_CSV_NAME}')

        if self.cfg.TEST.VAL_PRED_IMG_DIR:
            if self.cfg.TEST.SAVE_ALL_IMAGES:
                num_images = len(all_predictions)
            else:
                num_images = 10
            print(f'Number of plot images: {num_images}')
            for i in tqdm(range(num_images), total=num_images):
                gt_boxes = all_predictions[i]['gt_boxes'].copy()
                pred_boxes = all_predictions[i]['pred_boxes'].copy()
                scores = all_predictions[i]['scores'].copy()
                image_id = all_predictions[i]['image_id']
                indexes = np.where(scores>best_score_threshold)
                pred_boxes = pred_boxes[indexes]
                scores = scores[indexes]
                
                save_pred_images(train_image_dir, image_id, pred_boxes, scores, self.cfg.TEST.VAL_PRED_IMG_DIR, gt_boxes)