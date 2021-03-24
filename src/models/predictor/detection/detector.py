from __future__ import print_function

import glob
import os
import sys
from typing import Any, List, Optional, Tuple

import albumentations as A
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch._six
import torch.utils.data
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import DictConfig
from src.data.generator.detection.build import get_training_datasets
from src.models.modeling.detection.get_model import get_model
from src.models.utils.detection.metrics import calculate_final_score, run_ensemble_method
from src.utils.common import collate_fn
from src.utils.pytorch.utils import set_seed
from torch.utils.data import Dataset
from tqdm import tqdm


class Predictor:
    """ Run Object Detection model to predict on validation or test images """
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, weight_dir: Optional[str] = None):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.image_size = self.model_cfg.MODEL.INPUT_SIZE
        self.image_size_ratio = self.model_cfg.MODEL.OUTPUT_SIZE / self.image_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.device == torch.device('cpu'):
            print('Device: CPU')
        if weight_dir is None: 
            self.weight_dir = os.path.join(hydra.utils.get_original_cwd(), self.cfg.MODEL_PATH)
        else:
            self.weight_dir = weight_dir
        self.test_img_dir = hydra.utils.to_absolute_path(self.cfg.TEST.TEST_IMAGE_DIR)
        self.models = self._load_models()

        print(f'Test Images Path: {self.test_img_dir}')
        print(f'Test Annotations Path: {self.cfg.TEST.TEST_CSV_PATH}')


    def _make_results(self, results, outputs, image_ids, index, mode, clip_boxes=False, targets=None, score_threshold=None):
        """Make prediction list (bboxes(pascal-voc format), confidence scores, image ids, gt_boxes(optional))

        Args:
            results (list): prediction lists(bboxes)
            outputs (Any): model outputs 
            image_ids (list): image name
            index (int): image index
            mode (str): val or test
            clip_boxes (bool, optional): to clip bbox value for gt image size. Defaults to False.
            targets (list, optional): ground truth boxes. Defaults to None.
            score_threshold (float, optional): threshold for confidence score. Defaults to None.

        Raises:
            NotImplementedError: model shoud be EfficientDet or Faster RCNN

        Returns:
            results [list]: prediction list (bboxes(pascal-voc format), confidence scores, class ids, image ids, gt_boxes(optional))
            indexes [list]: index list of images larger than score threshold.
            gt_flag [bool]: if gt bboxes are available set as True.
        """

        if 'EfficientDet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
            boxes = outputs[index].detach().cpu().numpy()[:,:4]    
            scores = outputs[index].detach().cpu().numpy()[:,4]
            class_ids = outputs[index].detach().cpu().numpy()[:,5]

            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

        elif 'fasterrcnn' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
            boxes = outputs[index]['boxes'].data.cpu().numpy() #preds
            scores = outputs[index]['scores'].data.cpu().numpy()
            class_ids = outputs[index]['labels'].data.cpu().numpy()
        else:
            print(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
            raise NotImplementedError

        if mode == 'test':
            indexes = np.where(scores>=score_threshold)[0]
            boxes = boxes[indexes].astype(np.int32)
            scores = scores[indexes]
            class_ids = class_ids[indexes].astype(np.int32)
        elif mode == 'val':
            indexes = None
            boxes = boxes.astype(np.int32)
            class_ids =  class_ids.astype(np.int32)
        else:
            raise NotImplementedError

        try:
            target = targets[index]['boxes'].cpu().numpy()
            target = (target*self.image_size_ratio).clip(min=0, max=self.model_cfg.MODEL.OUTPUT_SIZE-1).astype(int)
            gt_flag = True
        except TypeError:
            target =None
            gt_flag = False
#         preds_sorted_idx = np.argsort(scores)[::-1]
#         boxes = boxes[preds_sorted_idx]
        if clip_boxes:
            boxes = (boxes*self.image_size_ratio).clip(min=0, max=self.model_cfg.MODEL.OUTPUT_SIZE-1).astype(int)
        results.append({
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'gt_boxes': target,
            'image_id': image_ids[index],
        })
        return results, indexes, gt_flag
        
        
    def _predict_validation(self):
        """Predict on validation data

        Raises:
            NotImplementedError: model shoud be EfficientDet or Faster RCNN

        Returns:
            [list]: prediction list (bboxes(pascal-voc format), confidence scores, class ids, image ids, gt_boxes(optional))
        """
        
        predictions = []
        for fold_number in range(self.model_cfg.DATA.N_FOLD):
            print('-'*30, f'Prediction: Validation Fold {fold_number}', '-'*30)
            model = self.models[fold_number]
            
            datasets = get_training_datasets(self.model_cfg, fold_number, eval_oof=True)
            valid_dataset = datasets['valid']
                
            val_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=self.cfg.TEST.BATCH_SIZE,
                                                num_workers=self.model_cfg.SYSTEM.NUM_WORKERS,
                                                shuffle=False,
                                                collate_fn=collate_fn)

            print(f'Score Threshold for Validation: {self.cfg.TEST.DETECTION_THRESHOLD}')
            
            for images, targets, image_ids in tqdm(val_loader, total=len(val_loader)):
                # images = list(image.to(device) for image in images)
                images = torch.stack(images).to(self.device).float()

                if 'EfficientDet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    img_size = torch.tensor([images[0].shape[-2:]] * self.cfg.TEST.BATCH_SIZE, dtype=torch.float).to(self.device)
                    outputs = model(images, torch.tensor([1]*images.shape[0], dtype=torch.float).to(self.device), img_size)
                elif 'fasterrcnn' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    outputs = model(images)
                else:
                    print(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
                    raise NotImplementedError
                # results = []
                for i, image in enumerate(images):
                    predictions, _, _ = self._make_results(outputs=outputs, results=predictions, image_ids=image_ids, 
                                                   index=i, mode='val', clip_boxes=True, targets=targets, score_threshold=None)
                # predictions.append(results)
        return predictions


    def get_class_name(self, class_id):
        """Get class name from class id

        Args:
            class_id (int): class identifier

        Returns:
            [str]: class name if "label_mappings.yaml" is found, otherwise return "class_id"[int].
        """
        class_mapping_files = glob.glob("../../../*/label_mappings.yaml")
        try:
            latest_file = max(class_mapping_files, key=os.path.getctime)
            label_mappings = open(latest_file, "r+")
            label_to_num = yaml.load(label_mappings, Loader=yaml.SafeLoader)
            # print(f'Loaded class mapping: {label_mappings}')
            num_to_label = {v:k for k,v in label_to_num.items()}
            class_name = num_to_label[class_id]
            return class_name
        except Exception as e:
            print(f'{e} \n Can not find a class mapping yaml file.')
            print('showing class id.')
            return class_id


    def _save_pred_images(self, test_image_dir, image_id, boxes, scores, class_ids, pred_images_dir, gt_boxes=None, save_images=True, show_class_name=True):
        """Plot bboxes on images and and save them

        Args:
            test_image_dir (str): directory cotaining test images 
            image_id (str): image name
            boxes (list): boxes for an image
            scores (list): confidence score
            class_ids (list): object class id
            pred_images_dir (str): directory to save prediction images
            gt_boxes (list, optional): ground truth boxes. Defaults to None.
            save_images (bool, optional): if True, save prediction images. Defaults to True.
            show_class_name (bool, optional): if True, show class name next to cofidence score. Defaults to True.
                                              if False, show class id next to confidence score
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(24, 12))

        image_file_format = os.listdir(test_image_dir)[0].split('.')[-1]
        if image_file_format not in image_id and '.' not in image_id:
            img_name = f'{image_id}.{image_file_format}'
        else:
            img_name = image_id
        img_ = cv2.imread(f'{test_image_dir}/{img_name}')  # BGR
        if not os.path.exists(f'{test_image_dir}/{img_name}'):
            print(f'Can not find image: {test_image_dir}/{img_name}')
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = cv2.resize(img_, (self.model_cfg.MODEL.OUTPUT_SIZE, self.model_cfg.MODEL.OUTPUT_SIZE))
        for box, score, class_id in zip(boxes,scores, class_ids):
            cv2.rectangle(img_, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 2)
            if show_class_name:
                class_info = self.get_class_name(class_id)
            else:
                class_info = class_id
            cv2.putText(img_, f'{class_info}:{score:.2f}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if gt_boxes is not None:
                for gt_box in gt_boxes:
                    cv2.rectangle(img_, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 220), 2)
        ax.set_axis_off()
        if gt_boxes is None:
            ax.set_title(f"{image_id} \n RED: Predicted")
        else:
            ax.set_title(f"{image_id} \n RED: Predicted | BLUE - Ground-truth")

        ax.imshow(img_)
        if save_images:
            if image_file_format not in image_id and '.' not in image_id:
                img_id = image_id
            else:
                img_id = image_id.split('.')[0]
            if not os.path.exists(pred_images_dir):
                os.makedirs(pred_images_dir, exist_ok=True)
            fig.savefig(f'{pred_images_dir}/pred_{img_id}.jpg', bbox_inches='tight')
            plt.close(fig)


    def _find_best_threshold(self, predictions):
        """ Search best threshold for best score """
        best_final_score, best_score_threshold = 0, 0
        print('-'*20,'Finding best threshold for best score', '-'*20)
        for score_threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0]):
            final_score, _ = calculate_final_score(predictions, score_threshold)
            if final_score > best_final_score:
                best_final_score = final_score
                best_score_threshold = score_threshold

        return best_final_score, best_score_threshold


    def evaluate(self):
        """ Run out-of-fold evaluation to find best score threshold. """
        set_seed(self.cfg)

        train_image_dir = hydra.utils.to_absolute_path(self.model_cfg.DATA.TRAIN_IMAGE_DIR)

        if self.cfg.TEST.VISUALIZE_RESULTS:
            os.makedirs(self.cfg.TEST.VAL_PRED_IMG_DIR, exist_ok=True)
    
        # Prediction on Validation Dataset
        predictions = self._predict_validation()
        
        if self.cfg.TEST.FIND_BEST_THR:
            best_final_score, best_score_threshold = self._find_best_threshold(predictions)

            print('-'*30)
            print(f'[Model Name]: {self.cfg.MODEL_ID}')
            print(f'[Best Score Threshold]: {best_score_threshold}')
            print(f'[OOF Score]: {best_final_score:.4f}')
            print('-'*30)

            result = [(self.cfg.MODEL_ID, best_score_threshold, round(best_final_score, 4))]
            val_score_df = pd.DataFrame(result, columns=['model_name', 'best_score_threshold', 'oof_score'])
            val_score_df.to_csv(self.cfg.TEST.BEST_THR_CSV_NAME, index=False)
            print(f'Best Score Threshold csv saved to: {self.cfg.TEST.BEST_THR_CSV_NAME}')

        if self.cfg.TEST.VAL_PRED_IMG_DIR and self.cfg.TEST.VISUALIZE_RESULTS:
            if self.cfg.TEST.SAVE_ALL_IMAGES:
                num_images = len(predictions)
            else:
                num_images = 10
            print(f'Number of plot images: {num_images}')
            for i in tqdm(range(num_images), total=num_images):
                gt_boxes = predictions[i]['gt_boxes'].copy()
                pred_boxes = predictions[i]['boxes'].copy()
                scores = predictions[i]['scores'].copy()
                class_ids = predictions[i]['class_ids'].copy()
                image_id = predictions[i]['image_id']
                indexes = np.where(scores>best_score_threshold)[0]
                pred_boxes = pred_boxes[indexes]
                scores = scores[indexes]
                class_ids = class_ids[indexes]
                
                self._save_pred_images(train_image_dir, image_id, pred_boxes, scores, class_ids, self.cfg.TEST.VAL_PRED_IMG_DIR, gt_boxes)
                

    def _load_models(self):
        """ Load cross valitation models

        Returns:
            [list]: model list
        """
        
        models = []
        for fold_number in range(self.model_cfg.DATA.N_FOLD):
            weight_path = glob.glob(f'{self.weight_dir}/fold{fold_number}*.pth')
            if len(weight_path) > 1:
                print(f'Found more than one weight path: {weight_path}')
                sys.exit()
            elif len(weight_path)==0:
                weight_path = glob.glob(f'{self.weight_dir}/best_*_fold{fold_number}.pth')
                if len(weight_path)==0:
                    print(f' No weight path found for Fold {fold_number}: {weight_path}')
                    return models
            model = get_model(self.model_cfg, mode='test', finetuned_weights=weight_path[0], device=self.device)
            model.eval()
            if self.device == torch.device('cuda'):
                model.cuda()         

            models.append(model)

        return models


    def _get_threshold(self, verbose=False):
        if os.path.exists(self.cfg.TEST.BEST_THR_CSV_NAME):
            val_score_df = pd.read_csv(self.cfg.TEST.BEST_THR_CSV_NAME)

            target_df = val_score_df[val_score_df['model_name']==self.cfg.MODEL_ID]
            score_threshold = target_df.iloc[0,1]
            best_final_score = target_df.iloc[0,2]

            if score_threshold is not None and verbose:
                print('-'*30, 'Applying Best Threshold for Test Dataset', '-'*30)
                print(f'[Model Name]: {self.cfg.MODEL_ID}')
                print(f'[Best Score Threshold]: {score_threshold}')
                print(f'[Out Of Fold mAP]: {best_final_score:.4f}')
        else:
            score_threshold = self.cfg.TEST.DETECTION_THRESHOLD
        print('Score threshold:', score_threshold)
        return score_threshold


    def _make_ensemble_predictions(self, images, image_ids, models, gts, score_threshold=0.25):
        """ Using all cross-validataion mnodels to predict on validation data

        Args:
            images (list): images for prediction
            image_ids (list): image names
            models (list): cross-validation models
            gts (list): ground truth bboxes
            score_threshold (float, optional): [description]. Defaults to 0.25.

        Raises:
            NotImplementedError: model shoud be EfficientDet or Faster RCNN

        Returns:
            results [list]: prediction list (bboxes(pascal-voc format), confidence scores, class ids, image ids, gt_boxes(optional))
            indexes [list]: index list of images larger than score threshold.
            gt_flag [bool]: if gt bboxes are available set as True.
        """
        
        if type(images) == torch.Tensor:
            images = images.to(self.device).float()
        else:
            images = torch.stack(images).to(self.device).float()

        predictions = []
        for idx, model in enumerate(models):
            with torch.no_grad():
                if 'EfficientDet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    img_size = torch.tensor([images[0].shape[-2:]] * self.cfg.TEST.BATCH_SIZE, dtype=torch.float).to(self.device)
                    outputs = model(images, torch.tensor([1]*images.shape[0], dtype=torch.float).to(self.device), img_size)
                elif 'fasterrcnn' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    outputs = model(images)
                else:
                    print(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
                    raise NotImplementedError

                results = []
                for i in range(images.shape[0]):
                    results, indexes, gt_flag = self._make_results(outputs=outputs, results=results, image_ids=image_ids, 
                                              index=i, mode='test', targets=gts, score_threshold=score_threshold)
                predictions.append(results)
        return predictions, indexes, gt_flag


    def _postprocess_image(self, predictions, image_ids, image_index, results, preds_list, gt_flag):
        """Ensemble Boxes and return results list

        Args:
            predictions
            image_ids (list): image names
            image_index (int): index of images
            results [list]: prediciton list for dataframe output
            preds_list [list]: prediciton list for evaluation
            gt_flag [bool]: if gt bboxes are available set as True.

        Raises:
            ValueError: data format must be 'pascal_voc' or 'coco'
            ValueError: each box coordinates list length must be equal.
        Returns:
            results [list]: prediciton list for dataframe output
            preds_list [list]: prediciton list for evaluation
        """

        boxes, scores, labels = run_ensemble_method(predictions, image_index=image_index, method_name=self.cfg.TEST.ENSEMBLE_BOXES.NAME, image_size=self.image_size, iou_thr=self.cfg.TEST.ENSEMBLE_BOXES.IOU_THR, 
                                                    skip_box_thr=self.cfg.TEST.ENSEMBLE_BOXES.SKIP_BOX_THR, sigma=self.cfg.TEST.ENSEMBLE_BOXES.SIGMA, thresh=self.cfg.TEST.ENSEMBLE_BOXES.THRESH, weights=self.cfg.TEST.ENSEMBLE_BOXES.WEIGHTS)

        boxes = (boxes*self.image_size_ratio).astype(np.int32).clip(min=0, max=self.model_cfg.MODEL.OUTPUT_SIZE-1)
        image_id = image_ids[image_index]

        if gt_flag:
            gt_boxes = [(prediction[image_index]['gt_boxes']).tolist() for prediction in predictions]
            gt_boxes = np.array(gt_boxes[0], np.int32)
        else:
            gt_boxes = None

        xmin_list, ymin_list, box2_list, box3_list = [],[],[],[]
            
        for batch_boxes in boxes.astype(int):
            xmin_list.append(batch_boxes[0])
            ymin_list.append(batch_boxes[1])
            if self.model_cfg.DATA.FORMAT == 'pascal_voc':
                box2_name = 'xmax'
                box3_name = 'ymax'
                box2_list.append(batch_boxes[2])
                box3_list.append(batch_boxes[3])
            elif self.model_cfg.DATA.FORMAT == 'coco':
                box2_name = 'width'
                box3_name = 'height'
                box2_list.append(batch_boxes[2]-batch_boxes[0])
                box3_list.append(batch_boxes[3]-batch_boxes[1])
            else:
                raise ValueError('Unknown data format. Set "pascal_voc" or "coco"')

        if len(xmin_list) == len(ymin_list) == len(box2_list) == len(box3_list):
            for box in range(len(xmin_list)):
                result = {
                    'image_id': image_id,
                    'image_path': self.test_img_dir + '/' + image_id,
                    'class_id': labels[box].astype(int),
                    'conf': scores[box],
                    'xmin': xmin_list[box],
                    'ymin': ymin_list[box],
                    box2_name: box2_list[box],
                    box3_name: box3_list[box],
                }
                results.append(result)
        else:
            raise ValueError(f'Each "xmin":{len(xmin_list)}, "ymin":{len(ymin_list)}, {box2_name}: {len(box2_list)}, {box3_name}: {len(box3_list)} length must be equal.')

        preds_list.append({
            'boxes': boxes,
            'class_ids': labels,
            'scores': scores,
            'gt_boxes': gt_boxes,
            'image_id': image_ids[image_index],
        })

        return results, preds_list


    def _apply_trasform(self, image):
        """ Apply transformations to an image

        Args:
            image (np.array): an input image for transformations

        Raises:
            NotImplementedError: Must load yaml file to apply Albumentations augmentation.
            NotImplementedError: Must use Albumentations for augmentation. 

        Returns:
            [Tensor]: a transformed image
        """

        if self.cfg.AUGMENTATION.FRAMEWORK == 'albumentations':
            test_augs = A.load(hydra.utils.to_absolute_path(self.cfg.AUGMENTATION.ALBUMENTATIONS.TEST.PATH), data_format='yaml')
        elif self.cfg.AUGMENTATION.FRAMEWORK == 'custom':
            print('Load yaml file to apply Albumentations augmentation.')
            raise NotImplementedError
        else:
            print('Detection task only supports augmentation using Albumentations')
            raise NotImplementedError

        image /= 255.0

        if test_augs:
            sample = {'image': image}
            sample = test_augs(**sample)
            image = sample['image']

        image = image.unsqueeze(dim=0)
        return image


    def _predict_single(self, models, images, image_ids, targets, score_threshold, results, preds_list):
        """ Predict on test images and make prediction results list

        Args:
            images (dataloader): test data loader
            models (list): cross-validation models
            score_threshold (float): confidence score threshold
            pred_images_dir (bool, optional): directory to save prediction images. Defaults to False.

        Returns:
            results [list]: prediciton list for dataframe output
            preds_list [list]: prediciton list for evaluation
            gt_flag [bool]: if gt bboxes are available set as True.
        """

        predictions, indexes, gt_flag = self._make_ensemble_predictions(images, image_ids, models, gts=targets, score_threshold=score_threshold)

        for i, image in enumerate(images):
            if i in indexes:
                results, preds_list = self._postprocess_image(predictions, image_ids, image_index=i, results=results, preds_list=preds_list, gt_flag=gt_flag)
        return results, preds_list, gt_flag


    def _predict_batch(self, test_data_loader, models, score_threshold):
        """ Predict on test dataset and make prediction results list

        Args:
            test_data_loader (Dataset): test data loader
            models (list): cross-validation models
            score_threshold (float): confidence score threshold
            pred_images_dir (bool, optional): directory to save prediction images. Defaults to False.

        Returns:
            results [list]: prediciton list for dataframe output
            preds_list [list]: prediciton list for evaluation
            gt_flag [bool]: if gt bboxes are available set as True.
        """

        results = []
        preds_list = []
        for images, targets, image_ids in tqdm(test_data_loader, total=len(test_data_loader)):
            results, preds_list, gt_flag = self._predict_single(models, images, image_ids, targets, score_threshold, results, preds_list)
        return results, preds_list, gt_flag


    def visualize_results(self, preds_list, gt_flag, score_threshold, num_plot_images=10, save_images=True):
        """ Visualize prediction results

        Args:
            preds_list (list): prediction results
            gt_flag (bool): plot gt bbox or not 
            score_threshold (float): confidence score threshold
            num_plot_images (int, optional): number of images to plot. if "TEST.SAVE_ALL_IMAGES" if False, Defaults to 10.
        """
        if self.cfg.TEST.TEST_PRED_IMG_DIR:
            if self.cfg.TEST.SAVE_ALL_IMAGES:
                num_images = len(preds_list)
            else:
                num_images = num_plot_images
            print(f'Number of plot images: {num_images}')
            test_image_dir = hydra.utils.to_absolute_path(self.cfg.TEST.TEST_IMAGE_DIR)

            for i in tqdm(range(num_images), total=num_images):
                if gt_flag:
                    gt_boxes = preds_list[i]['gt_boxes'].copy()
                else:
                    gt_boxes = None
                pred_boxes = preds_list[i]['boxes'].copy()
                scores = preds_list[i]['scores'].copy()
                class_ids = preds_list[i]['class_ids'].copy()
                image_id = preds_list[i]['image_id']
                indexes = np.where(scores>score_threshold)[0]
                pred_boxes = pred_boxes[indexes]
                scores = scores[indexes]
                class_ids = class_ids[indexes]
                self._save_pred_images(test_image_dir, image_id, pred_boxes, scores, class_ids, self.cfg.TEST.TEST_PRED_IMG_DIR, gt_boxes=gt_boxes, save_images=save_images)


    def get_final_score(self, preds_list, score_threshold):
        """ Run calculater to get score for prediction results

        Args:
            preds_list (list): prediction results
            score_threshold (float): confidence score threshold
        
        Raises:
            ValueError: The ensemble method name must be "WBF" or "NMW" or "SoftNMS" or "NMS".
        """
        final_score, conf_mats = calculate_final_score(preds_list, score_threshold)

        print('-'*30, 'Test Score', '-'*30)
        print(f'[Model Name]: {self.cfg.MODEL_ID}')
        print(f'[IoU Threshold]: {self.cfg.TEST.ENSEMBLE_BOXES.IOU_THR}')
        print(f'[Ensemble Boxes Method]: {self.cfg.TEST.ENSEMBLE_BOXES.NAME}')
        print(f'[Ensemble Weights]: {self.cfg.TEST.ENSEMBLE_BOXES.WEIGHTS}')

        if self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'WBF' or self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'NMW':
            #TODO: add bayesian optimization
            print(f'[Skip Box Threshold]: {self.cfg.TEST.ENSEMBLE_BOXES.SKIP_BOX_THR}')
        elif self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'SoftNMS':
            print(f'[Sigma]: {self.cfg.TEST.ENSEMBLE_BOXES.SIGMA}')
            print(f'[Threshold]: {self.cfg.TEST.ENSEMBLE_BOXES.THRESH}')

        print(f'[Score Threshold]: {score_threshold}')
        print(f'[mAP]: {final_score:.4f}')
        print('-'*60)

        if self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'WBF' or self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'NMW':
            result = [(self.cfg.MODEL_ID, score_threshold, round(final_score, 4), self.cfg.TEST.ENSEMBLE_BOXES.WEIGHTS, self.cfg.TEST.ENSEMBLE_BOXES.IOU_THR, self.cfg.TEST.ENSEMBLE_BOXES.SKIP_BOX_THR)]
            score_df = pd.DataFrame(result, columns=['model_name', 'score_threshold', 'score', 'ensemble_weight', 'iou_threshold', 'skip_box_threshold'])
        elif self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'SoftNMS':
            result = [(self.cfg.MODEL_ID, score_threshold, round(final_score, 4), self.cfg.TEST.ENSEMBLE_BOXES.WEIGHTS, self.cfg.TEST.ENSEMBLE_BOXES.IOU_THR, self.cfg.TEST.ENSEMBLE_BOXES.SIGMA, self.cfg.TEST.ENSEMBLE_BOXES.THRESH)]
            score_df = pd.DataFrame(result, columns=['model_name', 'score_threshold', 'score', 'ensemble_weight', 'iou_threshold', 'sigma', 'thresh'])
        elif self.cfg.TEST.ENSEMBLE_BOXES.NAME == 'NMS':
            result = [(self.cfg.MODEL_ID, score_threshold, round(final_score, 4), self.cfg.TEST.ENSEMBLE_BOXES.WEIGHTS, self.cfg.TEST.ENSEMBLE_BOXES.IOU_THR)]
            score_df = pd.DataFrame(result, columns=['model_name', 'score_threshold', 'score', 'ensemble_weight', 'iou_threshold'])
        else:
            raise ValueError('Ensemble Method name should be "WBF" or "NMW" or "SoftNMS" or "NMS"')

        score_df.to_csv(self.cfg.TEST.TEST_SCORE_CSV_NAME, index=False)
        print(f'Test score csv saved to: {self.cfg.TEST.TEST_SCORE_CSV_NAME}')

        return conf_mats


    def make_result_df(self, results, preds_list, gt_flag, score_threshold):
        """ Make dataframe containing predictions (if gt is available, it also includes gt bboxes and scores 

        Args:
            results (Optional[List], optional): result for saving as csv. Defaults to None.
            preds_list (Optional[List], optional): result for visualization and calculating scores. Defaults to None.
            gt_flag (bool): plot gt bbox or not 
            score_threshold (float): confidence score threshold

        Raises:         
            ValueError: data format must be 'pascal_voc' or 'coco'

        Returns:
            [pd.DataFrame]: result dataframe. Iif gt information is available, it includes gt bboxes, label and score.
        """
        if self.model_cfg.DATA.FORMAT == 'pascal_voc':
            preds_df_columns=['image_id', 'image_path', 'class_id', 'conf', 'xmin', 'ymin', 'xmax', 'ymax']
        elif self.model_cfg.DATA.FORMAT == 'coco':
            preds_df_columns=['image_id', 'image_path', 'class_id', 'conf', 'xmin', 'ymin', 'width', 'height']
        else:
            raise ValueError('Unknown data format. Set "pascal_voc" or "coco"')

        preds_df = pd.DataFrame(results, columns=preds_df_columns)

        if gt_flag:
            conf_mats = self.get_final_score(preds_list, score_threshold)
            score_df_columns=['image_id', 'confmat']
            score_df = pd.DataFrame(conf_mats, columns=score_df_columns)
        
            result_df = preds_df.merge(score_df, on='image_id', how='left')

            gt_df = self.get_gt_df()

            df = gt_df.merge(result_df, on='image_id', how='left')
            print('Made result dataframe with GT.')
            return df
        else:
            print('Made result dataframe [* No GT]')
            return preds_df


    def get_gt_df(self):
        """ Extract groud truth info from test dataframe

        Raises:         
            ValueError: data format must be 'pascal_voc' or 'coco'

        Returns:
            gt_df (pd.DataFrame): dataframe containing ground truth information
        """

        gt_df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.TEST.TEST_CSV_PATH), dtype={'image_id': str})

        if self.model_cfg.DATA.FORMAT == 'pascal_voc':
            target_cols = ['image_id', 'label', 'class_id', 'LabelName', 'xmin', 'ymin', 'xmax', 'ymax']
        elif self.model_cfg.DATA.FORMAT == 'coco':
            target_cols = ['image_id', 'label', 'class_id', 'LabelName', 'xmin', 'ymin', 'width', 'height']
        else:
            raise ValueError('Unknown data format. Set "pascal_voc" or "coco"')

        extract_cols = []
        for col in target_cols:
            if col in gt_df.columns:
                extract_cols.append(col)
                        
        gt_df = gt_df[extract_cols]
        gt_df = gt_df.rename(columns={'label': 'gt_class_id', 'class_id': 'gt_class_id', 'LabelName': 'gt_class_id', 'xmin': 'gt_xmin', 
                                    'ymin': 'gt_ymin', 'xmax': 'gt_xmax', 'ymax': 'gt_ymax','width': 'gt_width', 'height': 'gt_height'})
        
        return gt_df


    def predict(self, test_data_loader: Dataset):
        """ Run prediction on test dataset

        Args:
            test_data_loader (Dataset): dataset to predict.

        Raises:
            ValueError: Must set "test_data_loader" for dataset prediciton.

        Returns:
            df (pd.DataFrame): prediction results dataframe.
        """
                
        set_seed(self.cfg)

        if self.cfg.TEST.VISUALIZE_RESULTS:
            os.makedirs(self.cfg.TEST.TEST_PRED_IMG_DIR, exist_ok=True)
        
        score_threshold = self._get_threshold()

        if test_data_loader is not None:
            results, preds_list, gt_flag = self._predict_batch(test_data_loader=test_data_loader, models=self.models, score_threshold=score_threshold)

            if self.cfg.TEST.VISUALIZE_RESULTS:
                self.visualize_results(preds_list, gt_flag, score_threshold, num_plot_images=10, save_images=True)

            df = self.make_result_df(results, preds_list, gt_flag, score_threshold)
            return df

        else:
            raise ValueError('Set "test_data_loader" in arguments')


    def predict_image(self, image: Tuple, image_id: Optional[List]=['sample_image_1'], gt: Optional[Tuple]=None, results: Optional[List]=None, preds_list: Optional[List]=None, is_eval: Optional[bool]=False):
        """ Run prediction on a test image

        Args:
            image (Tuple): an image for prediction.
            image_id (List): an image name.
            gt (Tuple): ground truth bbox coordinates.
            results (Optional[List], optional): result for saving as csv. Defaults to None.
            preds_list (Optional[List], optional): result for visualization and calculating scores. Defaults to None.
            is_eval (Optional[bool], optional): if True, return prediction list for evaluation. Defaults to False.

        Raises:
            ValueError: Must set "image" for prediction.

        Returns:
            results (list): prediction results list to save in csv file.
        """
                
        set_seed(self.cfg)

        if self.cfg.TEST.VISUALIZE_RESULTS:
            os.makedirs(self.cfg.TEST.TEST_PRED_IMG_DIR, exist_ok=True)
        
        score_threshold = self._get_threshold()

        image = self._apply_trasform(image)

        if image is not None:
            results, preds_list, gt_flag = self._predict_single(self.models, image, image_id, gt, score_threshold, results, preds_list)
            if is_eval:
                return results, preds_list, gt_flag, score_threshold
            else:
                return results
        else:
            raise ValueError('Set "image" in arguments')
