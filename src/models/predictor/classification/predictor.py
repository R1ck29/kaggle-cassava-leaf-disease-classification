from __future__ import print_function

import glob
import logging
import os
import sys
from typing import Any, List, Optional, Tuple

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch._six
import torch.multiprocessing
import torch.utils.data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from src.data.generator.classification.build import (get_training_datasets,
                                                     load_augs)
from src.data.generator.classification.dataset import get_transform
from src.models.modeling.classification.get_model import get_model
from src.models.predictor.classification.mappings import person_attribute_names
from src.models.utils.classification.metrics import (get_pedestrian_metrics,
                                                     make_conf_mats)
from src.utils.pytorch.utils import set_seed
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


class Predictor:
    """ Run Classification model to predict on validation or test images """
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, weight_dir: Optional[str] = None):
        self.cfg = cfg
        self.model_cfg = model_cfg
        # A logger for this file
        self.log = logging.getLogger(__name__)
        self.image_size_height = self.model_cfg.MODEL.INPUT_SIZE.HEIGHT
        self.image_size_width = self.model_cfg.MODEL.INPUT_SIZE.WIDTH
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.device == torch.device('cpu'):
            self.log.info('Device: CPU')
        if weight_dir is None: 
            self.weight_dir = os.path.join(hydra.utils.get_original_cwd(), self.cfg.MODEL_PATH)
        else:
            self.weight_dir = weight_dir
        self.test_img_dir = hydra.utils.to_absolute_path(self.cfg.TEST.TEST_IMAGE_DIR)
        self.models = self._load_models()

        self.log.info(f'Test Images Path: {self.test_img_dir}')
        self.log.info(f'Test Annotations Path: {self.cfg.TEST.TEST_CSV_PATH}')


    def _split_labels(self, logits, data_type):
        """split logits into two attribute(age and gender)

        Args:
            logits (torch,tensor): logits from model
            data_type (str): dataset name

        Raises:
            NotImplementedError: dataset_type must be 'PA100k', 'PETA', 'RAP', 'RAP2'

        Returns:
            [torch.tensor]: logits for age class and gender class
        """
        if data_type == 'PA100k':
            gender_start_index = 3
        elif data_type == 'PETA':
            gender_start_index = 4
        elif data_type == 'RAP':
            gender_start_index = 3
        elif data_type == 'RAP2':
            gender_start_index = 4
        elif data_type == 'person_attribute':
            gender_start_index = 5
        elif data_type == 'person_attribute_demo':
            gender_start_index = 5
        else:
            raise NotImplementedError

        valid_logits_age = logits[:, :gender_start_index]
        valid_logits_gender = logits[:, gender_start_index:]
        return valid_logits_age, valid_logits_gender


    def _make_probabilities(self, logits, function_type, data_type):
        """Covert logits to probabilities using Softmax or Sigmoid function

        Args:
            logits (torch.tensor): logits from model
            function_type (str): softmax of sigmoid

        Raises:
            NotImplementedError: Activation function must be Softmax or Sigmoid

        Returns:
            [torch.tensor]: probabilities for each class
        """
        valid_logits_age, valid_logits_gender = self._split_labels(logits, data_type)
        if function_type == 'softmax':
            softmax = nn.Softmax(dim=1)
            valid_probs_age = softmax(valid_logits_age) #softmax(valid_logits_age) #torch.sigmoid(valid_logits_age)
            valid_probs_gender = softmax(valid_logits_gender) #torch.sigmoid(valid_logits_gender) #softmax(valid_logits_gender)
        elif function_type == 'sigmoid':
            valid_probs_age = torch.sigmoid(valid_logits_age)
            valid_probs_gender = torch.sigmoid(valid_logits_gender)
        else:
            raise NotImplementedError

        valid_probs = torch.cat([valid_probs_age, valid_probs_gender], 1)

        return valid_probs, valid_probs_age, valid_probs_gender


    def _make_results(self, results, class_ids, gt_labels, outputs, image_ids, mode, targets=None, threshold=None):
        """Make prediction list (image ids, gt labels, class ids)

        Args:
            results (list): prediction lists(labels)
            class_ids (list): stores predicted class ids. (ex.[0,1,0,0])
            gt_labels (list): stores gt class ids. (ex.[0,1,0,0])
            outputs (Any): model outputs 
            image_ids (list): image name
            mode (str): val or test
            targets (list, optional): ground truth boxes. Defaults to None.
            threshold (float, optional): threshold for confidence score. Defaults to None.

        Raises:
            NotImplementedError: model shoud be ResNet

        Returns:
            results [list]: prediction list(image ids, gt_label, class ids)
        """
        if 'resnet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME or 'efficientnet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
            valid_probs, valid_probs_age, valid_probs_gender = self._make_probabilities(outputs, function_type='softmax', data_type=self.model_cfg.DATA.DATA_ID)
            if targets[0] is None:
                self.gt_flag = False
                gt_age = None
                gt_gender = None
            else:
                self.gt_flag = True
                if isinstance(targets, tuple):
                    targets = torch.tensor(targets).detach().cpu().numpy()
                else:
                    targets = targets.detach().cpu().numpy()
                gt_age, gt_gender = self._split_labels(targets, self.model_cfg.DATA.DATA_ID)

                targets[targets == -1] = 0
                gt_age[gt_age == -1] = 0
                gt_gender[gt_gender == -1] = 0
                
                for i in range(targets.shape[0]):
                    gt1 = np.argwhere(targets[i] == 1.0).reshape(-1).tolist()
                    gt_str = " ".join(list(map(str, gt1)))
                    gt_labels.append(gt_str)

                # argsorted = preds_probs.argsort(axis=1)
                # binarized_prediction = self.binarize_prediction(preds_probs, threshold, argsorted, n_classes=self.model_cfg.DATA.NUM_CLASSES)
                # argsorted_age = preds_probs_age.argsort(axis=1)
                # binarized_prediction_age = self.binarize_prediction(preds_probs_age, threshold, argsorted_age, n_classes=5)
                # argsorted_gender = preds_probs_gender.argsort(axis=1)
                # binarized_prediction_gender = self.binarize_prediction(preds_probs_gender, threshold, argsorted_gender, n_classes=2)

            valid_probs = valid_probs.cpu().detach().numpy()
            valid_probs_age = valid_probs_age.cpu().detach().numpy()
            valid_probs_gender = valid_probs_gender.cpu().detach().numpy()

            if mode == 'test':
                valid_probs = (valid_probs > threshold).astype(int)
                valid_probs_age = (valid_probs_age > threshold).astype(int)
                valid_probs_gender = (valid_probs_gender > threshold).astype(int)

            for i in range(valid_probs.shape[0]):
                pred1 = np.argwhere(valid_probs[i] == 1.0).reshape(-1).tolist()
                pred_str = " ".join(list(map(str, pred1)))
                class_ids.append(pred_str)

        else:
            self.log.error(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
            raise NotImplementedError

        results.append({
            'image_id': image_ids,
            'gt_labels': gt_labels,
            'class_ids': class_ids,
            'gt_eval': targets,
            'gt_age_eval': gt_age,
            'gt_gender_eval': gt_gender,
            'preds_eval': valid_probs,
            'preds_age_eval': valid_probs_age,
            'preds_gender_eval': valid_probs_gender
        })
        return results


    def _predict_validation(self):
        """Predict on validation data

        Raises:
            NotImplementedError: model shoud be ResNet

        Returns:
            [list]: prediction list (image ids, gt_label, class ids)
        """
        
        predictions = []
        gt_labels = []
        class_ids = []
        for fold_number in range(self.model_cfg.DATA.N_FOLD):
            # self.log.info('-'*30, f'Prediction: Validation Fold {fold_number}', '-'*30)
            self.log.info(f'Prediction: Validation Fold {fold_number}')
            model = self.models[fold_number]
            
            datasets = get_training_datasets(self.model_cfg, fold_number, eval_oof=True)
            valid_dataset = datasets['valid']
                
            val_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=self.cfg.TEST.BATCH_SIZE,
                                                num_workers=self.model_cfg.SYSTEM.NUM_WORKERS,
                                                shuffle=False
                                                )
            
            for images, targets, image_ids in tqdm(val_loader, total=len(val_loader)):
                if type(images) == torch.Tensor:
                    images = images.to(self.device).float()
                else:
                    images = torch.stack(images).to(self.device).float()

                if 'resnet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME or 'efficientnet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    outputs = model(images)
                else:
                    self.log.error(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
                    raise NotImplementedError

                gt_labels = []
                class_ids = []
                predictions = self._make_results(outputs=outputs, results=predictions, class_ids=class_ids, gt_labels=gt_labels, image_ids=image_ids,
                                                     mode='val', targets=targets, threshold=None)
        return predictions


    def _convert_id_to_name(self, id_str_list, attr_name_mappings):
        """convert class id to class name

        Args:
            id_str_list (list): class id list 
            attr_name_mappings (dict): class name list for each dataset type

        Returns:
            [list]: class name list
        """
        label_ids = id_str_list.split()
        map_object = map(int, label_ids)
        id_num_list = list(map_object)

        attr_name_list = [attr_name_mappings[int(i)] for i in id_num_list]
        return attr_name_list


    def _save_pred_images(self, test_image_dir, image_id, class_ids, pred_images_dir, gt_labels=None, save_images=True):
        """Plot labels on images and and save them

        Args:
            test_image_dir (str): directory cotaining test images 
            image_id (str): image name
            class_ids (list): object class id
            pred_images_dir (str): directory to save prediction images
            gt_labels (list, optional): ground truth labels. Defaults to None.
            save_images (bool, optional): if True, save prediction images. Defaults to True.

        """
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))

        image_file_format = os.listdir(test_image_dir)[0].split('.')[-1]
        if image_file_format not in image_id and '.' not in image_id:
            img_name = f'{image_id}.{image_file_format}'
        else:
            img_name = image_id

        img_ = cv2.imread(f'{test_image_dir}/{img_name}')
        if img_ is None:
            self.log.error(f'Can not find image: {img_name}')
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = cv2.resize(img_, (self.model_cfg.MODEL.OUTPUT_SIZE.WIDTH, self.model_cfg.MODEL.OUTPUT_SIZE.HEIGHT))

        pred_attr_names = self._convert_id_to_name(class_ids, person_attribute_names[self.model_cfg.DATA.DATA_ID])
        pred_list_half_size = int(len(pred_attr_names)/2)

        ax.set_axis_off()
        if gt_labels is not None:
            gt_attr_names = self._convert_id_to_name(gt_labels, person_attribute_names[self.model_cfg.DATA.DATA_ID])
            gt_list_half_size = int(len(gt_attr_names)/2)

            ax.set_title("image id: {}\ngt: {}\n{}\n\npred: {}\n{}".format(image_id, gt_attr_names[:gt_list_half_size], 
                            gt_attr_names[gt_list_half_size:], pred_attr_names[:pred_list_half_size], pred_attr_names[pred_list_half_size:]))
        else:
            ax.set_title("image id: {}\npred: {}\n{}".format(image_id, pred_attr_names[:pred_list_half_size], pred_attr_names[pred_list_half_size:]))

        ax.imshow(img_)
        if save_images:
            if image_file_format not in image_id and '.' not in image_id:
                img_id = image_id
            else:
                img_id = image_id.split('.')[0]
            fig.savefig(f'{pred_images_dir}/pred_{img_id}.jpg')
            plt.close(fig)


    # def binarize_prediction(probabilities, threshold: float, argsorted=None,
    #                         min_labels=1, max_labels=10):
    #     """ Return matrix of 0/1 predictions, same shape as probabilities.
    #     """
    #     assert probabilities.shape[1] == N_CLASSES
    #     if argsorted is None:
    #         argsorted = probabilities.argsort(axis=1)
    #     max_mask = _make_mask(argsorted, max_labels)
    #     min_mask = _make_mask(argsorted, min_labels)
        
    #     prob_mask = []
    #     for prob in probabilities:
    #         prob_mask.append(prob > prob.max()/7)
            
    #     prob_mask = np.array(prob_mask, dtype=np.int)
        
    #     return (max_mask & prob_mask) | min_mask

    def binarize_prediction(self, probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10, n_classes=None):
        """ Return matrix of 0/1 predictions, same shape as probabilities.
        """
        assert probabilities.shape[1] == n_classes
        if argsorted is None:
            argsorted = probabilities.argsort(axis=1)
        max_mask = self._make_mask(argsorted, max_labels)
        min_mask = self._make_mask(argsorted, min_labels)
        prob_mask = probabilities > threshold
        return (max_mask & prob_mask) | min_mask


    def _make_mask(self, argsorted, top_n: int):
        mask = np.zeros_like(argsorted, dtype=np.uint8)
        col_indices = argsorted[:, -top_n:].reshape(-1)
        row_indices = [i // top_n for i in range(len(col_indices))]
        mask[row_indices, col_indices] = 1
        return mask


    def _run_score_calculation(self, predictions, threshold=0.5):
        """calculate scores with gt and predictions

        Args:
            predictions (list): list of prediction
            threshold (float, optional): threshold for prediction. Defaults to 0.5.

        Returns:
            [dict]: contains Accuracy, Recall, Precision, F1 scores
        """
        gt_label = self._make_series(predictions, 'gt_eval')
        preds_probs = self._make_series(predictions, 'preds_eval')
        gt_age = self._make_series(predictions, 'gt_age_eval')
        preds_probs_age = self._make_series(predictions, 'preds_age_eval')
        gt_gender = self._make_series(predictions, 'gt_gender_eval')
        preds_probs_gender = self._make_series(predictions, 'preds_gender_eval')

        scores = get_pedestrian_metrics(gt_label, preds_probs, threshold=threshold)
        scores_age = get_pedestrian_metrics(gt_age, preds_probs_age, threshold=threshold)
        scores_gender = get_pedestrian_metrics(gt_gender, preds_probs_gender, threshold=threshold)
        
        return scores, scores_age, scores_gender


    def _find_best_threshold(self, predictions):
        """ Search best threshold for best score """
        best_final_score, best_threshold = 0, 0
        best_acc, best_f1 = 0, 0
        # self.log.info('-'*20,'Finding best threshold for best score', '-'*20)
        self.log.info('Finding best threshold for best score')
        counter = 0
        # for threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0]):
        threshold_range = [0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.4, 0.45, 0.5]
        for threshold in tqdm(threshold_range, total=len(threshold_range)):
            valid_result, _, _ = self._run_score_calculation(predictions, threshold=threshold)
            ma = valid_result.ma
            acc = valid_result.instance_acc
            f1 = valid_result.instance_f1

            if (ma > best_final_score) and (acc > best_acc) and (f1 > best_f1):
                counter += 1
                best_final_score = ma
                best_acc = acc
                best_f1 = f1
                best_threshold = threshold
        if counter == 1:
            self.log.warning('Any threshold gave same score.')
            self.log.warning('Setting threshold to 0.5')
            best_threshold = 0.5 

        return best_final_score, best_acc, best_f1, best_threshold


    def evaluate(self):
        """ Run out-of-fold evaluation to find best score threshold. """
        set_seed(self.cfg)
        self.mode = 'valid'

        train_image_dir = hydra.utils.to_absolute_path(self.model_cfg.DATA.TRAIN_IMAGE_DIR)

        if self.cfg.TEST.VISUALIZE_RESULTS:
            os.makedirs(self.cfg.TEST.VAL_PRED_IMG_DIR, exist_ok=True)
    
        # Prediction on Validation Dataset
        predictions = self._predict_validation()

        if self.cfg.TEST.FIND_BEST_THR:
            best_final_score, best_acc, best_f1, best_threshold = self._find_best_threshold(predictions)
            # self.log.info('-'*30)
            self.log.info(f'[Model Path]: {self.cfg.MODEL_PATH}') #f'[Model Name]: {self.cfg.MODEL_ID}')
            self.log.info(f'[Best Threshold]: {best_threshold}')
            self.log.info(f'[OOF mA]: {best_final_score:.4f}')
            self.log.info(f'[OOF Accuracy]: {best_acc:.4f}')
            self.log.info(f'[OOF F1]: {best_f1:.4f}')
            # self.log.info('-'*30)
            result = [(self.cfg.MODEL_ID, best_threshold, round(best_final_score, 4))]
            val_score_df = pd.DataFrame(result, columns=['model_name', 'best_threshold', 'oof_score'])
            val_score_df.to_csv(self.cfg.TEST.BEST_THR_CSV_NAME, index=False)
            self.log.info(f'Best Threshold csv saved to: {self.cfg.TEST.BEST_THR_CSV_NAME}')
            threshold = best_threshold
        else:
            self.log.info(f'Threshold for Validation: {self.cfg.TEST.THRESHOLD}')
            threshold = self.cfg.TEST.THRESHOLD

        val_result = self._make_result_df(predictions, threshold)
        val_pkl_name = 'valid_result.pkl'
        val_result.to_pickle(val_pkl_name)
        self.log.info(f'Validation Results pickle saved to: {val_pkl_name}')

        if self.cfg.TEST.VAL_PRED_IMG_DIR and self.cfg.TEST.VISUALIZE_RESULTS:
            if self.cfg.TEST.SAVE_ALL_IMAGES:
                num_images = len(predictions)
            else:
                num_images = 10
            self.log.info(f'Number of plot images: {num_images}')
            for i in tqdm(range(num_images), total=num_images):
                try:
                    gt_labels = predictions[i]['gt_labels'][i]
                    class_ids = predictions[i]['class_ids'][i]
                    image_ids = predictions[i]['image_id'][i]
                    # indexes = np.where(scores>best_threshold)[0]
                    # pred_boxes = pred_boxes[indexes]
                    # scores = scores[indexes]
                    # class_ids = class_ids[indexes]
                    
                    self._save_pred_images(train_image_dir, image_ids, class_ids, self.cfg.TEST.VAL_PRED_IMG_DIR, gt_labels=gt_labels)
                except IndexError as index_error:
                    self.log.error(f'Index {i}: {index_error}')
                    self.log.error('Finished Plotting pred images')
                    break
                

    def _load_models(self):
        """ Load cross valitation models

        Returns:
            [list]: model list
        """
        
        models = []
        for fold_number in range(self.model_cfg.DATA.N_FOLD):
            if self.model_cfg.DATA.N_FOLD == 1 and fold_number == 1:
                self.log.info(f'Loading model only Fold {fold_number}')
                break
            monitor_name = self.cfg.TEST.BEST_WEIGHT_TYPE
            weight_path = glob.glob(f'{self.weight_dir}/fold{fold_number}*.pth')
            if len(weight_path) > 1:
                self.log.error(f'Found more than one weight path: {weight_path}')
                sys.exit()
            elif len(weight_path)==0 or monitor_name not in weight_path[0]:
                self.log.warning(f'No weight path found : fold{fold_number}*_{monitor_name}.pth')
                weight_path = glob.glob(f'{self.weight_dir}/best_{monitor_name}_fold{fold_number}.pth')
                if len(weight_path)==0:
                    self.log.warning(f'No weight path found : {self.weight_dir}/best_{monitor_name}_fold{fold_number}.pth')
                    return models
                elif monitor_name not in weight_path[0]:
                    self.log.warning(f'[{monitor_name}] should be in {weight_path[0]}')
            self.weight_path = weight_path[0]
            model = get_model(self.model_cfg, mode='test', finetuned_weights=weight_path[0], device=self.device)
            self.log.info(f'Loaded model Fold {fold_number}')
            model.eval()
            if self.device == torch.device('cuda'):
                model.cuda()         

            models.append(model)
        return models


    def _get_threshold(self, verbose=False):
        """Get the threshold for prediction

        Args:
            verbose (bool, optional): if true, print threshold and score. Defaults to False.

        Returns:
            [float]: the threshold for prediction
        """
        if os.path.exists(self.cfg.TEST.BEST_THR_CSV_NAME):
            val_score_df = pd.read_csv(self.cfg.TEST.BEST_THR_CSV_NAME)

            target_df = val_score_df[val_score_df['model_name']==self.cfg.MODEL_ID]
            threshold = target_df.iloc[0,1]
            best_final_score = target_df.iloc[0,2]

            if threshold is not None and verbose:
                # self.log.info('-'*30, 'Applying Best Threshold for Test Dataset', '-'*30)
                self.log.info('Applying Best Threshold for Test Dataset')
                self.log.info(f'[Model Path]: {self.cfg.MODEL_PATH}') #f'[Model Name]: {self.cfg.MODEL_ID}')
                self.log.info(f'[Best Threshold]: {threshold}')
                self.log.info(f'[Out Of Fold mA]: {best_final_score:.4f}')
        else:
            threshold = self.cfg.TEST.THRESHOLD
        return threshold


    def _make_ensemble_predictions(self, images, image_ids, models, gts, results, threshold=0.25):
        """ Using all cross-validataion mnodels to predict on validation data

        Args:
            images (list): images for prediction
            image_ids (list): image names
            models (list): cross-validation models
            gts (list): ground truth label
            results [list]: prediction list (image ids, class ids, gt_label)
            threshold (float, optional): [description]. Defaults to 0.25.

        Raises:
            NotImplementedError: model shoud be ResNet

        Returns:
            results [list]: prediction list (image ids, class ids, gt_label)
        """
        
        if type(images) == torch.Tensor:
            images = images.to(self.device).float()
        else:
            images = torch.stack(images).to(self.device).float()

        ensemble_predictions = []
        for idx, model in enumerate(models):
            with torch.no_grad():
                if 'resnet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME or 'efficientnet' in self.model_cfg.MODEL.BACKBONE.CLASS_NAME:
                    outputs = model(images)
                else:
                    self.log.error(f'Model is not defined. {self.model_cfg.MODEL.BACKBONE.CLASS_NAME}')
                    raise NotImplementedError

                # for i in range(images.shape[0]):
                gt_labels = []
                class_ids = []
                results = self._make_results(outputs=outputs, results=results, class_ids=class_ids, gt_labels=gt_labels, image_ids=image_ids,
                                            mode='test', targets=gts, threshold=threshold)

                # TODO: apply np.mean to each model prediction
                # if idx >= 1:
                #     self.log.info('making ensemble result list')
                # ensemble_predictions.append(results)
        # if len(models) > 1:
        return results #ensemble_predictions


    def _apply_trasform(self, image):
        """ Apply transformations to an image

        Args:
            image (np.array): an input image for transformations

        Raises:
            NotImplementedError: Must use Albumentations for augmentation. 

        Returns:
            [Tensor]: a transformed image
        """
        if 'albumentations_classification' in self.cfg.AUGMENTATION.FRAMEWORK:
            self.log.info('Using albumentations for transformations')
            test_augs = load_augs(self.model_cfg['ALBUMENTATIONS']['TEST']['AUGS'])
        elif self.cfg.AUGMENTATION.FRAMEWORK == 'custom':
            _, test_augs = get_transform(self.model_cfg)
        else:
            self.log.error('Classification task only supports augmentation using Albumentations')
            raise NotImplementedError

        # image /= 255.0

        if test_augs:
            if 'albumentations_classification' in self.cfg.AUGMENTATION.FRAMEWORK:
                albu_dict = {'image': image}
                transorm = test_augs(**albu_dict)
                image = transorm['image']
            elif self.cfg.AUGMENTATION.FRAMEWORK == 'custom':
                image = test_augs(image)

        image = image.unsqueeze(dim=0)
        return image


    def _predict_batch(self, test_data_loader, models, threshold):
        """ Predict on test dataset and make prediction results list

        Args:
            test_data_loader (Dataset): test data loader
            models (list): cross-validation models
            threshold (float): confidence score threshold

        Returns:
            results [list]: prediciton list for dataframe output
        """

        results = []
        for images, targets, image_ids in tqdm(test_data_loader, total=len(test_data_loader)):
            results = self._make_ensemble_predictions(images, image_ids, models, gts=targets, results=results, threshold=threshold)
        return results


    def visualize_results(self, preds_list, num_plot_images=10, save_images=True):
        """ Visualize prediction results

        Args:
            preds_list (list): prediction results
            gt_flag (bool): plot gt label or not 
            num_plot_images (int, optional): number of images to plot. if "TEST.SAVE_ALL_IMAGES" if False, Defaults to 10.
        """
        if self.cfg.TEST.TEST_PRED_IMG_DIR:
            if self.cfg.TEST.SAVE_ALL_IMAGES:
                num_images = len(preds_list)
            else:
                num_images = num_plot_images
            self.log.info(f'Number of plot images: {num_images}')
            test_image_dir = hydra.utils.to_absolute_path(self.cfg.TEST.TEST_IMAGE_DIR)

            for i in tqdm(range(num_images), total=num_images):
                try:
                    if self.gt_flag:
                        gt_labels = preds_list[i]['gt_labels'][i]
                    else:
                        gt_labels = None
                    class_ids = preds_list[i]['class_ids'][i]
                    image_ids = preds_list[i]['image_id'][i]

                    self._save_pred_images(test_image_dir, image_ids, class_ids, self.cfg.TEST.TEST_PRED_IMG_DIR, gt_labels=gt_labels, save_images=save_images)
                except IndexError as index_error:
                    self.log.error(f'Index {i}: {index_error}')
                    self.log.error('Finished Plotting pred images')
                    break

    def _print_results(self, score, threshold, name):
        """print classification metrics

        Args:
            score (dict): contains classification metrics scores
            threshold (float): threshold for prediction
            name (str): model name
        """
        # self.log.info('-'*30, f'{name}', '-'*30)
        self.log.info(f'{name}')
        self.log.info(f'[Model Path]: {self.cfg.MODEL_PATH}') #{self.cfg.MODEL_ID}
        self.log.info(f'[Threshold]: {threshold}')
        self.log.info(f'[mA]: {score.ma:.4f}')
        self.log.info(f'[Accuracy]: {score.instance_acc:.4f}')
        self.log.info(f'[Precision]: {score.instance_prec:.4f}')
        self.log.info(f'[Recall]: {score.instance_recall:.4f}')
        self.log.info(f'[F1]: {score.instance_f1:.4f}')
        # self.log.info('-'*60)


    def _get_final_score(self, preds_list, threshold):
        """ Run calculater to get score for prediction results

        Args:
            preds_list (list): prediction results
            threshold (float): confidence score threshold

        Returns:
            [dict]: evaluation scores
        """
        final_score, scores_age, scores_gender = self._run_score_calculation(preds_list, threshold=threshold)

        self._print_results(final_score, threshold, name='All')
        self._print_results(scores_age, threshold, name='Age')
        self._print_results(scores_gender, threshold, name='Gender')

        result = [(self.cfg.MODEL_ID, threshold, round(final_score.ma, 4), round(final_score.instance_acc, 4), round(final_score.instance_prec, 4),
         round(final_score.instance_recall, 4), round(final_score.instance_f1, 4), round(scores_age.ma, 4), round(scores_age.instance_acc, 4), round(scores_age.instance_prec, 4),
         round(scores_age.instance_recall, 4), round(scores_age.instance_f1, 4), round(scores_gender.ma, 4), round(scores_gender.instance_acc, 4), round(scores_gender.instance_prec, 4),
         round(scores_gender.instance_recall, 4), round(scores_gender.instance_f1, 4))]
        score_df = pd.DataFrame(result, columns=['model_name', 'threshold', 'ma', 'accuracy', 'precision', 'recall', 'f1', 
                                                'age_ma', 'age_accuracy', 'age_precision', 'age_recall', 'age_f1', 'gender_ma', 'gender_accuracy', 'gender_precision', 'gender_recall', 'gender_f1'])
        if self.mode == 'test':
            score_df.to_csv(self.cfg.TEST.TEST_SCORE_CSV_NAME, index=False)
            self.log.info(f'Score csv saved to: {self.cfg.TEST.TEST_SCORE_CSV_NAME}')
        elif self.mode == 'valid':
            score_df.to_csv(self.cfg.TEST.VALID_SCORE_CSV_NAME, index=False)
            self.log.info(f'Score csv saved to: {self.cfg.TEST.VALID_SCORE_CSV_NAME}')
        return final_score, scores_age, scores_gender


    def _make_series(self, results, key_name):
        """make series of data for dataframe

        Args:
            results (list): prediction list
            key_name (str): key name for values

        Returns:
            [array]: numpy array of predictions
        """
        key_values = [results[i][key_name] for i in range(len(results))]
        series = np.concatenate(key_values, axis=0)
        return series


    def _plot_confusion_matrix_multi_label(self, data, labels, output_filename=None):
        """Plot confusion matrix using heatmap.
    
        Args:
            data (list of list): List of lists with confusion matrix data.
            labels (list): Labels which will be plotted across x and y axis.
            output_filename (str): Path to output file.
    
        """
        sn.set(color_codes=True)
        plt.figure(1, figsize=(12, 9))
    
        plt.title("Confusion Matrix")
    
        sn.set(font_scale=1.4)
        ax = sn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'},fmt="d")
    
        # ax.set_xticklabels(labels)
        # ax.set_yticklabels(labels)
    
        ax.set(ylabel="True Label", xlabel="Predicted Label", title=labels)

        if output_filename is not None:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            self.log.info(f'Confusion Matrix saved to: {hydra.utils.to_absolute_path(output_filename)}')
            plt.close()
        else:
            plt.show()


    def _plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title=None, multi_label=False,
                                index = None,
                                cmap=plt.cm.Blues,
                                output_filename=None):
        """
        Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
                
            # Compute confusion matrix        
        if multi_label:
            cm = multilabel_confusion_matrix(y_true, y_pred)
        else:
            cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        
        if index is not None:
            cm = cm[index]
            classes = classes[index]
            
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.log.info("Normalized confusion matrix")
        else:
            self.log.info('Confusion matrix, without normalization')
            

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, fontsize=15)
        plt.yticks(tick_marks, fontsize=15)
        plt.xlabel('Predicted label',fontsize=25)
        plt.ylabel('True label', fontsize=25)
        plt.title(title, fontsize=30)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.15)
        cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
        cbar.ax.tick_params(labelsize=20)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        fontsize=20,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if output_filename is not None:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            self.log.info(f'Confusion Matrix saved to: {hydra.utils.to_absolute_path(output_filename)}')
            plt.close()
        return ax


    def _make_confusion_matrix(self, predictions, threshold=0.5):
        """make confusion matrix and save as images

        Args:
            predictions (list): model predictions
            threshold (float, optional): the threshold for predictions. Defaults to 0.5.

        Returns:
            [array]: array of confusion matrix (ex. [tp, fp, fn, tn])
        """
        gt_label = self._make_series(predictions, 'gt_eval')
        preds_probs = self._make_series(predictions, 'preds_eval')
        gt_age = self._make_series(predictions, 'gt_age_eval')
        preds_probs_age = self._make_series(predictions, 'preds_age_eval')
        gt_gender = self._make_series(predictions, 'gt_gender_eval')
        preds_probs_gender = self._make_series(predictions, 'preds_gender_eval')

        conf_mats_age = make_conf_mats(gt_age, preds_probs_age, threshold=threshold)
        conf_mats_gender = make_conf_mats(gt_gender, preds_probs_gender, threshold=threshold)

        pred_label = preds_probs > threshold
        pred_label_age = preds_probs_age > threshold
        pred_label_gender = preds_probs_gender > threshold

        cm = multilabel_confusion_matrix(gt_label, pred_label)

        labels_all = np.array(person_attribute_names[self.model_cfg.DATA.DATA_ID])

        cm_dir = f'./confusion_matrix/{self.mode}'
        if not os.path.exists(cm_dir):
            os.makedirs(cm_dir, exist_ok=True)

        for i in range(len(labels_all)):
            self._plot_confusion_matrix_multi_label(cm[i], labels_all[i], output_filename=f'{cm_dir}/cm_multi_label_{labels_all[i]}.png')

        self._plot_confusion_matrix(gt_age.argmax(axis=1), pred_label_age.argmax(axis=1), 
                                    classes=labels_all[:5],
                                    normalize=False,
                                    multi_label=False,
                                    output_filename=f'{cm_dir}/cm_age.png',
                                    title='Confusion matrix')

        self._plot_confusion_matrix(gt_gender.argmax(axis=1), pred_label_gender.argmax(axis=1), 
                                    classes=labels_all[-2:],
                                    normalize=False,
                                    multi_label=False,
                                    output_filename=f'{cm_dir}/cm_gender.png',
                                    title='Confusion matrix')
        return conf_mats_age, conf_mats_gender


    def get_gt_df(self):
        """ Extract groud truth info from test dataframe

        Returns:
            gt_df (pd.DataFrame): dataframe containing ground truth information
        """

        gt_df = pd.read_csv(hydra.utils.to_absolute_path(self.cfg.TEST.TEST_CSV_PATH), dtype={'image_id': str})

        target_cols = ['image_id', 'attr_name','source','image_path']


        extract_cols = []
        for col in target_cols:
            if col in gt_df.columns:
                extract_cols.append(col)
                        
        gt_df = gt_df[extract_cols]
        gt_df = gt_df.rename(columns={'attr_name': 'gt_attr_name'})
        
        return gt_df


    def _make_result_df(self, results, threshold):
        """ Make dataframe containing predictions (if gt is available, it also includes gt and scores 

        Args:
            results (Optional[List], optional): result for saving as csv. Defaults to None.
            threshold (float): confidence score threshold

        Returns:
            [pd.DataFrame]: result dataframe. If gt information is available, it includes gt labels and confusion matrix.
        """
        image_ids = self._make_series(results, 'image_id')
        class_ids = self._make_series(results, 'class_ids')

        if self.gt_flag:
            # self.log.info('-'*30, f'Prediction on {self.mode}', '-'*30)
            self.log.info(f'Prediction on {self.mode}')
            final_score, scores_age, scores_gender = self._get_final_score(results, threshold)

            gt_labels = self._make_series(results, 'gt_labels')
            conf_mats_age, conf_mats_gender = self._make_confusion_matrix(results)

            result_df = pd.DataFrame({'image_id': image_ids, 'gt_label': gt_labels, 'class_id': class_ids, 'conf_mat_age': conf_mats_age, 'conf_mat_gender': conf_mats_gender})

            gt_df = self.get_gt_df()

            df = gt_df.merge(result_df, on='image_id', how='left')
            self.log.info('Made result dataframe with GT.')
            return df
        else:
            result_df = pd.DataFrame({'image_id': image_ids, 'class_id': class_ids})
            self.log.info('Made result dataframe [* No GT]')
            
            return result_df


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

        self.mode='test'

        if self.cfg.TEST.VISUALIZE_RESULTS:
            os.makedirs(self.cfg.TEST.TEST_PRED_IMG_DIR, exist_ok=True)
        
        threshold = self._get_threshold()

        self.log.info(f'Loaded model weight: {self.weight_path}')

        if test_data_loader is not None:
            results = self._predict_batch(test_data_loader=test_data_loader, models=self.models, threshold=threshold)

            if self.cfg.TEST.VISUALIZE_RESULTS:
                self.visualize_results(results, num_plot_images=10, save_images=True)

            df = self._make_result_df(results, threshold)
            return df

        else:
            raise ValueError('Set "test_data_loader" in arguments')


    def predict_image(self, image: Tuple, image_id: Optional[List]=['sample_image_1'], gt: Optional[Tuple]=None, results: Optional[List]=None, preds_list: Optional[List]=None, is_eval: Optional[bool]=False):
        """ Run prediction on a test image

        Args:
            image (Tuple): an image for prediction.
            image_id (List): an image name.
            gt (Tuple): ground truth labels.
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
        
        threshold = self._get_threshold()

        image = self._apply_trasform(image)

        self.log.info(f'Loaded model weight: {self.weight_path}')

        if image is not None:
            results = self._make_ensemble_predictions(image, image_id, self.models, gts=gt, results=results, threshold=threshold)
            if is_eval:
                return results, preds_list, threshold
            else:
                return results
        else:
            raise ValueError('Set "image" in arguments')
