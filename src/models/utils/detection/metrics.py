import numba
import numpy as np
from ensemble_boxes import *
from numba import jit


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


## MAP calculation
@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_metrics(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates metrics for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) TP, FP, FN
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):
        #TODO: fix
        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx, threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp, fp, fn


@jit(nopython=True)
def calculate_image_metrics(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision and Confusion Matrix[TP, FP, FN, TN].

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision, Confusion Matrix[TP, FP, FN, TN]
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    image_tp = 0.0
    image_fp = 0.0
    image_fn = 0.0
    image_conf_mat = np.zeros(4)
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None
    for threshold in thresholds:
        tp, fp, fn = calculate_metrics(gts.copy(), preds, threshold=threshold, form=form, ious=ious)
        precision_at_threshold = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
        image_precision += precision_at_threshold / n_threshold
        image_tp += tp / n_threshold
        image_fp += fp / n_threshold
        image_fn += fn / n_threshold
    image_conf_mat[0] = image_tp
    image_conf_mat[1] = image_fp
    image_conf_mat[2] = image_fn

    return image_precision, image_conf_mat


def calculate_final_score(all_predictions, score_threshold, data_form='pascal_voc', iou_thr_list=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    final_scores = []
    conf_mats = []
    # Numba typed list!
    iou_thresholds = numba.typed.List()
    for x in iou_thr_list:
        iou_thresholds.append(x)
        
    for i in range(len(all_predictions)):
        gt_boxes = all_predictions[i]['gt_boxes'].copy()
        pred_boxes = all_predictions[i]['boxes'].copy()
        scores = all_predictions[i]['scores'].copy()
        image_id = all_predictions[i]['image_id']

        indexes = np.where(scores>score_threshold)
        pred_boxes = pred_boxes[indexes]
        scores = scores[indexes]
        
        # bugfix: conf should be descending
        rank = np.argsort(scores)[::-1]
        scores = scores[rank]
        pred_boxes = pred_boxes[rank]

        image_precision, image_conf_mat = calculate_image_metrics(gt_boxes, pred_boxes,thresholds=iou_thresholds,form=data_form)
        final_scores.append(image_precision)
        conf_mat = {'image_id': image_id,'confmat': image_conf_mat}
        conf_mats.append(conf_mat)

    return np.mean(final_scores), conf_mats


def run_ensemble_method(predictions, image_index, method_name, image_size=1024, iou_thr=0.6, skip_box_thr=0.0, sigma=0.5, thresh=0.001, weights=None):
    """Run One of Bboxes Ensemble Methods available below

    Args:
        predictions (list): prediction list containing boxes, scores, and class_ids
        image_index (int): index number of images
        method_name (str): "WBF" or "NMW" or "SoftNMS" or "NMS"
        image_size (int, optional): model input image size. Defaults to 1024.
        iou_thr (float, optional): IoU value for boxes to be a match. Defaults to 0.6.
        skip_box_thr (float, optional): exclude boxes with score lower than this variable. Defaults to 0.45.
        sigma (float, optional): Sigma value for SoftNMS. Defaults to 0.5.
        thresh (float, optional): threshold for boxes to keep (important for SoftNMS). Defaults to 0.001.
        weights (list, optional): list of weights for each model. Default: None, which means weight == 1 for each model. Defaults to None.

    Raises:
        ValueError: The ensemble method name must be "WBF" or "NMW" or "SoftNMS" or "NMS".

    Returns:
        [list]: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        [list]: scores: confidence scores
        [list]: labels: boxes labels
    """
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [prediction[image_index]['class_ids'].tolist()  for prediction in predictions]
    if method_name == 'WBF':
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif method_name == 'NMW':
        boxes, scores, labels = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif method_name == 'SoftNMS':
        boxes, scores, labels = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=thresh)
    elif method_name == 'NMS':
        boxes, scores, labels = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    else:
        raise ValueError('Ensemble Method name should be "WBF" or "NMW" or "SoftNMS" or "NMS"')
    boxes = boxes*(image_size-1)
    return boxes, scores, labels