import os
from os.path import join, dirname
import sys
import cv2
import munkres
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from torchvision.transforms import transforms

sys.path.append(join(dirname(__file__), "../.."))
from src.models.modeling.keypoint.higher_hrnet import get_pose_net
from src.models.predictor.keypoint.evaluate import Evaluator
from src.models.utils.keypoint import get_multi_scale_size, resize_align_multi_scale, get_multi_stage_outputs, aggregate_results, get_final_preds, bbox_iou
from src.visualize import keypoint_visualization

def py_max_match(scores):
    m = munkres.Munkres()
    assoc = m.compute(scores)
    assoc = np.array(assoc).astype(np.int32)
    return assoc

# derived from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/
class HeatmapParser(object):
    def __init__(self,
                 num_joints=17,
                 joint_set='coco',
                 max_num_people=30,
                 nms_kernel=5, nms_stride=1, nms_padding=2,
                 detection_threshold=0.1, tag_threshold=1., use_detection_val=True, ignore_too_much=True
                 ):
        """
        Heatmap Parser running on pytorch
        """
        assert joint_set in ('coco', 'crowdpose', 'mpii', 'hand'), joint_set

        self.num_joints = num_joints
        self.joint_set = joint_set
        self.max_num_people = max_num_people
        self.tag_per_joint = True
        self.maxpool = torch.nn.MaxPool2d(nms_kernel, nms_stride, nms_padding)
        self.detection_threshold = detection_threshold
        self.tag_threshold = tag_threshold
        self.use_detection_val = use_detection_val
        self.ignore_too_much = ignore_too_much

    def nms(self, det):
        maxm = self.maxpool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match_by_tag_torch(self, data):
        joint_order = keypoint_visualization.joints_dict()[self.joint_set]['order']

        tag_k, loc_k, val_k = data
        device = tag_k.device
        default_ = torch.zeros((self.num_joints, 3 + tag_k.shape[2]), device=device)

        loc_k = loc_k.float()
        joint_k = torch.cat((loc_k, val_k[..., None], tag_k), dim=2)  # nx30x2, nx30x1, nx30x1

        joint_dict = defaultdict(lambda: default_.clone().detach())
        tag_dict = {}
        for i in range(self.num_joints):
            idx = joint_order[i]

            tags = tag_k[idx]
            joints = joint_k[idx]
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if i == 0 or len(joint_dict) == 0:
                for tag, joint in zip(tags, joints):
                    key = tag[0]
                    joint_dict[key.item()][idx] = joint
                    tag_dict[key.item()] = [tag]
            else:
                grouped_keys = list(joint_dict.keys())[:self.max_num_people]
                grouped_tags = [torch.mean(torch.as_tensor(tag_dict[i]), dim=0, keepdim=True) for i in grouped_keys]

                if self.ignore_too_much and len(grouped_keys) == self.max_num_people:
                    continue

                grouped_tags = torch.as_tensor(grouped_tags, device=device)
                if len(grouped_tags.shape) < 2:
                    grouped_tags = grouped_tags.unsqueeze(0)

                diff = joints[:, None, 3:] - grouped_tags[None, :, :]
                diff_normed = torch.norm(diff, p=2, dim=2)
                diff_saved = diff_normed.clone().detach()

                if self.use_detection_val:
                    diff_normed = torch.round(diff_normed) * 100 - joints[:, 2:3]

                num_added = diff.shape[0]
                num_grouped = diff.shape[1]

                if num_added > num_grouped:
                    diff_normed = torch.cat(
                        (diff_normed, torch.zeros((num_added, num_added - num_grouped), device=device) + 1e10),
                        dim=1
                    )

                pairs = py_max_match(diff_normed.detach().cpu().numpy())
                for row, col in pairs:
                    if (
                            row < num_added
                            and col < num_grouped
                            and diff_saved[row][col] < self.tag_threshold
                    ):
                        key = grouped_keys[col]
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key].append(tags[row])
                    else:
                        key = tags[row][0].item()
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key] = [tags[row]]
                        
        ret = torch.stack([joint_dict[i] for i in joint_dict])
        return ret

    def match_torch(self, tag_k, loc_k, val_k):
        match = lambda x: self.match_by_tag_torch(x)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k_torch(self, det, tag):
        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)
        val_k, ind = det.topk(self.max_num_people, dim=2)

        tag = tag.view(tag.size(0), tag.size(1), w * h, -1)
        if not self.tag_per_joint:
            tag = tag.expand(-1, self.num_joints, -1, -1)

        tag_k = torch.stack(
            [torch.gather(tag[:, :, :, i], 2, ind) for i in range(tag.size(3))],
            dim=3
        )

        # added to reduce the number of unique tags
        tag_k = (tag_k * 10).round() / 10  # ToDo parametrize this

        x = ind % w
        y = (ind // w).long()

        ind_k = torch.stack((x, y), dim=3)

        ret = {
            'tag_k': tag_k,
            'loc_k': ind_k,
            'val_k': val_k
        }

        return ret

    def adjust_torch(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        # print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0] = y + 0.5
                        ans[batch_id][people_id, joint_id, 1] = x + 0.5
        return ans

    def refine_torch(self, det, tag, keypoints):
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].type(torch.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = torch.tensor(tags, device=tag.device).mean(dim=0, keepdim=True)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(dim=2) ** 0.5)
            tmp2 = tmp - torch.round(tt)

            def unravel_index(index, shape):
                out = []
                for dim in reversed(shape):
                    out.append(index % dim)
                    index = index // dim
                return tuple(reversed(out))

            # find maximum position
            y, x = unravel_index(torch.argmax(tmp2), tmp.shape)
            xx = x.clone().detach()
            yy = y.clone().detach()
            x = x.float()
            y = y.float()
            # detection score at maximum position
            val = tmp[yy, xx]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = torch.tensor(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def parse(self, det, tag, adjust=True, refine=True):
        try:
            ans = self.match_torch(**self.top_k_torch(det, tag))

            if adjust:
                ans = self.adjust_torch(ans, det)

            scores = [i[:, 2].mean() for i in ans[0]]

            if refine:
                # for each image
                for i in range(len(ans)):
                    # for each detected person
                    for j in range(len(ans[i])):
                        det_ = det[i]
                        tag_ = tag[i]
                        ans_ = ans[i][j]
                        if not self.tag_per_joint:
                            tag_ = torch.repeat(tag_, (self.num_joints, 1, 1, 1))
                        ans[i][j] = self.refine_torch(det_, tag_, ans_)
            return ans, scores
        
        except RuntimeError:
            return [], []

    
    
class Predictor:
    def __init__(self,
                 cfg,
                 model_cfg,
                 interpolation=cv2.INTER_LINEAR,
                 return_heatmaps=False,
                 return_bounding_boxes=False,
                 filter_redundant_poses=True,
                 head_neck = None):
        
        self.nof_joints = model_cfg.MODEL.NUM_JOINTS
        self.resolution = model_cfg.MODEL.INPUT_SIZE
        self.interpolation = interpolation
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.filter_redundant_poses = filter_redundant_poses
        self.max_batch_size = cfg.TEST.BATCH_SIZE
        if cfg.SYSTEM.DEVICE == 'GPU':
            self.device =  torch.device('cuda')
        else:
            self.device = torch.device("cpu")
        
        dir_path = join(dirname(__file__), '../../../../models')
        self.model = get_pose_net(model_cfg, dir_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()

        self.output_parser = HeatmapParser(num_joints=self.nof_joints,
                                           joint_set=cfg.DATA.FORMAT,
                                           max_num_people=cfg.DATA.MAX_NUM_PEOPLE,
                                           ignore_too_much=True,
                                           detection_threshold=cfg.TEST.DETECTION_THRESHOLD)
        
        self.evaluator = Evaluator(num_joints=self.nof_joints,
                                   conf_thresh=cfg.TEST.DETECTION_THRESHOLD,
                                   static_thresh=cfg.TEST.PCK_THRESHOLD,
                                   head_neck=head_neck,
                                   h_ratio=cfg.TEST.PCK_FACTOR,
                                   oks_ratio=cfg.TEST.OKS_FACTOR,
                                   oks_thresh=cfg.TEST.OKS_THRESHOLD)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def predict_image(self, image):
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')
            
    def predict(self, dataloader, metrics='oks'):
        results = []
        for idx, (img, joints, fname) in tqdm(enumerate(dataloader), total=len(dataloader), ascii=True):
            img = np.asarray(img)
            joints = np.asarray(joints)

            pts = self.predict_image(img)

            for pt, joint, path in zip(pts, joints, fname):
                confmat = self.evaluator.calc_confmat(pt, joint, metrics=metrics)
                results.append([idx, path, joint, pt, confmat])

        df = pd.DataFrame(results, columns=(['ID', 'input_path', 'gt', 'predict_result', 'evaluate_result']))
        return df
            
    def _predict_single(self, image):
        ret = self._predict_batch(image[None, ...])
        if len(ret) > 1:  # heatmaps and/or bboxes and joints
            ret = [r[0] for r in ret]
        elif len(ret) == 0:  # empty list
            ret = ret
        else:  # joints only
            ret = ret[0]
        return ret

    def _predict_batch(self, image):
        with torch.no_grad():

            heatmaps_list = None
            tags_list = []

            # scales and base (size, center, scale)
            scales = (1,)  # ToDo add support to multiple scales

            scales = sorted(scales, reverse=True)
            base_size, base_center, base_scale = get_multi_scale_size(
                image[0], self.resolution, 1, 1
            )

            # for each scale (at the moment, just one scale)
            for idx, scale in enumerate(scales):
                # rescale image, convert to tensor, move to device
                images = list()
                for img in image:
                    image, size_resized, _, _ = resize_align_multi_scale(
                        img, self.resolution, scale, min(scales), interpolation=self.interpolation
                    )
                    #image = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
                    image = self.transform(image).unsqueeze(dim=0)
                    image = image.to(self.device)
                    images.append(image)
                images = torch.cat(images)

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    self.model, images, with_flip=False, project2image=True, size_projected=size_resized,
                    nof_joints=self.nof_joints, max_batch_size=self.max_batch_size
                )

                # aggregate the multiple heatmaps and tags
                heatmaps_list, tags_list = aggregate_results(
                    scale, heatmaps_list, tags_list, heatmaps, tags, with_flip=False, project2image=True
                )

            heatmaps = heatmaps_list.float() / len(scales)
            tags = torch.cat(tags_list, dim=4)
            
            grouped, scores = self.output_parser.parse(
                heatmaps, tags, adjust=True, refine=True  # ToDo parametrize these two parameters
            )

            # get final predictions
            final_results = get_final_preds(
                grouped, base_center, base_scale, [heatmaps.shape[3], heatmaps.shape[2]]
            )

            if self.filter_redundant_poses:
                final_pts = []
                # for each image
                for i in range(len(final_results)):
                    final_pts.insert(i, list())
                    # for each person
                    for pts in final_results[i]:
                        if len(final_pts[i]) > 0:
                            diff = np.mean(np.abs(np.array(final_pts[i])[..., :2] - pts[..., :2]), axis=(1, 2))
                            if np.any(diff < 3):  # average diff between this pose and another one is less than 3 pixels
                                continue
                        final_pts[i].append(pts)
                final_results = final_pts

            pts = []
            boxes = []
            for i in range(len(final_results)):
                pts.insert(i, np.asarray(final_results[i]))
                pts[i][..., [0, 1]] = pts[i][..., [1, 0]]  # restoring (y, x) order as in SimpleHRNet
                pts[i] = pts[i][..., :3]

                if self.return_bounding_boxes:
                    left_top = np.min(pts[i][..., 0:2], axis=1)
                    right_bottom = np.max(pts[i][..., 0:2], axis=1)
                    # [x1, y1, x2, y2]
                    boxes.insert(i, np.stack(
                        [left_top[:, 1], left_top[:, 0], right_bottom[:, 1], right_bottom[:, 0]], axis=-1
                    ))

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]