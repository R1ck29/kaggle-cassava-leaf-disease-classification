import cv2
import os
import argparse
import json
import numpy as np
from imageio import imsave
from multiprocessing import Pool

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras_deeplab_v3_plus import model as model_deeplab

from generator import Generator, Transform
import label as lb
import label_JFEE as lb_JFEE
from utils import file_df_from_path


def save_diff_pics(arg):
    y_pred, diff_path, y_gt = arg
    y_diff = (y_pred == y_gt)
    acc = np.mean(y_diff)
    print(y_diff)
    plt.imshow(y_diff, 'gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.savefig(os.path.join(diff_path, key + '.png'), bbox_inches='tight')
    plt.close()

def save_mask_pics(arg):
    y_pred, maskpath, img = arg
    color_pred = lb_JFEE.color_map_mask[y_pred]
    color_pred = cv2.resize(color_pred, img.shape[1::-1])

    img = (img * 255).astype(np.uint8)  # tmp
    plt.imshow(color_pred, vmin=0, vmax=255)
    plt.imshow(img, alpha=0.3, vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.savefig(maskpath, bbox_inches='tight')
    plt.close()
    print(maskpath)

    # y_mask = cv2.addWeighted(color_pred,0.5,img,0.5,0, dtype=cv2.CV_32F)
    # imsave(maskpath,y_mask)


def save_pred(arg):
    y_pred, y_color, conf, pred_path, col_path, conf_path = arg
    imsave(pred_path, y_pred)
    imsave(col_path, y_color)
    imsave(conf_path, conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--input_path')
    parser.add_argument('-imp', '--input_img_path')
    parser.add_argument('-op', '--output_path')
    parser.add_argument('-bs', '--batch_size', type=int, default=14)
    parser.add_argument('-j', '--jobs', type=int)
    args = parser.parse_args()

    input_path = args.input_path
    input_img_path = args.input_img_path
    output_path = args.output_path
    batch_size = args.batch_size
    jobs = args.jobs

    splitid = config_train['splitid']  # tmp
    path_split = os.path.join('../data/preprocess/split', splitid)  # tmp

    file_df = file_df_from_path(input_img_path)
    file_df.columns = ['imgpath', '_', 'imgid']

    # 入出力ディレクトリ
    print('output_path is', output_path)
    os.makedirs(os.path.join(output_path, 'predict'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'confidence'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'predict_mask'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'gt_mask'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'diff'), exist_ok=True)


    preds = model.predict(imgs)

    y_preds = preds.argmax(axis=-1).astype(np.uint8)
    y_gts = gts.argmax(axis=-1).astype(np.uint8)  # tmp

    # cityscapes
    # y_preds = lb.oid2otid[y_preds].astype(np.uint8) #tmp
    # y_colors = lb.tid2color[y_preds].astype(np.uint8)
    # JFEE
    y_colors = lb_JFEE.color_map_mask[y_preds].astype(np.uint8)  # tmp
    confs = (255 * preds.max(axis=-1)).astype(np.uint8)

    pred_savepaths = [os.path.join(output_path, 'predict', key + '.png') for key in keys]
    col_savepaths = [os.path.join(output_path, 'color', key + '.png') for key in keys]
    conf_savepaths = [os.path.join(output_path, 'confidence', key + '.png') for key in keys]
    pred_mask_paths = [os.path.join(output_path, 'predict_mask', key + '.png') for key in keys]
    gt_mask_paths = [os.path.join(output_path, 'gt_mask', key + '.png') for key in keys]
    diff_paths = [os.path.join(output_path, 'diff', key + '.png') for key in keys]
    # maskpaths = [os.path.join(output_path, 'predict_mask', '_'.join(key.split('_')[:-1]) + '.png') for key in keys] #tmp cityscapes

    p.map(save_pred, zip(y_preds, y_colors, confs, pred_savepaths, col_savepaths, conf_savepaths))
    p.map(save_mask_pics, zip(y_gts, gt_mask_paths, imgs))
    p.map(save_mask_pics, zip(y_preds, pred_mask_paths, imgs))
    p.map(save_diff_pics, zip(y_preds, diff_paths, y_gts))