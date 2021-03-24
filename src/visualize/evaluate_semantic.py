import os
import argparse
import cv2
import numpy as np
import pandas as pd
from imageio import imread

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from configs import cfg, update_config
from configs.default import get_cfg_defaults
from src.utils.utils import file_df_from_path
import src.utils.label_cityscapes as lb
import src.utils.label_project as lb_prj
from src.data.generator.segmentation.callbacks import IoUCallback
from src.visualize.monitor_log import draw_best_iou

class dummy(object):
    ''' IoUコールバック内の関数を呼び出すためのダミークラス。
    '''
    def __init__(self, num_classes):
        '''
        Parameters
        ----------
        num_classes : int
            予測クラス数(background含める)
        '''
        self.num_classes = num_classes

if __name__ == '__main__':
    ## オプション引数の定義
    parser = argparse.ArgumentParser()
    # SSのバリデーションデータ格納ディレクトリ
    parser.add_argument('-iv', '--input_validation_semantic_path')
    # SSの予測結果格納ディレクトリ
    parser.add_argument('-is', '--input_semantic_path')
    # SSのモデルディレクトリ
    parser.add_argument('-ip', '--input_path')
    args = parser.parse_args()
    
    input_validation_semantic_path = args.input_validation_semantic_path
    input_semantic_path = args.input_semantic_path
    input_path = args.input_path

    # 訓練時configファイルの読み込みと変数への展開
    cfg = update_config(os.path.join(input_path, 'config.yaml'))

    # バリデーションデータのファイルパス情報の読込み
    val_df = file_df_from_path(input_validation_semantic_path)
    val_df['filetype'] = 'validation_semantic'
    if cfg.dataset['data_id'] == 'cityscapes':
        val_df['fileid'] = val_df['fileid'].str.replace('_gtFine_labelIds', '') #for cityscapes
    else:
        val_df['fileid'] = val_df['fileid'].str.replace('_annot', '')

    # 予測結果のファイルパス情報の読込み
    sem_df = file_df_from_path(input_semantic_path)
    sem_df['filetype'] = 'semantic'
    
    # テーブルの結合
    file_df = pd.concat([sem_df, val_df], axis=0)
    file_df = file_df.pivot(index='fileid', columns='filetype', values='filepath').dropna()
    file_df.reset_index(inplace=True)

    if cfg.dataset['num_classes'] == 34:
        dummy_class = 20
    else:
        dummy_class = cfg.dataset['num_classes']

    # IoU計算のためのインスタンス作成
    iou = IoUCallback('', dummy(dummy_class), save=False)

    # 各画像ごとにIoUを計算
    dfs = []
    for imgid, row in file_df.iterrows():
        print(row.validation_semantic)
        print(row.semantic)
        gt_sem = imread(row.validation_semantic)
        sem = imread(row.semantic)
        key = row.fileid

        gt_height, gt_width = gt_sem.shape[:2]
        sem = cv2.resize(sem, (gt_width, gt_height))

        # ラベル変換
        if cfg.dataset['data_id'] == 'cityscapes':
            if cfg.dataset['label_id'] == 'lb_raw':
                sem = lb.oid2otid[sem].astype(np.uint8)

            elif cfg.dataset['label_id'] == 'lb_honda':
                gt_sem = lb.oid2tid[gt_sem].astype(np.uint8)
                sem = lb.oid2tid[sem].astype(np.uint8)
        dfs.append(iou.calc_confusion(gt_sem, sem, key))
        print('key, gt, sem:', key, [np.amin(gt_sem),np.amax(gt_sem)],[np.amin(sem),np.amax(sem)])

    # データセット全体に対するIoUの計算結果の表、画像ごとのIoUの計算結果の表に変換
    conf_table, conf_table2 = iou.concat_confs(dfs)
        
    # mIoUの計算とラベル名の整理をし、グラフ描画
    iou_df = pd.DataFrame(conf_table.iou.tolist() + [np.nanmean(conf_table.iou.tolist())]).T
    print('iou_df.shape[-1]:', iou_df.shape[-1])
    print('iou_df:', iou_df)

    if cfg.dataset['data_id'] == 'cityscapes':
        #trainid
        if cfg.dataset['label_id'] == 'lb_origin' or cfg.dataset['label_id'] == 'lb_raw':
            iou_df.columns = ['road', 'sidewalk', 'building', 'wall', 'fence',
                              'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person',
                              'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bycycle'] + ['mean_IoU']

        #projectid
        elif cfg.dataset['label_id'] == 'lb_honda':
            iou_df.columns = ['car', 'truck', 'bus', 'motorcycle', 'bicycle',
                              'rider', 'human', 'traffic_light_car', 'traffic_light_human',
                              'traffic_sign', 'pole', 'white_line', 'stop_line', 'road_sign',
                              'road', 'sidewalk', 'building', 'other', 'inside'] + ['mean_IoU']
    else:
        iou_df.columns = lb_prj.lb_df.name.tolist() + ['mean_IoU']

    draw_best_iou(iou_df, savedir=os.path.join(input_semantic_path, '..'))

