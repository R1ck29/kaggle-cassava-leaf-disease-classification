
import os
from glob import glob
from sklearn.model_selection import train_test_split

from io import StringIO
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from imageio import imsave
from PIL import Image, ImageDraw
from multiprocessing import Pool
import hydra
import yaml

import label as lb
from utils import file_df_from_path

class ReadCvatPolygons(object):
    '''CVAT形式のポリゴンデータをカラーマップ画像へ変換。
    CVAT: https://github.com/opencv/cvat
    '''

    def __init__(self, background, name2id):
        self.background = background
        self.name2id = name2id

    def draw_polygons(self, polygons, datatype, instance_labels=None):
        img_height, img_width = int(polygons['height']), int(polygons['width'])

        # 多角形を描いていくための画像を定義
        labelimg = Image.new("L", (img_width, img_height), self.background)
        drawer = ImageDraw.Draw(labelimg)

        # 描くためのポリゴン情報をxmlから取得
        polys, labels, z_orders = [], [], []

        for polygon in polygons.findAll('polygon'):
            label = polygon['label']
            z_order = polygon['z_order']

            points = pd.read_csv(StringIO(polygon['points']), delimiter=';', header=None)
            points = points.T.iloc[:, 0].str.split(',', expand=True)
            # 多角形は3点以上から構成されるため、2点以下のポリゴン情報は描画しない
            if len(points) > 2:
                poly = list(map(tuple, points.values.astype(float)))
                polys.append(poly)
                labels.append(label)
                z_orders.append(z_order)
            else:
                print(label, z_order, points)

        # z_orderの型変換と、描画順の操作のためにソート
        # (z_orderが小さいほど画面奥のオブジェクト)
        z_orders = np.array(z_orders, dtype=int)
        zlp = sorted(zip(z_orders, labels, polys))

        # draw polygons
        if datatype == 'pixel':
            if instance_labels:
                for z_order, label, poly in zlp:
                    if label in instance_labels:
                        drawer.polygon(poly, fill=self.name2id[label])
                return np.array(labelimg)
            else:
                for z_order, label, poly in zlp:
                    drawer.polygon(poly, fill=self.name2id[label])
                return np.array(labelimg)

    def read_semantic(self, filepath, instance_labels=None):
        with open(filepath) as f:
            soup = BeautifulSoup(f, 'lxml')

        names, labelimgs = [], []
        for polygons in soup.find_all('image'):
            name = polygons['name']
            names.append(name)
            labelimg = self.draw_polygons(polygons, datatype='pixel', instance_labels=semantic_labels)
            labelimgs.append(labelimg)

        return names, labelimgs

if __name__ == '__main__':

    ## 各種パラメタなど
    # 画像保存に使用するCPUコア数
    jobs = os.cpu_count() // 2
    # アノテーション結果塗りつぶし画像のデフォルト値を定義
    background = 0
    # ラベルをIDに変換する辞書の定義
    name2id = {label.name: label.id for label in lb.name2lb.values()}
    # インスタンス化
    readcvat = ReadCvatPolygons(background, name2id)

    cfg_file = './configs/data/seg_pytorch.yaml'

    with open(cfg_file) as fp:
        cfg = yaml.load(fp)

    input_dir = cfg['DATA']['CVAT_INPUT']
    output_dir = cfg['DATA']['CVAT_OUTPUT']

    ################# semantic GT #################
    semantic_labels = name2id.keys()

    file_df = file_df_from_path(os.path.join(input_dir, 'cvat/'), ext='xml')

    output_path = os.path.join(output_dir, 'gt')
    output_path_color = os.path.join(output_dir, 'color')

    # 複数CPUによる並列処理のための関数化
    def save(row):
        filepath = row.filepath
        print('filepath:',filepath)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path_color, exist_ok=True)
        

        names, labelimgs = readcvat.read_semantic(filepath, semantic_labels)
        #print(names, labelimgs)

        for cnt, (name, limg) in enumerate(zip(names, labelimgs)):
            savename = name.split(os.sep)[-1][:-4] + '.png'
            #np.save(os.path.join(output_path, savename), limg)
            #np.save(os.path.join(output_path_color, savename),lb.id2color[limg].astype(np.uint8))
            imsave(os.path.join(output_path, savename), limg)
            imsave(os.path.join(output_path_color, savename),lb.id2color[limg].astype(np.uint8))
            print('save:',os.path.join(output_path_color, savename))

        np.save(os.path.join(os.path.dirname(input_dir), 'split', 'colormap'), lb.id2color)

    rows = []
    for _, row in file_df.iterrows():
        rows.append(row)

    # 複数CPUによる並列処理
    p = Pool(jobs)
    p.map(save, rows)
    p.close()

    img_dir = sorted(glob(os.path.join(input_dir, 'images/*')))
    gt_dir = sorted(glob(os.path.join(output_dir, 'gt/*')))

    X_train, X_test, y_train, y_test = train_test_split(img_dir, gt_dir,
                                                    test_size=0.25)

    img_path, gt_path, split_list = [], [], []

    for idx, Xtr in enumerate(X_train):
        img_path.append(Xtr)
        gt_path.append(y_train[idx])
        split_list.append('train')
        
    for _idx, Xte in enumerate(X_test):
        img_path.append(Xte)
        gt_path.append(y_test[_idx])
        split_list.append('val')
        
    for _idx, Xte in enumerate(X_test):
        img_path.append(Xte)
        gt_path.append(y_test[_idx])
        split_list.append('test')

    df = pd.DataFrame(list(zip(img_path, gt_path, split_list)),
                    columns=['img_path', 'gt_path', 'split_name'])

    df.to_pickle(os.path.join(os.path.dirname(input_dir), 'split', '{}.pkl'.format(cfg['DATA']['CSV_PATH'])))
    