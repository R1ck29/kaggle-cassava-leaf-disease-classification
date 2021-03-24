#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from imageio import imread

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.utils import to_categorical


class Generator(object):
    ''' モデルの入力を順次生成するイテレータ。
    ユーザーが予め作成しているデータ分割方法(train_df.pkl, val_df.pkl)を使用し、訓練用/検証用データを生成する。
    '''

    def __init__(self, path_split, batch_size, num_classes, img_shape, gt_shape, transform):
        '''
        Parameters
        ----------
        path_split : str
            分割方法の記述されたファイルの保存ディレクトリ
        batch_size : int
            ミニバッチサイズ
        num_classes : int
            クラス数
        img_shape : tuple
            画像の高さ、画像の幅を要素としたタプル
        gt_shape : tuple
            教師ラベルの画像としての高さ、幅を要素としたタプル
        transform : Transform
            画像変換関数を定義するクラス
        '''

        if path_split:
            self.train_df = pd.read_pickle(os.path.join(path_split, 'train_df.pkl'))
            self.val_df = pd.read_pickle(os.path.join(path_split, 'val_df.pkl'))
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_shape = list(img_shape)
        self.gt_shape = list(gt_shape)
        self.transform = transform

    #     def generate(self, TRAIN, random_transform, key=False, oneloop=False, droplast=True):
    def generate(self, train, random_transform, random_crop=0.0, key=False, oneloop=False, droplast=True):
        ''' モデルの入力データ(画像, 教師ラベル)を指定のバッチサイズで生成する。
        フラグに応じて、data augumentationも行う。

        Parameters
        ----------
        train : bool
            True時にシャッフルした訓練用データを出力するためのフラグ
            False時には、検証用データを出力する。
        random_transform : bool
            True時にaugmentationする。
        random_crop : float
            ランダムクロップを実行する確率。0から1までの値とする。
        key : bool
            True時にファイル名情報を一緒に生成するためのフラグ
        oneloop : bool
            True時にファイルを一巡すると停止するためのフラグ
        droplast : bool
            True時に最後のチャネルを間引いた教師ラベルを出力する。

        Yields
        ------
        trans_X_batch : np.ndarray [shape=(self.BATCH_SIZE, img_height, img_width, 3)]
            画像のミニバッチ
        trans_y_batch : np.ndarray [shape=(self.BATCH_SIZE, img_height, img_width, 3)]
            教師ラベルのミニバッチ
        key_batch : str
            データ名
        '''
        while True:
            if train:
                target_df = shuffle(self.train_df.copy())
            else:
                target_df = self.val_df.copy()

            for idx in range(0, len(target_df), self.batch_size):
                batch_df = target_df.iloc[idx:idx + self.batch_size]
                # X_batch = np.array(list(map(imread, batch_df.input_path)))#tmp
                # y_batch = np.array(list(map(imread, batch_df.gt_path)))#tmp
                X_batch = np.array(list(map(imread, '../../../../' + batch_df.input_path)))  # hydra tmp
                y_batch = np.array(list(map(imread, '../../../../' + batch_df.gt_path)))  # hydra tmp

                trans_X_batch, trans_y_batch = [], []
                for num, (x, y) in enumerate(zip(X_batch, y_batch)):

                    if np.random.random() < random_crop:
                        resized_x, resized_y = self.transform.RANDOM_CROP(x, y, self.img_shape[:2])
                    else:
                        resized_x = self.transform.resize(x, self.img_shape[:2], None)
                        resized_y = self.transform.resize(y, self.gt_shape[:2], 'nearest')
                    if random_transform:
                        trans_x, trans_y = self.transform.random_transform(resized_x, resized_y)
                    else:
                        trans_x, trans_y = resized_x, resized_y
                    normalized_x = self.transform.normalize(trans_x)
                    trans_X_batch.append(normalized_x)

                    trans_y_batch.append(trans_y)
                trans_X_batch, trans_y_batch = np.array(trans_X_batch), np.array(trans_y_batch)
                bin_y_batch = to_categorical(trans_y_batch, num_classes=self.num_classes)
                if droplast:
                    bin_y_batch = bin_y_batch[..., :-1]

                if key:
                    key_batch = batch_df.input_id
                    yield trans_X_batch, bin_y_batch, key_batch
                else:
                    yield trans_X_batch, bin_y_batch

            if oneloop:
                if key:
                    key_batch = batch_df.input_id
                    return trans_X_batch, bin_y_batch, key_batch
                else:
                    return trans_X_batch, bin_y_batch

    def generate_from_df(self, target_df, key=False, crop=False):
        ''' 予測用のジェネレータ。file_dfを入力として、画像のみを出力する。

        Parameters
        ----------
        target_df : pd.DataFrame
            ファイル情報
        key : bool
            True時にファイル名情報を一緒に生成するためのフラグ
        crop : bool
            True時に上下それぞれ150ピクセル分クロップする

        Yields
        ------
        trans_X_batch : np.ndarray [shape=(self.BATCH_SIZE, img_height, img_width, 3)]
            画像のミニバッチ
        key_batch : str
            データ名
        '''
        from tqdm import tqdm
        for idx in tqdm(range(0, len(target_df), self.batch_size)):
            batch_df = target_df.iloc[idx:idx + self.batch_size]
            X_batch = np.array(list(map(imread, batch_df.input_path)))

            if crop:
                X_batch = X_batch[:, 150:-150, ...]

            trans_X_batch = []
            for num, x in enumerate(X_batch):
                trans_x = self.transform.resize(x, self.img_shape[:2], None)
                normalized_x = self.transform.normalize(trans_x)
                trans_X_batch.append(normalized_x)
            trans_X_batch = np.array(trans_X_batch)
            if key:
                key_batch = batch_df.input_id
                yield trans_X_batch, key_batch
            else:
                yield trans_X_batch

    def mini_batch(self, target_df, idx, key=False, crop=False):
        batch_df = target_df.iloc[idx:idx + self.batch_size]
        X_batch = np.array(list(map(imread, batch_df.input_path)))

        if crop:
            X_batch = X_batch[:, 150:-150, ...]

        trans_X_batch = []
        for num, x in enumerate(X_batch, idx + self.batch_size):
            trans_x = self.transform.resize(x, self.img_shape[:2], None)
            normalized_x = self.transform.normalize(trans_x)
            trans_X_batch.append(normalized_x)
        trans_X_batch = np.array(trans_X_batch)
        key_batch = batch_df.input_id
        return trans_X_batch, key_batch