import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

import sys
from os.path import join, dirname
sys.path.append(join(dirname(__file__), "../../"))
from tools.utils import covert_one_hot_to_name
from utils import set_seed, make_dir, make_df, extract_age_gender_labels

set_seed(seed=0)

group_order = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1, 2, 3, 0, 4, 5, 6, 7, 8, 43, 44,
               45, 46, 47, 48, 49, 50]


def generate_data_description(save_dir, reorder, n_classes=51, extract_label=False):
    """
    create a dataset description file, which consists of images, labels
    """

    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'rap'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_dataset')
    dataset.image_name = [data['RAP_annotation'][0][0][5][i][0][0] for i in range(41585)]
    raw_attr_name = [data['RAP_annotation'][0][0][3][i][0][0] for i in range(92)]
    # (41585, 92)
    raw_label = data['RAP_annotation'][0][0][1]
    dataset.label = raw_label[:, np.array(range(n_classes))]
    dataset.attr_name = [raw_attr_name[i] for i in range(n_classes)]

    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    if extract_label:
        dataset = extract_age_gender_labels(dataset, dataset_name=dataset.description)
        pkl_file_name = 'dataset_extracted.pkl'
    else:
        pkl_file_name = 'dataset.pkl'

    dataset.partition = EasyDict()
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_trainval = []

    for idx in range(5):
        trainval = data['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1
        test = data['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1

        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, pkl_file_name), 'wb+') as f:
        pickle.dump(dataset, f)

    val_index = len(dataset.partition.trainval[0])
    train_idx = int(val_index * 0.8)
    val_len = val_index - train_idx
    assert train_idx + val_len == val_index
    print('Train length:', train_idx)
    print('Val length:', val_len)
    print('Train Val length:', val_index)
    print('Test length:', len(dataset.partition.test[0]))

    all_data_len = val_index + len(dataset.partition.test[0])

    return dataset, train_idx, val_index, all_data_len


if __name__ == "__main__":
    save_dir = './data/RAP'
    reorder = True
    dataset, train_idx, val_index, all_data_len = generate_data_description(save_dir, reorder=reorder, n_classes=51, extract_label=True)
    make_df(dataset=dataset, save_dir=save_dir, train_idx=train_idx, val_index=val_index, all_data_len=all_data_len, kfold=True, version=5)
