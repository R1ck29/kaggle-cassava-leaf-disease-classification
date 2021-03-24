import os
import numpy as np
import pandas as pd
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

group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]


def generate_data_description(save_dir, reorder, n_classes=26, extract_label=False):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name

    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert dataset.label.shape == (100000, n_classes)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(n_classes)]

    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    if extract_label:
        dataset = extract_age_gender_labels(dataset, dataset_name=dataset.description)
        pkl_file_name = 'dataset_extracted.pkl'
    else:
        pkl_file_name = 'dataset.pkl'

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, pkl_file_name), 'wb+') as f:
        pickle.dump(dataset, f)

    return dataset


if __name__ == "__main__":
    save_dir = './data/PA100k/'
    reoder = True
    dataset = generate_data_description(save_dir, reorder=True, n_classes=26, extract_label=True)
    make_df(dataset=dataset, save_dir=save_dir, kfold=True, version=5)
