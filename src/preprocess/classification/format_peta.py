import os
import numpy as np
import pickle

from easydict import EasyDict
from scipy.io import loadmat

import sys
from os.path import join, dirname
sys.path.append(join(dirname(__file__), "../../"))
from tools.utils import covert_one_hot_to_name
from utils import set_seed, make_dir, make_df, extract_age_gender_labels


set_seed(seed=0)

# note: ref by annotation.md
group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


def generate_data_description(save_dir, reorder, n_classes=35, extract_label=False):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]

    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]

    # (19000, 35)
    dataset.label = raw_label[:, :n_classes]
    dataset.attr_name = raw_attr_name[:n_classes]
    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    if extract_label:
        dataset = extract_age_gender_labels(dataset, dataset_name=dataset.description)
        pkl_file_name = 'dataset_extracted.pkl'
    else:
        pkl_file_name = 'dataset.pkl'

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, pkl_file_name), 'wb+') as f:
        pickle.dump(dataset, f)

    train_idx = len(dataset.partition.train[0])
    val_len = len(dataset.partition.val[0])
    val_index = train_idx + val_len
    print('Train length:', train_idx)
    print('Val length:', val_len)
    print('Train Val length:', len(dataset.partition.trainval[0]))
    print('Test length:', len(dataset.partition.test[0]))

    assert val_index ==  len(dataset.partition.trainval[0])
    all_data_len = val_index + len(dataset.partition.test[0])

    return dataset, train_idx, val_index, all_data_len

if __name__ == "__main__":
    save_dir = './data/PETA'
    dataset, train_idx, val_index, all_data_len = generate_data_description(save_dir, reorder=True, n_classes=35, extract_label=True)
    make_df(dataset=dataset, save_dir=save_dir, train_idx=train_idx, val_index=val_index, all_data_len=all_data_len, kfold=True, version=5)