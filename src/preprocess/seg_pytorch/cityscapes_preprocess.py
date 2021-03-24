import os
from glob import glob
import argparse
import sys

import pandas as pd
import hydra


def recursive_glob(rootdir=".", suffix=""):
    """
    Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


@hydra.main(config_path="../../../configs", config_name="train_back")
def make_df(cfg):
    """
    Create a DataFrame that summarizes information about the dataset
    """
    
    images_base = os.path.join(cfg.DATA.path, "leftImg8bit")
    annotations_base = os.path.join(cfg.DATA.path, "gtFine")
    
    # get image file path
    files = {}
    for split in ['train', 'val']:
        files[split] = recursive_glob(rootdir=os.path.join(images_base, split), suffix=".png")
    
    print('Number of train images are {}'.format(len(files['train'])))
    print('Number of val images are {}'.format(len(files['val'])))
    
    # getting image_path, gt_path, split(train or val), cityname while loop
    img_path, gt_path, split_list, city_list = [], [], [], []
    for split in ['train', 'val']:
        for file_path in files[split]:
            train_val_split = file_path.split(os.sep)[5]
            city_name = file_path.split(os.sep)[6]
            lbl_path = os.path.join(
                annotations_base,
                train_val_split,
                city_name,
                os.path.basename(file_path)[:-15] + "gtFine_labelIds.png",
            )
            img_path.append(file_path)
            gt_path.append(lbl_path)
            split_list.append(train_val_split)
            city_list.append(city_name)
    
    df = pd.DataFrame(list(zip(img_path, gt_path, split_list, city_list)),
                     columns=['img_path', 'gt_path', 'split_name', 'city_name'])
    
    train_df = df.groupby('split_name').get_group('train')
    val_df = df.groupby('split_name').get_group('val')
    
    data_dir = os.path.join('../../../data', cfg.DATA.DATA_ID)
    split_id_dir = os.path.join(data_dir, 'split')
    
    # make split_id dir if it doesn't exists
    if os.path.exists(split_id_dir) == False:
        os.mkdir(split_id_dir)
    
    df.to_pickle(os.path.join(split_id_dir, '{}.pkl'.format(cfg.DATA.CSV_PATH)))
    
    
if __name__ == '__main__':
    make_df()