import os
from glob import glob

import hydra
import pandas as pd
from tqdm import tqdm

from .cvat_utils import read_cvat


def check_glob_list_size(target_path):
    """ Check files list length obtaind by glob
    Args:
        target_path (str): path to files. need to include '*' to specify target files.
    Raises:
        ValueError: file must be only one.
    Returns:
        list_len(int): if the list length is 0, return 0. if the list length is 1, return 1
        glob_list(list): files list in 'target_path'
    """
    glob_list = glob(target_path)
    if len(glob_list) > 1:
        print(f'Found {len(glob_list)} files in {target_path}')
        print(glob_list)
        raise ValueError(f'Only 1 file should be found. Check the path: {target_path}')
    elif len(glob_list) == 0:
        print(f'Found {len(glob_list)} files in {target_path}')
        list_len = 0
    else:
        print(f'Found {glob_list[0]} file  in {target_path}')
        list_len = 1
    return list_len, glob_list


def get_cvat_annotations(cfg, cvat_files, interim_path, out_csv_name='all_cvat'):
    """ Read all xml files as dataframes and concat them.
        Rename columns for post process
    Args:
        cfg (DictConfig): configs for preprocessing
        cvat_files (list): list of xml files
        interim_path (str): interim folder path to save csv
        out_csv_name (str, optional): output csv name. Defaults to 'all_cvat'.
    Returns:
        pd.DataFrame: all CVAT annotations 'cvat_files' concated.
    """

    df_list = []
    print(f'Loading {len(cvat_files)} Annotations...')
    for cvat_file in tqdm(cvat_files, total=len(cvat_files)):
        gt_df = read_cvat(cvat_file, task_type='object_detection')
        gt_df['name'] = gt_df['name'].str.split('.', expand = True)[0].astype(str)
        gt_df = gt_df.rename(columns={'name': 'image_id', 'width': 'image_width', 'height': 'image_height',
                                        'xtl': 'xmin', 'ytl': 'ymin', 'xbr': 'xmax', 'ybr': 'ymax'})
        df_list.append(gt_df)
    gt_all = pd.concat(df_list)
    print(f'After Renamed colmns.')
    print(gt_all.columns)
    gt_all.to_csv(f'{interim_path}{out_csv_name}.csv', index=False)
    print(f'------------ CVAT annotation csv saved to: {interim_path}{out_csv_name}.csv ------------')
    return gt_all


@hydra.main(config_path="../../../configs", config_name="train")
def main(cfg):
    # input
    print('-'*30, 'Converting xml files to csv files', '-'*30)
    cvat_file_path = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/raw/cvat_annotations/'
    interim_path = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/interim/'
    if not os.path.exists(interim_path):
        os.makedirs(interim_path, exist_ok=True)
    all_cvat_files = glob(cvat_file_path+ '*.xml')
    train_cvat_files = glob(cvat_file_path+ 'train/*.xml')
    test_cvat_files = glob(cvat_file_path+ 'test/*.xml')

    if len(train_cvat_files) > 0 and len(test_cvat_files) > 0:
        print('Found CVAT train and test folders.')
        train = get_cvat_annotations(cfg, train_cvat_files, interim_path, out_csv_name='train')
        test = get_cvat_annotations(cfg, test_cvat_files, interim_path, out_csv_name='test')
    elif len(all_cvat_files) > 0:
        print(f'Found all CVAT annotations in {cvat_file_path}')
        all_df = get_cvat_annotations(cfg, all_cvat_files, interim_path, out_csv_name='all_cvat')
    else:
        print(f'Can not find any CVAT annotations in {cvat_file_path}')

if __name__ == '__main__':
    main()