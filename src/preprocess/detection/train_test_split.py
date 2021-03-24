import os
import sys

import hydra
import pandas as pd
from sklearn.model_selection import train_test_split

from .make_folds import count_bbox, make_stratify_group


def make_train_test_df(cfg, all_df, out_dir, image_id_col_name='image_id'):
    """ Split 'all_df' to train and test dataframe

    Args:
        cfg (DictConfig): configs for preprocessing
        all_df (pd.DataFrame): dataframe to split
        out_dir (str): output directory for train and test dataframes
        image_id_col_name (str, optional): image id column name. Defaults to 'image_id'.

    Returns:
        pd.DataFrame: train and test dataframe
    """
    target_col_name = cfg.DATA.FOLD_TARGET_COL

    if target_col_name == 'class_id':
        if target_col_name in all_df.columns:
            if all_df.class_id.nunique() == 1:
                print('There is only one class')
                df_folds = count_bbox(all_df)

                # target class: stratify_group
                if 'source' in all_df.columns:
                    df_folds = make_stratify_group(all_df, df_folds, image_id_col_name,'source', 'count')
                    target_col_name = 'stratify_group'
                # target class: count 
                elif 'count' in df_folds.columns:
                    target_col_name = 'count'
            else:
                # target class: class_id 
                df_folds = None
        else:
            raise ValueError(f'Can not find "{target_col_name}" column.')
    elif target_col_name == 'count':
        if target_col_name in all_df.columns:
            df_folds = count_bbox(all_df)
        else:
            raise ValueError(f'Can not find "{target_col_name}" column.')

    elif target_col_name == 'stratify_group':
        df_folds = count_bbox(all_df)
        if 'source' in all_df.columns and 'count' in df_folds.columns:
            df_folds = make_stratify_group(all_df, df_folds, image_id_col_name,'source', 'count')
        else:
            raise ValueError(f'Can not find "source" column and "count" column to create "stratify_group" column.')
    else:
        if target_col_name in all_df.columns:
            # other column
            df_folds = None
        else:
            raise ValueError(f'Can not find "{target_col_name}" column.')
    
    if df_folds is not None:
        cols_to_use = df_folds.columns.difference(all_df.columns)
        all_df = all_df.merge(df_folds[cols_to_use], left_on=image_id_col_name, right_on=df_folds.index)

    print(f'Test size: {cfg.DATA.TEST_SIZE}')
    print(f'Target column: {target_col_name}')

    train, test = train_test_split(all_df, test_size=cfg.DATA.TEST_SIZE, random_state=cfg.SYSTEM.SEED, stratify=all_df[target_col_name])

    print(f'train df length: {len(train)}')
    print(f'test df length: {len(test)}')

    train.to_csv(out_dir + 'train.csv', index=False)
    test.to_csv(out_dir + 'test.csv', index=False)
    print(f'train df saved to: {out_dir}train.csv')
    print(f'test df saved to: {out_dir}test.csv')

    return train, test


@hydra.main(config_path="../../../configs", config_name="train")
def main(cfg):
    # input
    print('-'*30, f'Making train test dataset', '-'*30)
    interim_path = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/interim/'
    if not os.path.exists(interim_path):
        raise FileNotFoundError(f'Can not find a path: {interim_path}')

    input_path = interim_path + f'all{cfg.DATA.PROCESSED_CSV_NAME}'
    if not os.path.exists(input_path):
        print('You need to run "xml_to_csv.py" first.')
        sys.exit(0)

    print(f'Loading input file: {input_path}')

    all_df = pd.read_csv(input_path, dtype={'image_id': str})
    train, test = make_train_test_df(cfg, all_df, interim_path)


if __name__ == '__main__':
    main()
