import os
import sys

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold

from .build_features import get_annotations


def make_leave_one_out(train_df, val_df):
    """ Make Leave-One-Out dataframe with train and validation dataframe

    Args:
        train_df (pd.DataFrame): train dataframe
        val_df (pd.DataFrame): validation dataframe

    Returns:
        pd.DataFrame: train and validation dataframe with 'fold' column. 
                      validation data has 0, train data has 1 in 'fold' column.
    """
    train_df['fold'] = 1
    val_df['fold'] = 0
    train_folds = pd.concat([train_df, val_df], axis=0)
    return train_folds


def make_stratify_group(org_df, df, image_id_col_name, col_1='source', col_2='count'):
    """ Make grop for stratify cross validation

    Args:
        org_df (pd.DataFrame): original dataframe
        df (pd.DataFrame): copy of "org_df". bbox count column is included.
        image_id_col_name (str): image id column name
        col_1 (str, optional): first column name to make group. Defaults to 'source'.
        col_2 (str, optional): second column name to make group. Defaults to 'count'.

    Returns:
        [pd.DataFrame]: dataframe with stratify_group
    """
    print(f'Making "stratify_group" column using "{col_1}" and "{col_2}" columns.')
    df.loc[:, col_1] = org_df[[image_id_col_name, col_1]].groupby(image_id_col_name).min()[col_1]
    df.loc[:, 'stratify_group'] = np.char.add(
        df[col_1].values.astype(str),
        df[col_2].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    return df


def count_bbox(df, image_id_col_name='image_id'):
    """ Count bounding boxes per image and make "count" column.

    Args:
        df (pd.DataFrame): target dataframe
        image_id_col_name (str, optional): image id column name. Defaults to 'image_id'.

    Returns:
        [pd.DataFrame]: dataframe with "count" column
    """
    bbox_count_df = df[[image_id_col_name]].copy()
    bbox_count_df.loc[:, 'count'] = 1
    bbox_count_df = bbox_count_df.groupby(image_id_col_name).count()
    return bbox_count_df


def _assign_fold_number(n_fold, seed, train_val_df, target_col_name, df_folds=None, image_id_col_name='image_id'):
    """ Assign a fold number using StratifiedKFold

    Args:
        n_fold (int): number of fold to make. Defaults to 5.
        seed (int): seed for StratifiedKFold. Defaults to 42.
        train_val_df (pd.DataFrame): train and validation dataframe
        target_col_name (str): target column name for StratifiedKFold. Defaults to 'source'.
        df_folds (pd.DataFrame, optional): dataframe with bbox counts. Defaults to None.
        image_id_col_name (str, optional): image id column name. Defaults to 'image_id'.

    Raises:
        ValueError: "target_col_name" column must be in "df_folds" dataframe

    Returns:
        [pd.DataFrame]: dataframe with fold number
    """
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    print(f'Target column: {target_col_name}')

    if df_folds is not None:
        df_folds.loc[:, 'fold'] = 0
        if target_col_name in df_folds:
            for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds[target_col_name])):
                df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

            cols_to_use = df_folds.columns.difference(train_val_df.columns)
            fold_df = train_val_df.merge(df_folds[cols_to_use], left_on=image_id_col_name, right_on=df_folds.index)
        else:
            raise ValueError(f'Can not find "{target_col_name}" column.')
    else:
        unique_id_df = train_val_df[[image_id_col_name, target_col_name]].copy()
        unique_id_df.loc[:, 'fold'] = 0
        unique_id_df[target_col_name] = unique_id_df[target_col_name].astype(str)
        unique_id_df = unique_id_df.groupby([image_id_col_name]).count()

        for fold_number, (train_index, val_index) in enumerate(skf.split(X=unique_id_df.index, y=unique_id_df[target_col_name])):
            unique_id_df.loc[unique_id_df.iloc[val_index].index, 'fold'] = fold_number

        cols_to_use = unique_id_df.columns.difference(train_val_df.columns)
        fold_df = train_val_df.merge(unique_id_df[cols_to_use], left_on=image_id_col_name, right_on=unique_id_df.index)
    return fold_df


def make_folds(train_val_df, image_id_col_name='image_id', bbox_col_name='bbox', target_col_name='class_id', n_fold=5, seed=42, random_kfold=False):
    """ Make fold column for cross validation

    Args:
        train_val_df (pd.DataFrame): train and validation dataframe
        image_id_col_name (str, optional): image id column name. Defaults to 'image_id'.
        bbox_col_name (str, optional): bbox coordinates column name. Defaults to 'bbox'.
        target_col_name (str, optional): target column name for StratifiedKFold. Defaults to 'source'.
        n_fold (int, optional): number of fold to make. Defaults to 5.
        seed (int, optional): seed for StratifiedKFold. Defaults to 42.
        random_kfold (bool, optional): if True, run randoom cross validation. Defaults to False.

    Raises:
        ValueError: "target_col_name" must be found in "train_val_df" columns

    Returns:
        [pd.DataFrame]: dataframe with fold column
    """

    if bbox_col_name in train_val_df.columns:
        bboxs = np.stack(train_val_df[bbox_col_name].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
        for i, column in enumerate(['xmin', 'ymin', 'width', 'height']):
            train_val_df[column] = bboxs[:,i]
        train_val_df.drop(columns=[bbox_col_name], inplace=True)

    if n_fold == 1:
        print(f'n_fold is not 5')
        n_fold = 5
        print('-'*30, 'Changed "n_fold" to 5 to make "20%" of validation data.', '-'*30)

    if random_kfold:
        print('Splitting train data with Kfold')
        kf = KFold(n_splits=n_fold, shuffle=False, random_state=seed)
        fold_df = train_val_df.copy()
        fold_df.loc[:, 'fold'] = 0

        for fold_number, (train_index, val_index) in enumerate(kf.split(fold_df[image_id_col_name])):
            fold_df.loc[fold_df.iloc[val_index].index, 'fold'] = fold_number
        
    else:
        print('Splitting train data with Stratified KFold.')
        if target_col_name == 'class_id':
            if target_col_name in train_val_df.columns:
                if train_val_df.class_id.nunique() == 1:
                    print('There is only one class')
                    df_folds = count_bbox(train_val_df)

                    # target class: stratify_group
                    if 'source' in train_val_df.columns:
                        df_folds = make_stratify_group(train_val_df, df_folds, image_id_col_name,'source', 'count')
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
            if target_col_name in train_val_df.columns:
                df_folds = count_bbox(train_val_df)
            else:
                raise ValueError(f'Can not find "{target_col_name}" column.')

        elif target_col_name == 'stratify_group':
            df_folds = count_bbox(train_val_df)
            if 'source' in train_val_df.columns and 'count' in df_folds.columns:
                df_folds = make_stratify_group(train_val_df, df_folds, image_id_col_name,'source', 'count')
            else:
                raise ValueError(f'Can not find "source" column and "count" column to create "stratify_group" column.')
        else:
            if target_col_name in train_val_df.columns:
                # other column
                df_folds = None
            else:
                raise ValueError(f'Can not find "{target_col_name}" column.')

        fold_df = _assign_fold_number(n_fold=n_fold, seed=seed, train_val_df=train_val_df, target_col_name=target_col_name, 
                                                df_folds=df_folds, image_id_col_name=image_id_col_name)

    for fold in range(n_fold):
        n_images = len(fold_df[fold_df['fold']==fold][image_id_col_name])
        print(f'Fold {fold} has {n_images} images.')

    if fold_df.isnull().sum().sum() > 0:
        # Count the NaN under an entire DataFrame:
        print(f'Number of NaN in DataFrame : {fold_df.isnull().sum().sum()}')
        sys.exit(1)
    else:
        return fold_df


@hydra.main(config_path="../../../configs", config_name="train")
def main(cfg):
    print('-'*30, 'Making Train Val Folds', '-'*30)

    input_dir = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/interim/'

    train_df = get_annotations(cfg, mode='train', out_csv_name=cfg.DATA.CSV_PATH.split('/')[-1])
    valid_df = get_annotations(cfg, mode='valid', out_csv_name=cfg.DATA.CSV_PATH.split('/')[-1])

    if train_df is not None and valid_df is not None:
        train_folds = make_leave_one_out(train_df, valid_df)
    elif train_df is not None and valid_df is None:
        print('-'*30, f'Making {cfg.DATA.N_FOLD} Folds', '-'*30)
        train_folds = make_folds(train_df, image_id_col_name='image_id', bbox_col_name='bbox',
                                target_col_name=cfg.DATA.FOLD_TARGET_COL, n_fold=cfg.DATA.N_FOLD, seed=cfg.SYSTEM.SEED, random_kfold=cfg.DATA.RANDOM_KFOLD)
    else:
        raise ValueError(f'Can not make folds. check input files in {input_dir}')

    drop_col_list = ['id', 'ImageID', 'Source', 'Confidence', 'IsOccluded', 'IsGroupOf','IsTruncated', 'IsInside', 'IsDepiction', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']
    for col_name in drop_col_list:
        if col_name in train_folds.columns:
            train_folds = train_folds.drop([col_name], axis=1)
            print(f'Dropped "{col_name}" column.')

    print('-'*30, 'Output CSV Columns', '-'*30)
    print(train_folds.columns)
    
    folds_csv_path = hydra.utils.get_original_cwd() + '/' + cfg.DATA.CSV_PATH
    os.makedirs(os.path.dirname(folds_csv_path), exist_ok=True)
    train_folds.to_csv(folds_csv_path, index=False)
    print('Train Valid csv is saved to:', folds_csv_path)


if __name__ == '__main__':
    main()
