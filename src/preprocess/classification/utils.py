import os
import pandas as pd
import numpy as np
import random

import sys
from os.path import join, dirname
sys.path.append(join(dirname(__file__), "../../"))
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from tools.utils import covert_one_hot_to_name
import gc


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def make_test_split(df, target_col_name='attr_name', test_size=0.2, random_state=42):
    train, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[target_col_name].values
    )
    df_trin = train.reset_index(drop=True)
    df_test = test.reset_index(drop=True)
    return df_trin, df_test


def make_folds(df_train, target_col_name='attr_name', n_fold=5, seed=42):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    df_folds = df_train.copy()
    df_folds.loc[:, 'fold'] = None
    if target_col_name in df_folds:
        for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds[target_col_name])):
            df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
            fold_df_len = len(df_folds[df_folds['fold']==fold_number])
            print(f'Fold {fold_number} length : {fold_df_len}')
            print(f'Fold {fold_number} percentage : {(fold_df_len / len(df_folds))*100}%')
    return df_folds


def remove_gender_duplicates(duplicated_list, df):
    print(f'Removing Age Duplicates: {duplicated_list}')
    for attr_name in duplicated_list:
        print(attr_name, len(df[df['attr_name']==attr_name]))
        df['attr_name'] = df.attr_name.astype(str).str.replace(attr_name,attr_name.split(' ')[-1]) #.str.replace(']','').str.replace(',','')
        assert len(df[df['attr_name']==attr_name]) == 0
    return df


def add_gender_class(df, dataset, fill_attr_name, fill_class_id):
    no_gender_name = df[~df['attr_name'].str.contains('Female') & ~df['attr_name'].str.contains('Male')]['attr_name'].copy()
    existed_gender_id =  int(fill_class_id) - 1
    no_gender_id = df[~df['class_id'].str.contains(str(existed_gender_id)) & ~df['class_id'].str.contains(fill_class_id)]['class_id'].copy()
    if len(no_gender_name) > 0:
        print('-'*20, f'Filling Genger class Name for {dataset.description} dataset', '-'*20)
        print('Attribute Name to add: ', fill_attr_name)
        df.loc[~df['attr_name'].str.contains('Female') & ~df['attr_name'].str.contains('Male'), 'attr_name'] = no_gender_name + ' ' + fill_attr_name
        assert len(df.loc[~df['attr_name'].str.contains('Female') & ~df['attr_name'].str.contains('Male'), 'attr_name']) == 0

    if len(no_gender_id) > 0:
        print('-'*20, f'Filling Gender Class ID for {dataset.description} dataset', '-'*20)
        print('Attribute ID to add: ', fill_class_id)
        df.loc[~df['class_id'].str.contains(str(existed_gender_id)) & ~df['class_id'].str.contains(fill_class_id), 'class_id'] = no_gender_id + ' ' + fill_class_id
        assert len(df.loc[~df['class_id'].str.contains(str(existed_gender_id)) & ~df['class_id'].str.contains(fill_class_id), 'class_id']) == 0

    return df


def make_df(dataset, save_dir, train_idx=80000, val_index=90000, all_data_len=100000, kfold=False, n_fold=5, version=None):
    """[summary]

    Args:
        dataset (dict): dataset information
        save_dir (str): directory name for output csv
        train_idx (int, optional): [description]. Defaults to 80000.
        val_index (int, optional): [description]. Defaults to 90000.
        all_data_len (int, optional): [description]. Defaults to 100000.
        kfold (bool, optional): [description]. Defaults to False.
        n_fold (int, optional): Number to split train data. Defaults to 5
        version (int, optional): output csv version

    Raises:
        NotImplementedError: [description]
    """

    attr_ids, attr_names = covert_one_hot_to_name(dataset.image_name, dataset.label, dataset.attr_name)

    df = pd.DataFrame({'image_id': dataset.image_name, 'class_id': attr_ids, 'attr_name': attr_names}) #'class_id':class_ids,
    df['class_id'] = df.class_id.astype(str).str.replace('[','').str.replace(']','').str.replace(',','')

    if dataset.description == 'peta':
        fill_attr_name = 'personalFemale'
        fill_class_id = '5'
    elif dataset.description == 'pa100k':
        fill_attr_name = 'Male'
        fill_class_id = '4'
    elif dataset.description == 'rap':
        fill_attr_name = 'Male'
        fill_class_id = '4'
    elif dataset.description == 'rap2':
        fill_attr_name = 'Male'
        fill_class_id = '5'
    else:
        raise NotImplementedError

    if len(df[df['attr_name']=='']) > 0:
        print('-'*20, f'Filling Na for {dataset.description} dataset', '-'*20)
        df = df.replace({'class_id': {'': fill_class_id}, 'attr_name': {'': fill_attr_name}})
        assert len(df[df['attr_name']=='']) == 0
    
    # fix typo in class name
    if len(df[df['attr_name'] == 'Femal']) > 0:
        print('Found class name "Femal". Replacing with "Female"')
        df['attr_name'] = df['attr_name'].str.replace('Femal', 'Female')
        assert len(df[df['attr_name'] == 'Femal']) == 0

    df = add_gender_class(df, dataset, fill_attr_name, fill_class_id)

    if dataset.description == 'rap2':
        duplicated_list = ['Age31-45 Age46-60 Female','Age31-45 Age46-60 Male', 'Age17-30 Age31-45 Female', 'Age17-30 Age31-45 Male', 'AgeLess16 Age17-30 Male', 'AgeLess16 Age31-45 Male']
        df = remove_gender_duplicates(duplicated_list, df)

    elif dataset.description == 'peta':
        duplicated_list = ['personalLess30 personalLarger60 personalMale', 'personalLess30 personalLess60 personalMale', 'personalLess30 personalLess45 personalFemale']
        df = remove_gender_duplicates(duplicated_list, df)

    print('-'*30, 'Labels', '-'*30)
    for i in df['attr_name'].unique():
        print(i)
    print('-'*70)
    
    df['fold'] = None
    if kfold:
        print('Making Test Split')
        train_df, test_df = make_test_split(df)
        print(f'Making {n_fold} fold')
        train_df = make_folds(train_df, target_col_name='attr_name', n_fold=n_fold, seed=42)
    else:
        df['fold'][0:train_idx] = 1
        df['fold'][train_idx:val_index] = 0

        test_df = df[val_index:all_data_len]
        del test_df['fold']
        gc.collect()
        train_df = df[0:val_index]

    assert len(train_df) + len(test_df) == len(df)
    dataset_name = dataset.description

    if version is not None:
        all_csv_name = f'{save_dir}/{dataset_name}_v{version}.csv'
        train_csv_name = f'{save_dir}/train_v{version}.csv'
        test_csv_name = f'{save_dir}/test_v{version}.csv'
    else:
        all_csv_name = f'{save_dir}/{dataset_name}.csv'
        train_csv_name = f'{save_dir}/train.csv'
        test_csv_name = f'{save_dir}/test.csv'

    print('All df len: ', len(df))
    print('Train Val len: ', len(train_df))
    print('Test len: ', len(test_df))

    df.to_csv(all_csv_name, index=False)
    train_df.to_csv(train_csv_name, index=False)
    test_df.to_csv(test_csv_name, index=False)
    print(f'saved to {all_csv_name}')
    print(f'saved to {train_csv_name}')
    print(f'saved to {test_csv_name}')


def extract_age_gender_labels(dataset, dataset_name='pa100k'):
    print(f'Extracting Age and Gender Labels for {dataset_name} dataset')
    if dataset.description == 'pa100k':
        start_idx = 19
        end_idx = 23
    elif dataset.description == 'peta':
        start_idx = 30
        end_idx = 35
    elif dataset.description == 'rap':
        start_idx = 34
        end_idx = 38
    elif dataset.description == 'rap2':
        start_idx = 35
        end_idx = 40
    else:
        raise NotImplementedError

    dataset.label = dataset.label[:, start_idx:end_idx]
    dataset.attr_name = dataset.attr_name[start_idx:end_idx]
    print('-'*30, 'Attributes', '-'*30)
    print(dataset.attr_name)
    print('-'*80)
    return dataset