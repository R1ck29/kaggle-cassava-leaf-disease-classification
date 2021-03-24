import os
import pandas as pd

import hydra
import sys
from os.path import join, dirname
sys.path.append(join(dirname(__file__), "../../"))


def read_dataset_csv_files(version=5):
    dataset = 'PA100k'
    input_dir = hydra.utils.get_original_cwd() + f'/data/{dataset}/split/'
    df_train_pa100k = pd.read_csv(f'{input_dir}/train_v{version}.csv')
    df_test_pa100k = pd.read_csv(f'{input_dir}/test_v{version}.csv')
    df_all_pa100k = pd.concat([df_train_pa100k, df_test_pa100k])
    df_all_pa100k['source'] = dataset

    dataset = 'PETA'
    input_dir = hydra.utils.get_original_cwd() + f'/data/{dataset}/split/'
    df_train_peta = pd.read_csv(f'{input_dir}/train_v{version}.csv')
    df_test_peta = pd.read_csv(f'{input_dir}/test_v{version}.csv')
    df_all_peta = pd.concat([df_train_peta, df_test_peta])
    df_all_peta['source'] = dataset

    dataset = 'RAP'
    input_dir = hydra.utils.get_original_cwd() + f'/data/{dataset}/split/'
    df_train_rap = pd.read_csv(f'{input_dir}/train_v{version}.csv')
    df_test_rap = pd.read_csv(f'{input_dir}/test_v{version}.csv')
    df_all_rap = pd.concat([df_train_rap, df_test_rap])
    df_all_rap['source'] = dataset

    dataset = 'RAP2'
    input_dir = hydra.utils.get_original_cwd() + f'/data/{dataset}/split/'
    df_train_rap2 = pd.read_csv(f'{input_dir}/train_v{version}.csv')
    df_test_rap2 = pd.read_csv(f'{input_dir}/test_v{version}.csv')
    df_all_rap2 = pd.concat([df_train_rap2, df_test_rap2])
    df_all_rap2['source'] = dataset

    print('PA100k length: ', len(df_all_pa100k))
    print('PETA length: ', len(df_all_peta))
    print('RAP length: ', len(df_all_rap))
    print('RAP2 length: ', len(df_all_rap2))
    # df_all_data= pd.concat([df_all_pa100k, df_all_peta, df_all_rap, df_all_rap2])
    return df_all_pa100k, df_all_peta, df_all_rap, df_all_rap2


def add_image_path_col(df, dataset_name):
    df['image_path'] = hydra.utils.get_original_cwd() + f'/data/{dataset_name}/raw/images/' + df.image_id
    sample_path = df['image_path'].values[0]
    if os.path.exists(sample_path):
        print('Image Path col added!')
    else:
        raise ValueError(f'Image path is not valid. {sample_path}')
    return df


def extract_classes_peta(df_all_peta):
    peta_age = df_all_peta[df_all_peta['attr_name'].str.contains('personalLarger60') | df_all_peta['attr_name'].str.contains('personalLess60') | df_all_peta['attr_name'].str.contains('personalLess45')]
    print('PETA Gender Extrcted len: ', len(peta_age))
    peta_only_gender = df_all_peta[(df_all_peta['attr_name'] == 'personalMale') | (df_all_peta['attr_name'] == 'personalFemale')]
    print('PETA Only Age len: ', len(peta_only_gender))

    # Gender class change
    peta_age2 = peta_age.copy(deep=True)
    peta_age2['attr_name'] = peta_age2['attr_name'].str.replace('personalMale', 'Male')#.str.replace('personalFemale', 'Female')
    peta_age2['attr_name'] = peta_age2['attr_name'].str.replace('personalFemale', 'Female')

    peta_age2['class_id'] = peta_age2['class_id'].str.replace('4', '6')

    peta_only_gender2 = peta_only_gender.copy(deep=True)
    peta_only_gender2['attr_name'] = peta_only_gender2['attr_name'].str.replace('personalMale', 'Male')
    peta_only_gender2['attr_name'] = peta_only_gender2['attr_name'].str.replace('personalFemale', 'Female')

    peta_only_gender2['class_id'] = peta_only_gender2['class_id'].str.replace('4', '6')

    # Age Class
    peta_age2['attr_name'] = peta_age2['attr_name'].str.replace('personalLarger60', 'AgeOver60')
    peta_age2['class_id'] = peta_age2['class_id'].str.replace('3', '4')

    peta_age2['attr_name'] = peta_age2['attr_name'].str.replace('personalLess60', 'Age46-60')
    peta_age2['class_id']= peta_age2['class_id'].str.replace('2', '3')

    peta_age2['attr_name'] = peta_age2['attr_name'].str.replace('personalLess45', 'Age31-45')
    peta_age2['class_id'] = peta_age2['class_id'].str.replace('1', '2')

    peta_data = pd.concat([peta_age2, peta_only_gender2])

    return peta_data


def extract_classes_pa100k(df_all_peta):
    pa100k_age = df_all_peta[df_all_peta['attr_name'].str.contains('AgeLess18') | df_all_peta['attr_name'].str.contains('AgeOver60')]
    print('PA100k Gender Extrcted len: ', len(pa100k_age))
    pa100k_only_gender = df_all_peta[(df_all_peta['attr_name'] == 'Male') | (df_all_peta['attr_name'] == 'Female')]
    print('PA100k Only Age len: ', len(pa100k_only_gender))

    # Gender class change
    pa100k_age2 = pa100k_age.copy(deep=True)
    pa100k_age2['class_id'] = pa100k_age2['class_id'].str.replace('3', '5') # female
    pa100k_age2['class_id'] = pa100k_age2['class_id'].str.replace('4', '6') # male

    pa100k_only_gender2 = pa100k_only_gender.copy(deep=True)
    pa100k_only_gender2['class_id'] = pa100k_only_gender2['class_id'].str.replace('3', '5') # female
    pa100k_only_gender2['class_id'] = pa100k_only_gender2['class_id'].str.replace('4', '6') # male

    # Age Class
    pa100k_age2['class_id'] = pa100k_age2['class_id'].str.replace('0', '4') # over60

    pa100k_age2['attr_name'] = pa100k_age2['attr_name'].str.replace('AgeLess18', 'AgeLess16')
    pa100k_age2['class_id']= pa100k_age2['class_id'].str.replace('2', '0')

    pa100k_data = pd.concat([pa100k_age2, pa100k_only_gender2])

    return pa100k_data


def format_rap_dataset(df, dataset_name):
    df2 = df.copy(deep=True)
    if dataset_name == 'RAP':
        df2['class_id'] = df2['class_id'].str.replace('3', '5')
        df2['class_id'] = df2['class_id'].str.replace('4', '6')
    elif dataset_name == 'RAP2':
        df2['class_id'] = df2['class_id'].str.replace('5', '6')
        df2['class_id'] = df2['class_id'].str.replace('4', '5')
    else:
        raise NotImplementedError
    return df2

def make_test_split(df):
    train_val_df = df.dropna()
    df.loc[df['fold'].isnull(), 'fold'] = 'test'
    test_df = df[df['fold'] == 'test']
    del test_df['fold']
    for fold_number in df['fold'].unique():
        print(f'Fold {fold_number}: {len(df[df["fold"] == fold_number])}')
    print('Test len: ', len(test_df))

    assert len(test_df) + len(train_val_df) == len(df)
    return train_val_df, test_df


@hydra.main(config_path="../../../configs", config_name="train")
def main(cfg):
    df_all_pa100k, df_all_peta, df_all_rap, df_all_rap2 = read_dataset_csv_files()

    if 'image_path' not in df_all_pa100k.columns:
        df_all_pa100k = add_image_path_col(df_all_pa100k, 'PA100k')
    if 'image_path' not in df_all_peta.columns:
        df_all_peta = add_image_path_col(df_all_peta, 'PETA')
    if 'image_path' not in df_all_rap.columns:
        df_all_rap = add_image_path_col(df_all_rap, 'RAP')
    if 'image_path' not in df_all_rap2.columns:
        df_all_rap2 = add_image_path_col(df_all_rap2, 'RAP2')
    
    peta_data = extract_classes_peta(df_all_peta)
    pa100k_data = extract_classes_pa100k(df_all_pa100k)

    rap_data = format_rap_dataset(df_all_rap, 'RAP')
    rap2_data = format_rap_dataset(df_all_rap2, 'RAP2')

    df_all_data= pd.concat([pa100k_data, peta_data, rap_data, rap2_data])

    out_all_csv_path = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/split/{cfg.DATA.DATA_ID}.csv'
    os.makedirs(os.path.dirname(out_all_csv_path), exist_ok=True)

    df_all_data.to_csv(out_all_csv_path, index=False)
    print('All dataset merged: ', out_all_csv_path)

    train_val_df, test_df = make_test_split(df_all_data)

    out_train_csv_path = hydra.utils.get_original_cwd() + '/' + cfg.DATA.CSV_PATH
    os.makedirs(os.path.dirname(out_train_csv_path), exist_ok=True)
    train_val_df.to_csv(out_train_csv_path, index=False)
    print('Train Valid dataset saved to: ', out_train_csv_path)

    out_test_csv_path = hydra.utils.get_original_cwd() + '/' + cfg.TEST.TEST_CSV_PATH
    os.makedirs(os.path.dirname(out_test_csv_path), exist_ok=True)
    test_df.to_csv(out_test_csv_path, index=False)
    print('Test dataset saved to: ', out_test_csv_path)


if __name__ == '__main__':
    main()
