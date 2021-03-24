import os
import sys
import yaml
from glob import glob

import cv2
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .xml_to_csv import check_glob_list_size


def make_target_df(all_csv, img_list, image_id_col_name, drop_na=False):
    """ Make dataframe from image list and merge with annotation dataframe

    Args:
        all_csv (pd.DataFrame): annotation dataframe
        img_list (list): list of target images list
        image_id_col_name (str): column name of image identifier
        drop_na (bool, optional): set True to drop NA in dataframe. Defaults to False.

    Raises:
        ValueError: image name and image_id values are not same.

    Returns:
        pd.DataFrame: annotation dataframe linked with images list
    """
    all_images = pd.DataFrame([fns.split('/')[-1][:-4] for fns in img_list])
    all_images['image_path'] = [fns for fns in img_list]
    all_images.columns=[image_id_col_name, 'image_path']

    # Merge all images with the bounding boxes dataframe
    all_images = all_images.merge(all_csv, on=image_id_col_name, how='left')
    # print(f'{len(all_train_images) - len(all_csv)} images without target object.')

    all_images = all_images.rename(columns={image_id_col_name: 'image_id'})
    print(f'Renamed {image_id_col_name} to image_id')
    if drop_na:
        all_imgs_len = len(all_images)
        all_images.dropna(inplace=True)
        dropped_imgs_len = len(all_images)
        print('Removed {} images without target object.'.format(all_imgs_len - dropped_imgs_len))
        if len(all_images) == 0:
            raise ValueError('Image Name and ImgeID values mismatch!')

    return all_images


def _get_image_size(image_dir, df):
    """ Check image size and make image_height and image_width colmns

    Args:
        image_dir (str): directory to images
        df (pd.DataFrame): target dataframe to add image width and height columns.

    Returns:
        pd.DataFrame: dataframe with 'image_width' and 'image_height' columns.
    """
    # For Wheat train csv
    if 'width' in df.columns and 'height' in df.columns:
        print('Found width and height columns.')
        df = df.rename(columns={'width': 'image_width', 'height': 'image_height'})
        print('Renamed width to image_width')
        print('Renamed height to image_height')
    else:
        error_counter = 0
        for i, row in tqdm(df.iterrows(), total=len(df)):
            image_name = row['image_path'].split('/')[-1]
            if not os.path.exists(f'{image_dir}/{str(image_name)}'):
                print(f'File not found: {image_dir}/{str(image_name)}')
                sys.exit(1)
            img = cv2.imread(f'{image_dir}/{str(image_name)}')
            if img is None:
                print('Image is none')
                error_counter += 1
                continue

            h0, w0 = img.shape[:2]
            df.at[i,'image_height'] = h0
            df.at[i,'image_width'] = w0
        print(f'Found {error_counter} error images')
    return df


def _make_box_features(df, bbox_col_name='bbox'):
    """ Make bbox related columns in a dataframe.
        These columns will be added.
        - xmin, ymin, xmax, ymax, width, height, area(width * height)
    Args:
        df (pd.DataFrame): input dataframe
        bbox_col_name (str, optional): column name for bbox coordinates string. Defaults to 'bbox'.

    Returns:
        pd.DataFrame: dataframe with bbox related columns
    """
    if not 'xmin' in df.columns and 'XMin' in df.columns:
        df['xmin'] = df['XMin'] * df['image_width']
    if not 'xmax' in df.columns and 'XMax' in df.columns:
        df['xmax'] = df['XMax'] * df['image_width']
    if not 'ymin' in df.columns and 'YMin' in df.columns:
        df['ymin'] = df['YMin'] * df['image_height']
    if not 'ymax' in df.columns and 'YMax' in df.columns:
        df['ymax'] = df['YMax'] * df['image_height']

    if bbox_col_name in df.columns:
        print(f'Found {bbox_col_name} column.')
        # replace nan values with zeros
        df[bbox_col_name] = df[bbox_col_name].fillna('[0,0,0,0]')
        # split bbox column
        bbox_items = df[bbox_col_name].str.split(',', expand=True)
        df['xmin'] = bbox_items[0].str.strip('[ ').astype(float)
        df['ymin'] = bbox_items[1].str.strip(' ').astype(float)
        df['width'] = bbox_items[2].str.strip(' ').astype(float)
        df['height'] = bbox_items[3].str.strip(' ]').astype(float)
        df['ymax'] = df['ymin'] + df['height']
        df['xmax'] = df['xmin'] + df['width']
    else:
        df['width'] = df['xmax'] - df['xmin']
        df['height'] = df['ymax'] - df['ymin']

    # compute bounding box areas
    df['area'] = df['width'] * df['height']
    return df


def build_features(cfg, df, img_dir, output_dir, out_csv_name='_features.csv', mode='train'):
    """ Add bbox related features and save as csv file

    Args:
        cfg (DictConfig): configs for preprocessing
        df (pd.DataFrame): input dataframe
        img_dir (str): input image directory
        output_dir (str): directory to save output csv
        out_csv_name (str, optional): output csv name after 'mode' string. Defaults to '_person_v3.csv'.
        mode (str, optional): should be 'all' or 'train' or 'valid' or 'test'. 
                              output csv name starts with 'mode' string. Defaults to 'train'.
    Returns:
        [type]: [description]
    """
    print('-'*30, f'Getting Image Size: {mode}', '-'*30)
    df_2 = _get_image_size(img_dir, df)
    print('-'*30, f'Making Box cols: {mode}', '-'*30)
    df_3 = _make_box_features(df_2)

    if cfg.DATA.REMOVE_LARGE_BBOXES:
        large_bbox_area_threshold = cfg.DATA.LARGE_BBOX_AREA_THRESHOLD
        print('-'*30, f'Removing Box area over {large_bbox_area_threshold}', '-'*30)
        df_3 = _remove_large_box(df_3, image_id_col_name='image_id', bbox_col_name='bbox', large_bbox_area_threshold=large_bbox_area_threshold)
    
    if 'Source' in df.columns:
        df_3 = df_3.rename(columns={'Source': 'source'})
        print('Renamed "Source" to "source" column.')
    out_csv_path = output_dir + mode + out_csv_name
    df_3.to_csv(out_csv_path, index=False)
    print(f'{mode} df saved to: {out_csv_path}')
    return df_3


def extract_one_label(df, label_col_name='LabelName', target_label_id='/m/01g317', mode='train'):
    """ Extract one class(label) from dataframe (For Open Images Dataset)

    Args:
        df (pd.DataFrame): dataframe to extract rows
        label_col_name (str, optional): label column name. Defaults to 'LabelName'.
        target_label_id (str, optional): value in 'label_col_name' to extract. Defaults to '/m/01g317'.
        mode (str, optional): 'all' or 'train' or 'valid' or 'test'. Defaults to 'train'.

    Returns:
        pd.DataFrame: dataframe only containing 'target_label_id' in 'label_col_name' column.
    """

    print(f'{mode}: Found Label ID: "{target_label_id}" in {label_col_name} column.')
    if label_col_name in df.columns:
        single_class_df = df[df[label_col_name]==target_label_id]
        if len(single_class_df) > 0:
            print(f'Found {len(single_class_df)} rows in {mode} df.')
            print(f'Found {single_class_df.image_id.nunique()} image_ids in {mode} df')
            assert single_class_df[label_col_name].nunique() == 1
            return single_class_df
        else:
            print(f'No label matched: "{target_label_id}" in {label_col_name} column.')
            sys.exit(1)
    else:
        print(f'Can not find "{label_col_name}" columns.')
        sys.exit(1)


def remove_flagged_rows(df, remove_flagged_rows):
    """ Remove rows which flagged (For Open Images Dataset)

    Args:
        df (pd.DataFrame): dataframe to remove rows
        remove_flagged_rows (list): target columns list

    Returns:
        pd.DataFrame: dataframe that removed rows with flag
    """
    for col in remove_flagged_rows:
        if col in df.columns:
            print(f'Removing {col} with flag : {len(df[df[col]==1.0])} boxes.')
            df = df[df[col]==0.0]
            assert len(df[df[col]==1.0]) == 0
        else:
            print(f'Can not find {col} column.')
    return df


def plot_bboxes(df, pred_images_dir, img_dir, save_images=True):
    """ Plot bounding boxes in a image and save them.

    Args:
        df (pd.DataFrame): dataframe containing image ids
        pred_images_dir (str): path to save output images.
        img_dir (str): path to original images
        save_images (bool, optional): if True, save gt images
    """
    os.makedirs(pred_images_dir, exist_ok=True)
    # Annotate and plot
    image_path_list = df['image_path'].unique().tolist()
    image_list = [image_path.split('/')[-1] for image_path in image_path_list]

    for i, im_file in enumerate(image_list[:10], start=1):
        fig, ax = plt.subplots(1, 1, figsize=(30, 25))
        target_df = df[df['image_id']==im_file.split('.')[0]]

        img = cv2.imread(f'{img_dir}/{str(im_file)}')
        if img is None:
            print(f'{img_dir}/{str(im_file)} is None.')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i, row in target_df.iterrows():
            cv2.rectangle(img, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), color=(0,255,0), thickness=2)

        ax.set_axis_off()
        ax.set_title(f'{im_file}')
        ax.imshow(img)

        if save_images:
            fig.savefig(f'{pred_images_dir}/gt_{im_file}', bbox_inches='tight')
            plt.close(fig)
            print(f'Sample Image Saved to: {pred_images_dir}/gt_{im_file}')


def get_annotations(cfg, mode, out_csv_name):
    """ Get annotations csv from interim or raw folders.
        csv name must be start with {mode}.

    Args:
        cfg (DictConfig): configs for preprocessing
        mode (str): should be 'all' or 'train' or 'valid' or 'test'
        out_csv_name (str): output csv file name including extension.

    Raises:
        ValueError: can not find csv file in both interim and raw directories.

    Returns:
        pd.DataFrame: dataframe from raw or interim path.
    """
    interim_csv_path = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/interim/{mode}*.csv'
    interim_list_len, interim_csv_path_list  = check_glob_list_size(interim_csv_path)
    if interim_list_len == 0:
        raw_csv_path = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/raw/{mode}*.csv'
        list_len, raw_csv_path_list = check_glob_list_size(raw_csv_path)
        if list_len == 0:
            if mode == 'valid' or mode == 'all':
                print(f'"{mode}" csv not found.')
                annotation_df = None
                return annotation_df
            else:
                raise ValueError(f'Can not find a "{mode}" csv in both "raw" and "interim" directories.')
        else:
            print('-'*30, f'Found a {mode} csv file in "raw" directory', '-'*30)
            annotation_path = raw_csv_path_list[0]
    else:
        print('-'*30, f'Found a {mode} csv file in "iterim" directory', '-'*30)
        annotation_path = interim_csv_path_list[0]
        csv_name = annotation_path.split('/')[-1]
        if csv_name == out_csv_name:
            print('-'*20, f'Found output csv file named "{csv_name}" in "iterim" directory', '-'*20)
            sys.exit(0)
    print(f'Loading Annotations from: {annotation_path}')
    annotation_df = pd.read_csv(annotation_path, dtype={cfg.DATA.IMAGE_ID_COL_NAME: str})
    print(f'{mode} csv length: ', len(annotation_df))
    return annotation_df


def get_images(cfg):
    """ Glob the directories and get the lists of train and test images

    Args:
        cfg (DictConfig): configs for preprocessing

    Raises:
        ValueError: can not find train images in "train_img_dir"

    Returns:
        str, list: train and test images directories and image list in the directories.
    """
    train_img_dir = hydra.utils.get_original_cwd() + '/' + cfg.DATA.TRAIN_IMAGE_DIR
    train_fns = glob(hydra.utils.get_original_cwd() + '/' + cfg.DATA.TRAIN_IMAGE_DIR + '/*')
    if len(train_fns) == 0:
        raise ValueError(f'Set train images in {train_img_dir}')
    
    test_fns = glob(hydra.utils.get_original_cwd() + '/' + cfg.TEST.TEST_IMAGE_DIR + '/*')

    if len(test_fns) > 0:
        test_img_dir = hydra.utils.get_original_cwd() + '/' + cfg.TEST.TEST_IMAGE_DIR
    else:
        test_img_dir = hydra.utils.get_original_cwd() + '/' + cfg.DATA.TRAIN_IMAGE_DIR
        print(f'Test images are set in {test_img_dir}')

    if test_img_dir == train_img_dir:
        print(f'Train and test images are in the same directory. \n {train_img_dir}')

    return train_img_dir, test_img_dir, train_fns, test_fns


def _remove_large_box(df, image_id_col_name='image_id', bbox_col_name='bbox', large_bbox_area_threshold=200000):
    """ Remove rows taht has large bbox area

    Args:
        df (pd.DataFrame): target dataframe
        image_id_col_name (str, optional): column name for image identifier. Defaults to 'image_id'.
        bbox_col_name (str, optional): column name for bbox coordinates string. Defaults to 'bbox'.
        large_bbox_area_threshold (int, optional): to remove bbox area over this threshold. Defaults to 200000.

    Returns:
        pd.DataFrame: dataframe which contains area less than 'large_bbox_area_threshold'
    """
    large_bbox_df = df[df['area'] > large_bbox_area_threshold]
    df_new = df.copy()
    for index, row in large_bbox_df.iterrows():
        df_new = df_new[~((df_new[bbox_col_name]==row[bbox_col_name]) & (df_new[image_id_col_name]==row[image_id_col_name]))]
    print(f'Removed {len(df)-len(df_new)} large bboxes.')
    return df_new


def make_label_mappings(df, target_col='class_id'):
    """ Convert class name to id and save mappings in yaml file

    Args:
        df (pd.DataFrame): target dataframe
        target_col (str, optional): class identifier column name. Defaults to 'class_id'.

    Returns:
        [pd.DataFrame]: dataframe with class id
    """
    print('-'*30, f'Making label mappings', '-'*30)
    labels = df[target_col].unique().tolist()
    values = [i for i in range(1, len(labels) + 1)]
    label_to_num = dict(zip(labels, values))
    print('-'*25, 'class mappings', '-'*25)
    print(label_to_num)
    print('-'*65)

    with open('label_mappings.yaml', 'w') as outfile:
        yaml.dump(label_to_num, outfile, default_flow_style=False)
    print(f'Saved class mappings in "{os.getcwd()}/label_mappings.yaml"')
    # apply using map
    df[target_col] = df[target_col].map(label_to_num)
    return df


def make_class_col(df, target_col='class_id'):
    """ Make class id column

    Args:
        df (pd.DataFrame): dataframe to add "class_id" column
        target_col (str, optional): column name for class id. Defaults to 'class_id'.

    Returns:
        df (pd.DataFrame): dataframe with "class_id" column
    """
    print('-'*30, f'Making class id column', '-'*30)
    if not target_col in df.columns:
        found_flag = False
        label_col_names = ['class', 'LabelName', 'label']
        for col_name in label_col_names:
            if col_name in df.columns:
                df = df.rename(columns={col_name: target_col})
                print(f'Renamed "{col_name}" column to "{target_col}" column.')
                found_flag = True
        if not found_flag:
            print('No class column is found')
            print(f'Setting class: 1 in "{target_col}" column')
            df[target_col] = 1
            label_to_num = dict(object1=1)
            print('-'*10, f'Set class name: "object1" and class id: 1', '-'*10)
            with open('label_mappings.yaml', 'w') as outfile:
                yaml.dump(label_to_num, outfile, default_flow_style=False)
            print(f'Saved class mappings in "{os.getcwd()}/label_mappings.yaml"')
            return df

    if df[target_col].dtype != int:
        n_class = df[target_col].nunique()
        if n_class == 1:
            if '/m/01g317' == df[target_col].unique()[0]:
                class_name = 'person'
            else:
                class_name = df[target_col].unique()[0]
            label_to_num = {class_name: 1}
            print('-'*10, f'Set "{class_name}" class as 1 in {target_col} column.', '-'*10)
            df[target_col] = 1
            with open('label_mappings.yaml', 'w') as outfile:
                yaml.dump(label_to_num, outfile, default_flow_style=False)
            print(f'Saved class mappings in "{os.getcwd()}/label_mappings.yaml"')
        else:
            df = make_label_mappings(df)      

    return df


@hydra.main(config_path="../../../configs", config_name="train")
def main(cfg):
    print('-'*30, 'Building Features', '-'*30)
    # output 
    output_dir = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/interim/'
    plot_image_dir = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/interim/sample_images/'

    # get images list
    train_img_dir, test_img_dir, train_fns ,test_fns = get_images(cfg)
    print(f'Found {len(train_fns)} train images')
    print(f'Found {len(test_fns)} test images')

    mode_list = ['all', 'train', 'valid', 'test']
    for mode in mode_list:
        print('-'*30, f'Processing mode: {mode}', '-'*30)

        out_csv_name = mode + cfg.DATA.PROCESSED_CSV_NAME
        annotations = get_annotations(cfg, mode=mode, out_csv_name=out_csv_name)

        if annotations is None:
           print('-'*30, f'No annotations for "{mode}"', '-'*30)
           continue

        if mode == 'test':
            img_list=test_fns
        else:
            img_list=train_fns

        # make dataframe matched images
        target_df = make_target_df(annotations, img_list=img_list, image_id_col_name = cfg.DATA.IMAGE_ID_COL_NAME, drop_na=True)
        print(f'Target {mode} csv length: ', len(target_df))
        
        # remove rows using flags
        remove_flagged_col_list = ['IsTruncated', 'IsGroupOf', 'IsDepiction']
        target_df = remove_flagged_rows(target_df, remove_flagged_col_list)

        open_images_label_col_name='LabelName'
        if cfg.MODEL.NUM_CLASSES == 1:
            if open_images_label_col_name in target_df.columns and cfg.DATA.EXTRACT_ONE_CLASS:
                print('-'*30, f'Extracting 1 class from {target_df[open_images_label_col_name].nunique()} classes', '-'*30)
                target_label_id='/m/01g317' # person label
                target_df = extract_one_label(target_df, label_col_name=open_images_label_col_name, target_label_id=target_label_id, mode=mode)
            else:
                print('-'*30, 'Number of Class: 1', '-'*30)
        else:
            if open_images_label_col_name in target_df.columns:
                print('-'*30, f'Found {target_df[open_images_label_col_name].nunique()} Classes', '-'*30)

        # class column process 
        target_df = make_class_col(target_df)

        # build features for detection task
        if mode == 'test':        
            if ('bbox' in target_df.columns) or ('xmin' in target_df.columns) or ('XMin' in target_df.columns):
                target_df = build_features(cfg=cfg, df=target_df, img_dir=test_img_dir, output_dir=output_dir, out_csv_name=cfg.DATA.PROCESSED_CSV_NAME, mode=mode)
        else:
            target_df = build_features(cfg=cfg, df=target_df, img_dir=train_img_dir, output_dir=output_dir, out_csv_name=cfg.DATA.PROCESSED_CSV_NAME, mode=mode)
       
        # save images to check ground truth
        save_sample_images = True
        if save_sample_images:
            print('-'*30, 'Plotting Sample Images with Boxes', '-'*30)
            if mode == 'test': 
                if ('bbox' in target_df.columns) or ('xmin' in target_df.columns) or ('XMin' in target_df.columns):
                    plot_bboxes(target_df, plot_image_dir + 'test', test_img_dir)
            else:
                plot_bboxes(target_df, plot_image_dir + 'train', train_img_dir)
        
        print('-'*30, f'Done Processing {mode} !', '-'*30)

        if mode == 'all':
            break

    print('-'*30, f'Buiding Feature Finished Successfully!', '-'*30)


if __name__ == '__main__':
    main()
