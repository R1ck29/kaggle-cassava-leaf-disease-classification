import os
import sys
import cv2
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.models.predictor.classification.mappings import person_attribute_names
import os
import shutil
from glob import glob


def _convert_id_to_name(id_str_list, attr_name_mappings):
    """convert class id to class name

    Args:
        id_str_list (list): class id list 
        attr_name_mappings (dict): class name list for each dataset type

    Returns:
        [list]: class name list
    """
    label_ids = id_str_list.split()
    map_object = map(int, label_ids)
    id_num_list = list(map_object)

    attr_name_list = [attr_name_mappings[int(i)] for i in id_num_list]
    return attr_name_list


def copy_images(img_dir, demo_images_dir):
    print(f'Copying images...')
    print(f'From {img_dir}')
    print(f'To {demo_images_dir}')
    src_files = os.listdir(img_dir)
    for file_name in tqdm(src_files, total=len(src_files)):
        full_file_name = os.path.join(img_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, demo_images_dir)
    
    
def plot_bboxes(df, img_dir, demo_images_dir=None, resize_width=512, resize_height=512, save_images=True, resize=True, debug=False):
    """ Plot bounding boxes in a image and save them.

    Args:
        df (pd.DataFrame): dataframe containing image ids
        img_dir (str): path to original images
        demo_images_dir (str): path to save output images.
        save_images (bool, optional): if True, save gt images
    """
    # Annotate and plot
    # image_path_list = df['image_path'].unique().tolist()
    # image_list = [image_path.split('/')[-1] for image_path in image_path_list]
    image_path_list = glob(img_dir + '/*')
    image_list = [image_path.split('/')[-1] for image_path in image_path_list]

    if demo_images_dir is not None:
        print(f'Plot Images Saving to: {demo_images_dir}')

    image_file_format = os.listdir(img_dir)[0].split('.')[-1]

    detected_objects = 0
    for i, im_file in enumerate(tqdm(image_list, total=len(image_path_list)), start=1):
        if debug:
            if i > 5:
                print('Debug run is finished!')
                break
        fig, ax = plt.subplots(1, 1, figsize=(25, 20))

        # if image_file_format not in str(im_file):
        #     img_name = im_file + '.' + image_file_format
        # else:
        #     img_name = str(im_file)
        img = cv2.imread(f'{img_dir}/{im_file}')
        if img is None:
            print(f'{img_dir}/{im_file} is None.')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if resize:
            dim = (resize_width, resize_height)
            if i == 1:
                print(f'Resizing image to: {dim}')
            # resize image
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            font_scale = 0.4
            thickness = 1
        else:
            font_scale = 2
            thickness = 5
            
        #TODO: make options
        sample_img_id = df['image_id'].values[0]
        if image_file_format in sample_img_id:
            extension_flag = True
        else:
            extension_flag = False

        if extension_flag:
            target_df = df[df['image_id']==im_file]
        else:
            target_df = df[df['image_id']==im_file.split('.')[0]]

        for i, row in target_df.iterrows():
            pred_attr_names = _convert_id_to_name(row.attr_id, person_attribute_names['person_attribute_demo'])
            if 'Female' in pred_attr_names:
                gender_color = (255,0,0)
            else:
                gender_color = (0,0,255)
            cv2.rectangle(img, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), color=gender_color, thickness=2)
            pred_attr_name = ' '.join(pred_attr_names)
            cv2.putText(img, f'{pred_attr_name}', (int(row.xmin), int(row.ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, gender_color, thickness, cv2.LINE_AA)
            detected_objects += 1
        ax.set_axis_off()
        # ax.set_title(f'{im_file}')
        ax.imshow(img)
        if 'bmp' in im_file:
            im_file = im_file.replace('bmp', 'jpg')
        if save_images and demo_images_dir is not None:
            fig.savefig(f'{demo_images_dir}/{im_file}', bbox_inches='tight')
            plt.close(fig)
    
    print('Detected Objects: ', detected_objects)


@hydra.main(config_path="../../configs", config_name="test")
def main(cfg):
    print('-'*30, 'Visualizing Bboxes and Person Attribute', '-'*30)
    # detection result
    detection_result = pd.read_pickle(os.path.join(hydra.utils.get_original_cwd(), f'models/effdet_person_ca_v5/20201010_00_40_05/result/{cfg.TEST_ID}/result_2.pkl'))
    #TODO: change class id to extract
    # person_result = detection_result[detection_result['gt_class_id']==2]
    person_result = detection_result.sort_values(by=['image_id'])
    person_result = person_result.reset_index(drop=True)

    resize_width = 640 #512
    resize_height = 640 #512

    # classification result
    cls_result = pd.read_pickle(os.path.join(hydra.utils.get_original_cwd(), cfg.MODEL_PATH, f'result/{cfg.TEST_ID}/result.pkl'))
    cls_result = cls_result.sort_values(by=['image_id'])
    cls_result = cls_result.reset_index(drop=True)
    cls_result = cls_result.sort_values(by=['image_id'])
    cls_result = cls_result.rename(columns={'class_id': 'attr_id', 'image_id': 'object_id'})

    df = pd.merge(person_result, cls_result, left_on='object_id', right_on='object_id')
    df.to_pickle(os.getcwd() + f'/{cfg.TEST_ID}.pkl')

    assert df.isnull().sum(axis = 0).sum() == 0

    img_dir = os.path.join(hydra.utils.get_original_cwd(),f'data/{cfg.DATA.DATA_ID}/raw/images')

    demo_images_dir = os.path.join(os.getcwd(), 'demo_frames')
    if not os.path.exists(demo_images_dir):
        os.makedirs(demo_images_dir, exist_ok=True)

    # copy_images(img_dir, demo_images_dir)

    plot_bboxes(df, img_dir, demo_images_dir=demo_images_dir, resize_width=resize_width, resize_height=resize_height, save_images=True, resize=True, debug=False)


if __name__ == '__main__':
    main()
