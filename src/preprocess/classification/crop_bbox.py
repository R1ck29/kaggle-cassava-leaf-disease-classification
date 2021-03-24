import os

import hydra
import pandas as pd
from PIL import Image
from tqdm import tqdm


@hydra.main(config_path="../../../configs", config_name="test")
def main(cfg):
    print('-'*30, 'Cropping Bbox' , '-'*30)
    detection_result = pd.read_pickle(os.path.join(hydra.utils.get_original_cwd(), cfg.MODEL_PATH, f'result/{cfg.DATA.DATA_ID}/result.pkl'))
    # person_result = detection_result[detection_result['class_id']==2]

    person_result = detection_result.sort_values(by=['image_id'])

    person_result = person_result.reset_index(drop=True)

    # out_images_dir = '../../data/person_attribute_demo/raw/images/'
    output_dir = hydra.utils.get_original_cwd() + f'/data/{cfg.DATA.DATA_ID}/raw/person_images/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Saving to: ', output_dir)
    test_image_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.TEST.TEST_IMAGE_DIR)
    image_file_format = os.listdir(test_image_dir)[0].split('.')[-1]
        
    for i, row in tqdm(person_result.iterrows(), total=len(person_result)):
        if image_file_format not in row.image_path:
            extension = '.' + image_file_format
        else:
            extension = ''
        img_name = row.image_path + extension
        img=Image.open(img_name)
        width = cfg.MODEL.INPUT_SIZE
        height = cfg.MODEL.INPUT_SIZE
        img = img.resize((width, height))
        bbox = (row.xmin, row.ymin, row.xmax, row.ymax)
        
        crop_img=img.crop(bbox)
        if image_file_format not in row.image_path:
            image_id = row.image_id
        else:
            image_id = row.image_id.split('.')[0]
        crop_img = crop_img.save(os.path.join(output_dir, image_id+ f'_{i}.jpg')) 
        person_result.loc[i, 'object_id'] = image_id + f'_{i}.jpg'

    person_result.to_pickle(os.path.join(hydra.utils.get_original_cwd(), cfg.MODEL_PATH, f'result/{cfg.DATA.DATA_ID}/result_2.pkl'))

if __name__ == '__main__':
    main()
