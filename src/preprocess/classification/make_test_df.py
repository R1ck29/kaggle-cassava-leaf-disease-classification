from glob import glob
import os

import cv2
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


@hydra.main(config_path="../../../configs", config_name="test")
def main(cfg):
    print('-'*30, 'Making Test df without GT', '-'*30)

    img_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.TEST.TEST_IMAGE_DIR) #f'/data/person_attribute_demo/raw/images'
    test_fns = glob(img_dir + '/*')

    image_id_col_name = 'image_id'
    all_images = pd.DataFrame([fns.split('/')[-1] for fns in test_fns]) #[:-4]
    all_images['image_path'] = [fns for fns in test_fns]
    all_images.columns=[image_id_col_name, 'image_path']

    if cfg.DATA.DATA_ID != 'person_attribute_demo':
        all_images['class_id'] = 0

    out_path = os.path.join(hydra.utils.get_original_cwd(), cfg.TEST.TEST_CSV_PATH) #f'/data/person_attribute_demo/split/test_v1.csv'
    csv_dir = os.path.join(hydra.utils.get_original_cwd(), f'data/{cfg.DATA.DATA_ID}/split/')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    all_images.to_csv(out_path, index=False)
    print(f'Test csv Saved to: {out_path}')
    print('-'*30, 'Done Making Test df without GT', '-'*30)


if __name__ == '__main__':
    main()
