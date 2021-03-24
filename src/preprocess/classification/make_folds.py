import os
import sys
from glob import glob

import hydra
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from tqdm import tqdm


@hydra.main(config_path="../../../configs", config_name="train")
def main(cfg):
    print('-'*30, 'Making Train Validation df', '-'*30)

    csv_dir = os.path.join(hydra.utils.get_original_cwd(), f'data/{cfg.DATA.DATA_ID}/split/')
    out_path = os.path.join(hydra.utils.get_original_cwd(), cfg.DATA.CSV_PATH)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    train_df = pd.read_csv(csv_dir + 'merged_v2.csv')
    
    folds = StratifiedKFold(n_splits=cfg.DATA.N_FOLD, shuffle=True, random_state=cfg.SYSTEM.SEED).split(np.arange(train_df.shape[0]), train_df[cfg.DATA.FOLD_TARGET_COL].values)

    fold_df = train_df.copy()
    for fold, (trn_idx, val_idx) in enumerate(folds):
        fold_df.loc[val_idx,'fold'] = fold

    img_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.DATA.TRAIN_IMAGE_DIR)
    train_fns = glob(img_dir + '/*')

    image_id_col_name = 'image_id'
    all_images = pd.DataFrame([fns.split('/')[-1] for fns in train_fns])
    all_images['image_path'] = [fns for fns in train_fns]
    all_images.columns=[image_id_col_name, 'image_path']

    assert len(fold_df) == len(all_images)

    df = pd.merge(fold_df, all_images, on='image_id', how='left')

    df = df.rename(columns={'label': 'class_id'})

    if df.isnull().sum().sum() > 0:
        # Count the NaN under an entire DataFrame:
        print(f'Number of NaN in DataFrame : {df.isnull().sum().sum()}')
        sys.exit(1)
    else:
        for fold in range(5):
            fold_len = len(df[df['fold']==fold]) / len(df)
            print(f'Fold {fold}: ', fold_len * 100, '%')
        df.to_csv(out_path, index=False)
        print(f'Train Val csv Saved to: {out_path}')
        print('-'*30, 'Done Making Train Val df', '-'*30)


if __name__ == '__main__':
    main()
