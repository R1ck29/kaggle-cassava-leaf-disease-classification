import os
from os.path import join, dirname, abspath
import sys
import datetime
import argparse
import pandas as pd
import hydra
import gc

root = abspath(join(dirname(__file__), '../..'))
sys.path.append(root)
from src.utils.common import save_env, set_gpu


@hydra.main(config_path="../../configs", config_name="train")
def main(cfg):
    # フレームワーク共通の処理
    data_path = join(root, "data")
    output_path = os.getcwd()
        
    # save execution environment
    save_env(cfg, root, output_path)
    
    # フレームワーク毎の処理
    print('FRAMEWORK: ', cfg.FRAMEWORK)
    if cfg.FRAMEWORK == 'pytorch':
        import torch
        import pytorch_lightning as pl
        from src.utils.pytorch import get_callback, create_model, save_model
        from src.utils.pytorch.utils import set_seed

        gpu_id = [int(gid) for gid in str(cfg.SYSTEM.GPUS)]
        set_gpu(gpu_id)
        
        # cudnn related setting
        torch.backends.cudnn.enabled = cfg.SYSTEM.CUDNN.ENABLED
            
        if cfg.SYSTEM.SEED:
            set_seed(cfg)
            
        for fold in range(cfg.DATA.N_FOLD):
            # Make Callback
            loggers, lr_logger, model_checkpoint, early_stopping = get_callback(cfg, output_path, fold)

            # define model
            model = create_model(cfg, fold, data_path=data_path, output_path=output_path)
            
            trainer = pl.Trainer(logger=loggers,
                                early_stop_callback=early_stopping,
                                checkpoint_callback=model_checkpoint,
                                callbacks=[lr_logger],
                                **cfg.TRAINER)

            trainer.fit(model)

            save_model(cfg, output_path, fold)

            del model, trainer, loggers, lr_logger, model_checkpoint, early_stopping
            torch.cuda.empty_cache()
            gc.collect()
        
    elif cfg.FRAMEWORK == 'tensorflow':
        from src.utils.tensorflow import get_generator, create_model, get_callback
        import tensorflow as tf
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        set_gpu(cfg.SYSTEM.GPUS)
        device = [int(gid) for gid in cfg.SYSTEM.GPUS.split(',')]
        
        # fold loop
        for fold in range(cfg.DATA.N_FOLD):
            # データ生成関数の定義
            gen = get_generator(cfg)

            # コールバックのインスタンス化
            cbs = get_callback(output_path,cfg,gen)

            # define model
            model = create_model(cfg,device)

            # do training
            model.fit_generator(gen.generate(train=True,
                                random_transform=True,
                                random_crop=cfg.TRAIN['RANDOM_CROP']),
                                steps_per_epoch=gen.train_df.shape[0]//cfg.TRAIN['BATCH_SIZE']*len(device),
                                epochs=cfg.TRAIN['EPOCHS'],
                                validation_data=gen.generate(train=False,
                                                             random_transform=False),
                                validation_steps=gen.val_df.shape[0],
                                callbacks=cbs)

if __name__ == '__main__':
    main()