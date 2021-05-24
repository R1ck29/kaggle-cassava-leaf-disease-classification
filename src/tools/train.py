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


if __name__ == '__main__':
    main()