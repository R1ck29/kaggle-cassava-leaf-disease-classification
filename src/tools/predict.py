import os
from os.path import join, dirname, abspath
import sys
import datetime
import pandas as pd
from omegaconf import OmegaConf
import hydra

root = abspath(join(dirname(__file__), '../..'))
sys.path.append(root)
from src.utils.common import set_gpu
    
@hydra.main(config_path="../../configs", config_name="test")
def main(cfg):
    # フレームワーク共通の処理
    model_cfg_path = join(hydra.utils.get_original_cwd(), cfg.MODEL_PATH,'.hydra/config.yaml')
    model_cfg = OmegaConf.load(model_cfg_path)
    cwd = os.getcwd()
    
    # set gpu
    if cfg.SYSTEM.DEVICE == 'GPU':
        gpu_id = [int(gid) for gid in str(cfg.SYSTEM.GPUS)]
        set_gpu(gpu_id)
    
    # フレームワーク毎の処理
    print('FRAMEWORK: ', cfg.FRAMEWORK)
    if cfg.FRAMEWORK == 'pytorch':
        from src.utils.pytorch import get_test_data_loader, load_predictor, set_seed
        import torch
        torch.backends.cudnn.enabled = cfg.SYSTEM.CUDNN.ENABLED
        
        # load dataset
        test_dataloader = get_test_data_loader(cfg, model_cfg)
        
        if cfg.SYSTEM.SEED:
            set_seed(cfg)
        
        # load predictor
        predictor = load_predictor(cfg, model_cfg)

        if cfg.TASK == 'detection' and cfg.TEST.FIND_BEST_THR:
            print('-'*30, 'Evaluating on Validation dataset', '-'*30)
            predictor.evaluate()
        elif cfg.TASK == 'classification' and cfg.TEST.VALID_PREDICTION:
            print('-'*30, 'Evaluating on Validation dataset', '-'*30)
            predictor.evaluate()
        
        # Predict
        df = predictor.predict(test_dataloader)
        save_path = join(cwd, 'result.pkl')
        df.to_pickle(save_path)
        print("saved result at %s"%save_path)

    
if __name__ == '__main__':
    main()