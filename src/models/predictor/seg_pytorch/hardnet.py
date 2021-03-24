import os
from os.path import join, dirname
import sys
import numpy as np
import pandas as pd
import timeit
import time
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix

import torch
from pytorch_lightning.core.lightning import LightningModule

sys.path.append(join(dirname(__file__), "../../../.."))
from src.models.modeling.seg_pytorch.hardnet import hardnet
from src.data.generator.seg_pytorch import make_dataloader
from src.models.modeling.seg_pytorch import get_model
from src.utils.pytorch.utils import RunningScore, AverageMeter, fuse_bn_recursively
from src.data.transforms.build import get_composed_augmentations



def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

        
class Predictor:
    def __init__(self, cfg, model_cfg):
        super().__init__()
        
        self.cfg = cfg
        self.model_cfg = model_cfg
        
        if torch.cuda.is_available():
            if cfg.SYSTEM.DEVICE == 'GPU':
                self.device =  torch.device('cuda')
        else:
            self.device = 'cpu'
        
        # hardnetの呼び出し
        self.model = get_model(self.model_cfg.MODEL.arch, self.model_cfg.MODEL.n_classes).apply(weights_init).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print( 'Parameters:',total_params )
        
        # 重みの呼び出し
        state_path = os.path.join(self.cfg.MODEL_PATH, 'state.pth')
        
        if torch.cuda.is_available():
            weight = torch.load(state_path)
        else:
            weight = torch.load(state_path, map_location=torch.device('cpu'))
        
        self.model.load_state_dict(weight)
        
        if self.model_cfg.MODEL.fuse_bn:
            self.model = fuse_bn_recursively(self.model)
        
        self.model_cfg.TEST.BATCH_SIZE = 1
        
        # 推論モード
        self.model.eval()
        
    def predict_image(self, arr):
        
        h, w, c = arr.shape
        
        img = np.array(arr, dtype=np.uint8)
        
        
        img = np.array(Image.fromarray(img).resize(
                (w, h)))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
    
        
        value_scale = 255
        mean = [0.406, 0.456, 0.485]
        mean = [item * value_scale for item in mean]
        std = [0.225, 0.224, 0.229]
        std = [item * value_scale for item in std]
        img = (img - mean) / std
        
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        img = torch.from_numpy(img).float().to(self.device)
        
        out = self.model(img)
        pred = out.data.max(1)[1].cpu().numpy()
        
        return pred
        
    
    def predict(self, dataloader=None):
        
        if dataloader == None:
            dataloader = self.val_data
        
        running_metrics = RunningScore(self.model_cfg.MODEL.n_classes)
        total_time, cnt = 0, 0

        torch.backends.cudnn.benchmark=True
        
        IDs, img_paths, label_paths, arrays, conf_mats = [], [], [], [], []

        for i, (images, labels, fname, tup) in tqdm(enumerate(dataloader)):
            images = images.to(self.device)
            img_path, lbl_path = tup

            if i == 0:
              with torch.no_grad():
                outputs = self.model(images)        
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()

            with torch.no_grad():
              outputs = self.model(images)

            pred = outputs.data.max(1)[1].cpu().numpy()

            gt = labels.numpy()
            s = np.sum(gt==pred) / (1024*2048)
            running_metrics.update(gt, pred)
            
            cm = multilabel_confusion_matrix(np.squeeze(gt).ravel(), np.squeeze(pred).ravel())
    
            IDs.append(fname[0])
            img_paths.append(img_path[0])
            label_paths.append(lbl_path[0])
            arrays.append([pred])
            conf_mats.append([cm])
            
            cnt += 1
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            elapsed_time = time.perf_counter() - start_time
            total_time += elapsed_time
        
        score, class_iou = running_metrics.get_scores()
        print("Total Frame Rate = %.2f fps" %(cnt / total_time))
        
        for k, v in score.items():
            print(k, v)

        for i in range(self.model_cfg.MODEL.n_classes):
            print(i, class_iou[i])
        
        return pd.DataFrame(list(zip(IDs, img_paths, label_paths, arrays, conf_mats)), columns=['ID', 'ImgPath', 'GtPath', 'Array', 'Confmat'])

        
    
    

    