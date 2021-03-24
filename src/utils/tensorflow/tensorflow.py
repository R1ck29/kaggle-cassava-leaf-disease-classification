import os
import random
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.generator.segmentation.generator import Generator
from src.data.transforms.transform import Transform
from src.models.modeling.segmentation.keras_deeplab_v3_plus.model import Deeplabv3

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.compat.v1.keras.callbacks import Callback, LearningRateScheduler

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

SEED = 123
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['HOROVOD_FUSION_THRESHOLD']='0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_generator(cfg):
    if cfg.TASK == 'detection':
        raise NotImplementedError
        
    elif cfg.TASK == 'segmentation':
        trans = Transform()
        split_path = os.path.join(cfg.MODEL.BASE_PATH, '../data', cfg.DATASET['DATA_ID'], 'split',
                                  cfg.DATASET['SPLIT_ID'])
        gen = Generator(split_path,
                        cfg.TRAIN['BATCH_SIZE'],
                        cfg.DATASET['NUM_CLASSES'],
                        cfg.DATASET['INPUT_SHAPE'],
                        cfg.DATASET['INPUT_SHAPE'],  # old gen_iou gt_shape
                        transform=trans)
        
    elif cfg.TASK == 'keypoint':
        raise NotImplementedError
    
    return gen

def create_model(cfg,device):
    print('Loading model ...')

    # モデルのインスタンス化
    with tf.device('/gpu:0'):
        deeplab_model = Deeplabv3(input_shape=cfg.DATASET['INPUT_SHAPE'],
                                  classes=cfg.DATASET['NUM_CLASSES'] - 1,
                                  backbone=cfg.MODEL['BACKBONE'],
                                  weights='cityscapes')
    # モデルパラメタの読み込み
    deeplab_model.load_weights(os.path.join(cfg.MODEL.BASE_PATH, cfg.MODEL['MODEL_PATH']),
                               by_name=True)

    gpus = len(device)
    if gpus>=2:
        deeplab_model = multi_gpu_model(deeplab_model, gpus=gpus)
    else:
        pass

    # define model
    deeplab_model.compile(optimizer=cfg.TRAIN['OPTIMIZER'],
                          loss=cfg.TRAIN['LOSS'],
                          metrics=['accuracy'])
    return deeplab_model

class IoUCallback(Callback):
    def __init__(self, filedir, generator, save=True):
        '''
        Parameters
        ----------
        filedir : str
            保存先ディレクトリ名
        generator : generator.Generator
            ジェネレータ
        save : bool
            True時に計算結果を保存する
        '''
        super().__init__()
        self.filedir = os.path.join(filedir, 'iou')
        os.makedirs(self.filedir, exist_ok=True)
        self.gen = generator
        self.save = save
        self.num_classes = self.gen.num_classes
        if self.save:
            self.modelpath = os.path.relpath(os.path.join(self.filedir, '..', 'best_iou_model.h5'))
            self.max_miou = 0

    def calc_confusion(self, gt, pred, key):
        '''教師ラベルと予測結果を入力とし、各クラスごとに混同行列の要素を計算する。

        Parameters
        ----------
        gt : np.ndarray [shape=(height, width, NUM_CLASSES)]
            教師ラベル(正解データ)
        pred : np.ndarray [shape=(height, width, NUM_CLASSES)]
            予測結果ラベル
        key : str
            データ名

        Returns
        -------
        confusion : pd.DataFrame
            表形式の混同行列
        '''
        gt_unlabeled = (gt == self.num_classes - 1)
        confusion = []
        for cls in range(self.num_classes - 1):
            gt_cls = (gt == cls)
            pred_cls = (pred == cls) & (gt_unlabeled == False)

            tp = np.sum(gt_cls & pred_cls)
            fp = np.sum((~gt_cls) & pred_cls)
            fn = np.sum(gt_cls & (~pred_cls))
            tn = np.sum((~gt_cls) & (~pred_cls))
            confusion.append([key, cls, tp, fp, fn, tn])
        return pd.DataFrame(confusion)

    def concat_confs(self, confs):
        '''表形式の混同行列から、データセット全体のIoU、画像ごとのIoUを計算する。

        Parameters
        ----------
        confs : pd.DataFrame
            表敬式の混同行列

        Returns
        -------
        conf_table : pd.DataFrame
            データセット全体のIoU
        conf_table2 : pd.DataFrame
            各画像ごとのIoU
        '''
        conf_df = pd.concat(confs, axis=0)
        conf_df.columns = ['key', 'cls', 'tp', 'fp', 'fn', 'tn']
        conf_table = conf_df.groupby(['cls']).sum()
        conf_table['iou'] = conf_table.tp / (conf_table.tp + conf_table.fp + conf_table.fn)
        conf_table2 = conf_df.groupby(['key', 'cls']).sum()
        conf_table2['iou'] = conf_table2.tp / (conf_table2.tp + conf_table2.fp + conf_table2.fn)
        return conf_table, conf_table2

    def on_epoch_end(self, epoch, logs={}):
        '''毎エポック終了時に評価データに対するIoUを算出する。

        Paramters
        ---------
        epoch: integer
            index of epoch.
        logs: dict
            Currently no data is passed to this argument for this method
            but that may change in the future.
        '''
        dfs = []
        for _, (imgs, gts, keys) in enumerate(self.gen.generate(train=False,
                                                                key=True,
                                                                oneloop=True,
                                                                random_transform=False)):
            # argmaxによる1チャネル化
            preds = self.model.predict(imgs).argmax(axis=-1).astype(np.uint8)
            gts = gts.argmax(axis=-1).astype(np.uint8)
            # バッチ内の処理
            for gt, pred, key in zip(gts, preds, keys):
                # confusion計算
                pred = self.gen.transform.resize(pred, gt.shape, 'nearset')
                dfs.append(self.calc_confusion(gt, pred, key))

        conf_table, conf_table2 = self.concat_confs(dfs)
        # 全サンプルの集計値
        conf_table.to_csv(os.path.join(self.filedir, 'iou_total_' + str(epoch).zfill(6) + '.log'))
        # 各サンプルごとの集計値
        conf_table2.to_csv(os.path.join(self.filedir, 'iou_sample_' + str(epoch).zfill(6) + '.log'))
        miou = np.nanmean(conf_table.iou)
        iou_class = conf_table.iou.tolist()
        with open(os.path.join(self.filedir, '..', 'iou.log'), 'ab') as f:
            np.savetxt(f,
                       np.array(iou_class + [miou]).reshape((1, -1)),
                       fmt='%.6f', delimiter=',')
        print('val_mIoU : {:5f}'.format(miou))

        self.max_miou_old, self.max_miou = self.max_miou, max(miou, self.max_miou)
        if self.max_miou > self.max_miou_old:
            print('val_mIoU improved from {:5f} to {:5f}'.format(self.max_miou_old, self.max_miou))
            if self.save:
                print('saving model to', self.modelpath)
                self.model.save(self.modelpath)
        else:
            print('val_mIoU did not improve from {:5f}'.format(self.max_miou))


# LRスケジューラ
def get_scheduler(cfg):
    step = cfg.TRAIN.LR_STEP
    factor = cfg.TRAIN.LR_FACTOR

    def scheduler(epoch, lr):
        if epoch >= step:
            lr *= factor
        return lr

    return scheduler


class Customlogs(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_epoch_end(self, epoch, logs={}):
        import datetime
        date = datetime.datetime.today().strftime("%Y%m%d_%H%M")
        logs['date'] = date

        scheduler = get_scheduler(self.cfg)
        lr = scheduler(epoch, self.cfg.TRAIN.LR)
        logs['LR'] = lr


def get_callback(output_path, cfg, gen_iou=None, patience=None):
    '''各種コールバックを作成する。
    Parameters
    ----------
    output_path : str
        モデルと訓練履歴を保存するディレクトリ名
    gen_iou : iterator
        IoU計算に使用するデータのジェネレータ
    patience : int > 0
        val_lossが改善しなくなってから、訓練を打ち切るまでのエポック数

    Returns
    -------
    cbs : list
        コールバックのインスタンスのリスト
    '''
    cbs = []
    # LRスケジューラの設定
    # cbs += [LearningRateScheduler(scheduler,verbose=1)]
    cbs += [LearningRateScheduler(get_scheduler(cfg), verbose=1)]

    cbs += [Customlogs(cfg)]

    # 訓練履歴(loss, acc)の保存
    # trainlog_%sなどでfoldidを指定、modelも
    cbs += [CSVLogger(os.path.join(output_path, 'trainlog.log'), append=False)]
    # lossが最も低いモデルを保存
    cbs += [ModelCheckpoint(filepath=os.path.join(output_path,
                                                  'best_loss_model.h5'),
                            verbose=1, save_best_only=True, monitor='val_loss')]
    # accuracyが最も高いモデルを保存
    cbs += [ModelCheckpoint(filepath=os.path.join(output_path,
                                                  'best_acc_model.h5'),
                            verbose=1, save_best_only=True, monitor='val_acc')]
    # 最終エポックのモデルを上書き保存
    cbs += [ModelCheckpoint(os.path.join(output_path, 'last_epoch_model.h5'),
                            verbose=1, save_weights_only=False)]
    # IoUの計算
    if gen_iou:
        cbs += [IoUCallback(filedir=output_path, generator=gen_iou)]
    # val_lossがpatienceエポック経過後に下がらない場合に訓練を終了
    if patience:
        cbs += [EarlyStopping(monitor='val_loss', patience=patience)]

    return cbs
