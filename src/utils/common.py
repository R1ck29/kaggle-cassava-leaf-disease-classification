import collections
import colorsys
import importlib
import os
from os.path import join
import random
import shutil
from glob import glob
from itertools import product
from typing import Any, Dict, Generator, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

def set_gpu(gpu_id):
    gpu_id = [str(i) for i in gpu_id]
    gpu_id = ",".join(gpu_id)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def save_env(cfg, root, output_path):
    """Config、ソースコード、ライブラリ、Pretrain済み重みを保存する。

    Args:
        cfg (CfgNode): config
        input_path (str): ソースコードのパス
        output_path (str): 出力先のパス
    """
    # save source code
    shutil.copytree(join(root, "src"), join(output_path, 'code/src'))
    
    # save dependencies
    req_path = os.path.join(output_path, 'requirements.txt')
    command = 'pip freeze > out_path'
    command = command.replace('out_path', req_path)
    os.system(command)

    
def split_extract(series, char, position=slice(None)):
    return series.str.split(char, expand=True).iloc[:, position]

def join_str(df, char):
    a = ''
    for i in range(len(df.columns)):
        a += df.iloc[:, i] + char
    return a.str[:-len(char)]


def file_df_from_paths(paths):
    df = pd.DataFrame(paths)
    df.columns = ['filepath']
    df['filename'] = split_extract(df.iloc[:, 0], os.sep, -1)
    df['fileid'] = df.filename.apply(lambda x: os.path.splitext(x)[0])
    return df

def file_df_from_path(path):
    ext_list = ['png','jpg']
    paths = []
    for ext in ext_list:
        paths.extend(sorted(glob(os.path.join(path, '**', '*.' + ext), recursive=True)))
    df = file_df_from_paths(paths)
    return df

def to_timedelta(series):
    base_time = pd.to_datetime('00:00:00', format='%H:%M:%S')
    return pd.to_datetime(series, format='%H:%M:%S') - base_time


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    This function is quoted from keras 2.2.4 (keras.utils.to_categorical).
    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1

    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    print('num_classes: ', num_classes)
          
    categorical[np.arange(n), y] = 1 
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def fixed_colors(bright=True):
    """
    Generate fixed colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    N = 300
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    
    colors2 = []
    for i in range(30):
        colors2 += colors[i::30]
    return np.array(colors2)


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)

def save_useful_info():
    shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'src'), os.path.join(os.getcwd(), 'code/src'))
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(), 'src/tools/train_detection.py'),
                 os.path.join(os.getcwd(), 'code'))


def collate_fn(batch):
    return tuple(zip(*batch))


def product_dict(**kwargs: Dict) -> Generator:
    """
    Convert dict with lists in values into lists of all combinations

    This is necessary to convert config with experiment values
    into format usable by hydra
    Args:
        **kwargs:

    Returns:
        list of lists

    ---
    Example:
        >>> list_dict = {'a': [1, 2], 'b': [2, 3]}
        >>> list(product_dict(**list_dict))
        >>> [['a=1', 'b=2'], ['a=1', 'b=3'], ['a=2', 'b=2'], ['a=2', 'b=3']]

    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        zip_list = list(zip(keys, instance))
        yield [f'{i}={j}' for i, j in zip_list]


def config_to_hydra_dict(cfg: DictConfig) -> Dict:
    """
    Convert config into dict with lists of values, where key is full name of parameter

    This fuction is used to get key names which can be used in hydra.

    Args:
        cfg:

    Returns:
        converted dict

    """
    experiment_dict = {}
    for k, v in cfg.items():
        for k1, v1 in v.items():
            experiment_dict[f'{k}.{k1}'] = v1

    return experiment_dict


def flatten_omegaconf(d, sep='_'):
    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=''):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}
    # obj = {k: v for k, v in obj.items()}

    return obj


def freeze_until(net, param_name: str = None):
    """
    Freeze net until param_name

    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD

    Args:
        net:
        param_name:

    Returns:

    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
