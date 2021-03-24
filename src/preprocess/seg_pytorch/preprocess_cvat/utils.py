import os
from glob import glob
import pandas as pd

def split_extract(series, char, position=slice(None)):
    return series.str.split(char, expand=True).iloc[:, position]

def file_df_from_paths(paths):
    df = pd.DataFrame(paths)
    df.columns = ['filepath']
    df['filename'] = split_extract(df.iloc[:, 0], os.sep, -1)
    df['fileid'] = df.filename.apply(lambda x: os.path.splitext(x)[0])
    return df

def file_df_from_path(path, ext='xml'):
    paths = sorted(glob(os.path.join(path, '*.' + ext), recursive=True))
    df = file_df_from_paths(paths)
    return df