from collections import namedtuple
import numpy as np
import pandas as pd

def make_x2y(x2lb, numbers, towards_var):
    '''x2yを作成する。orig_id -> colorに変換するためのnp.arrayを生成する。
    '''
    x2y = [None] * (max(numbers) + 1)
    for num in numbers:
        x2y[num] = getattr(x2lb[num], towards_var)
    x2y = np.array(x2y)
    return x2y

# ラベルの定義
#lb_df = pd.DataFrame([['id','trainid','name','name_jp','R','G','B'],
#                      ['0','0','background','その他','0','0','0'],
#                      ['1','1','plane','飛行機','255','0','0'],
#                      ['2','2','chair','椅子','0','255','0'],
#                      ['3','3','dog','犬','0','0','255'],
#                      ['4','4','bus','椅子','255','255','0'],
#                      ['5','5','car','車','0','255','255']
#                      ])

lb_df = pd.DataFrame([['id','trainid','name','name_jp','R','G','B'],
                      ['0','0','background','その他','0','0','0'],
                      ['1','1','glass','緑','0','255','0'],
                      ['2','2','water','水','0','0','255'],
                      ['3','3','outer','舗装','255','0','0']
                      ])

# dataframeの整理
lb_df.columns = lb_df.iloc[0]
lb_df = lb_df.iloc[1:].copy()

for col in lb_df.columns:
    try:
        lb_df[col] = lb_df[col].astype(int)
    except:
        pass

# RGB値の統合
colors = []
for _, row in lb_df.iterrows():
    color = row.R, row.G, row.B
    colors.append(color)
lb_df['color'] = colors

# namedtupleの定義
Label = namedtuple('Label', ['name', 'name_jp', 'id', 'tid', 'color'])
labels = []
for _, row in lb_df.iterrows():
    labels.append(Label(*row[['name', 'name_jp', 'id', 'trainid', 'color']].tolist()))

name2lb = {label.name:label for label in labels}
id2lb = {label.id:label for label in labels}
# tid2lb = {label.tid:label for label in labels}

ids = sorted(set(lb_df.id))
# tids = sorted(set(lb_df.trainid))

# 本プロジェクト->本プロジェクトの各種変換行列(x2y)の定義
id2color = make_x2y(id2lb, ids, 'color')
# tid2color = make_x2y(tid2lb, tids, 'color')