import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
# from lxml import etree
import xml.etree.ElementTree as et


def read_cvat(filepath_xml, task_type=None, parent_type=None, object_type=None):
    """ Convert a xml file exported from CVAT to DataFrame.
    Args:
        filepath_xml (str): path to xml file
        task_type (str, optional): 'object_detection' or 'tracking' or 'segmentation'. Defaults to None.
        parent_type (str, optional): 'image' or 'track'. Defaults to None.
        object_type (str, optional): 'box' or 'polygon'. Defaults to None.
    Raises:
        Warning: specify task_type or parent_type and object_type in arguments
    """

    if task_type:
        if task_type == 'object_detection':
            parent_type, object_type = 'image', 'box'
        elif task_type == 'tracking':
            parent_type, object_type = 'track', 'box'
        elif task_type == 'segmentation':
            parent_type, object_type = 'image', 'polygon'
    else:
        if (parent_type is None) or (object_type is None):
            raise Warning('missing arguments; specify task_type or parent_type and object_type')

    # read xml file (annotation data)
    with open(filepath_xml, 'r') as f:
        soup = BeautifulSoup(f, 'lxml')

    # parse the xml
    parents = soup.find_all(parent_type)
    tmp_dfs = []
    for parent in parents:
        children = parent.find_all(object_type)
        tmp_df = pd.DataFrame([{**parent.attrs,
                                **child.attrs} for child in children])
        tmp_dfs.append(tmp_df)
    gt_df = pd.concat(tmp_dfs, axis=0)

    for col in gt_df.columns:
        try:
            gt_df[col] = gt_df[col].astype(float)
            gt_df[col] = gt_df[col].astype(int)
        #             print(col, 'is changed to int')
        except:
            pass
    #             print(col, 'is skipped.')
    return gt_df


def align_trackdf(gt_df):
    """make columns for tracking annotations
    Args:
        gt_df (pd.DataFrame): dataframe made by read_cvat function
    Returns:
        pd.DataFrame: dataframe with column names and added soem columns.
    """
    gt_df.sort_values(['frame', 'id'], inplace=True)
    gt_df.columns = ['frameid', 'trackid',
                     'keyframe', 'label', 'occluded', 'outside',
                     'xmax', 'xmin', 'ymax', 'ymin']
    gt_df['boxid'] = np.arange(len(gt_df))
    gt_df['conf'] = 1
    gt_df['height'] = gt_df.ymax - gt_df.ymin
    gt_df['width'] = gt_df.xmax - gt_df.xmin
    return gt_df


def create_cvat(dummy_header_xml, gt_df, savepath, parent_type=None, object_type=None):
    """ convert dataframe to xml file for uploading to CVAT
    Args:
        dummy_header_xml (str): dummy xml file path
        gt_df (pd.DataFrame): dataframe to convert
        savepath (str): path to save xml file
        parent_type (str, optional): 'image' or 'track'. Defaults to None.
        object_type (str, optional): 'box' or 'polygon'. Defaults to None.
    """
    # read xml file (annotation data)
    with open(dummy_header_xml, 'r') as f:
        soup = BeautifulSoup(f, 'lxml')

    # read format from the dummy cvat xml file
    parent = soup.find(parent_type)
    parent_keys = parent.attrs.keys()
    obj_keys = parent.findAll(object_type)[0].attrs.keys()
    del parent

    tree = et.parse(dummy_header_xml)
    root = tree.getroot()

    # remove all annotation data from xml
    # header like meta information remains
    for el in root.findall(parent_type):
        root.remove(el)

    # add track
    for parent_id in sorted(set(gt_df.id.values)):
        partial_df = gt_df[gt_df.id == parent_id].copy()

        parent = et.SubElement(root, parent_type)
        parent.attrib = partial_df.iloc[0][['id', 'label']].to_dict()

        # add box
        child = et.SubElement(parent, object_type)
    tree.write(savepath)

# gt_df = read_cvat(filepath_xml, task_type='tracking')
# create_cvat(gt_df, filepath_xml, '../misc/test.xml', parent_type='track', object_type='box')