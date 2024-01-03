# -----------------------------------------------------------------------------------
# References:
# cascadepsp: https://github.com/hkchengrex/CascadePSP/blob/83cc3b8783b595b2e47c75016f93654eaddb7412/util/boundary_modification.py
# -----------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import random
import math
import json
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    data_root = 'data/'
    thin_object_txt = 'data/thin_object/list/train.txt'
    thin_object_root = 'data/thin_object/images'
    dis_root = 'data/dis/DIS-TR/im'
    collection_json_file = 'data/collection_hr.json'
    collection = dict(dis=[], thin=[])

    print('----------start collecting dis---------------')
    dis_all_files = os.listdir(dis_root)
    with tqdm(total=len(dis_all_files)) as p:
        for filename in dis_all_files:
            mask_file = filename.replace('jpg', 'png')
            mask_name = 'dis/DIS-TR/gt/' + mask_file
            img_name = 'dis/DIS-TR/im/' + filename
            coarsename = 'dis/DIS-TR/coarse/' + mask_file
            expandname = 'dis/DIS-TR/coarse_expand/' + mask_file
            img = cv2.imread(os.path.join(dis_root, filename))
            assert img is not None
            h, w = img.shape[:2]
            img_info = {'filename': img_name,
                        'maskname': mask_name,
                        'coarsename': coarsename,
                        'expandname': expandname,
                        'height': h, 'width': w}
            collection['dis'].append(img_info)
            p.update()
    
    print('----------start collecting thin---------------')
    f=open(thin_object_txt)
    thin_all_files=[]
    for line in f:
        thin_all_files.append(line.strip())
    with tqdm(total=len(dis_all_files)) as p:
        for filename in thin_all_files:
            mask_file = filename
            mask_name = 'thin_object/masks/' + mask_file
            filename = filename.replace('png', 'jpg')
            img_name = 'thin_object/images/' + filename
            coarsename = 'thin_object/coarse/' + mask_file
            expandname = 'thin_object/coarse_expand/' + mask_file
            img = cv2.imread(os.path.join(thin_object_root, filename))
            assert img is not None
            h, w = img.shape[:2]
            img_info = {'filename': img_name,
                        'maskname': mask_name,
                        'coarsename': coarsename,
                        'expandname': expandname,
                        'height': h, 'width': w}
            collection['thin'].append(img_info)
            p.update()
    
    print('----------writing json file---------------')
    with open(collection_json_file, 'w') as f:
        json.dump(collection, f)