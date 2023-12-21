# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import os
import tempfile
import warnings
from collections import OrderedDict

from torch.utils.data import Dataset

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose


@DATASETS.register_module()
class PascalDataset(Dataset):

    def __init__(self,
                 data_root,
                 pipeline,
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        self.data_root = data_root
        self.test_mode = test_mode
        assert self.test_mode
        self.file_client = mmcv.FileClient(**file_client_args)

        self.data_infos = self.load_datas(self.data_root)

        # processing pipeline
        self.pipeline = Compose(pipeline)
    
    def load_datas(self, data_root):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        all_files = os.listdir(data_root)
        data_infos = []
        for filename in all_files:
            if '_im' in filename:
                data_info = dict()
                data_info['filename'] = filename
                data_info['gtname'] = filename.replace('_im', '_gt')
                data_info['maskname'] = filename.replace('_im', '_seg')
                data_infos.append(data_info)
        return data_infos

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        # img_info = self.data_infos[idx]

        # img_info = dict(
        #     filename = 'dis/DIS-TR/im/4#Architecture#5#GasStation#3650661576_fbd6a86403_o.jpg',
        #     maskname = 'dis/DIS-TR/coarse/4#Architecture#5#GasStation#3650661576_fbd6a86403_o.png')
        img_info = self.data_infos[idx]
        coarse_info=dict(masks = img_info['maskname'])
        results = dict(img_info=img_info, coarse_info=coarse_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.data_root
        # results['img_prefix'] = 'data'
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
    

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)


