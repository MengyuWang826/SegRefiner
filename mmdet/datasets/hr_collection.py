# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import Dataset
import json
import mmcv
import numpy as np
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class HRCollectionDataset(Dataset):

    CLASSES = ('1', '2')

    def __init__(self,
                 data_root,
                 pipeline,
                 collection_datasets,
                 collection_json,
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        self.data_root = data_root
        self.test_mode = test_mode
        self.file_client = mmcv.FileClient(**file_client_args)

        print('loading collection datasets')
        self.data_infos = self.load_collection(collection_datasets, collection_json)
        print('loading collection datasets done')

        # processing pipeline
        self.pipeline = Compose(pipeline)

        if not test_mode:
            self._set_group_flag()

    def load_collection(self, collection_datasets, collection_json):
        collect = json.load(open(collection_json))
        data_infos = []
        for dataset_name in collection_datasets:
            data_infos.extend(collect[dataset_name])
        return data_infos
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

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
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info, ann_info=img_info)
        self.pre_pipeline(results)
        data = self.pipeline(results)
        return data
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.data_root
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []