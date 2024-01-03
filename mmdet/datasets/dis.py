# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class DISDataset(Dataset):

    CLASSES = ('1', '2')

    def __init__(self,
                 data_root,
                 img_root,
                 coarse_root,
                 pipeline,
                 testsets=[],
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        self.data_root = data_root
        self.img_root = img_root
        self.coarse_root = coarse_root
        self.test_mode = test_mode
        assert self.test_mode
        self.testsets = testsets
        self.file_client = mmcv.FileClient(**file_client_args)

        self.load_data()

        # processing pipeline
        self.pipeline = Compose(pipeline)
    
    def load_data(self):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        assert len(self.testsets) > 0
        self.data_infos = []
        for testset in self.testsets:
            img_dir = os.path.join(self.img_root, testset, 'im')
            coarse_dir = os.path.join(self.coarse_root, testset)
            all_files = os.listdir(os.path.join(self.data_root, coarse_dir))
            for filename in all_files:
                if '.png' in filename:
                    data_info = dict()
                    data_info['maskname'] = os.path.join(coarse_dir, filename)
                    data_info['filename'] = os.path.join(img_dir, filename.replace('.png', '.jpg'))
                    data_info['testset'] = testset
                    self.data_infos.append(data_info)
    
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
        coarse_info=dict(masks = img_info['maskname'])
        results = dict(img_info=img_info, coarse_info=coarse_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.data_root
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


