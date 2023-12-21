import json
import mmcv
import os
import os.path as osp
import numpy as np
from lvis import LVIS
from .api_wrappers import COCO
from numpy.core.fromnumeric import shape
from .builder import DATASETS
from .pipelines import Compose
from .lvis import LVISV1Dataset


@DATASETS.register_module()
class LVISRefine(LVISV1Dataset):

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ann_ids = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 512 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            ann_id = str(ann['id'])
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_ann_ids.append(ann_id)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            return None
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        chosen_idx = np.random.choice(len(gt_ann_ids))
        ann = dict(
            ann_id=gt_ann_ids[chosen_idx],
            bboxes=gt_bboxes[chosen_idx],
            labels=gt_labels[chosen_idx],
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann[chosen_idx],
            seg_map=seg_map)

        return ann
 

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys 
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info is None:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

