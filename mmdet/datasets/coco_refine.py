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
from .coco import CocoDataset


@DATASETS.register_module()
class CocoRefine(CocoDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def __init__(self, coarse_file, **kwargs):
        self.coarse_infos = self.load_coarse(coarse_file)
        super(CocoRefine, self).__init__(**kwargs)
        print(f'testing total {len(self)} imgs....')
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = list(self.coarse_infos) 
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        # data_infos = data_infos[:10]
        return data_infos
    
    def load_coarse(self, coarse_file):
        coarse_dts = json.load(open(coarse_file))
        coarse_infos = {}
        for dt in coarse_dts:
            img_id = dt['image_id']
            if img_id not in coarse_infos:
                coarse_infos.update({img_id: []})
            coarse_infos[img_id].append(dt)
        return coarse_infos

    def get_coarse_info(self, img_info):
        img_id = img_info['id']
        coarse_masks = []
        bboxes = []
        lables = []
        coarse_dts = self.coarse_infos[img_id] 
        for dt in coarse_dts:
            label = self.cat2label[dt['category_id']]
            coarse_masks.append(dt['segmentation'])
            bbox = dt['bbox']
            bbox.append(dt['score'])
            bbox.append(label)
            bboxes.append(bbox)
            lables.append(label)
        bboxes = np.array(bboxes)
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        return dict(masks=coarse_masks, bboxes=bboxes, lables=lables)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        coarse_info = self.get_coarse_info(img_info)
        results = dict(img_info=img_info, coarse_info=coarse_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

