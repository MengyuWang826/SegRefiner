# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import cv2
import mmcv
import numpy as np
import random
import math
import torch
import torch.nn.functional as F
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

from PIL import Image


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)        

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative value to
            absolute value. Only used in OpenImage Dataset.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """
        h, w = results['img_info']['height'], results['img_info']['width']
        if 'maskname' in results['ann_info']:
            filename = osp.join(results['img_prefix'], results['ann_info']['maskname'])
            mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            mask = mask.astype(np.float32) / 255
            mask = (mask >= 0.5).astype(np.uint8)
            gt_masks = BitmapMasks([mask], mask.shape[-2], mask.shape[-1])
        else:
            gt_masks = results['ann_info']['masks']
            gt_masks = BitmapMasks([self._poly2mask(gt_masks, h, w)], h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 file_client_args=dict(backend='disk')):
        if rgb2id is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(LoadPanopticAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=True,
            denorm_bbox=False,
            file_client_args=file_client_args)

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            results = self._load_masks_and_semantic_segs(results)

        return results


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground truth
            boxes. Default: (1., 1.)
        min_gt_mask_area (int): Minimum foreground area of ground truth masks.
            Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: True
        by_mask (bool): Filter instances with masks not meeting
            min_gt_mask_area threshold. Default: False
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Default: True
    """

    def __init__(self,
                 min_gt_bbox_wh=(1., 1.),
                 min_gt_mask_area=1,
                 by_box=True,
                 by_mask=False,
                 keep_empty=True):
        # TODO: add more filter options
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    def __call__(self, results):
        if self.by_box:
            assert 'gt_bboxes' in results
            gt_bboxes = results['gt_bboxes']
            instance_num = gt_bboxes.shape[0]
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            instance_num = len(gt_masks)

        if instance_num == 0:
            return results

        tests = []
        if self.by_box:
            w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            tests.append((w > self.min_gt_bbox_wh[0])
                         & (h > self.min_gt_bbox_wh[1]))
        if self.by_mask:
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        keep = keep.nonzero()[0]

        keys = ('gt_bboxes', 'gt_labels', 'gt_masks')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]
        if keep.size == 0:
            if self.keep_empty:
                return None
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(min_gt_bbox_wh={self.min_gt_bbox_wh},' \
            f'min_gt_mask_area={self.min_gt_mask_area},' \
            f'by_box={self.by_box},' \
            f'by_mask={self.by_mask},' \
            f'always_keep={self.always_keep})'

@PIPELINES.register_module()
class LoadCoarseMasks:
    def __init__(self,
                 with_bbox=False,
                 with_lable=False,
                 test_mode=False):
        self.test_mode = test_mode
        self.with_bbox = with_bbox
        self.with_lable = with_lable
    
    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def __call__(self, results):
        h, w = results['img_shape'][:2]
        if not self.test_mode:
            gt_mask = (results['gt_masks'].masks[0] * 255).astype(np.uint8)
            coarse_masks = modify_boundary(gt_mask)
            results['coarse_masks'] = BitmapMasks([coarse_masks], coarse_masks.shape[-2], coarse_masks.shape[-1])
        else:
            coarse_masks = results['coarse_info']['masks']
            if isinstance(coarse_masks, str):
                filename = osp.join(results['img_prefix'], coarse_masks)
                mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                assert mask is not None
                mask = mask.astype(np.float32) / 255
                mask = (mask >= 0.5).astype(np.uint8)
                results['coarse_masks'] = BitmapMasks([mask], mask.shape[-2], mask.shape[-1])
            else:
                new_coarse_masks = []
                for mask in coarse_masks:
                    new_coarse_masks.append(self._poly2mask(mask, h, w))
                results['coarse_masks'] = BitmapMasks(new_coarse_masks, h, w)
        results['mask_fields'].append('coarse_masks')
        if self.with_bbox:
            bboxes = results['coarse_info']['bboxes']
            results['dt_bboxes'] = bboxes
            assert len(results['dt_bboxes']) == len(results['coarse_masks'])
        if self.with_lable:
            lables = results['coarse_info']['lables']
            results['dt_lables'] = lables
            assert len(results['dt_lables']) == len(results['coarse_masks'])
        return results

@PIPELINES.register_module()
class LoadObjectData:
    def __init__(self,
                 pad_size=40):
        self.pad_size = pad_size

    def _mask2bbox(self, mask):
        x_any = mask.any(axis=0)
        y_any = mask.any(axis=1)
        x = np.where(x_any)[0]
        y = np.where(y_any)[0]
        x_1, x_2 = x[0], x[-1] + 1
        y_1, y_2 = y[0], y[-1] + 1
        return x_1, y_1, x_2, y_2
    
    def _get_object_crop_coor(self, x_1, x_2, w, object_size):
        x_start = int(max(object_size/2, x_2 - object_size/2))
        x_end = int(min(x_1 + object_size/2, w - object_size/2))
        x_c = np.random.randint(x_start, x_end + 1)
        x_1_ob = max(int(x_c - object_size/2), 0)
        x_2_ob = min(int(x_c + object_size/2), w)
        return x_1_ob, x_2_ob
    
    def ramdom_crop_object(self, results):
        h, w = results['img_shape'][:2]
        x_1, y_1, x_2, y_2 = self._mask2bbox(results['gt_masks'].masks[0])
        object_h = min(y_2 - y_1 + self.pad_size,  h)
        object_w = min(x_2 - x_1 + self.pad_size, w)
        x_1_ob, x_2_ob = self._get_object_crop_coor(x_1, x_2, w, object_w)
        y_1_ob, y_2_ob = self._get_object_crop_coor(y_1, y_2, h, object_h)
        results['object_img'] = results['img'][y_1_ob: y_2_ob, x_1_ob: x_2_ob, :]
        object_gt_mask = results['gt_masks'].masks[:, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
        results['object_gt_masks'] = BitmapMasks(object_gt_mask, object_gt_mask.shape[-2], object_gt_mask.shape[-1])
        object_coarse_mask = results['coarse_masks'].masks[:, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
        results['object_coarse_masks'] = BitmapMasks(object_coarse_mask, object_coarse_mask.shape[-2], object_coarse_mask.shape[-1])
        return results

    def __call__(self, results):
        results = self.ramdom_crop_object(results)
        del results['ann_info']
        del results['mask_fields']
        results['mask_fields'] = ['object_gt_masks', 'object_coarse_masks']
        del results['img']
        results['img_shape'] = results['object_img'].shape
        results['ori_shape'] = results['object_img'].shape
        del results['img_fields']
        results['img_fields'] = ['object_img']
        del results['gt_masks']
        del results['coarse_masks']
        return results

@PIPELINES.register_module()
class LoadPatchData:
    def __init__(self,
                 object_size,
                 patch_size,
                 use_expand_coarsemask=True,
                 expand_prob=0.5,
                 pad_fraction=1/8):
        self.object_size = object_size
        self.patch_size = patch_size
        self.pad_fraction = pad_fraction
        self.use_expand_coarsemask = use_expand_coarsemask
        self.expand_prob = expand_prob

    def _mask2bbox(self, mask):
        x_any = mask.any(axis=0)
        y_any = mask.any(axis=1)
        x = np.where(x_any)[0]
        y = np.where(y_any)[0]
        x_1, x_2 = x[0], x[-1] + 1
        y_1, y_2 = y[0], y[-1] + 1
        return x_1, y_1, x_2, y_2
    
    def _get_patch_crop_coor(self, patch_size, h, w):
        margin_h = max(h - patch_size, 0)
        margin_w = max(w - patch_size, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        x_1_pt, x_2_pt = offset_w, offset_w + patch_size
        y_1_pt, y_2_pt = offset_h, offset_h + patch_size
        return x_1_pt, y_1_pt, x_2_pt, y_2_pt
    
    def _get_object_coor(self, x_1, x_2, x_max):
        if x_1 == 0:
            x_1_ob = 0
        else:
            x_1_ob = np.random.randint(0, x_1)
        if x_2 == x_max:
            x_2_ob = x_max
        else:
            x_2_ob = np.random.randint(x_2, x_max)
        return x_1_ob, x_2_ob
    
    def ramdom_crop_object(self, results):
        h, w = results['img_shape'][:2]
        x_1, y_1, x_2, y_2 = self._mask2bbox(results['gt_masks'].masks[0])
        x_1_ob, x_2_ob = self._get_object_coor(x_1, x_2, w)
        y_1_ob, y_2_ob = self._get_object_coor(y_1, y_2,  h)
        results['object_img'] = results['img'][y_1_ob: y_2_ob, x_1_ob: x_2_ob, :]
        object_gt_mask = results['gt_masks'].masks[:, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
        results['object_gt_masks'] = BitmapMasks(object_gt_mask, object_gt_mask.shape[-2], object_gt_mask.shape[-1])
        object_coarse_mask = results['coarse_masks'].masks[:, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
        results['object_coarse_masks'] = BitmapMasks(object_coarse_mask, object_coarse_mask.shape[-2], object_coarse_mask.shape[-1])
        return results
    
    def ramdom_crop_patch(self, results):
        h, w = results['ori_shape'][:2]
        if h < self.patch_size or w < self.patch_size:
            patch_size = int(min(h, w) / 2)
        else:
            patch_size = self.patch_size
        while True:
            x_1_pt, y_1_pt, x_2_pt, y_2_pt = self._get_patch_crop_coor(patch_size, h, w)
            patch_gt_mask = results['gt_masks'].masks[:, y_1_pt: y_2_pt, x_1_pt: x_2_pt]
            if patch_gt_mask.any() and (not patch_gt_mask.all()):
                if np.random.rand() <= self.expand_prob:
                    patch_coarse_mask = results['coarse_masks'].masks[0, y_1_pt: y_2_pt, x_1_pt: x_2_pt]
                else:
                    patch_coarse_mask = modify_boundary(patch_gt_mask[0] * 255)
                results['patch_gt_masks'] = BitmapMasks(patch_gt_mask, patch_gt_mask.shape[-2], patch_gt_mask.shape[-1])
                results['patch_img'] = results['img'][y_1_pt: y_2_pt, x_1_pt: x_2_pt, :]
                results['patch_coarse_masks'] = BitmapMasks([patch_coarse_mask], patch_coarse_mask.shape[-2], patch_coarse_mask.shape[-1])
                return results
    
    def load_coarse_mask(self, results):
        filename = osp.join(results['img_prefix'], results['img_info']['coarsename'])
        if self.use_expand_coarsemask:
            flag = np.random.rand() <= self.expand_prob
            if flag:
                filename = osp.join(results['img_prefix'], results['img_info']['expandname'])
        coarse_mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        assert coarse_mask is not None
        coarse_mask = coarse_mask.astype(np.float32) / 255
        coarse_mask = (coarse_mask >= 0.5).astype(np.uint8)
        results['coarse_masks'] = BitmapMasks([coarse_mask], coarse_mask.shape[-2], coarse_mask.shape[-1])
        return results

    def __call__(self, results):
        results = self.load_coarse_mask(results)
        results = self.ramdom_crop_object(results)
        results = self.ramdom_crop_patch(results)
        del results['ann_info']
        del results['mask_fields']
        results['mask_fields'] = ['object_gt_masks', 'object_coarse_masks', 'patch_gt_masks', 'patch_coarse_masks']
        del results['img']
        results['img_shape'] = results['object_img'].shape
        results['ori_shape'] = results['object_img'].shape
        results['patch_shape'] = results['patch_img'].shape
        del results['img_fields']
        results['img_fields'] = ['object_img', 'patch_img']
        del results['gt_masks']
        del results['coarse_masks']
        return results


# -----------------------------------------------------------------------------------
# References:
# cascadepsp: https://github.com/hkchengrex/CascadePSP/blob/83cc3b8783b595b2e47c75016f93654eaddb7412/util/boundary_modification.py
# -----------------------------------------------------------------------------------
def get_random_structure(size):
    # The provided model is trained with 
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target = 0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate. 
    # input: np array of size [H,W] image
    # output: same shape as input
    
    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #only modified contours is needed actually. 
    sampled_contours = []   
    modified_contours = [] 

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        #remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)
        
        idx_dist = []
        for i in range(number_of_vertices-number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
            
        idx_dist = sorted(idx_dist, key=lambda x:x[1])
        
        remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]
        
       #remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
        contour = new_contour
        

        #sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            #modify contours
            for idx, coor in enumerate(modified_contour):

                change = np.random.normal(0,move_rate) # 0.1 means change position of vertex to 10 percent farther from center
                x,y = coor[0]
                new_x = x + (x-center[0]) * change
                new_y = y + (y-center[1]) * change

                modified_contour[idx] = [new_x,new_y]
        modified_contours.append(modified_contour)
        
    #draw boundary
    gt = np.copy(image)
    image = np.zeros((image.shape[0], image.shape[1], 3))

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    if len(image.shape) == 3:
        image = image[:, :, 0]
    image = perturb_seg(image, iou_target)

    image = image / 255
    image = (image >= 0.5).astype(np.uint8)
    
    return image