import torch
import torch.nn.functional as F
import numpy as np
from .segrefiner_base import SegRefiner
from ..builder import DETECTORS
import numpy as np

def mask2bbox(mask):
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    x_1, x_2 = x[0], x[-1] + 1
    y_1, y_2 = y[0], y[-1] + 1
    return x_1, y_1, x_2, y_2


@DETECTORS.register_module()
class SegRefinerInstance(SegRefiner):

    def simple_test_instance(self, 
                             img_metas, 
                             img, 
                             coarse_masks,
                             dt_bboxes,
                             **kwargs):
        """Test without augmentation."""
        
        coarse_masks, tiny_coarse_masks, dt_bboxes = self._filter_tiny_instance(coarse_masks, dt_bboxes)

        if len(coarse_masks) == 0:
            assert len(tiny_coarse_masks) > 0
            img_masks = tiny_coarse_masks
        else:
            current_device = img.device
            img_masks = coarse_masks
            object_imgs, object_masks, object_coors = self._get_object_input(img, coarse_masks, img_metas, current_device)
            batch_max = self.test_cfg.get('batch_max', 0)
            num_ins = len(object_masks)
            indices = list(range(self.num_timesteps))[::-1]
            indices = indices[:self.step]
            if num_ins <= batch_max:
                xs = [(object_masks, object_imgs, None)]
            else:
                xs = []
                for idx in range(0, num_ins, batch_max):
                    end = min(num_ins, idx + batch_max)
                    xs.append((object_masks[idx: end], object_imgs[idx:end], None))
            res, _ = self.p_sample_loop(xs, 
                                        indices, 
                                        current_device, 
                                        use_last_step=True)
            img_masks = _do_paste_mask(res, object_coors, img_metas)

            img_masks = img_masks >= 0.5
            img_masks = img_masks.cpu().numpy().astype(np.uint8)
            img_masks = np.concatenate((img_masks, tiny_coarse_masks), axis=0)
            
        assert len(img_masks) == len(dt_bboxes)
        bboxes = dt_bboxes[:, :5]
        labels = dt_bboxes[:, 5]
        labels = labels.astype(int)
        bbox_results = self._format_bboxes_results(bboxes, labels)
        mask_results = self._format_mask_results(img_masks, labels)
        return [(bbox_results, mask_results)]
    
    def _filter_tiny_instance(self, coarse_masks, dt_bboxes):
        area_thr = self.test_cfg.get('area_thr', 0)
        valid_idx = coarse_masks[0].areas >= area_thr
        invalid_idx = ~ valid_idx
        tiny_dt_bboxes = dt_bboxes[0][invalid_idx]
        dt_bboxes = dt_bboxes[0][valid_idx]
        dt_bboxes = np.concatenate((dt_bboxes, tiny_dt_bboxes), axis=0)
        return coarse_masks[0].masks[valid_idx], coarse_masks[0].masks[invalid_idx], dt_bboxes
    
    def _get_object_input(self, img, coarse_masks, img_metas, current_device):
        img_h, img_w = img_metas[0]['img_shape'][:2]
        object_imgs, object_masks, object_coors = [], [], []
        model_size = self.test_cfg.get('model_size', 256)
        pad_width = self.test_cfg.get('pad_width', 0)

        for mask in coarse_masks:
            x_1, y_1, x_2, y_2 = mask2bbox(mask)
            x_1_ob = max(0, x_1 - pad_width)
            x_2_ob = min(img_w, x_2 + pad_width)
            y_1_ob = max(0, y_1 - pad_width)
            y_2_ob = min(img_h, y_2 + pad_width)
            object_img = img[:, :, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
            object_mask = torch.tensor(mask[y_1_ob: y_2_ob, x_1_ob: x_2_ob], 
                                        device=current_device,
                                        dtype=torch.float32)
            object_imgs.append(F.interpolate(object_img, size=(model_size, model_size), mode='bilinear'))
            object_masks.append(F.interpolate(object_mask.unsqueeze(0).unsqueeze(0), size=(model_size, model_size), mode='bilinear'))
            object_coors.append(torch.tensor((x_1_ob, y_1_ob, x_2_ob, y_2_ob), device=current_device))
        object_imgs = torch.cat(object_imgs, dim=0)
        object_masks = torch.cat(object_masks, dim=0)
        object_coors = torch.stack(object_coors, dim=0)
        object_masks = (object_masks>= 0.5).float()
        return object_imgs, object_masks, object_coors
    
    def _format_bboxes_results(self,bboxes, labels):
        cls_bboxes = []
        for i in range(self.num_classes):
            cur_idx = (labels == i)
            cls_bboxes.append(bboxes[cur_idx])
        return cls_bboxes
    
    def _format_mask_results(self,masks, labels):
        cls_masks = [[] for _ in range(self.num_classes)]
        for i in range(len(masks)):
            cls_masks[labels[i]].append(masks[i])
        return cls_masks

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError


def _do_paste_mask(masks, object_coors, img_metas):
    device = masks.device
    img_h, img_w = img_metas[0]['ori_shape'][:2]
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(object_coors, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    return img_masks[:, 0]
   