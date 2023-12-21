# -----------------------------------------------------------------------------------
# References:
# cascadepsp: https://github.com/hkchengrex/CascadePSP/blob/83cc3b8783b595b2e47c75016f93654eaddb7412/eval_post.py
# -----------------------------------------------------------------------------------
import argparse
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm 


def compute_boundary_acc(gt, seg, mask):
    gt = gt.astype(np.uint8)
    gt = gt.astype(np.uint8)
    seg = seg.astype(np.uint8)
    mask = mask.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    seg_acc = [None] * num_steps
    mask_acc = [None] * num_steps
    bnd_regions = []

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (curr_radius*2+1, curr_radius*2+1))
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        bnd_regions.append(boundary_region)

        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = ((gt_in_bound) * (seg_in_bound) + (1-gt_in_bound) * (1-seg_in_bound)).sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1-gt_in_bound) * (1-mask_in_bound)).sum()

        seg_acc[i] = num_seg_gd_pix / num_edge_pixels
        mask_acc[i] = num_mask_gd_pix / num_edge_pixels

    return sum(seg_acc)/num_steps, sum(mask_acc)/num_steps, bnd_regions

def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)
    
    return intersection, union 

def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluation a json file')
    parser.add_argument('--gt', help='direction of big gt segmentations', default='data/dis/DIS-VD/gt')
    parser.add_argument('--coarse', help='direction of big coarse segmentations', default='data/dis/coarse/isnet/DIS-VD')
    parser.add_argument('--refine', help='direction of big refined segmentations', default='dis_isnet/DIS-VD')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gt_root = args.gt
    coarse_root = args.coarse
    refine_root = args.refine

    total_new_i = 0
    total_new_u = 0
    total_old_i = 0
    total_old_u = 0

    total_old_correct_pixels = 0
    total_new_correct_pixels = 0
    total_num_pixels = 0

    total_num_images = 0
    total_seg_acc = 0
    total_mask_acc = 0

    small_objects = 0

    all_h = 0
    all_w = 0
    all_max = 0

    all_gts, all_olds, all_refines = [], [], []
    for file_name in os.listdir(gt_root):
        if '.png' in file_name:
            all_gts.append(file_name)

    with tqdm(total=len(all_gts)) as p:
        for gt_file in all_gts:
            gt = np.array(Image.open(os.path.join(gt_root, gt_file)
                                        ).convert('L'))

            seg = np.array(Image.open(os.path.join(coarse_root, gt_file)
                                        ).convert('L'))

            mask = np.array(Image.open(os.path.join(refine_root, gt_file)
                                        ).convert('L'))

            """
            Compute IoU and boundary accuracy
            """
            gt = gt >= 128
            seg = seg >= 128
            mask = mask >= 128

            old_i, old_u = get_iu(gt, seg)
            new_i, new_u = get_iu(gt, mask)

            total_new_i += new_i
            total_new_u += new_u
            total_old_i += old_i
            total_old_u += old_u

            seg_acc, mask_acc, bnd_regions = compute_boundary_acc(gt, seg, mask)
            total_seg_acc += seg_acc
            total_mask_acc += mask_acc
            total_num_images += 1

            # for idx, bnd_region in enumerate(bnd_regions):
            #     file = os.path.join(refine_root, gt_file.replace('_gt', f'_bnd{idx}'))
            #     bnd_region = (bnd_region*255).astype(np.uint8)
            #     Image.fromarray(bnd_region).save(file)

            p.update()
        

    new_iou = total_new_i/total_new_u
    old_iou = total_old_i/total_old_u
    new_mba = total_mask_acc/total_num_images
    old_mba = total_seg_acc/total_num_images

    print('New IoU  : ', new_iou)
    print('Old IoU  : ', old_iou)
    print('IoU Delta: ', new_iou-old_iou)

    print('New mBA  : ', new_mba)
    print('Old mBA  : ', old_mba)
    print('mBA Delta: ', new_mba-old_mba)
