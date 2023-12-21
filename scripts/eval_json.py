import json
import argparse
import os
from lvis import LVIS, LVISEval
from tqdm import tqdm


def val_lvis(result_path, ann_path, val_type):
    print('############# start evaluation #############')
    lvis_eval = LVISEval(ann_path, result_path, val_type)
    lvis_eval.run()
    lvis_eval.print_results()

def format_dt_json(gt_json, dt_coco_json, dt_lvis_json):
    print('############# start transforming #############')
    gt = LVIS(gt_json)
    dts = json.load(open(dt_coco_json))
    new_dts = []

    with tqdm(total=len(dts)) as p:
        for dt in dts:
            if dt['image_id'] in gt.imgs and dt['category_id'] in gt.cats:
                new_dts.append(dt)
            p.update() 

    with open(dt_lvis_json, 'w') as f:
        json.dump(new_dts, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluation a json file')
    parser.add_argument('--gt', help='json file of lvis annotation of coco val set', default='data/lvis/lvis_v1_val_cocofied.json')
    parser.add_argument('--dt', help='json file of dt results in coco format', default='maskrcnn.segm.json')
    parser.add_argument('--dt_out', help='output json file of dt results in lvis format', default='new_json/refine/maskrcnn.segm.json')
    parser.add_argument('--iou_type', help='evaluation mask or boundary metrics', default='segm')
    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = parse_args()
    assert args.iou_type in {'segm', 'boundary'}

    # transform to lvis format
    if not os.path.exists(args.dt_out):
        format_dt_json(args.gt, args.dt, args.dt_out)
    
    # evaluation
    val_lvis(args.dt_out, args.gt, args.iou_type)
