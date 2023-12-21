_base_ = [
    './segrefiner_lr.py'
]

object_size = 256
task='instance'

model = dict(
    type='SegRefinerInstance',
    test_cfg=dict(
        pad_width=20,
        model_size=object_size,
        batch_max=32,
        area_thr=512)) 

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', test_mode=True, with_bbox=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks', 'dt_bboxes'])
]


dataset_type = 'CocoRefine'
img_root = '/share/project/datasets/MSCOCO/coco2017/'
ann_root = '/share/project/datasets/MSCOCO/coco2017/annotations/'

data = dict(
    train=dict(),
    val=dict(),
    test=dict(
        pipeline = test_pipeline,
        type = dataset_type,
        ann_file = ann_root + 'instances_val2017.json',
        coarse_file = 'all_json/coarse_coco/maskrcnn_3x_r50.json',
        img_prefix = img_root + 'val2017'),
    test_dataloader = dict(
        samples_per_gpu=1,
        workers_per_gpu=1))
