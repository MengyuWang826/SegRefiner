_base_ = [
    './segrefiner_lr.py'
]

object_size = 256
task='semantic'

model = dict(
    type='SegRefinerSemantic',
    task=task,
    test_cfg=dict(
        model_size=object_size,
        fine_prob_thr=0.9,
        iou_thr=0.3,
        batch_max=32))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', test_mode=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks'])
]

data = dict(
    train = dict(),
    val = dict(),
    test = dict(
        type='DISDataset',
        data_root = 'data',
        img_root = 'dis',
        coarse_root = 'dis/coarse/isnet',
        pipeline = test_pipeline,
        testsets = ['DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4', 'DIS-VD']),
    test_dataloader = dict(
        samples_per_gpu=1,
        workers_per_gpu=1))
