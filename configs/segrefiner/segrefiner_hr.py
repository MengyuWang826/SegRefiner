_base_ = [
    './segrefiner_lr.py'
]

object_size = 256
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True),
    dict(type='LoadPatchData', object_size=object_size, patch_size = object_size),
    dict(type='Resize', img_scale=(object_size, object_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_coarse_masks',
                               'patch_img', 'patch_gt_masks', 'patch_coarse_masks'])]


dataset_type = 'HRCollectionDataset'
data_root = 'data/'
train_dataloader=dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
data = dict(
    _delete_=True,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        collection_datasets=['thin', 'dis'],
        collection_json=data_root + 'collection_hr.json'),
    train_dataloader=train_dataloader,
    val=dict(), 
    test=dict())

max_iters = 40000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.5,
    by_epoch=False,
    step=[20000, 35000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)
