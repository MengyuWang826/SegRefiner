_base_ = [
    '../_base_/default_runtime.py'
]

object_size = 256
task='instance'

model = dict(
    type='SegRefiner',
    task=task,
    step=6,
    denoise_model=dict(
        type='DenoiseUNet',
        in_channels=4,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        learn_time_embd=True,
        channel_mult = (1, 1, 2, 2, 4, 4),
        dropout=0.0),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,
            stop=0,
            num_timesteps=6),
        diff_iter=False),
    # model training and testing settings
    test_cfg=dict()) 

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True),
    dict(type='LoadCoarseMasks'),
    dict(type='LoadObjectData'),
    dict(type='Resize', img_scale=(object_size, object_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_coarse_masks'])]


dataset_type = 'LVISRefine'
img_root = '/share/project/datasets/MSCOCO/coco2017/'
ann_root = '/share/project/datasets/LVIS/'
train_dataloader=dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
data = dict(
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'lvis_v1_train.json',
        img_prefix=img_root),
    train_dataloader=train_dataloader,
    val=dict(),
    test=dict())

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    weight_decay=0,
    eps=1e-8,
    betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 120000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.5,
    by_epoch=False,
    step=[80000, 100000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)



log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ])
interval = 5000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=20)
