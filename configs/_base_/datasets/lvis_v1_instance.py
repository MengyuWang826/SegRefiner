# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV1Dataset'
data_root = 'data/coco/'
json_root = 'data/refine_annotations/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=json_root + 'lvis_v1_train_easy.json',
            img_prefix=data_root)),
    val=dict(
        type=dataset_type,
        ann_file=json_root + 'lvis_val_easy.json',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        ann_file=json_root + 'lvis_val_easy.json',
        img_prefix=data_root))
evaluation = dict(metric=['bbox', 'segm'])
