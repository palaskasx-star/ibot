dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# Pipeline changes: 
# 1. 'Normalize' and 'Pad' move to the model's data_preprocessor (defined in the model file)
# 2. 'DefaultFormatBundle' and 'Collect' become 'PackDetInputs'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True), # 'img_scale' renamed to 'scale'
    dict(type='RandomFlip', prob=0.5),                       # 'flip_ratio' renamed to 'prob'
    dict(type='PackDetInputs')                               # Combines bundle and collect
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have ground truth, LoadAnnotations is skipped
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# MMDet 3.x uses separate dataloader dicts
train_dataloader = dict(
    batch_size=8,        # replaced samples_per_gpu
    num_workers=25,       # replaced workers_per_gpu
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

# Evaluation is now handled by an Evaluator object
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False)

test_evaluator = val_evaluator
