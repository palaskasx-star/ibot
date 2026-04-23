_base_ = [
    '../_base_/models/cascade_mask_rcnn_vit_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]

# ViT-Small specific logic
model = dict(
    backbone=dict(
        _delete_=True,
        type='TimmViTWithFPN',
        model_name='vit_small_patch16_224',
        pretrained=True,
        with_fpn=True,
        out_indices=[3, 5, 7, 11],
        drop_path_rate=0.1), # Lower drop path for smaller models
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384], # ViT-Small dim is 384
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ])
)

# Pipeline Update
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment',
         policies=[
             [dict(type='Resize', scale=[(480, 1333), (800, 1333)], multiscale_mode='range', keep_ratio=True)],
             [
                 dict(type='Resize', scale=[(400, 1333), (600, 1333)], multiscale_mode='range', keep_ratio=True),
                 dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                 dict(type='Resize', scale=[(480, 1333), (800, 1333)], multiscale_mode='range', keep_ratio=True)
             ]
         ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# Optimizer Wrapper (AMP Enabled)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0001, 
        betas=(0.9, 0.999), 
        weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.75) # Higher decay rate for ViT-S vs ViT-B
)

# Schedulers
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True, milestones=[27, 33], gamma=0.1)
]

# Training Loops
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
