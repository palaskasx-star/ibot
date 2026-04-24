_base_ = [
    '../_base_/models/cascade_mask_rcnn_vit_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]

# 1. Model Overwrites
model = dict(
    backbone=dict(
        _delete_=True,
        type='TimmViTWithFPN', # Ensure your custom class is registered in MMDet 3.x registry
        model_name='vit_base_patch16_224',
        pretrained=True,
        with_fpn=True,
        out_indices=[3, 5, 7, 11],
        drop_path_rate=0.2),
    neck=dict(in_channels=[768, 768, 768, 768]),
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

# 2. Pipeline Update (Using the new AutoAugment syntax)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

# 3. Optimizer & Layer Decay (The 3.x way)
optim_wrapper = dict(
    type='AmpOptimWrapper', # Replaces EpochBasedRunnerAmp + use_fp16=True
    optimizer=dict(
        type='AdamW', 
        lr=0.0002, 
        betas=(0.9, 0.999), 
        weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65)
)

# 4. Learning Rate Scheduler (Replacing lr_config)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500), # Warmup
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True, milestones=[27, 33], gamma=0.1)
]

# 5. Runner Loops (Replacing workflow/runner)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
