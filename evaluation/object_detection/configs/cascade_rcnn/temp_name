_base_ = [
    '../_base_/models/cascade_mask_rcnn_vit_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]

# CRITICAL: Tell MMDet to load your custom timm model file before building.
# Change 'my_custom_modules.tiny_vit' to the actual python path where 
# you saved the @register_model code.
custom_imports = dict(
    imports=['my_custom_modules.tiny_vit'], 
    allow_failed_imports=False
)

# 1. ViT-Tiny Specific Model Overwrites
model = dict(
    backbone=dict(
        _delete_=True,
        type='TimmViTWithFPN',
        model_name='vit_tiny_patch16_dinov3', # Your registered custom model
        pretrained=True, # Will try to load weights if available, or initialize from scratch
        with_fpn=True,
        out_indices=[3, 5, 7, 11],
        drop_path_rate=0.05), # Lowered for the Tiny model to prevent underfitting
    
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192], # Adjusted to match ViT-Tiny's embed_dim=192
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

# 2. Pipeline Update
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

# 3. Optimizer Wrapper (AMP Enabled & Adjusted Layer Decay)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0002, 
        betas=(0.9, 0.999), 
        weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.85) # Increased to 0.85 since Tiny models need less aggressive layer decay
)

# 4. Schedulers
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True, milestones=[27, 33], gamma=0.1)
]

# 5. Training Loops
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
