# 1. Inherit everything from your heavy 36-epoch config
_base_ = ['./vit_small_giou_4conv1f_coco_3x.py']

# 2. Override only the specific parts for Model A
model = dict(
    backbone=dict(model_name='vit_small_patch16_dinov3')
)

train_dataloader = dict(batch_size=8)
optim_wrapper = dict(optimizer=dict(lr=0.0002))
train_cfg = dict(max_epochs=12)

# 3. Safely override the scheduler without CLI parser crashes
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]
