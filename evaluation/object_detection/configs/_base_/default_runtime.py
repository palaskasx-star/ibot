# 1. Default Hooks: Consolidates checkpoint, logger, and other runtime hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'), # New: for drawing boxes during eval
    # Replaces custom_hooks = [dict(type='NumClassCheckHook')]
    num_class_check=dict(type='NumClassCheckHook')
)

# 2. Environment Settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 3. Visualizer: New in 3.x for logging to Tensorboard/WandB
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

# 4. Logging and Loading
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# In 3.x, load_from is still used, but 'resume_from' is replaced 
# by 'resume=True' in the runner or command line.
load_from = None
resume = False # Replaces resume_from logic

# workflow = [('train', 1)] is deprecated. 
# Training loops are now defined in the main config (train_cfg).
default_scope = 'mmdet'
