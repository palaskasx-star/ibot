import json
from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.dist import get_dist_info

def get_num_layer_for_vit(var_name, num_max_layer):
    # Adjusting strings to match the "self.vit" wrapper from our previous Backbone class
    if var_name in ("backbone.vit.cls_token", "backbone.vit.pos_embed", "backbone.vit.mask_token"):
        return 0
    elif "backbone.vit.patch_embed" in var_name:
        return 0
    elif "backbone.vit.blocks" in var_name:
        # Expected format: backbone.vit.blocks.0.norm1.weight
        layer_id = int(var_name.split('vit.blocks.')[1].split('.')[0])
        return layer_id + 1
    else:
        # For the Neck (FPN) and ROI Heads
        return num_max_layer - 1

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list."""
        parameter_groups = {}
        
        # In 3.x, paramwise_cfg is accessed via self.paramwise_cfg
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        
        # base_lr and base_wd are handled by the parent class
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue 
            
            # 1. Decide Weight Decay
            if len(param.shape) == 1 or name.endswith(".bias") or \
               'pos_embed' in name or 'cls_token' in name:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            # 2. Get Layer ID and calculate Scale
            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                # LLRD Formula: scale = rate^(max_layers - current_layer - 1)
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_mult": scale, # MMDet 3.x uses lr_mult instead of manual lr calculation
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)

        # Optional: Print groups for debugging (Rank 0 only)
        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Build LayerDecayOptimizerConstructor: {layer_decay_rate} rate, {num_layers} layers")
        
        params.extend(parameter_groups.values())
