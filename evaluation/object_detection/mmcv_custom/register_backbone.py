import torch
import torch.nn as nn
import timm

# 1. Update Imports: MMDet 3.x uses the central MODELS registry
from mmdet.registry import MODELS
from mmengine.logging import print_log
from mmengine.model import BaseModule

# 2. Inherit from BaseModule for better weight initialization handling
@MODELS.register_module()
class TimmViTWithFPN(BaseModule):
    def __init__(self,
                 model_name='vit_base_patch16_224',
                 pretrained=True,
                 with_fpn=True,
                 frozen_stages=-1,
                 out_indices=[3, 5, 7, 11],
                 init_cfg=None, # Added for 3.x compatibility
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices

        # Load the timm model
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            dynamic_img_size=True, 
            **kwargs
        )
        
        self.patch_size = self.vit.patch_embed.patch_size[0]
        embed_dim = self.vit.embed_dim

        # FPN Rescaling Logic (Remains the same as your logic)
        if with_fpn:
            self._build_fpn_rescaler(embed_dim)

        self._freeze_stages()

    def _build_fpn_rescaler(self, embed_dim):
        if self.patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.patch_size == 8:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))

    def _freeze_stages(self):
        # Your existing freeze logic is correct for timm-style ViT
        if self.frozen_stages >= 0:
            self.vit.patch_embed.eval()
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            # ... (rest of your freezing logic)

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        
        # We use timm's internal methods to handle the embeddings
        x = self.vit.patch_embed(x)
        
        # --- FIX 2: Handle DINOv3 / EVA tuple returns (x, rope) ---
        pos_embed_out = self.vit._pos_embed(x)
        if isinstance(pos_embed_out, tuple):
            x, rope = pos_embed_out
        else:
            x = pos_embed_out
            rope = None
            
        features = []
        for i, blk in enumerate(self.vit.blocks):
            # --- FIX 3: Pass rope to block if it exists ---
            if rope is not None:
                x = blk(x, rope=rope)
            else:
                x = blk(x)
                
            if i in self.out_indices:
                num_prefix = self.vit.num_prefix_tokens
                xp = x[:, num_prefix:, :]
                
                if hasattr(self.vit, 'norm') and self.vit.norm is not None:
                    xp = self.vit.norm(xp)
                    
                xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp)       
                features.append(xp.contiguous())

        if self.with_fpn:
            features = [
                self.fpn1(features[0]), 
                self.fpn2(features[1]), 
                self.fpn3(features[2]), 
                self.fpn4(features[3])
            ]

        return tuple(features)
