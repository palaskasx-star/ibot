import torch
import torch.nn as nn
import timm

from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES

@BACKBONES.register_module()
class TimmViTWithFPN(nn.Module):
    def __init__(self,
                 model_name='vit_base_patch16_224', # Any timm model name
                 pretrained=True,
                 with_fpn=True,
                 frozen_stages=-1,
                 out_indices=[3, 5, 7, 11],
                 **kwargs):
        super().__init__()
        
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices

        # 1. Load the timm model natively
        # num_classes=0 strips the classification head.
        # dynamic_img_size=True allows timm to natively handle MMDetection's varying image sizes!
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            dynamic_img_size=True, 
            **kwargs
        )
        
        # 2. Dynamically grab the patch size and embed dim from the timm model
        self.patch_size = self.vit.patch_embed.patch_size[0]
        embed_dim = self.vit.embed_dim

        # 3. Keep the exact FPN logic from the iBOT paper
        if with_fpn and self.patch_size == 16:
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
            
        elif with_fpn and self.patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')

        self._freeze_stages()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.vit.patch_embed.eval()
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            if hasattr(self.vit, 'cls_token'):
                self.vit.cls_token.requires_grad = False
            if hasattr(self.vit, 'pos_embed'):
                self.vit.pos_embed.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == len(self.vit.blocks):
                norm_layer = getattr(self.vit, 'norm', None)
                if norm_layer:
                    norm_layer.eval()
                    for param in norm_layer.parameters():
                        param.requires_grad = False

            m = self.vit.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        # We don't need complex loading logic anymore. 
        # timm.create_model(pretrained=True) already downloaded and loaded the weights!
        logger = get_root_logger()
        if pretrained is None:
            logger.info(f"Using timm's native pretrained weights.")
        else:
            logger.info(f"Note: Custom pretrained path {pretrained} ignored. Using timm native weights.")

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        
        # Native timm forward features up to the blocks
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        
        features = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Safely drop prefix tokens (CLS token, Distillation token, etc.)
                # timm natively knows how many extra tokens the model uses.
                num_prefix = self.vit.num_prefix_tokens
                xp = x[:, num_prefix:, :]
                
                # Apply normalization if the specific timm model has it
                if hasattr(self.vit, 'norm') and self.vit.norm is not None:
                    xp = self.vit.norm(xp)
                    
                # Reshape flat sequence back to 2D image map
                xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp)       
                features.append(xp.contiguous())

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        return tuple(features)
