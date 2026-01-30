# Modified from:
#   LightningDiT: https://github.com/hustvl/LightningDiT/blob/main/vavae/ldm/models/foundation_models.py

import timm
import torch
import torch.nn as nn

from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


def get_dinov2_base_encoder():
    """
    Load the DINOv2 pretrained ViT-B encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_base_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


def get_dinov3_encoder():
    """
    Load the DINOv3 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_dinov3.lvd1689m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


def get_clip_encoder():
    """
    Load the CLIP pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_clip_224.openai", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


def get_siglip2_encoder():
    """
    Load the SigLIP2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_siglip_256.v2_webli", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model


def create_foundation_model(type):
    # assert type in ['mae', 'dinov2', 'dinov2_b', 'dinov3', 'clip', 'siglip2'], f"Unsupported foundation model type: {type}"

    if type == 'mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024
    elif type == 'dinov2_b':
        return get_dinov2_base_encoder(), 768
    elif type == 'dinov3':
        return get_dinov3_encoder(), 1024
    elif type == 'clip':
        return get_clip_encoder(), 1024
    elif type == 'siglip2':
        return get_siglip2_encoder(), 1024
    else:
        raise NotImplementedError


class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim

        # --------------------------------------------------------------------------
        # ImageNet specifics
        self.register_buffer('imnet_mean', torch.tensor(IMAGENET_DEFAULT_MEAN), persistent=False)
        self.register_buffer('imnet_std', torch.tensor(IMAGENET_DEFAULT_STD), persistent=False)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # CLIP input specifics
        self.register_buffer('clip_mean', torch.tensor(CLIP_DEFAULT_MEAN), persistent=False)
        self.register_buffer('clip_std', torch.tensor(CLIP_DEFAULT_STD), persistent=False)
        # --------------------------------------------------------------------------

    def norm_imnet_img(self, x):
        return (x - self.imnet_mean.view(1, -1, 1, 1)) / self.imnet_std.view(1, -1, 1, 1)

    def unnorm_imnet_img(self, x):
        return x * self.imnet_std.view(1, -1, 1, 1) + self.imnet_mean.view(1, -1, 1, 1)

    def norm_clip_img(self, x):
        return (x - self.clip_mean.view(1, -1, 1, 1)) / self.clip_std.view(1, -1, 1, 1)

    def unnorm_clip_img(self, x):
        return x * self.clip_std.view(1, -1, 1, 1) + self.clip_mean.view(1, -1, 1, 1)

    def forward_mae(self, x):
        b, c, h, w = x.shape
        x = self.norm_imnet_img(x)
        x = self.model.forward_features(x)
        return (x[:, :1].transpose(2, 1).reshape(b, -1, 1, 1), x[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2))

    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = self.norm_imnet_img(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = self.model.forward_features(x)
        return (x[:, :1].transpose(2, 1).reshape(b, -1, 1, 1), x[:, -(h//16 * w//16):].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2))

    def forward_dinov3(self, x):
        b, c, h, w = x.shape
        x = self.norm_imnet_img(x)
        x = self.model.forward_features(x)
        return (x[:, :1].transpose(2, 1).reshape(b, -1, 1, 1), x[:, -(h//16 * w//16):].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2))

    def forward_clip(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = self.norm_clip_img(x)
        x = self.model.forward_features(x)
        return (x[:, :1].transpose(2, 1).reshape(b, -1, 1, 1), x[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2))

    def forward_siglip2(self, x):
        b, c, h, w = x.shape
        x = (x - 0.5) * 2
        x = self.model.forward_features(x).reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
        return (x.mean(dim=(2, 3), keepdim=True), x)

    def forward(self, x):
        with torch.no_grad():
            if 'mae' in self.type:
                return self.forward_mae(x)
            elif 'dinov2' in self.type:
                return self.forward_dinov2(x)
            elif 'dinov3' in self.type:
                return self.forward_dinov3(x)
            elif 'clip' in self.type:
                return self.forward_clip(x)
            elif 'siglip2' in self.type:
                return self.forward_siglip2(x)
            else:
                raise NotImplementedError
