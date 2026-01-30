"""Building blocks for ResTok.

Copyright (2026) Kuaishou Technology and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Modified from:
    TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/blocks.py
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import partial
import os
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, use_fused_attn
from timm.layers.mlp import Mlp
from timm.layers.drop import DropPath
from modeling.modules.rope import EmbedNDHybrid2DMaker, EmbedNDHybridMulti2DMaker, apply_rope, apply_rope_single


class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        qkv_bias_separate: bool = False,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        qk_norm: bool = False,
        scale_norm: bool = True,
        rotate_half: bool = False,
    ) -> None:
        super().__init__()
        if scale_norm or qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(attn_dim))
                self.register_buffer('k_bias', torch.zeros(attn_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(attn_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, attn_dim, bias=False)
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(attn_dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor = None,
        attn_mask: Tensor = None,
        return_attn_weights: bool = False,
        average_attn_weights: bool = True,
        causal: bool = False,
        pe: Tensor = None,
        q_pe: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if pe is not None:
            q, k = apply_rope(q, k, pe)

        if q_pe is not None:
            q = apply_rope_single(q, q_pe)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=causal,
        )

        if return_attn_weights:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if causal or attn_mask is not None:
                mask = (
                    torch.tril(torch.ones_like(attn)) if causal else attn_mask.logical_not().float()
                )
                mask = mask.logical_not().float()
                attn += torch.where(mask == 1.0, float("-inf"), mask)

            attn = attn.softmax(-1)
            if average_attn_weights:
                attn = attn.mean(dim=1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return (x, k.mean(1), attn) if return_attn_weights else (x, k.mean(1), None)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        drop_path_prob: float = 0.0,
        head_dim: int = 64,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_mult: int = 4,
        init_values=1.0, # 1.0e-05, # layer-scale
        norm_layer=partial(nn.LayerNorm, eps=1e-5), # dinov3 config
    ):
        super().__init__()

        self.attn = Attention(
            embed_dim, embed_dim // head_dim,
            attn_drop=attn_drop, proj_drop=proj_drop,
            norm_layer=norm_layer,
            qk_norm=True, # different from dinov3
            qkv_bias=True, scale_norm=False, # dinov3 config
        )
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_mult), act_layer=nn.GELU, drop=proj_drop)

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        self.gamma_1 = nn.Parameter(init_values * torch.ones(embed_dim)) if init_values is not None else None
        self.gamma_2 = nn.Parameter(init_values * torch.ones(embed_dim)) if init_values is not None else None

        self.drop_path1 = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        self.init_parameter()

    def init_parameter(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_linear)

    def forward(self, x, attn_mask=None, pe=None, causal=False,
                return_attn=False, avg_attn=True, return_metric=False, **kwargs):

        attn, metric, attn_weights = self.attn(
            self.norm1(x), attn_mask=attn_mask, pe=pe,
            return_attn_weights=return_attn, causal=causal,
            average_attn_weights=avg_attn,
        )

        if self.gamma_1 is None:
            x = x + self.drop_path1(attn)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * attn)
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))

        if return_metric:
            return x, metric, attn_weights
        else:
            return x, attn_weights


def sample_multi_level_1d_tokens(x, patch_sizes, residual=False):
    x_groups = []
    x_rest = x
    H, W = x.shape[2], x.shape[3]
    for size in patch_sizes:
        cur_group_x = F.interpolate(x_rest, size=size, mode='area')
        x_groups.append(einops.rearrange(cur_group_x, "b c h w -> b (h w) c"))
        if residual:
            x_rest = x_rest - F.interpolate(cur_group_x, size=(H, W), mode='nearest')

    x_1d = torch.cat(x_groups, dim=1)
    return x_1d


def multi_level_1d_features_to_2d_maps_avg(x, patch_sizes, orig_size, residual=False):
    n_sum = 0
    x_2d = None
    for size in patch_sizes:
        h, w = size
        x_up = F.interpolate(einops.rearrange(x[:,n_sum:n_sum+h*w], "b (h w) c -> b c h w", h=h, w=w), size=orig_size, mode='nearest')
        x_2d = x_up if x_2d is None else x_2d + x_up
        n_sum += h*w

    if not residual:
        x_2d = x_2d / len(patch_sizes)

    return x_2d


def build_hierarchical_causal_mask(
    block_sizes: Union[List[int], List[Tuple[int, int]]], # [w] or [h, w]
    device=None,
):
    """
    Create a hierarchical causal attention mask.

    Args:
        block_sizes (List[int]): token count per block (e.g., [4, 16, 64])
        device (torch.device or None): device to put the tensor on

    Returns:
        mask (torch.Tensor): shape [1, 1, H, W], float32, 0 or -inf
    """
    levels_h = []
    levels_w = []
    for level_idx, size in enumerate(block_sizes):
        if isinstance(size, (list, tuple)):
            levels_h.append(torch.full((size[0],), level_idx, dtype=torch.long))
            levels_w.append(torch.full((size[-1],), level_idx, dtype=torch.long))
        else:
            levels_h.append(torch.full((size,), level_idx, dtype=torch.long))
            levels_w.append(torch.full((size,), level_idx, dtype=torch.long))

    # Concatenate and create level tensor: [L]
    level_h_tensor = torch.cat(levels_h, dim=0)  # [H]
    level_w_tensor = torch.cat(levels_w, dim=0)  # [W]
    H = level_h_tensor.shape[0]
    W = level_w_tensor.shape[0]

    # Expand to pairwise comparison
    level_h_tensor = level_h_tensor.view(1, H, 1)
    level_w_tensor = level_w_tensor.view(1, 1, W)

    # Create attention mask
    mask = torch.where(level_h_tensor >= level_w_tensor, 0.0, float('-inf'))  # [1, H, W] lower triangular matrix

    # Add batch and head dimension for compatibility: [1, 1, H, W]
    mask = mask.unsqueeze(0)

    if device is not None:
        mask = mask.to(device)

    return mask # float32


class ResTokTransformerBlock(TransformerBlock):
    def forward(
            self,
            x: torch.Tensor,
            num_image_tokens: int,
            attn_mask: torch.Tensor = None,
            pe: torch.Tensor = None,
            causal: bool = False,
            return_attn: bool = False,
            avg_attn: bool = True,
            reduction_ratio: float = 0.0,
            residual_image_tokens: bool = False,
            return_metric: bool = False,
    ):
        attn_output, metric, attn_output_weights = self.attn(
            x=self.norm1(x),
            attn_mask=attn_mask,
            pe=pe,
            causal=causal,
            return_attn_weights=return_attn,
            average_attn_weights=avg_attn,
        )
        if self.gamma_1 is None:
            x = x + self.drop_path1(attn_output)
        else:
            x = x + self.drop_path1(self.gamma_1 * attn_output)

        if reduction_ratio > 0:
            x_merged = F.interpolate(
                x[:,:num_image_tokens].transpose(2, 1),
                size=int(num_image_tokens * (1 - reduction_ratio)),
                mode="nearest",
            ).transpose(2, 1).contiguous()

            if residual_image_tokens:
                x[:,:num_image_tokens] = x[:,:num_image_tokens] - \
                    F.interpolate(x_merged.transpose(2, 1), size=num_image_tokens, mode="nearest").transpose(2, 1).contiguous()

            x = torch.cat((x_merged, x), dim=1)

        if self.gamma_2 is None:
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        if return_metric:
            return x, metric, attn_output_weights
        else:
            return x, attn_output_weights


class ResTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.num_min_tokens = config.model.vq_model.num_min_tokens

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                # "small": 384, # TiTok's setting
                "small": 512, # GigaTok's setting
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 6,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                # "small": 12, # TiTok's setting
                "small": 8, # GigaTok's setting
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.vae_encoder = config.model.vq_model.get("vae_encoder", False)
        if self.vae_encoder:
            self.patch_embed = nn.Conv2d(
                in_channels=256, out_channels=self.width,
                kernel_size=1, stride=1, bias=True)
        else:
            embed_args = {}
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC')) # flatten deferred until after pos embed
            self.patch_embed = PatchEmbed(
                img_size=config.dataset.preprocessing.get("crop_size", 256),
                patch_size=16,
                in_chans=3,
                embed_dim=self.width,
                dynamic_img_pad=False,
                bias=True,
                **embed_args,
            )

        scale = self.width ** -0.5
        self.class_embedding = None
        self.ln_pre = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(ResTokTransformerBlock(
                self.width, head_dim=self.width//self.num_heads, mlp_mult=4.0, init_values=1.,
            ))
        self.residual_image_tokens = config.model.vq_model.get("residual_image_tokens", True)
        self.residual_latent_tokens = config.model.vq_model.get("residual_latent_tokens", True)
        self.ln_post = nn.LayerNorm(self.width)

        merge_tokens = config.model.vq_model.get("merge_tokens", False)
        hierarchy_stages = config.model.vq_model.get("hierarchy_stages", 4)
        if merge_tokens and hierarchy_stages > 0:
            merge_indices = [int(i/hierarchy_stages*self.num_layers)-1 for i in range(1, hierarchy_stages)]
        else:
            merge_indices = []
        self.reduction_ratio = [config.model.vq_model.get("reduction_ratio", 0.5) if i in merge_indices else 0 for i in range(self.num_layers)]

        self.level_embedding = nn.ParameterList([
            scale * torch.randn(1, self.width) for _ in range(len(merge_indices) + 1)])

        self.latent_hierarchy = config.model.vq_model.get("latent_hierarchy", True)
        self.nested_dropout_list = [self.num_latent_tokens]
        while self.nested_dropout_list[-1] // 2 > 1:
            self.nested_dropout_list.append(self.nested_dropout_list[-1] // 2)
        self.nested_dropout_list.append(1)
        self.nested_dropout_list = self.nested_dropout_list[::-1]
        self.nested_dropout_blocks = [self.nested_dropout_list[0]] + \
            [self.nested_dropout_list[i+1]-self.nested_dropout_list[i]
            for i in range(len(self.nested_dropout_list)-1)]
        self.nested_dropout_list = [x for x in self.nested_dropout_list if x >= self.num_min_tokens]
        self.latent_token_level_embedding = nn.ParameterList([
            scale * torch.randn(1, self.width) for _ in range(len(self.nested_dropout_blocks))])

        pooling_ps = {
            1: (1, 1),
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (4, 8),
            64: (8, 8),
            128: (8, 16),
            256: (16, 16),
        }
        self.pooling_patch_sizes = [pooling_ps[pn] for pn in self.nested_dropout_blocks]
        self.n_levels = len(self.nested_dropout_blocks)
        self.multi_quant = config.model.vq_model.get("multi_quant", True)
        if self.multi_quant:
            self.conv_out = nn.ModuleList([
                nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)
                for _ in range(self.n_levels)])
        else:
            self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

        self.rope2d = EmbedNDHybridMulti2DMaker(
            self.width,
            config.model.vq_model.get("rope_base_len", 10000),
            [self.width//self.num_heads//2 for _ in range(2)],
        )

    def forward(self, pixel_values, latent_tokens=None, return_attn=False, avg_attn=True):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        if self.vae_encoder:
            x = self.patch_embed(x)
        else:
            x = self.patch_embed(x).permute(0, 3, 1, 2)
        height_base, width_base = x.shape[-2:]
        hw_list = [(height_base, width_base),]

        if latent_tokens is None:
            if self.latent_hierarchy:
                latent_tokens = sample_multi_level_1d_tokens(x, self.pooling_patch_sizes, residual=self.residual_latent_tokens)
            else:
                latent_tokens = F.interpolate(x.flatten(2), size=self.num_latent_tokens, mode='area').transpose(2, 1)
        else:
            latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        length = latent_tokens.shape[1] # [N, L, D]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]

        hiera_level = 0
        x = x + self.level_embedding[hiera_level].to(x.dtype)
        hiera_level = hiera_level + 1

        num_image_tokens = int(x.shape[1])

        n_sum = 0
        for i, n in enumerate(self.nested_dropout_blocks):
            latent_tokens[:, n_sum:n_sum+n] = latent_tokens[:, n_sum:n_sum+n] + self.latent_token_level_embedding[i].to(latent_tokens.dtype)
            n_sum = n_sum + n
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        if return_attn:
            all_attn_weights = []

        image_tokens_blocks = [num_image_tokens]
        latent_tokens_blocks = self.nested_dropout_blocks
        attn_mask = build_hierarchical_causal_mask(image_tokens_blocks + latent_tokens_blocks, device=x.device)

        for i in range(self.num_layers):
            reduction_ratio = self.reduction_ratio[i]
            rope2d = self.rope2d(hw_list=hw_list, text_length=length, text_first=False, device=x.device)
            x, attn_weights = self.blocks[i](
                x,
                num_image_tokens,
                attn_mask=attn_mask,
                pe=rope2d,
                return_attn=return_attn,
                avg_attn=avg_attn,
                reduction_ratio=reduction_ratio,
                residual_image_tokens=self.residual_image_tokens,
            )
            if return_attn:
                all_attn_weights.append(attn_weights)
            if reduction_ratio > 0:
                num_image_tokens = int(num_image_tokens * (1 - reduction_ratio))
                image_tokens_blocks = [num_image_tokens] + image_tokens_blocks
                attn_mask = build_hierarchical_causal_mask(image_tokens_blocks + latent_tokens_blocks, device=x.device)
                x[:, :num_image_tokens] = x[:, :num_image_tokens] + self.level_embedding[hiera_level].to(x.dtype)
                hiera_level = hiera_level + 1
                hw_list.insert(0, [hw_list[0][0], int(hw_list[0][1] * (1 - reduction_ratio))])

        latent_tokens = x[:, -length:]

        result_dict = dict()
        if return_attn:
            result_dict["all_attn_weights"] = all_attn_weights

        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, length, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, length, self.width, 1).permute(0, 2, 1, 3)

        if self.multi_quant:
            n_sum = 0
            tmp = []
            for i, n in enumerate(self.nested_dropout_blocks):
                tmp.append(self.conv_out[i](latent_tokens[:, :, n_sum:n_sum+n]))
                n_sum = n_sum + n
            latent_tokens = torch.cat(tmp, dim=2)
        else:
            latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, length)

        image_tokens = x[:, :-length]
        num_image_tokens_list = image_tokens_blocks[1:]
        result_dict["image_tokens"] = []
        n_sum = 0
        for i, n in enumerate(image_tokens_blocks):
            result_dict["image_tokens"].append(image_tokens[:,n_sum:n_sum+n])
            n_sum = n_sum + n
        result_dict["first_image_tokens"] = einops.rearrange(result_dict["image_tokens"][-1], "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size)
        result_dict["last_image_tokens"] = einops.rearrange(result_dict["image_tokens"][0], "b (1 w) c -> b c 1 w")
        result_dict["first_image_tokens_sum"] = einops.rearrange(sum([
            F.interpolate(x.transpose(2, 1), size=result_dict["image_tokens"][-1].shape[1], mode="nearest").transpose(2, 1) for x in result_dict["image_tokens"]
        ]), "b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size)

        return latent_tokens, result_dict


class ResTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.num_min_tokens = config.model.vq_model.num_min_tokens
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.learnable_dec_image_tokens = config.model.vq_model.get("learnable_dec_image_tokens", True)
        self.residual_latent_tokens = config.model.vq_model.get("residual_latent_tokens", True)

        self.width = {
                # "small": 384, # TiTok's setting
                "small": 512, # GigaTok's setting
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 6,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                # "small": 12, # TiTok's setting
                "small": 8, # GigaTok's setting
                "base": 12,
                "large": 16,
            }[self.model_size]

        scale = self.width ** -0.5
        # add mask token and query pos embed
        self.latent_mask_token = nn.Parameter(scale * torch.randn(1, 1, self.token_size))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width)) if self.learnable_dec_image_tokens else None
        self.dec_feat_sup = config.model.vq_model.get("decoder_feature_supervision", False) and (config.model.vq_model.get("use_vf", None) is not None)
        self.feat_mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width)) if self.dec_feat_sup and config.model.vq_model.get("learnable_dec_feat_tokens", True) else None
        self.ln_pre = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(
                self.width, head_dim=self.width//self.num_heads, mlp_mult=4.0, init_values=1.,
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            # self.ffn = nn.Sequential(
            #     nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
            #     nn.Tanh(),
            #     nn.Conv2d(2 * self.width, 256, 1, padding=0, bias=True),
            # )
            self.ffn = nn.Conv2d(self.width, 256, 1, padding=0, bias=True)
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

        self.nested_dropout_list = [self.num_latent_tokens]
        while self.nested_dropout_list[-1] // 2 > 1:
            self.nested_dropout_list.append(self.nested_dropout_list[-1] // 2)
        self.nested_dropout_list.append(1)
        self.nested_dropout_list = self.nested_dropout_list[::-1]
        self.nested_dropout_blocks = [self.nested_dropout_list[0]] + \
            [self.nested_dropout_list[i+1]-self.nested_dropout_list[i]
            for i in range(len(self.nested_dropout_list)-1)]
        self.nested_dropout_list = [x for x in self.nested_dropout_list if x >= self.num_min_tokens]
        self.latent_token_level_embedding = nn.ParameterList([
            scale * torch.randn(1, self.width) for _ in range(len(self.nested_dropout_blocks))])

        pooling_ps = {
            1: (1, 1),
            2: (1, 2),
            4: (2, 2),
            8: (2, 4),
            16: (4, 4),
            32: (4, 8),
            64: (8, 8),
            128: (8, 16),
            256: (16, 16),
        }
        self.pooling_patch_sizes = [pooling_ps[pn] for pn in self.nested_dropout_blocks]
        self.n_levels = len(self.nested_dropout_blocks)
        self.multi_quant = config.model.vq_model.get("multi_quant", True)
        if self.multi_quant:
            self.decoder_embed = nn.ModuleList([nn.Linear(
                self.token_size, self.width, bias=True)
                for _ in range(self.n_levels)])
        else:
            self.decoder_embed = nn.Linear(
                self.token_size, self.width, bias=True)

        self.rope2d = EmbedNDHybrid2DMaker(
            self.width,
            config.model.vq_model.get("rope_base_len", 10000),
            [self.width//self.num_heads//2 for _ in range(2)],
        )

        n = len(self.nested_dropout_list)
        drop_prob = config.model.vq_model.get("drop_prob", 0.2)
        if config.model.vq_model.get("drop_latent_tokens", False) and drop_prob > 0.0:
            weights = [2 ** i for i in range(n-1)] + [sum(2 ** i for i in range(n-1)) * (int(1/drop_prob) - 1)]
            total = sum(weights)
            self.nested_dropout_probs = [w / total for w in weights]

    def forward(self, z_quantized, num_latent_tokens=32, return_attn=False, avg_attn=True):
        N, C, H, W = z_quantized.shape
        if self.training:
            assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD

        if isinstance(num_latent_tokens, int):
            latent_mask_token = self.latent_mask_token.repeat(x.shape[0], self.num_latent_tokens - num_latent_tokens, 1).to(x.dtype)
            x = torch.cat([x[:, :num_latent_tokens], latent_mask_token], dim=1)
        else:
            n_vec = torch.tensor(num_latent_tokens, device=x.device, dtype=torch.long)
            keep = (torch.arange(self.num_latent_tokens, device=x.device).view(1, -1) < n_vec.view(-1, 1)).unsqueeze(-1)
            x = torch.where(keep, x, self.latent_mask_token.to(x).expand_as(x))

        if self.multi_quant:
            n_sum = 0
            tmp = []
            for i, n in enumerate(self.nested_dropout_blocks):
                tmp.append(self.decoder_embed[i](x[:, n_sum:n_sum+n]))
                n_sum = n_sum + n
            x = torch.cat(tmp, dim=1)
        else:
            x = self.decoder_embed(x)
        batchsize, seq_len, _ = x.shape

        if self.mask_token is not None:
            mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        else:
            mask_tokens = multi_level_1d_features_to_2d_maps_avg(
                x, self.pooling_patch_sizes, (self.grid_size, self.grid_size), residual=self.residual_latent_tokens,
            ).permute(0, 2, 3, 1).reshape(batchsize, self.grid_size**2, self.width)

        if self.dec_feat_sup:
            if self.feat_mask_token is not None:
                feat_mask_tokens = self.feat_mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
            else:
                feat_mask_tokens = multi_level_1d_features_to_2d_maps_avg(
                    x, self.pooling_patch_sizes, (self.grid_size, self.grid_size), residual=self.residual_latent_tokens,
                ).permute(0, 2, 3, 1).reshape(batchsize, self.grid_size**2, self.width)
            mask_tokens = torch.cat([mask_tokens, feat_mask_tokens], dim=0)
            x = torch.cat([x, x], dim=0)

        n_sum = 0
        for i, n in enumerate(self.nested_dropout_blocks):
            x[:, n_sum:n_sum+n] = x[:, n_sum:n_sum+n] + self.latent_token_level_embedding[i].to(x.dtype)
            n_sum = n_sum + n
        x = torch.cat([x, mask_tokens], dim=1)

        rope2d = self.rope2d(h=self.grid_size, w=self.grid_size, text_length=seq_len, text_first=True).to(x.device)

        x = self.ln_pre(x)
        if return_attn:
            all_attn_weights = []
        attn_mask = None
        for i in range(self.num_layers):
            x, attn_weights = self.blocks[i](
                x,
                attn_mask=attn_mask,
                pe=rope2d,
                return_attn=return_attn,
                avg_attn=avg_attn,
            )
            if return_attn:
                all_attn_weights.append(attn_weights)

        dec_result_dict = dict()
        if return_attn:
            dec_result_dict["all_attn_weights"] = all_attn_weights

        x = x[:, seq_len:]
        if self.dec_feat_sup:
            x, feat_mask_tokens = torch.chunk(x, 2, dim=0)
            dec_result_dict["dec_vf_feat"] = feat_mask_tokens.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x, dec_result_dict
