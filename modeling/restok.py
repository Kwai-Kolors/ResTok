"""This file contains the model definition of ResTok.

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
    TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/titok.py
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import ResTokEncoder, ResTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Encoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from timm.models.layers import trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ResTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2601.03955", "image-tokenization", "visual-tokenizer", "image-reconstruction"], repo_url="https://github.com/Kwai-Kolors/ResTok", paper_url="https://arxiv.org/abs/2601.03955", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self.encoder = ResTokEncoder(config)
        self.decoder = ResTokDecoder(config)

        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.drop_latent_tokens = config.model.vq_model.get("drop_latent_tokens", True)
        self.test_num_latent_tokens = config.model.vq_model.get("test_num_latent_tokens", self.num_latent_tokens)
        scale = self.encoder.width ** -0.5
        self.learnable_latent_tokens = config.model.vq_model.get("learnable_latent_tokens", False)
        if self.learnable_latent_tokens:
            self.latent_tokens = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, self.encoder.width))

        self.vae_encoder = config.model.vq_model.get("vae_encoder", False)
        if self.vae_encoder:
            self.pixel_encoder = Pixel_Encoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256}))

        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            entropy_loss_ratio=config.model.vq_model.entropy_loss_ratio,
            use_l2_norm=config.model.vq_model.use_l2_norm,
            )

        self.enc_feat_sup = config.model.vq_model.get("encoder_feature_supervision", True)
        use_vf = config.model.vq_model.get("use_vf", None)
        if use_vf is not None:
            self.use_vf = use_vf
            self.reverse_proj = config.model.vq_model.get("reverse_proj", True)
            from modeling.modules.foundation_models import aux_foundation_model
            print(f"Using {use_vf} as auxiliary feature.")
            self.foundation_model = aux_foundation_model(use_vf)
            self.foundation_model.eval()
            self.foundation_model.requires_grad_(False)
            vf_feature_dim = self.foundation_model.feature_dim
            if self.reverse_proj:
                if self.enc_feat_sup:
                    self.vf_proj = torch.nn.Conv2d(self.encoder.width, vf_feature_dim, kernel_size=1, bias=False)
                if self.decoder.dec_feat_sup:
                    self.vf_proj_dec = torch.nn.Conv2d(self.decoder.width, vf_feature_dim, kernel_size=1, bias=False)
            else:
                self.mlp_proj = torch.nn.Conv2d(vf_feature_dim, self.encoder.width, kernel_size=1, bias=True)
        else:
            self.use_vf = None

        # ImageNet specifics
        self.register_buffer('imnet_mean', torch.tensor(IMAGENET_DEFAULT_MEAN), persistent=False)
        self.register_buffer('imnet_std', torch.tensor(IMAGENET_DEFAULT_STD), persistent=False)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, return_attn=False, avg_attn=True):
        x = (x - self.imnet_mean.view(1, -1, 1, 1)) / self.imnet_std.view(1, -1, 1, 1)
        z, extra_tokens = self.encoder(
            pixel_values=self.pixel_encoder(x) if self.vae_encoder else x,
            latent_tokens=self.latent_tokens if self.learnable_latent_tokens else None,
            return_attn=return_attn,
            avg_attn=avg_attn,
        )
        z_quantized, result_dict = self.quantize(z)

        result_dict.update(extra_tokens)
        return z_quantized, result_dict

    def decode(self, z_quantized, num_latent_tokens=None, return_attn=False, avg_attn=True):
        if num_latent_tokens is None:
            num_latent_tokens = self.test_num_latent_tokens

        decoded, dec_result_dict = self.decoder(z_quantized, num_latent_tokens, return_attn, avg_attn)

        # decoded = self.pixel_decoder(decoded)
        with torch.autocast("cuda", enabled=False):
            decoded = self.pixel_decoder(decoded.float())
        return decoded, dec_result_dict

    def decode_tokens(self, tokens, num_latent_tokens=None, return_attn=False, avg_attn=True):
        if num_latent_tokens is None:
            num_latent_tokens = self.test_num_latent_tokens

        tokens = tokens.squeeze(1) # when [B, 1, N] then -> [B, N]
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        decoded, dec_result_dict = self.decode(z_quantized, num_latent_tokens, return_attn, avg_attn)
        return decoded if not return_attn else (decoded, dec_result_dict["all_attn_weights"])

    def forward(self, x, num_latent_tokens=None):
        z_quantized, result_dict = self.encode(x)

        if num_latent_tokens is None:
            if self.training:
                if self.drop_latent_tokens:
                    if hasattr(self.decoder, "nested_dropout_probs"):
                        nested_dropout_index = random.choices(range(len(self.decoder.nested_dropout_list)), weights=self.decoder.nested_dropout_probs, k=x.shape[0])
                    else:
                        nested_dropout_index = [random.randint(0, len(self.decoder.nested_dropout_list) - 1) for _ in range(len(self.decoder.nested_dropout_list))]
                    nested_dropout_remain = [self.decoder.nested_dropout_list[i] for i in nested_dropout_index]
                    result_dict["nested_dropout_index"] = nested_dropout_index
                else:
                    nested_dropout_remain = self.num_latent_tokens
            else:
                nested_dropout_remain = self.test_num_latent_tokens
        else:
            nested_dropout_remain = num_latent_tokens

        decoded, dec_result_dict = self.decode(z_quantized, num_latent_tokens=nested_dropout_remain, return_attn=False)
        result_dict.update(dec_result_dict)
        result_dict["nested_dropout_remain"] = nested_dropout_remain
        result_dict["test_min_encoding_indices"] = result_dict["min_encoding_indices"][:,:,:(num_latent_tokens if num_latent_tokens is not None else self.test_num_latent_tokens)]

        if self.use_vf is not None:
            with torch.no_grad():
                aux_cls, aux_feature = self.foundation_model(x)
            if not self.reverse_proj:
                aux_feature = self.vf_proj(aux_feature)
            else:
                if self.enc_feat_sup:
                    result_dict["last_image_tokens_proj"] = self.vf_proj(result_dict["last_image_tokens"])
                if self.decoder.dec_feat_sup:
                    result_dict["dec_vf_feat_proj"] = self.vf_proj_dec(result_dict["dec_vf_feat"])
            result_dict["aux_feature"] = aux_feature
            result_dict["aux_cls"] = aux_cls

        return decoded, result_dict
