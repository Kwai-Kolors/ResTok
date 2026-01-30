"""This file contains the model definition of LlamaGen with HAR.

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
    LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
"""

from dataclasses import dataclass
from typing import Optional, Callable, Union, List, Tuple


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
import torch.distributed as dist

import json
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from pathlib import Path
from modeling.modules.base_model import BaseModel
from .generate import generate_har as _generate


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    ar_token_num: int = 4
    spe_token_num: int = (4 - 1) + 4 + 8 + 16 + 32 + 64


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class SpecialTokenEmbedding(nn.Module):
    def __init__(self, num_special_tokens, hidden_size):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(num_special_tokens, hidden_size)

    def forward(self):
        special_tokens = torch.arange(self.num_special_tokens, device=self.special_embeddings.weight.device)
        special_embeddings = self.special_embeddings(special_tokens)
        return special_embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        if k_out.dtype != k_val.dtype:
            k_val_dtype = k_val.dtype
            v_val_dtype = v_val.dtype
            k_out[:, :, input_pos] = k_val.to(k_out.dtype)
            v_out[:, :, input_pos] = v_val.to(v_out.dtype)
            k_out.to(k_val_dtype)
            v_out.to(v_val_dtype)
        else:
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val
        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2601.03955", "llamagen", "autoregressive", "image-generation"], repo_url="https://github.com/Kwai-Kolors/ResTok", paper_url="https://arxiv.org/abs/2601.03955", license="apache-2.0"):
    def __init__(self, config: ModelArgs):
        super().__init__()
        config = OmegaConf.create(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.ar_token_num = config.ar_token_num
        self.spe_token_num = config.spe_token_num

        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        self.spe_tok_embeddings = SpecialTokenEmbedding(self.spe_token_num, config.dim)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # hierarchical mask
        self.nested_dropout_list = [self.block_size]
        while self.nested_dropout_list[-1] // 2 > 1:
            self.nested_dropout_list.append(self.nested_dropout_list[-1] // 2)
        self.nested_dropout_list.append(1)
        self.nested_dropout_list = self.nested_dropout_list[::-1]
        self.nested_dropout_blocks = [self.nested_dropout_list[0]] + \
            [self.nested_dropout_list[i+1]-self.nested_dropout_list[i]
            for i in range(len(self.nested_dropout_list)-1)]
        self.group_mask = build_hierarchical_causal_mask([1 for _ in range(self.cls_token_num-1)] + self.nested_dropout_blocks).squeeze(dim=(0,1))
        ar_mask = build_hierarchical_causal_mask([1 for _ in range(self.ar_token_num)]).squeeze(dim=(0,1))
        self.group_mask[self.cls_token_num-1:self.cls_token_num-1+self.ar_token_num, self.cls_token_num-1:self.cls_token_num-1+self.ar_token_num] = ar_mask
        self.group_mask[self.cls_token_num-1:self.cls_token_num-1+self.ar_token_num, self.cls_token_num-1+self.ar_token_num:] = float('-inf')

        # 1d rotary pos embedding
        self.freqs_cis = precompute_freqs_cis_1d(self.cls_token_num + sum(self.nested_dropout_blocks) - 1, self.config.dim // self.config.n_head, self.config.rope_base)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

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

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        # hierarchical mask
        group_mask = build_hierarchical_causal_mask([1 for _ in range(self.cls_token_num-1)] + self.nested_dropout_blocks + [128]).squeeze(dim=(0,1))[:max_seq_length,:max_seq_length]
        ar_mask = build_hierarchical_causal_mask([1 for _ in range(self.ar_token_num)]).squeeze(dim=(0,1))
        group_mask[self.cls_token_num-1:self.cls_token_num-1+self.ar_token_num, self.cls_token_num-1:self.cls_token_num-1+self.ar_token_num] = ar_mask
        group_mask[self.cls_token_num-1:self.cls_token_num-1+self.ar_token_num, self.cls_token_num-1+self.ar_token_num:] = float('-inf')
        self.causal_mask = group_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1).to(dtype)

        # 1d rotary pos embedding
        self.freqs_cis = precompute_freqs_cis_1d(max_seq_length, self.config.dim // self.config.n_head, self.config.rope_base)

    def disable_caches(self):
        self.max_batch_size = -1
        self.max_seq_length = -1
        for b in self.layers:
            b.attention.kv_cache = None
        del self.causal_mask
        self.freqs_cis = precompute_freqs_cis_1d(self.cls_token_num + sum(self.nested_dropout_blocks) - 1, self.config.dim // self.config.n_head, self.config.rope_base)

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):

        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(cond_embeddings.shape[0], -1, -1)
            # token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            token_embeddings_first, token_embeddings_last = token_embeddings[:,:self.ar_token_num], token_embeddings[:,self.ar_token_num:]
            token_embeddings_first = torch.cat((token_embeddings_first, spe_embeddings[:,:self.ar_token_num-1]), dim=1)

            n_sum = 0
            token_embeddings_last_list = []
            for i, n in enumerate(self.nested_dropout_blocks):
                if n_sum + n <= self.ar_token_num:
                    n_sum = n_sum + n
                else:
                    tmp = torch.stack([
                        token_embeddings_last[:, n_sum-self.ar_token_num:n_sum-self.ar_token_num+n], spe_embeddings[:, n_sum-1:n_sum-1+n]
                    ], dim=1) # dim=2: interleaved
                    token_embeddings_last_list.append(tmp.reshape(token_embeddings.shape[0], -1, self.config.dim))
                    n_sum = n_sum + n
            token_embeddings_last = torch.cat(token_embeddings_last_list, dim=1)

            token_embeddings = torch.cat((cond_embeddings, token_embeddings_first, token_embeddings_last), dim=1)
            token_embeddings = token_embeddings[:,:self.cls_token_num + self.block_size - 1]
            # always use the cond_embeddings with positional embeddings for adaLN
            cond_embeddings = token_embeddings[:,:self.cls_token_num]
            h = self.tok_dropout(token_embeddings)

            mask = self.group_mask[:token_embeddings.shape[1], :token_embeddings.shape[1]]
            batch_size = cond_embeddings.shape[0]
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            mask = mask.to(h.device)

        else:
            if cond_idx is not None: # prefill in inference
                self.start = False
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                # spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
                # token_embeddings = torch.cat((token_embeddings, spe_embeddings), dim=1)
            else: # decode_n_tokens(kv cache) in inference
                if idx.shape[1] == 1 and not self.start: # ar phase
                    token_embeddings = self.tok_embeddings(idx)
                    if self.ar_token_num == 1:
                        self.start = True
                        self.prev_spe_token_num = self.ar_token_num
                        self.prev_spe_token_num_sum = self.ar_token_num - 1
                elif idx.shape[1] > 1 and not self.start:
                    token_embeddings = self.tok_embeddings(idx)
                    spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
                    token_embeddings = torch.cat((token_embeddings[:,-1:], spe_embeddings[:,:self.ar_token_num-1]), dim=1)
                    self.start = True
                    self.prev_spe_token_num = self.ar_token_num
                    self.prev_spe_token_num_sum = self.ar_token_num - 1
                else:
                    token_embeddings = self.tok_embeddings(idx)
                    spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
                    token_embeddings = torch.stack((
                        token_embeddings, spe_embeddings[:,self.prev_spe_token_num_sum:self.prev_spe_token_num_sum+self.prev_spe_token_num]
                    ), dim=1) # dim=2: interleaved
                    token_embeddings = token_embeddings.flatten(1,2)
                    self.prev_spe_token_num_sum += self.prev_spe_token_num
                    self.prev_spe_token_num *= 2

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)

        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]].to(h.device)
        else:
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        # output layers
        h = self.norm(h)
        logits = self.output(h).float()

        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self,
                 condition,
                 num_sample_steps,
                 randomize_temperature=1.0,
                 guidance_scale=2.0,
                 cfg_interval=-1,
                 guidance_decay="constant",
                 cfg_schedule_kwargs={},
                 top_k=0,
                 top_p=1.0,
                 kv_cache=True,
                 **kwargs):
        return _generate(
            self, condition, max_new_tokens=num_sample_steps,
            cfg_scale=guidance_scale, cfg_interval=cfg_interval,
            cfg_schedule=guidance_decay, cfg_schedule_kwargs=cfg_schedule_kwargs,
            ar_token_num=self.ar_token_num,
            temperature=randomize_temperature, top_k=top_k,
            top_p=top_p, sample_logits=True, 
        )

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis_1d(seq_len: int, n_elem: int, base: int = 10000):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cache


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    # print(x_out2)
    return x_out2.type_as(x)


#################################################################################
#                                    Utils                                      #
#################################################################################


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

    return mask  # float32


#################################################################################
#                                GPT Configs                                    #
#################################################################################
def parse_kwargs(config):
    if config.model.generator.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = config.model.generator.dropout_p
    return {
        'vocab_size': config.model.generator.vocab_size,
        'block_size': config.model.generator.block_size,
        'num_classes': config.model.generator.num_classes,
        'cls_token_num': config.model.generator.cls_token_num,
        'model_type': config.model.generator.gpt_type,
        'resid_dropout_p': dropout_p,
        'ffn_dropout_p': dropout_p,
        'drop_path_rate': config.model.generator.drop_path_rate,
        'token_dropout_p': config.model.generator.token_dropout_p,
        'ar_token_num': config.model.generator.get("ar_token_num", 4),
        'spe_token_num': config.model.generator.get("spe_token_num", 127),
    }

### text-conditional
def GPT_7B(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(config):
    kwargs = parse_kwargs(config)
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}
