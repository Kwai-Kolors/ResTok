"""This file contains the sampling utils of LlamaGen with HAR.

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
    PAR: https://github.com/YuqingWang1029/PAR/blob/main/autoregressive/models/generate.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def sample_multi(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, token_num=4, sample_logits=True):
    logits = logits[:, -1*token_num:, :]
    batch_size, num_samples, vocab_size = logits.shape
    logits = logits.reshape(batch_size * num_samples, vocab_size) / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    idx = idx.view(batch_size, num_samples)
    probs = probs.view(batch_size, num_samples, vocab_size)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def prefill_with_schedule(
    model, cond_idx: torch.Tensor, input_pos: torch.Tensor,
    cfg_scale: float, cfg_interval: int,
    cfg_schedule: str = "constant", cfg_schedule_kwargs: dict = {},
    **sampling_kwargs):

    if cfg_schedule == 'constant':
        cfg = constant_schedule(0, cfg_scale)
    elif cfg_schedule == 'linear':
        cfg = linear_schedule(0, cfg_scale)
    elif cfg_schedule == 'linear_re':
        cfg = linear_re_schedule(0, cfg_scale)
    elif cfg_schedule == 'linear_constant':
        cfg = linear_constant_schedule(0, cfg_scale, **cfg_schedule_kwargs)
    elif cfg_schedule == 'triangular':
        cfg = triangular_schedule(0, cfg_scale, **cfg_schedule_kwargs)
    elif cfg_schedule == 'rectangular':
        cfg = rectangular_schedule(0, cfg_scale, **cfg_schedule_kwargs)
    elif cfg_schedule == 'step':
        cfg = step_schedule(0, cfg_scale, **cfg_schedule_kwargs)
    elif cfg_schedule == 'cosine':
        cfg = cosine_schedule(0, cfg_scale)
    elif cfg_schedule == 'cosine_re':
        cfg = cosine_re_schedule(0, cfg_scale)

    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, cond_idx: torch.Tensor = None, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=cond_idx, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=cond_idx, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_one_token_multi(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, token_num: int, cond_idx: torch.Tensor = None, **sampling_kwargs):
    # assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=cond_idx, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=cond_idx, input_pos=input_pos)
    return sample_multi(logits, token_num=token_num, **sampling_kwargs)


# a series of cfg scheduling functions
def constant_schedule(ratio, cfg_scale, **kwargs):
    return cfg_scale


def linear_schedule(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * ratio + 1e-4


def linear_re_schedule(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * (1.0 - ratio) + 1e-4


def linear_constant_schedule(ratio, cfg_scale, peak=0.5, **kwargs):
    if ratio < peak:
        return 1.0 + (cfg_scale - 1.0) * (ratio / peak)
    else:
        return cfg_scale


def triangular_schedule(ratio, cfg_scale, peak=0.5, **kwargs):
    if ratio < peak:
        return 1.0 + (cfg_scale - 1.0) * (ratio / peak) + 1e-4
    else:
        return 1.0 + (cfg_scale - 1.0) * ((1.0 - ratio) / (1.0 - peak)) + 1e-4


def rectangular_schedule(ratio, cfg_scale, window_start=0.05, window_end=1.0, **kwargs):
    if window_start <= ratio <= window_end:
        return cfg_scale
    else:
        if "min_cfg_scale" in kwargs:
            return kwargs["min_cfg_scale"] if kwargs["min_cfg_scale"] != 1.0 else kwargs["min_cfg_scale"] + 1e-4
        else:
            return 1.0 + 1e-4


def step_schedule(ratio, cfg_scale, window_start, **kwargs):
    if cfg_scale == 1.0:
        return cfg_scale
    elif ratio >= window_start:
        return cfg_scale
    else:
        if "min_cfg_scale" in kwargs:
            return kwargs["min_cfg_scale"] if kwargs["min_cfg_scale"] != 1.0 else kwargs["min_cfg_scale"] + 1e-4
        else:
            return 1.0 + 1e-4


def cosine_re_schedule(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * (0.5 * (1 + math.cos(math.pi * ratio)))


def cosine_schedule(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * (0.5 * (1 - math.cos(math.pi * ratio)))


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int, cfg_schedule: str = "constant", cfg_schedule_kwargs: dict = {},
    **sampling_kwargs):
    """
    Args:
        cfg_interval: -1 means always cfg, other value bigger than -1 means only first cfg_interval tokens are cfg
    """
    # reference: 
    # https://github.com/Pepper-lll/LMforImageGeneration/blob/9aed8d795ce2c79cbfaa75884c09fa2e1fc744ee/llama/ar_model.py#L510
    valid_schedules = ["constant", "linear", "linear_re", "linear_constant", "triangular", "rectangular", "cosine", "cosine_re", "sigmoid", "step"]
    assert cfg_schedule in valid_schedules, f"cfg_schedule must be one of {valid_schedules}"

    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        if "num_steps" in cfg_schedule_kwargs:
            ratio = 1. * ( i + 1 ) / (cfg_schedule_kwargs["num_steps"] - 1)
            if "max_new_tokens" in sampling_kwargs:
                sampling_kwargs.pop("max_new_tokens")
        elif "max_new_tokens" in sampling_kwargs:
            ratio = 1. * ( i + 1 ) / (sampling_kwargs.pop("max_new_tokens") - 1)
        else:
            ratio = 1. * ( i + 1 ) / num_new_tokens
        if cfg_schedule == 'constant':
            cfg = constant_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear':
            cfg = linear_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear_re':
            cfg = linear_re_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear_constant':
            cfg = linear_constant_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'triangular':
            cfg = triangular_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'rectangular':
            cfg = rectangular_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'step':
            cfg = step_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'cosine':
            cfg = cosine_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'cosine_re':
            cfg = cosine_re_schedule(ratio, cfg_scale)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg, cfg_flag, **sampling_kwargs
            )
            input_pos += 1

            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)

    return new_tokens, new_probs


def decode_n_tokens_multi(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int, spe_token_num: int, **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(0, num_new_tokens, spe_token_num):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token_multi(
                model, cur_token, input_pos, cfg_scale, cfg_flag, token_num=spe_token_num, **sampling_kwargs
            )
            input_pos += spe_token_num
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token #.view(-1, 1)
    
    return new_tokens, new_probs


def decode_n_tokens_har(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, ar_token_num: int,
    cfg_scale: float, cfg_interval: int, cfg_schedule: str = "constant", cfg_schedule_kwargs: dict = {},
    **sampling_kwargs):
    """
    Args:
        cfg_interval: -1 means always cfg, other value bigger than -1 means only first cfg_interval tokens are cfg
    """
    # reference: 
    # https://github.com/Pepper-lll/LMforImageGeneration/blob/9aed8d795ce2c79cbfaa75884c09fa2e1fc744ee/llama/ar_model.py#L510
    valid_schedules = ["constant", "linear", "linear_re", "linear_constant", "triangular", "rectangular", "cosine", "cosine_re", "sigmoid", "step"]
    assert cfg_schedule in valid_schedules, f"cfg_schedule must be one of {valid_schedules}"

    new_tokens, new_probs = [], []
    cfg_flag = True
    token_num = ar_token_num

    i = ar_token_num - 1
    cnt = ar_token_num - 1
    while i < num_new_tokens + ar_token_num - 1:
        if "num_steps" in cfg_schedule_kwargs:
            ratio = 1. * ( cnt + 1 ) / (cfg_schedule_kwargs["num_steps"] - 1)
        else:
            ratio = 1. * ( i + 1 ) / (num_new_tokens + ar_token_num - 1)
        if cfg_schedule == 'constant':
            cfg = constant_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear':
            cfg = linear_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear_re':
            cfg = linear_re_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear_constant':
            cfg = linear_constant_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'triangular':
            cfg = triangular_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'rectangular':
            cfg = rectangular_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'step':
            cfg = step_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'cosine':
            cfg = cosine_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'cosine_re':
            cfg = cosine_re_schedule(ratio, cfg_scale)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token_multi(
                model, cur_token, input_pos, cfg, cfg_flag, token_num=token_num, **sampling_kwargs
            )
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token #.view(-1, 1)
            i += token_num
            cnt += 1
            input_pos = torch.cat([input_pos, input_pos + token_num]) + token_num
            token_num *= 2

    return new_tokens, new_probs


@torch.no_grad()
def generate(
        model,
        cond,
        max_new_tokens,
        emb_masks=None,
        cfg_scale=1.0,
        cfg_interval=-1,
        cfg_schedule="constant",
        cfg_schedule_kwargs={},
        prefilled_guidance=False,
        ar_token_num=4, # not used in vanilla ar
        **sampling_kwargs):
    device = cond.device
    if model.model_type == 'c2i':
        if not prefilled_guidance:
            if cfg_scale > 1.0:
                cond_null = torch.ones_like(cond, device=device) * model.num_classes
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = 1
        else:
            raise NotImplementedError
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond, device=device) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill_with_schedule(
        model,
        cond_combined,
        input_pos,
        cfg_scale,
        cfg_interval,
        cfg_schedule=cfg_schedule,
        cfg_schedule_kwargs=cfg_schedule_kwargs,
        **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(
                            model, 
                            next_token, 
                            input_pos, 
                            max_new_tokens-1, 
                            cfg_scale, 
                            cfg_interval, 
                            cfg_schedule=cfg_schedule,
                            cfg_schedule_kwargs=cfg_schedule_kwargs,
                            # cond_idx=cond_combined, # not used for inference with kv_cache
                            **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    with torch.device(device):
        model.disable_caches()

    return seq[:, T:]


@torch.no_grad()
def generate_multi(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, ar_token_num=4, spe_token_num=3, **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    spe_token_num = spe_token_num+1

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, ar_token_num-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:T+1+len(generated_tokens)] = torch.cat(generated_tokens, dim=1)
    input_pos = torch.tensor([input_pos[-1]+i for i in range(spe_token_num)], device=device, dtype=torch.int)
    next_token = torch.cat(generated_tokens, dim=1)[:,-spe_token_num:]
    generated_tokens, _ = decode_n_tokens_multi(model, next_token, input_pos, max_new_tokens - ar_token_num, cfg_scale, cfg_interval, spe_token_num, **sampling_kwargs)
    seq[:, T+ar_token_num:] = torch.cat(generated_tokens, dim=1)

    with torch.device(device):
        model.disable_caches()

    return seq[:, T:]


@torch.no_grad()
def generate_har(
        model,
        cond,
        max_new_tokens,
        emb_masks=None,
        cfg_scale=1.0,
        cfg_interval=-1,
        cfg_schedule="constant",
        cfg_schedule_kwargs={},
        prefilled_guidance=False,
        ar_token_num=4,
        **sampling_kwargs):

    device = cond.device
    if model.model_type == 'c2i':
        if not prefilled_guidance:
            if cfg_scale > 1.0:
                cond_null = torch.ones_like(cond, device=device) * model.num_classes
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = 1
        else:
            raise NotImplementedError
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond, device=device) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)

    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill_with_schedule(
        model,
        cond_combined,
        input_pos,
        cfg_scale,
        cfg_interval,
        cfg_schedule=cfg_schedule,
        cfg_schedule_kwargs=cfg_schedule_kwargs,
        **sampling_kwargs)
    seq[:, T:T+1] = next_token

    if ar_token_num > 1:
        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        generated_tokens, _ = decode_n_tokens(
                                model,
                                next_token,
                                input_pos,
                                ar_token_num-1,
                                cfg_scale,
                                cfg_interval,
                                cfg_schedule=cfg_schedule,
                                cfg_schedule_kwargs=cfg_schedule_kwargs,
                                # cond_idx=cond_combined, # not used for inference with kv_cache
                                max_new_tokens=max_new_tokens,
                                **sampling_kwargs)
        seq[:, T+1:T+1+len(generated_tokens)] = torch.cat(generated_tokens, dim=1)

    if max_new_tokens > ar_token_num:
        input_pos = torch.tensor([input_pos[-1]+i for i in range(ar_token_num)], device=device, dtype=torch.int)
        if ar_token_num > 1:
            next_token = torch.cat([next_token, *generated_tokens], dim=1)[:,-ar_token_num:]
        generated_tokens, _ = decode_n_tokens_har(
            model,
            next_token,
            input_pos,
            max_new_tokens - ar_token_num,
            ar_token_num,
            cfg_scale,
            cfg_interval,
            cfg_schedule=cfg_schedule,
            cfg_schedule_kwargs=cfg_schedule_kwargs,
            **sampling_kwargs)
        seq[:, T+ar_token_num:] = torch.cat(generated_tokens, dim=1)[:,:max_new_tokens - ar_token_num]

    with torch.device(device):
        model.disable_caches()

    return seq[:, T:]
