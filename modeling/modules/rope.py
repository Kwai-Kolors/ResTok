# ROPE from Flux
# https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return (
        xq_out.reshape(*xq.shape).type_as(xq),
        xk_out.reshape(*xk.shape).type_as(xk),
    )


def apply_rope_single(xq: Tensor, freqs_cis: Tensor) -> Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.size(-1)
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class EmbedND1DMaker(EmbedND):
    def forward(self, length: int, device=None) -> torch.Tensor:
        ids = torch.arange(length, device=device, dtype=torch.float).unsqueeze(-1)
        ids = ids.unsqueeze(0)

        n_axes = ids.size(-1)
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class EmbedND2DMaker(EmbedND):
    def forward(self, height: int, width: int, device=None) -> Tensor:
        ids = torch.stack(
            torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float),
                torch.arange(width, device=device, dtype=torch.float),
            ),
            dim=-1,
        )
        ids = ids.flatten(0, 1).unsqueeze(0)

        n_axes = ids.size(-1)
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class EmbedND3DMaker(EmbedND):
    def forward(self, t: int, h: int, w: int, device=None) -> Tensor:
        tt = torch.arange(t, device=device, dtype=torch.float)
        hh = torch.arange(h, device=device, dtype=torch.float)
        ww = torch.arange(w, device=device, dtype=torch.float)

        ids = torch.stack(
            torch.meshgrid(tt, hh, ww, indexing="ij"),
            dim=-1
        )  # (T, H, W, 3)
        ids = ids.reshape(-1, 3).unsqueeze(0)  # (1, T*H*W, 3)

        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class EmbedNDHybrid2DMaker(EmbedND):
    def forward(
        self,
        h: int,
        w: int,
        text_length: int,
        text_first: bool,
        device=None,
    ) -> Tensor:
        """
        Generate RoPE embeddings for a hybrid sequence that includes both vision (H, W) and text tokens.

        Args:
            h (int): Height of the visual grid.
            w (int): Width of the visual grid.
            text_length (int): Number of text tokens.
            text_first (bool): If True, text tokens precede vision tokens; otherwise, they follow.
            device (torch.device, optional): The device on which tensors are allocated.

        Returns:
            Tensor: RoPE embeddings of shape (1, 1, total_len, head_dim//2, 2, 2)
        """

        # 1. Generate position indices for vision tokens: (h, w, 2) -> (h*w, 2)
        hh = torch.arange(h, device=device, dtype=torch.float)
        ww = torch.arange(w, device=device, dtype=torch.float)

        vision_ids = torch.stack(torch.meshgrid(hh, ww), dim=-1).reshape(-1, 2) # shape: (vision_len, 2)

        # 2. Generate position indices for text tokens: repeat scalar pos along 2 axes
        if text_length > 0:
            text_ids = torch.arange(text_length, device=device, dtype=torch.float).view(-1, 1)
            text_ids = text_ids.repeat(1, 2)  # shape: (text_len, 2)
        else:
            text_ids = torch.empty(0, 2, device=device)

        # 3. Apply proper positional offset
        if text_first:
            # Text appears first: shift vision IDs by text length
            vision_ids = vision_ids + text_length
            all_ids = torch.cat([text_ids, vision_ids], dim=0)  # (total_len, 2)
        else:
            # Vision appears first: shift text IDs by max vision ID + 1
            max_vid = vision_ids.max() if vision_ids.numel() > 0 else 0
            text_ids = text_ids + (max_vid + 1)
            all_ids = torch.cat([vision_ids, text_ids], dim=0)  # (total_len, 2)

        all_ids = all_ids.unsqueeze(0)  # (1, total_len, 2)

        # 4. Generate RoPE embeddings for each axis and concatenate along channel dimension
        emb = torch.cat(
            [rope(all_ids[..., i], self.axes_dim[i], self.theta) for i in range(2)],
            dim=-3,  # concatenating along num_heads
        )  # shape: (1, total_len, head_dim//2, 2, 2)

        return emb.unsqueeze(1)  # shape: (1, 1, total_len, head_dim//2, 2, 2)


class EmbedNDHybridMulti2DMaker(EmbedND):
    def forward(
        self,
        hw_list: list[tuple[int, int]],  # [(h1, w1), (h2, w2), ...]
        text_length: int,
        text_first: bool,
        device=None,
    ) -> Tensor:
        """
        Generate RoPE embeddings for a hybrid sequence that includes:
        - multiple vision grids (each HxW)
        - optional text tokens

        Args:
            hw_list (list[tuple[int, int]]): List of (height, width) for each vision grid.
            text_length (int): Number of text tokens.
            text_first (bool): If True, text tokens precede vision tokens; otherwise, they follow.
            device (torch.device, optional): The device on which tensors are allocated.

        Returns:
            Tensor: RoPE embeddings of shape (1, 1, total_len, head_dim//2, 2, 2)
        """

        # --- Step 1: Initialize position IDs
        all_ids_list = []
        offset = 0

        # (a) Add text tokens first if text_first=True
        if text_first and text_length > 0:
            text_ids = torch.arange(text_length, device=device, dtype=torch.float).view(-1, 1)
            text_ids = text_ids.repeat(1, 2)  # (text_len, 2)
            all_ids_list.append(text_ids)
            offset += text_length

        # (b) Process each 2D visual block
        for (h, w) in hw_list:
            hh = torch.arange(h, device=device, dtype=torch.float)
            ww = torch.arange(w, device=device, dtype=torch.float)
            vision_ids = torch.stack(torch.meshgrid(hh, ww), dim=-1).reshape(-1, 2)  # (h*w, 2)
            vision_ids = vision_ids + offset
            all_ids_list.append(vision_ids)
            offset += vision_ids.numel() // 2  # increase by h*w

        # (c) Add text tokens at the end if text_first=False
        if not text_first and text_length > 0:
            text_ids = torch.arange(text_length, device=device, dtype=torch.float).view(-1, 1)
            text_ids = text_ids.repeat(1, 2)
            text_ids = text_ids + offset
            all_ids_list.append(text_ids)
            offset += text_length

        # --- Step 2: Concatenate all positions
        all_ids = torch.cat(all_ids_list, dim=0).unsqueeze(0)  # shape: (1, total_len, 2)

        # --- Step 3: Generate RoPE embeddings per axis
        emb = torch.cat(
            [rope(all_ids[..., i], self.axes_dim[i], self.theta) for i in range(2)],
            dim=-3,
        )  # shape: (1, total_len, head_dim/2, 2, 2)

        return emb.unsqueeze(1)  # shape: (1, 1, total_len, head_dim/2, 2, 2)


class EmbedNDHybrid3DMaker(EmbedND):
    def forward(
        self,
        t: int,
        h: int,
        w: int,
        text_length: int,
        text_first: bool,
        device=None,
    ) -> Tensor:
        """
        Generate RoPE embeddings for a hybrid sequence that includes both vision (T, H, W) and text tokens.

        Args:
            t (int): Temporal length (number of frames). Set t=1 for 2D images.
            h (int): Height of the visual grid.
            w (int): Width of the visual grid.
            text_length (int): Number of text tokens.
            text_first (bool): If True, text tokens precede vision tokens; otherwise, they follow.
            device (torch.device, optional): The device on which tensors are allocated.

        Returns:
            Tensor: RoPE embeddings of shape (1, 1, total_len, head_dim//3, 3, 2)
        """

        # 1. Generate position indices for vision tokens: (t, h, w, 3) -> (t*h*w, 3)
        tt = torch.arange(t, device=device, dtype=torch.float)
        hh = torch.arange(h, device=device, dtype=torch.float)
        ww = torch.arange(w, device=device, dtype=torch.float)

        vision_ids = torch.stack(
            torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1
        ).reshape(-1, 3)  # shape: (vision_len, 3)

        # 2. Generate position indices for text tokens: repeat scalar pos along 3 axes
        if text_length > 0:
            text_ids = torch.arange(text_length, device=device, dtype=torch.float).view(-1, 1)
            text_ids = text_ids.repeat(1, 3)  # shape: (text_len, 3)
        else:
            text_ids = torch.empty(0, 3, device=device)

        # 3. Apply proper positional offset
        if text_first:
            # Text appears first: shift vision IDs by text length
            vision_ids = vision_ids + text_length
            all_ids = torch.cat([text_ids, vision_ids], dim=0)  # (total_len, 3)
        else:
            # Vision appears first: shift text IDs by max vision ID + 1
            max_vid = vision_ids.max() if vision_ids.numel() > 0 else 0
            text_ids = text_ids + (max_vid + 1)
            all_ids = torch.cat([vision_ids, text_ids], dim=0)  # (total_len, 3)

        all_ids = all_ids.unsqueeze(0)  # (1, total_len, 3)

        # 4. Generate RoPE embeddings for each axis and concatenate along channel dimension
        emb = torch.cat(
            [rope(all_ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            dim=-3,  # concatenating along num_heads
        )  # shape: (1, total_len, head_dim//3, 3, 2)

        return emb.unsqueeze(1)  # shape: (1, 1, total_len, head_dim//3, 3, 2)
