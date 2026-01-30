# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/quantizer/quantizer.py

from typing import Mapping, Text, Tuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from accelerate.utils.operations import gather
from torch.cuda.amp import autocast


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """
    modified from llamagen and magvit
    Args:
        affinity: (b, n, n), the affinity matrix, where affinity[i, j] is the affinity 
                between encoed vector i and codebook vector j
        loss_type: how to turn the affinity into probability distribution
    """
    # shape: (b n) n
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    # target_probs.shape: (b, n, n), and sum(target_probs, dim=-1) = 1
    avg_probs = torch.mean(target_probs, dim=0) # (,n)
    # average entropy corresponeds (negatively) to the diversity of indices for a single position
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    # sample entropy is the confidence for the quantization process
    # (bn, n) -> (bn) -> avg 
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 entropy_loss_ratio: float = 0.0,
                 use_l2_norm: bool = False,
                 clustering_vq: bool = False,
                 ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.entropy_loss_ratio = entropy_loss_ratio

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')
        unnormed_z_flattened = z_flattened

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        if self.clustering_vq and self.training:
            with torch.no_grad():
                # Gather distance matrix from all GPUs.
                encoding_indices = gather(min_encoding_indices)
                if len(min_encoding_indices.shape) != 1:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook.
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                avg_probs = torch.mean(encodings, dim=0)
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay)
                # Closest sampling to update the codebook.
                all_d = gather(d)
                all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                indices = torch.argmin(all_d, dim=0)
                random_feat = all_unnormed_z_flattened[indices]
                # Decay parameter based on the average usage.
                decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) /
                                   (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.token_size)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
        loss = loss + entropy_loss

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy_loss=entropy_loss,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]),
            distances=rearrange(d, '(b h w) n -> b n h w', b=z.shape[0], h=z.shape[1], w=z.shape[2]),
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized
