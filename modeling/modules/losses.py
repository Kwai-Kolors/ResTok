# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/losses.py

from itertools import chain
from typing import Mapping, Text, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

from timm.layers.helpers import to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator
from .discriminator_dino import DinoDisc
from .diff_aug import DiffAugment


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class ReconstructionLoss(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.dino_disc = loss_config.get("dino_disc", False)
        if self.dino_disc:
            self.discriminator = DinoDisc(norm_type="bn")
            self.daug = DiffAugment
        else:
            self.discriminator = NLayerDiscriminator()
            self.daug = None

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        # vf loss
        self.use_vf = config.model.vq_model.get("use_vf", None)
        self.enc_feat_sup = config.model.vq_model.get("encoder_feature_supervision", True) and (config.model.vq_model.get("use_vf", None) is not None)
        self.dec_feat_sup = config.model.vq_model.get("decoder_feature_supervision", False) and (config.model.vq_model.get("use_vf", None) is not None)
        if self.use_vf:
            self.vf_weight = loss_config.get("vf_weight", 1)
            self.cos_margin = loss_config.get("cos_margin", 0)
            self.cos_weight = loss_config.get("cos_weight", 1)

        self.config = config

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        weighted_reconstruction_loss = reconstruction_loss * self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions, training=True).mean()

        # vf loss
        vf_loss = None
        cos_sim = None
        cos_sim_dec = None
        if self.use_vf:
            if self.enc_feat_sup:
                cos_sim = torch.nn.functional.cosine_similarity(
                    extra_result_dict["aux_cls"].mean(dim=(2,3)),
                    extra_result_dict["last_image_tokens_proj"].mean(dim=(2,3)) if "last_image_tokens_proj" in extra_result_dict else extra_result_dict["last_image_tokens"].mean(dim=(2,3)),
                )
                vf_loss = torch.nn.functional.relu(1 - self.cos_margin - cos_sim).mean() if vf_loss is None else vf_loss + torch.nn.functional.relu(1 - self.cos_margin - cos_sim).mean()
            else:
                cos_sim = None
            if self.dec_feat_sup:
                cos_sim_dec = torch.nn.functional.cosine_similarity(
                    extra_result_dict["aux_feature"],
                    extra_result_dict["dec_vf_feat_proj"] if "dec_vf_feat_proj" in extra_result_dict else extra_result_dict["dec_vf_feat"]
                )
                vf_loss = torch.nn.functional.relu(1 - self.cos_margin - cos_sim_dec).mean() if vf_loss is None else vf_loss + torch.nn.functional.relu(1 - self.cos_margin - cos_sim_dec).mean()
            else:
                cos_sim_dec = None

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            if self.daug is not None:
                reconstructions = self.daug(reconstructions, policy='color,translation,cutout_0.2', prob=0.5)
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (
            weighted_reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )

        if self.use_vf:
            total_loss = total_loss + self.vf_weight * vf_loss

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            weighted_reconstruction_loss=weighted_reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            entropy_loss=extra_result_dict["entropy_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
            dropout_remain=torch.tensor(
                float(extra_result_dict["nested_dropout_remain"] if isinstance(extra_result_dict["nested_dropout_remain"], int) else sum(extra_result_dict["nested_dropout_remain"])/len(extra_result_dict["nested_dropout_remain"])),
                device=reconstructions.device,
            ),
        )

        if vf_loss is not None:
            loss_dict["vf_loss"] = (self.vf_weight * vf_loss).detach()
        if cos_sim is not None:
            loss_dict["vf_cos"] = (cos_sim.mean()).detach()
            if extra_result_dict["aux_cls"].shape[1] == extra_result_dict["last_image_tokens"].shape[1]:
                real_cos_sim = torch.nn.functional.cosine_similarity(
                    extra_result_dict["aux_cls"].mean(dim=(2,3)),
                    extra_result_dict["last_image_tokens"].mean(dim=(2,3)),
                )
                loss_dict["vf_real_cos"] = (real_cos_sim.mean()).detach()
        if cos_sim_dec is not None:
            loss_dict["vf_cos_dec"] = (cos_sim_dec.mean()).detach()
            if extra_result_dict["aux_feature"].shape[1] == extra_result_dict["dec_vf_feat"].shape[1]:
                real_cos_sim_dec = torch.nn.functional.cosine_similarity(
                    extra_result_dict["aux_feature"],
                    extra_result_dict["dec_vf_feat"],
                )
                loss_dict["vf_real_cos_dec"] = (real_cos_sim_dec.mean()).detach()

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        reconstructions = reconstructions.detach()
        if self.daug is not None:
            real_images = self.daug(real_images, policy='color,translation,cutout_0.2', prob=0.5)
            reconstructions = self.daug(reconstructions, policy='color,translation,cutout_0.2', prob=0.5)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions)

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits.permute(0, 2, 1).contiguous() if logits.shape[-2] == labels.shape[-1] else logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        loss = loss.mean(dim=-1).mean()
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
