# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/utils/train_utils.py

import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
import open_clip
from timm.utils import accuracy, AverageMeter

from dataset import SimpleImageDataset, PretokenizedDataSetJSONL, PretokenizedWebDataset
from dataset.imagenet import CustomDataset
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from torch.optim import AdamW, SGD
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss, ARLoss
from modeling.restok import ResTok
from autoregressive.models.gpt_vanilla import GPT_models as GPT_models_vanilla
from autoregressive.models.gpt_har import GPT_models as GPT_models_har
from evaluator import VQGANEvaluator, DistributedVQGANEvaluator
from demo_util import get_restok_tokenizer, sample_fn

from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation, make_viz_from_samples_t2i_generation
from torchinfo import summary
from tqdm import tqdm


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_clip_model():
    clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip.visual
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip.transformer.batch_first = False
    clip.eval()
    clip.requires_grad_(False)
    return clip, tokenizer


def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="restok"):
    """Creates ResTok model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "restok":
        model_cls = ResTok
        loss_cls = ReconstructionLoss
    elif model_type == "llamagen":
        if config.model.generator.get("har", False):
            model_cls = GPT_models_har[config.model.generator.model_type]
        else:
            model_cls = GPT_models_vanilla[config.model.generator.model_type]
        loss_cls = ARLoss
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.model.vq_model.get("pretrained_vf_weight", ""):
        if "dinov3" in config.model.vq_model.pretrained_vf_weight:
            from safetensors.torch import load_file
            pretrained_vf_weight = load_file(
                config.model.vq_model.pretrained_vf_weight, device="cpu",
            )
            if config.model.vq_model.get("vae_encoder", True):
                model_weight = {"encoder." + k:v for k,v in pretrained_vf_weight.items() if ("blocks." in k and int(k.split(".")[1]) < config.model.vq_model.get("pretrained_vf_layers", 9999))}
            else:
                model_weight = {"encoder." + k:v for k,v in pretrained_vf_weight.items() if (("blocks." in k and int(k.split(".")[1]) < config.model.vq_model.get("pretrained_vf_layers", 9999)) or "patch_embed." in k)}
            msg = model.load_state_dict(model_weight, strict=False)
            logger.info(f"loading vf weight from {config.model.vq_model.pretrained_vf_weight}, msg: {msg}")
        del model_weight

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=config.training.get("ema_decay", 0.999),
                             model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config) if loss_cls is not None else None

    # Print Model for sanity check.
    if accelerator.is_main_process:
        if model_type in ["restok"]:
            input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
            model_summary_str = summary(
                model, input_size=input_size, depth=5,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"))
            logger.info(model_summary_str)
        elif model_type in ["llamagen"]:
            input_size = (1, config.model.vq_model.num_latent_tokens)
            input_data = [
                torch.randint(0, config.model.vq_model.codebook_size, input_size),
                torch.ones(1, dtype=int)
            ]
            model_summary_str = summary(
                model, input_data=input_data, depth=7,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"))
            logger.info(model_summary_str)
        else:
            raise NotImplementedError

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="restok", need_discrminator=True):
    """Creates optimizer for ResTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    elif optimizer_type == "sgd":
        optimizer_cls = SGD
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n
               or 'mask_latent_token' in n or 'rope' in n or 'freqs_cis' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    if optimizer_type == "adamw":
        optimizer = optimizer_cls(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        optimizer = optimizer_cls(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=learning_rate,
            momentum=optimizer_config.momentum
        )

    if need_discrminator:
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for ResTok and discrminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes, # a bug in hf accelerator, should be multiplied by num_processes to get the correct max_train_steps when using DDP
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=(config.training.max_train_steps - config.losses.discriminator_start) * accelerator.num_processes,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # T2I uses a pretokenized dataset for speed-up.
    if dataset_config.get("pretokenization", "") != "" and dataset_config.get("dataset_with_text_label", False) is True:
        dataset = PretokenizedWebDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
            random_resized_crop=preproc_config.get("random_resized_crop", None),
            resize_shorter_edge=preproc_config.resize_shorter_edge,
            crop_size=preproc_config.crop_size,
            random_crop=preproc_config.random_crop,
            random_flip=preproc_config.random_flip,
            normalize_mean=preproc_config.normalize_mean,
            normalize_std=preproc_config.normalize_std,
            process_recap=preproc_config.get("preproc_recap", True),
            use_recap_prob=preproc_config.get("use_recap_prob", 0.95)
        )
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    # SimpleImageDataset without pretokenization
    elif dataset_config.get("pretokenization", "") == "" and dataset_config.get("dataset_with_text_label", False) is False:
        dataset = SimpleImageDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
            random_resized_crop=preproc_config.get("random_resized_crop", None),
            resize_with_crop=preproc_config.get("resize_with_crop", None),
            resize_shorter_edge=preproc_config.resize_shorter_edge,
            crop_size=preproc_config.crop_size,
            random_crop=preproc_config.random_crop,
            random_flip=preproc_config.random_flip,
            dataset_with_class_label=dataset_config.get("dataset_with_class_label", True),
            dataset_with_text_label=dataset_config.get("dataset_with_text_label", False),
            res_ratio_filtering=preproc_config.get("res_ratio_filtering", False),
        )
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    # potentially, use a pretokenized dataset for ImageNet speed-up.
    else:
        if dataset_config.get("pretokenization", ""):
            # train_dataloader = DataLoader(
            #     PretokenizedDataSetJSONL(dataset_config.pretokenization),
            #     batch_size=config.training.per_gpu_batch_size,
            #     shuffle=True, drop_last=True, pin_memory=True)
            train_dataset = CustomDataset(os.path.join(dataset_config.pretokenization, "imagenet256_codes"), os.path.join(dataset_config.pretokenization, "imagenet256_labels"))
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.training.per_gpu_batch_size,
                shuffle=True,
                num_workers=dataset_config.num_workers_per_gpu,
                pin_memory=True,
                drop_last=True,
            )
            train_dataloader.num_batches = math.ceil(
                config.experiment.max_train_examples / total_batch_size_without_accum)
            eval_dataloader = None

    return train_dataloader, eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    if config.model.vq_model.get("quantize_mode", "vq") == "vq":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=True,
            enable_codebook_entropy_measure=True,
            num_codebook_entries=config.model.vq_model.codebook_size
        )
    elif config.model.vq_model.get("quantize_mode", "vq") == "vae":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=False,
            enable_codebook_entropy_measure=False,
        )
    else:
        raise NotImplementedError
    return evaluator


def create_distributed_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    if config.model.vq_model.get("quantize_mode", "vq") == "vq":
        evaluator = DistributedVQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=True,
            enable_codebook_entropy_measure=True,
            num_codebook_entries=config.model.vq_model.codebook_size,
            is_distributed=True,
        )
    elif config.model.vq_model.get("quantize_mode", "vq") == "vae":
        evaluator = DistributedVQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=False,
            enable_codebook_entropy_measure=False,
            is_distributed=True,
        )
    else:
        raise NotImplementedError
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    model_type="restok"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)

    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([model, loss_module]):
            if model_type in ["restok"]:
                reconstructed_images, extra_results_dict = model(images)
                autoencoder_loss, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator",
                )
            else:
                raise NotImplementedError

            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
                discriminator_logs["lr/discriminator"] = discriminator_lr_scheduler.get_last_lr()[0]

                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)

                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]

                log_msg = (
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                )
                if config.losses.get("perceptual_loss", None) is not None:
                    log_msg = log_msg + f"Perc Loss: {autoencoder_logs['train/perceptual_loss']:0.4f} "
                if config.model.vq_model.get("use_vf", None):
                    log_msg = log_msg + f"VF Loss: {autoencoder_logs['train/vf_loss']:0.4f} "
                logger.info(log_msg)
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                reconstruct_images(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())


            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        model_type=model_type,
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        model_type=model_type,
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= min(config.training.max_train_steps, config.training.get("early_stop_steps", config.training.max_train_steps)):
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break

            if global_step >= config.training.get("early_stop_steps", config.training.max_train_steps):
                accelerator.print(
                    f"Finishing training: Global step is >= Early stop steps: {global_step} >= {config.training.get('early_stop_steps', config.training.max_train_steps)}"
                )
                break


    return global_step


def train_one_epoch_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    global_step,
                    model_type="llamagen"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        if config.dataset.params.get("pretokenization", ""):
            # the data is already pre-tokenized
            input_tokens, conditions = batch
            input_tokens = input_tokens.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            conditions = conditions.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            # tokenize on the fly
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                conditions = batch["class_id"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )

                # Encode images on the flight.
                with torch.no_grad():
                    tokenizer.eval()
                    result_dict = tokenizer.encode(images)[1]
                    input_tokens = result_dict["min_encoding_indices"].reshape(images.shape[0], -1)
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)


        if model_type == "maskgit":
            # Randomly masking out input tokens.
            masked_tokens, masks = unwrap_model.masking_input_tokens(
                input_tokens)
        elif model_type in ["llamagen"]:
            pass
        else:
            raise NotImplementedError


        with accelerator.accumulate([model]):

            if model_type == "maskgit":
                logits = model(masked_tokens, conditions,
                            cond_drop_prob=config.model.generator.class_label_dropout)
                loss, loss_dict= loss_module(logits, input_tokens, weights=masks)
            if model_type in ["llamagen"]:
                logits, _ = model(input_tokens, conditions)
                loss, loss_dict = loss_module(logits, input_tokens)
            # Gather the losses across all processes for logging.
            gen_logs = {}
            for k, v in loss_dict.items():
                gen_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {gen_logs['train/loss']:0.4f} "
                    f"Accuracy: {gen_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break

            if global_step >= config.training.get("early_stop_steps", config.training.max_train_steps):
                accelerator.print(
                    f"Finishing training: Global step is >= Early stop steps: {global_step} >= {config.training.get('early_stop_steps', config.training.max_train_steps)}"
                )
                break


    return global_step


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
    model_type="restok",
    len_dataset=None,
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        try:
            total_images = len(eval_loader.dataset)
        except:
            try:
                total_images = len(eval_loader) * eval_loader.batch_size * accelerator.num_processes
            except:
                total_images = len_dataset
        pbar = tqdm(total=total_images, desc="Eval", unit="img", dynamic_ncols=True)
    else:
        pbar = None

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )

        original_images = torch.clone(images)
        if model_type in ["restok"]:
            reconstructed_images, model_dict = local_model(images)
        else:
            raise NotImplementedError

        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        # Quantize to uint8
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        original_images = torch.clamp(original_images, 0.0, 1.0)

        if isinstance(model_dict, dict): 
            # For VQ model.
            evaluator.update(original_images, reconstructed_images.squeeze(2), model_dict["test_min_encoding_indices"] if "test_min_encoding_indices" in model_dict else model_dict["min_encoding_indices"])
        else:
            # For VAE model.
            evaluator.update(original_images, reconstructed_images.squeeze(2), None)

        if pbar is not None:
            pbar.update(images.size(0) * accelerator.num_processes)

    if pbar is not None:
        pbar.close()

    model.train()
    return evaluator.result()


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    model_type="restok"):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)

    if model_type in ["restok"]:
        reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)[0]
    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


@torch.no_grad()
def generate_images(model, tokenizer, accelerator, 
                    global_step, output_dir, logger, config=None):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    generated_image = sample_fn(
        accelerator.unwrap_model(model),
        tokenizer,
        guidance_scale=config.model.generator.get("guidance_scale", 3.0),
        guidance_decay=config.model.generator.get("guidance_decay", "constant"),
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0),
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0),
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False),
        num_sample_steps=config.model.generator.get("num_steps", 8),
        device=accelerator.device,
        return_tensor=True
    )
    images_for_saving, images_for_logging = make_viz_from_samples_generation(
        generated_image)

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model, keep_torch_compile=False)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(unwrapped_model), # state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)
