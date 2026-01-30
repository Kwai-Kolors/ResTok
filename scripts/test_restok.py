# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/scripts/train_titok.py

import math
import os
import pprint
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config,
    get_restok_tokenizer,
    create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    eval_reconstruction)


def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # for deterministic inference
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    # tracker = "tensorboard"
    # if config.training.enable_wandb:
    #     tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        # log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="ResTok", log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers(
        #     project_name=config.experiment.project,
        #     init_kwargs={
        #         "tensorboard": {
        #             "log_dir": f"./tb_logs/{config.experiment.project}/runs/{config.experiment.name}",
        #         },
        #         "wandb": {
        #             "name": config.experiment.name,
        #         },
        #         "comet_ml": {
        #             "experiment_name": config.experiment.name,
        #         },
        #     },
        # )
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    accelerator.wait_for_everyone()

    model = get_restok_tokenizer(config)

    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model = accelerator.prepare(model)

    # Start training.
    logger.info("***** Running test *****")

    with torch.no_grad():
        eval_scores = eval_reconstruction(model, eval_dataloader, accelerator, evaluator, len_dataset=50_000)
    logger.info(pprint.pformat(eval_scores))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()