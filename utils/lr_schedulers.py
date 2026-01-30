# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/utils/lr_schedulers.py

import math
import numpy as np
from enum import Enum
from typing import Optional, Union

import torch


class SchedulerType(Enum):
    COSINE = "cosine"
    CONSTANT = "constant"
    WSD = "wsd"


def _warmup_lr(base_lr, warmup_length, step, init_div_factor=100):
    ratio = (step / warmup_length) + (1 - step / warmup_length) / init_div_factor
    return base_lr * ratio
    # return base_lr * (step + 1) / warmup_length


def get_wsd_schedule_with_warmup(
        optimizer: torch.optim.Optimizer, 
        num_warmup_steps: int,
        num_training_steps: int,
        base_lr: float = 1e-4,
        end_lr: float = 0.0,
        last_epoch: int = -1,
        final_lr_factor=None,
        init_div_factor=100,
        fract_decay=0.2,
        decay_type="sqrt",
    ):
    """
    Adapted from https://github.com/epfml/schedules-and-scaling/src/optim/utils.py
    This is a function that returns a function that adjusts the learning rate of the optimizer.
    Args:
        num_training_steps: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        num_warmup_steps: length of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    """
    n_anneal_steps = int(fract_decay * num_training_steps)
    n_hold = num_training_steps - n_anneal_steps

    if final_lr_factor is None:
        if end_lr > 0:
            final_lr_factor = max(end_lr / base_lr, 0.0)
        else:
            final_lr_factor = 0.0

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            lr = _warmup_lr(base_lr, num_warmup_steps, current_step, init_div_factor=init_div_factor)
        elif current_step < n_hold:
            lr = base_lr
        else:
            if decay_type == "linear":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (current_step - n_hold) / n_anneal_steps
                )
                lr = base_lr * lr_factor

            elif decay_type == "exp":
                lr = final_lr_factor ** ((current_step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                lr = base_lr * (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + np.cos(np.pi * (current_step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "square":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * max(
                    1 - ((current_step - n_hold) / n_anneal_steps) ** 2, 0
                )

                lr = base_lr * lr_factor

            elif decay_type == "sqrt":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * max(
                    1 - np.sqrt((current_step - n_hold) / n_anneal_steps), 0
                )

                lr = base_lr * lr_factor

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Creates a cosine learning rate schedule with warm-up and ending learning rate.

    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """UViT: Creates a constant learning rate schedule with warm-up.

    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule_with_warmup,
    SchedulerType.WSD: get_wsd_schedule_with_warmup,
}

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Retrieves a learning rate scheduler from the given name and optimizer.

    Args:
        name: A string or SchedulerType, the name of the scheduler to retrieve.
        optimizer: torch.optim.Optimizer. The optimizer to use with the scheduler.
        num_warmup_steps: An integer, the number of warmup steps.
        num_training_steps: An integer, the total number of training steps.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Returns:
        A instance of torch.optim.lr_scheduler.LambdaLR

    Raises:
        ValueError: If num_warmup_steps or num_training_steps is not provided.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        base_lr=base_lr,
        end_lr=end_lr,
    )