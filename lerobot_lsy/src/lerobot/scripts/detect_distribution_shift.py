import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
from dataclasses import dataclass
from pathlib import Path
import copy

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.datasets.factory import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.optimizers import OptimizerConfig, AdamWConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.schedulers import LRSchedulerConfig, LRScheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed, load_rng_state
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    load_training_step,
    load_optimizer_state,
    load_scheduler_state
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


@dataclass
class DetectShiftConfig:
    detect_disctribution_shift_steps:int = 200
    detect_disctribution_shift_batch_size: int = 32
    detect_disctribution_shift_num_workers: int = 16
    detect_disctribution_shift_log_freq:int = 10


def detect_distribution_shift(cfg: DetectShiftConfig,
                              wandb_logger: WandBLogger,
                              global_steps: int,
                              policy: PreTrainedPolicy, 
                              peft_modules: list,
                              dataset: LeRobotDataset, 
                              device):

    infer_metrics = {
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        peft_module.track_z_score(True)
        for discriminator_id in range(peft_module.num_discriminators):
            key = peft_module.layer_name + "_" + peft_module.layer_id + "_" + discriminator_id

            infer_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            infer_metrics[f"z_score_{key}"] = AverageMeter(f"z_score_{key}", ":.3f")
    
    detect_tracker = MetricsTracker(
        cfg.detect_disctribution_shift_batch_size, dataset.num_frames, dataset.num_episodes, infer_metrics, initial_step=0
    )

    detect_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.detect_disctribution_shift_num_workers,
        batch_size=cfg.detect_disctribution_shift_batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    detect_iter = cycle(detect_dataloader)

    policy.eval()

    z_scores_sum = {}
    losses_sum = {}

    step = 0

    # infer on new dataset only for 1 epoch
    for _ in range(cfg.detect_disctribution_shift_steps):
        batch = next(detect_iter)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        with torch.inference_mode():
            _, _ = policy.forward(batch)


        for peft_module in peft_modules:
            state_dict = peft_module.state_dict
            for discriminator_id in range(peft_module.num_discriminators):

                discriminator_state_dict = state_dict[f"discriminator_{discriminator_id}"]

                loss = discriminator_state_dict["loss"].item()
                z_score = discriminator_state_dict["z_score"].item()

                key = peft_module.layer_name + "_" + peft_module.layer_id

                if key in z_scores_sum:
                    if len(z_scores_sum[key]) < discriminator_id:
                        z_scores_sum[key].append(z_score)
                        losses_sum[key].append(loss)
                    else:
                        z_scores_sum[key][discriminator_id] += z_score
                        losses_sum[key][discriminator_id] += loss
                else:
                    z_scores_sum[key] = [z_score]
                    losses_sum[key] = [loss]

                log_key = key  + "_" + discriminator_id
                detect_tracker.__setattr__(f"loss_{log_key}", loss)
                detect_tracker.__setattr__(f"z_score_{log_key}", z_score)

        step += 1
        detect_tracker.step()
        is_log_step = cfg.detect_disctribution_shift_log_freq > 0 and step % cfg.detect_disctribution_shift_log_freq == 0

        if is_log_step:
            logging.info(detect_tracker)
            if wandb_logger:
                wandb_log_dict = detect_tracker.to_dict()
                wandb_step = step
                if global_steps > 0:
                    wandb_step += global_steps
                wandb_logger.log_dict(wandb_log_dict, wandb_step, mode='continual_learning')
            detect_tracker.reset_averages()

    z_scores_mean = {}
    losses_mean = {}

    for peft_module in peft_modules:
        peft_module.track_z_score(False)

        key = peft_module.layer_name + "_" + peft_module.layer_id

        z_scores_mean_current_layer = torch.tensor(z_scores_sum[key], device="cpu") / step
        losses_mean_current_layer = torch.tensor(losses_sum[key], device="cpu") / step

        logging.info(f"Distribution shift of {key}")
        logging.info(f"Average z_scores: {z_scores_mean_current_layer.tolist()}")
        logging.info(f"Average losses: {losses_mean_current_layer.tolist()}")

        z_scores_mean[key] = z_scores_mean_current_layer
        losses_mean[key] = losses_mean_current_layer

    return z_scores_mean, losses_mean, global_steps + step