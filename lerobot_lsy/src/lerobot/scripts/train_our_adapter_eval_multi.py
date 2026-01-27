#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
import copy
import re

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.optimizers import OptimizerConfig, AdamWConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.schedulers import LRSchedulerConfig, LRScheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
# from lerobot.scripts.eval import eval_policy
from lerobot.scripts.eval_peft import eval_policy_with_env_init
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

from peft import get_peft_model, PeftConfig, PeftModel
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

class PeftWrapperPolicy(torch.nn.Module):
    policy: PreTrainedPolicy

    def __init__(self, policy: PreTrainedPolicy):
        super().__init__()
        self._policy = policy

    @property
    def policy(self):
        return self.policy


@dataclass
class PEFTTrainPipelineConfig(TrainPipelineConfig):
    peft_cfg_path: Path | None = None
    peft_weight_path: Path | None = None

    detect_disctribution_shift_steps:int = 200
    detect_disctribution_shift_batch_size: int = 32
    detect_disctribution_shift_num_workers: int = 16
    detect_disctribution_shift_log_freq:int = 10
    
    train_discriminators_steps: int = 2000
    train_discriminators_batch_size: int = 32
    train_discriminators_num_workers: int = 16
    train_discriminators_log_freq: int = 50
    train_discriminators_save_freq: int = 2000
    train_discriminators_eval_freq: int = 2000
    train_discriminator_optimizer: OptimizerConfig = \
        AdamWConfig(
            lr = 0.0005,
            weight_decay=0.01,
            grad_clip_norm=10.0,
            betas=(0.9,0.999),
            eps=1e-08
        )
    train_discriminator_lr_scheduler: LRSchedulerConfig = None

    maximum_expand: int = 100
    expand_threshold: float = 0.0
    at_least_expand: Literal["shallowest", "deepest"] = field(
        default="shallowest", metadata={"help": "At least expand which layer. Can be 'shallowest' or 'deepest'"}
    )

    max_episodes_rendered: int = 4

    def __post_init__(self):
        assert self.peft_cfg_path or self.peft_weight_path, "One from (peft_cfg_path,peft_weight_path) must be specified"


def set_peft_module_train(peft_modules:list, train: bool = True):
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_modules[0].peft_config.peft_type]
    for peft_module in peft_modules:
        for name, module in peft_module.named_modules():
            if prefix in name or name == '':
                module.train(train)
            if 'base_layer' in name:
                module.train(False)
    return peft_modules


def detect_distribution_shift(cfg: PEFTTrainPipelineConfig,
                              wandb_logger: WandBLogger,
                              global_steps: int,
                              policy: PreTrainedPolicy, 
                              peft_modules: list,
                              dataset, 
                              device):

    infer_metrics = {
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        peft_module.track_z_score(True)
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"

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
            info_dicts = peft_module.info_dicts
            for discriminator_id in range(peft_module.num_discriminators):

                discriminator_info_dict = info_dicts[f"discriminator_{discriminator_id}"]

                loss = discriminator_info_dict["loss"].mean().item()
                z_score = discriminator_info_dict["z_score"].mean().item()

                key = f"{peft_module.layer_name}.{peft_module.layer_id}"

                if key in z_scores_sum:
                    if len(z_scores_sum[key]) <= discriminator_id:
                        z_scores_sum[key].append(z_score)
                        losses_sum[key].append(loss)
                    else:
                        z_scores_sum[key][discriminator_id] += z_score
                        losses_sum[key][discriminator_id] += loss
                else:
                    z_scores_sum[key] = [z_score]
                    losses_sum[key] = [loss]

                log_key = f"{key}.{discriminator_id}"
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

        key = f"{peft_module.layer_name}.{peft_module.layer_id}"

        z_scores_mean_current_layer = torch.tensor(z_scores_sum[key], device="cpu") / step
        losses_mean_current_layer = torch.tensor(losses_sum[key], device="cpu") / step

        logging.info(f"Distribution shift of {key}")
        logging.info(f"Average z_scores: {[f'{z_score:.4f}' for z_score in z_scores_mean_current_layer.tolist()]}")
        logging.info(f"Average losses: {[f'{loss:.4f}' for loss in losses_mean_current_layer.tolist()]}")

        z_scores_mean[key] = z_scores_mean_current_layer
        losses_mean[key] = losses_mean_current_layer

    return z_scores_mean, losses_mean, global_steps + step


def load_discriminator_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the info_dict to.
        scheduler (LRScheduler | None): The scheduler to load the info_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            info_dict loaded.
    """
    training_state_dir = checkpoint_dir / "discriminator_training_state"
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    peft_modules: list,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    # policy.train()
    peft_modules = set_peft_module_train(peft_modules)
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        policy_loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        if peft_modules[0]._train_discriminator:
            discriminators_loss = []
            for peft_module in peft_modules:
                # discriminator_id = peft_module._forwarded_discriminator_id
                for discriminator_id in range(peft_module.num_discriminators):
                    discriminator_info_dict = peft_module.info_dicts[f"discriminator_{discriminator_id}"]

                    
                    discriminator_running_mean = discriminator_info_dict["running_mean"]
                    discriminator_running_std = discriminator_info_dict["running_std"]
                    discriminator_num_batches_tracked = discriminator_info_dict["num_batches_tracked"]

                    key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"
                    
                    train_metrics.__setattr__(f"running_mean_{key}", discriminator_running_mean.item())
                    train_metrics.__setattr__(f"running_std_{key}", discriminator_running_std.item())
                    train_metrics.__setattr__(f"num_batches_tracked_{key}", discriminator_num_batches_tracked.item())

                    if discriminator_id == peft_module._forwarded_discriminator_id:
                        discriminator_loss = discriminator_info_dict["loss"].mean()
                        discriminators_loss.append(discriminator_loss)
                        train_metrics.__setattr__(f"loss_{key}", discriminator_loss.item())
            loss = sum(discriminators_loss)
        else:
            loss = policy_loss

    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: PEFTTrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_envs = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        task_list = [task.strip() for task in cfg.env.task.split(",") if task.strip()]
        if not task_list:
            raise ValueError("No valid tasks found in env.task")

        # env_cfg = copy.deepcopy(cfg.env)

        eval_envs = {}
        for task in task_list:
            env_cfg = copy.deepcopy(cfg.env)
            env_cfg.task = task
            # eval_env = make_env(
            #     env_cfg, 
            #     n_envs=cfg.eval.batch_size, 
            #     use_async_envs=cfg.eval.use_async_envs
            # )
            # eval_envs[task] = eval_env
            eval_envs[task] = env_cfg

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    policy.eval()

    logging.info("Wrapping policy with peft module")

    peft_wrapper_policy = PeftWrapperPolicy(policy=policy)
    
    if cfg.peft_weight_path:
        peft_policy = PeftModel.from_pretrained(peft_wrapper_policy, cfg.peft_weight_path, is_trainable=True, autocast_adapter_dtype=False)
        peft_config = peft_policy.peft_config["default"]
    else:
        peft_cfg = PeftConfig.from_pretrained(cfg.peft_cfg_path)
        peft_cfg.inference_mode = False
        peft_policy = get_peft_model(peft_wrapper_policy, peft_cfg)
        peft_config = peft_policy.peft_config["default"]

    peft_modules = peft_policy.base_model.adapter_layers

    step = 0  # number of policy updates (forward + backward + optim)

    logging.info("Explore new task")

    adapter_params = []
    discriminator_params = []

    # new_task_id start from 0
    new_task_id = peft_config.num_learned_task

    if new_task_id == 0:
        logging.info("Learning the first new task")
        logging.info("Expand all finetuned layers with new adapter and new discriminator")

        for peft_module in peft_modules:
            adapter_param, discriminator_param = \
                peft_module.add_adapter_and_discriminator(new_task_id)
            adapter_params += adapter_param
            discriminator_params += discriminator_param

            key = f"{peft_module.layer_name}.{peft_module.layer_id}"

            peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
            peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1

            peft_config.structure[key] = \
                [peft_module.num_adapters, peft_module.num_discriminators]
            logging.info(f"Add both adapter and discriminator into layer {key}")
            logging.info(f"Only forward adapter id: {peft_module._forwarded_adapter_id}")
            logging.info(f"Only forward discriminator id: {peft_module._forwarded_discriminator_id}")
    else:
        z_scores_mean, losses_mean, step = \
            detect_distribution_shift(
                cfg,
                wandb_logger,
                step,
                policy,
                peft_modules,
                dataset,
                device
            )
        
        only_forward_ids = []
        to_expand_or_not = []
        
        for peft_module in peft_modules:
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            logging.info(f"For layer {key}")

            z_scores_mean_current_layer = z_scores_mean[key]
            losses_mean_current_layer = losses_mean[key]

            closest_discriminator_id = torch.argmin(losses_mean_current_layer).item()
            connected_adapter_id = peft_module.get_adapter_id_by_discriminator_id(closest_discriminator_id)
            only_forward_ids.append(connected_adapter_id)

            expand_the_module = False

            if all(z_scores_mean_current_layer > cfg.expand_threshold):
                logging.info(f"All z-scores in layer {key} exceed threshold {cfg.expand_threshold}")
                logging.info(f"Will try to add new adapter and new discriminator in layer {key}")
                expand_the_module = True
            else:
                logging.info(f"At least one z_score in layer {key} is lower than threshold {cfg.expand_threshold}")
                logging.info(f"Will try to only add new discriminator in layer {key}")
            
            if expand_the_module:
                if sum(to_expand_or_not) < cfg.maximum_expand:
                    logging.info("The number of new adapter is within limit")
                else:
                    logging.info("The number of new adapter reaches the expansion limit")
                    expand_the_module = False

            to_expand_or_not.append(expand_the_module)
            
        if sum(to_expand_or_not) == 0:
            logging.info("No layer have expansion signal. But still expand")
            if cfg.at_least_expand == "shallowest":
                logging.info("Still expand the shallowest layer.")
                to_expand_or_not[0] = True
                only_forward_ids[0] = -1
            elif cfg.at_least_expand == "deepest":
                logging.info("Still expand the deepest layer.")
                to_expand_or_not[-1] = True
                only_forward_ids[-1] = -1
        
        
        for peft_module_id, (to_expand, only_forward_id) in enumerate(zip(to_expand_or_not, only_forward_ids)):
            peft_module = peft_modules[peft_module_id]

            peft_module.train_discriminator(False)
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            logging.info(f"For layer {key}")

            if to_expand:
                adapter_param, discriminator_param = \
                    peft_module.add_adapter_and_discriminator(new_task_id)
                adapter_params += adapter_param
                discriminator_params += discriminator_param

                peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
                logging.info(f"Add both adapter and discriminator into layer {key}")
            else:
                discriminator_param = \
                    peft_module.add_discriminator(only_forward_id, new_task_id)
                discriminator_params += discriminator_param
        
                peft_module._forwarded_adapter_id = only_forward_id
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1

                attached_adapter_task_id = peft_module.our_adapter_func_adapters[peft_module.adapter_name][only_forward_id].func_adapter.task_id.item()
                logging.info(f"Only add discriminator into layer {key}, attatch it with adapter of task_id {attached_adapter_task_id}")
            
            logging.info(f"Only forward adapter id: {peft_module._forwarded_adapter_id}")
            logging.info(f"Only forward discriminator id: {peft_module._forwarded_discriminator_id}")
            peft_module._active_task = new_task_id
            peft_config.structure[key] = \
                [peft_module.num_adapters, peft_module.num_discriminators]

    peft_config.num_learned_task += 1

    logging.info("Creating optimizer and scheduler")
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    adapter_optimizer = cfg.optimizer.build(adapter_params)
    if cfg.scheduler:
        adapter_lr_scheduler = cfg.scheduler.build(adapter_optimizer, cfg.steps)
    else:
        adapter_lr_scheduler = None

    discriminator_optimizer = cfg.train_discriminator_optimizer.build(discriminator_params)
    if cfg.train_discriminator_lr_scheduler:
        discriminator_lr_scheduler = cfg.train_discriminator_lr_scheduler.build(discriminator_optimizer, cfg.steps)
    else:
        discriminator_lr_scheduler = None

    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    if cfg.resume:
        step, adapter_optimizer, adapter_lr_scheduler = load_training_state(cfg.checkpoint_path, adapter_optimizer, adapter_lr_scheduler)
        step, discriminator_optimizer, discriminator_lr_scheduler = load_discriminator_training_state(cfg.checkpoint_path, discriminator_optimizer, discriminator_lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_adapter_params = sum(p.numel() for p in adapter_params)
    num_discriminator_params = sum(p.numel() for p in discriminator_params)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_adapter_params=} ({format_big_number(num_adapter_params)})")
    logging.info(f"{num_discriminator_params=} ({format_big_number(num_discriminator_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # policy.train()
    peft_modules = set_peft_module_train(peft_modules)

    train_adapter_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_adapter_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_adapter_metrics, initial_step=step
    )

    train_discriminator_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"
            train_discriminator_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            train_discriminator_metrics[f"running_mean_{key}"] = AverageMeter(f"running_mean_{key}", ":.3f")
            train_discriminator_metrics[f"running_std_{key}"] = AverageMeter(f"running_std_{key}", ":.3f")
            train_discriminator_metrics[f"num_batches_tracked_{key}"] = AverageMeter(f"num_batches_tracked_{key}", ":.0f")

    train_discriminator_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_discriminator_metrics, initial_step= step + cfg.steps
    )

    logging.info("Start offline training on a fixed dataset")
    logging.info("Training func adapters")
    for peft_module in peft_modules:
        peft_module.train_discriminator(False)
        peft_module.update_stats(False)
    optimizer = adapter_optimizer
    lr_scheduler = adapter_lr_scheduler
    train_tracker = train_adapter_tracker
    wandb_mode="train"
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq

    init_step = step
    for _ in range(init_step, init_step + cfg.steps + cfg.train_discriminators_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        if step == init_step + cfg.steps:
            logging.info("Training discriminator")
            for peft_module in peft_modules:
                peft_module.train_discriminator(True)
                peft_module.update_stats(True)
            optimizer = discriminator_optimizer
            lr_scheduler = discriminator_lr_scheduler
            train_tracker = train_discriminator_tracker
            wandb_mode="train_discriminator"
            log_freq = cfg.train_discriminators_log_freq
            save_freq = cfg.train_discriminators_save_freq

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            peft_modules,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )


        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()

        if step <= init_step + cfg.steps:
            is_log_step = log_freq > 0 and step % log_freq == 0
            is_saving_step = (step - init_step) % save_freq == 0 or step == init_step + cfg.steps
            is_eval_step = cfg.eval_freq > 0 and (step - init_step) % cfg.eval_freq == 0
        else:
            is_log_step = log_freq > 0 and (step - init_step) % log_freq == 0
            is_saving_step = (step - init_step) % save_freq == 0 or step == init_step + cfg.steps + cfg.train_discriminators_steps
            is_eval_step = cfg.train_discriminators_eval_freq > 0 and (step - init_step) % cfg.train_discriminators_eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step, mode=wandb_mode)
            train_tracker.reset_averages()

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            
            logging.info("Stopping gradients during evaluation")
            to_train_module_list = []
            for peft_module in peft_modules:
                for name, parameter in peft_module.named_parameters():
                    if parameter.requires_grad:
                        to_train_module_list.append(name)
                        parameter.requires_grad = False

            eval_infos = {}

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }

            for task in eval_envs.keys():
                # eval_env = eval_envs[task]
                eval_env_cfg = eval_envs[task]
                logging.info(f"Eval task {task}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    
                    # eval_info = eval_policy(
                    #     eval_env,
                    #     policy,
                    #     cfg.eval.n_episodes,
                    #     videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                    #     max_episodes_rendered=cfg.max_episodes_rendered,
                    #     start_seed=cfg.seed,
                    # )
                    eval_info = eval_policy_with_env_init(
                        eval_env_cfg,
                        cfg.eval.batch_size,
                        False,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                        max_episodes_rendered=cfg.max_episodes_rendered,
                        start_seed=cfg.seed,
                    )
                    eval_infos[task] = eval_info

                eval_metrics[f"avg_sum_reward_{task}"] = AverageMeter(f"∑rwrd_{task}", ":.3f")
                eval_metrics[f"pc_success_{task}"] = AverageMeter(f"success_{task}", ":.1f")
                eval_metrics[f"eval_s_{task}"] = AverageMeter(f"eval_s_{task}", ":.3f")

            logging.info("Restoring gradients during evaluation")
            for peft_module in peft_modules:
                for name, parameter in peft_module.named_parameters():
                    if name in to_train_module_list:
                        parameter.requires_grad = True

            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )

            sum_avg_sum_reward = 0.0
            sum_pc_success = 0.0
            sum_eval_s = 0.0

            for task in eval_infos.keys():
                eval_info = eval_infos[task]
                avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                pc_success = eval_info["aggregated"].pop("pc_success")
                eval_s = eval_info["aggregated"].pop("eval_s")
                
                sum_avg_sum_reward += avg_sum_reward
                sum_pc_success += pc_success
                sum_eval_s += eval_s

                eval_tracker.__setattr__(f"avg_sum_reward_{task}", avg_sum_reward)
                eval_tracker.__setattr__(f"pc_success_{task}", pc_success)
                eval_tracker.__setattr__(f"eval_s_{task}", eval_s)

            mean_avg_sum_reward = sum_avg_sum_reward / len(eval_infos.keys())
            mean_pc_success = sum_pc_success / len(eval_infos.keys())

            eval_tracker.avg_sum_reward = mean_avg_sum_reward
            eval_tracker.pc_success = mean_pc_success
            eval_tracker.eval_s = sum_eval_s

            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][-1], step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

            peft_policy.save_pretrained(str(checkpoint_dir / "adapter"))

            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            # if wandb_logger:
            #     wandb_logger.log_policy(checkpoint_dir)

            

    # if eval_env:
    #     eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
