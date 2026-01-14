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
from typing import Any
from pathlib import Path
import copy
from dataclasses import dataclass

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from safetensors.torch import save_file, load_file

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

@dataclass
class PackNetTrainPipelineConfig(TrainPipelineConfig):
    current_task: int = 0
    prune_ratio: float = 0.75
    post_prune_steps: int = 20000
    ignore_modules: str | None = None

    max_episodes_rendered: int = 100


@torch.no_grad()
def prune(cfg: PackNetTrainPipelineConfig, policy: torch.nn.Module, previous_mask: dict):
    """
    Prune cfg.prune_ratio of current_task's weights (by magnitude).
    mask keys are layer names (from named_modules).
    Returns new mask dict.
    """
    current_masks = {}

    for name, module in policy.named_modules():
        if not name in previous_mask.keys():
            continue

        weight = module.weight.data
        layer_mask = previous_mask[name].clone().to(weight.device)

        # Select weights belonging to current task
        select = layer_mask.eq(cfg.current_task + 1)
        
        if select.sum().item() == 0:
            current_masks[name] = layer_mask
        else:
            tensor = weight[select]
            abs_tensor = tensor.abs()

            top_k = int(round(cfg.prune_ratio * tensor.numel()))
            if top_k <= 0:
                current_masks[name] = layer_mask
                continue
            if top_k >= tensor.numel():
                top_k = tensor.numel() - 1  # keep at least one

            cutoff = abs_tensor.view(-1).kthvalue(top_k).values.item()

            # prune where |w| <= cutoff among current task
            remove = (weight.abs().le(cutoff)) & select
            layer_mask[remove] = 0
            weight[remove] = 0.0

            current_masks[name] = layer_mask

    return current_masks


def mask_gradient(policy: torch.nn.Module, mask: dict, current_task: int):
    """
    Zero gradients for all weights not assigned to current_task.
    mask: dict mapping layer_name -> mask tensor
    """

    for name, module in policy.named_modules():
        # Conv/Linear family → gate gradients by mask
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)):
            if module.weight.grad is not None and name in mask.keys():
                layer_mask = mask[name].to(module.weight.grad.device)
                module.weight.grad.data[layer_mask.ne(current_task + 1)] = 0
                if module.bias is not None and module.bias.grad is not None and name in mask.keys():
                    module.bias.grad.data.fill_(0)

        # Normalization layers → freeze grads entirely
        elif "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
            if module.weight is not None and module.weight.grad is not None:
                module.weight.grad.data.fill_(0)
            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad.data.fill_(0)

        # Other modules (activations, Dropout, GELU, Identity, LayerScale, Embedding, ParameterDict, etc.)
        # have no gradients to handle, so just skip them.



def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    mask: torch.nn.ParameterDict,
    current_task: int,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    mask_gradient(policy, mask, current_task)

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
def train(cfg: PackNetTrainPipelineConfig):
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
            eval_envs[task] = env_cfg

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    ignore_modules = [ignore_module.strip() for ignore_module in cfg.ignore_modules.split(",") if ignore_module.strip()]

    if cfg.current_task > 0:
        logging.info("Loading previous mask")

        mask = load_file(Path(cfg.policy.pretrained_path) / "mask.safetensors", cfg.policy.device)

        # for name in mask.keys():
        #     layer_mask = mask[name]
        #     layer_mask[layer_mask.eq(0)] = cfg.current_task + 1

        for name, module in policy.named_modules():
            if any(ignore_module in name for ignore_module in ignore_modules): 
                for parameter in module.parameters():
                    parameter.requires_grad = False
                module.eval()
                logging.info(f"Skip module {name}")
                continue
            if name in mask.keys():
                layer_mask = mask[name]
                layer_mask[layer_mask.eq(0)] = cfg.current_task + 1
            elif "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    else:
        logging.info("Creating first mask")
        mask = {}
        for name, module in policy.named_modules():
            if any(ignore_module in name for ignore_module in ignore_modules): 
                # for parameter in module.parameters():
                #     parameter.requires_grad = False
                module.eval()
                logging.info(f"Skip module {name}")
                continue
            
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)):
                if module.weight.requires_grad:
                    mask[name] = torch.ones_like(module.weight, dtype=torch.int8, device=module.weight.device) # all 1
            elif "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()
    
    cfg.optimizer.grad_clip_norm = 100.0

    logging.info("Creating optimizer and scheduler")
    pre_prune_optimizer, pre_prune_lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=True)

    logging.info("Duplicate optimizer and scheduler")
    post_prune_optimizer, post_prune_lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, pre_prune_optimizer, pre_prune_lr_scheduler = load_training_state(cfg.checkpoint_path, pre_prune_optimizer, pre_prune_lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
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

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    optimizer = pre_prune_optimizer
    lr_scheduler = pre_prune_lr_scheduler

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps + cfg.post_prune_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            mask=mask,
            current_task=cfg.current_task,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps or step == cfg.steps + cfg.post_prune_steps
        is_eval_step = cfg.eval_freq > 0 and (step % cfg.eval_freq == 0 or step == cfg.steps)

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            eval_infos = {}

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }

            for task in eval_envs.keys():
                eval_env_cfg = eval_envs[task]
                logging.info(f"Eval task {task}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    
                    eval_info = eval_policy_with_env_init(
                        eval_env_cfg,
                        cfg.eval.batch_size,
                        False,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                        max_episodes_rendered=100,
                        start_seed=cfg.seed,
                    )
                    eval_infos[task] = eval_info

                eval_metrics[f"avg_sum_reward_{task}"] = AverageMeter(f"∑rwrd_{task}", ":.3f")
                eval_metrics[f"pc_success_{task}"] = AverageMeter(f"success_{task}", ":.1f")
                eval_metrics[f"eval_s_{task}"] = AverageMeter(f"eval_s_{task}", ":.3f")

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
                if step <= cfg.steps:
                    mode = "eval"
                else:
                    mode = "eval_after_prune"
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode=mode)
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode=mode)

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)

            save_file(mask, str(checkpoint_dir / PRETRAINED_MODEL_DIR / "mask.safetensors"))

            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if step == cfg.steps:
            logging.info("Prune the mask")
            mask = prune(cfg, policy, mask)
            optimizer = post_prune_optimizer
            lr_scheduler = post_prune_lr_scheduler

            logging.info("Start post fine-tuning after prune")

        

    # if eval_env:
    #     eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
