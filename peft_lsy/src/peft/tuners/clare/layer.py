# peft/tuners/our_adapter/layer.py

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state
import einops
from .config import CLAREConfig, FuncAdapterConfig, CLAREModuleConfig
from .discriminator import Discriminator, BatchedAutoEncoderSmall, get_discriminaor_class
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from .lora_layer import LoRALinear, LoRAMultiheadAttention
from .func_adapter import FuncAdapter, LoRAFuncAdapter

STACK_FORWARD = False

class ConvHelper(nn.Module):
    """Swap dims: (B, T, D) <-> (B, D, T)."""
    def forward(self, x):
        return x.transpose(1, 2)


# class FuncAdapterWrapper(nn.Module):
#     def __init__(self, 
#                  config: CLAREConfig, 
#                  adapter: nn.Module):
#         super().__init__()

#         self.add_zero_init_conv_layer = config.add_zero_init_conv_layer
#         self.func_adapter = None  # Will be set below

#         if config.add_zero_init_conv_layer:

#             conv_layer = nn.Conv1d(
#                 in_channels=config.out_feature_dim, 
#                 out_channels=config.out_feature_dim,
#                 kernel_size=1,
#                 padding=0
#             )

#             # Initialize weights and bias to zero
#             nn.init.constant_(conv_layer.weight, 0.0)
#             if conv_layer.bias is not None:
#                 nn.init.constant_(conv_layer.bias, 0.0)

#             self.func_adapter = nn.Sequential(
#                 adapter,
#                 ConvHelper(),
#                 conv_layer,
#                 ConvHelper()
#             )
#         else:
#             self.func_adapter = adapter

#     def forward(self, x):
#         if x.ndim == 2 and self.add_zero_init_conv_layer:
#             x = x.squeeze(0)
#             y = self.func_adapter(x)
#             y = y.unsqueeze(0)
#             return y
#         else:
#             return self.func_adapter(x)

def general_set_module(base_layer: nn.Module, submodule_name: str, new_submodule: nn.Module):
    if submodule_name == '':
        return "self", new_submodule
    else:
        base_layer.set_submodule(submodule_name, new_submodule)
        return submodule_name, base_layer


def general_get_module(base_layer: nn.Module, submodule_name: str):
    if submodule_name == 'self':
        return base_layer
    else:
        return base_layer.get_submodule(submodule_name)


class LoRAFuncAdapterWrapper(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lora_module = nn.Linear(input_size, output_size, bias=False)
        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))

    def forward(self, x):
        return self.lora_module(x)


# ---- Layer wrapper: base + adapter ----
class CLARELayer(nn.Module, BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        peft_config: CLAREConfig,
        module_config: CLAREModuleConfig,
        adapter_name: str,
        layer_name: str,
        layer_id: int,
        base_layer_name: str,
        num_adapters: int,
        num_discriminators: int
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.peft_config = peft_config
        self.module_config = module_config
        self.adapter_name = adapter_name
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.base_layer_name = base_layer_name
        self.num_adapters = num_adapters
        self.num_discriminators = num_discriminators
        self.use_lora = self.module_config.func_adapter_cfg.use_lora

        self._base_layer_device = next(self.base_layer.parameters()).device
        self._base_layer_dtype = next(self.base_layer.parameters()).dtype

        def submodule_name_match(submodule_name: str, lora_module_name_list: list[str]) -> bool:
            for registered_name in lora_module_name_list:
                if submodule_name == registered_name or submodule_name.startswith(registered_name + "."):
                    return True
            return False

        # create adapters
        if self.use_lora:
            self.lora_module_name_list = []
            new_func_adapters_list = nn.ModuleList([])

            lora_func_adapter_template = LoRAFuncAdapter(self.module_config.func_adapter_cfg)
            
            for name, module in self.base_layer.named_modules():
                if not submodule_name_match(name, self.lora_module_name_list): 
                    # only conside nn.Linear
                    if isinstance(module, nn.Linear):
                        # Replace the original base layer with lora compatiable layer
                        lora_wrapped_module = LoRALinear(module, self.module_config.func_adapter_cfg)
                        name, self.base_layer = general_set_module(self.base_layer, name, lora_wrapped_module)

                        # record name of lora wrapped module
                        self.lora_module_name_list.append(name)

                        if num_adapters > 0:
                            lora_func_adapter_template.layer_wise_lora_adapters[name.replace(".", "_")] = nn.ModuleDict({
                                "lora_a" : nn.Linear(lora_wrapped_module.in_features, lora_wrapped_module.rank, bias=False),
                                "lora_b" : nn.Linear(lora_wrapped_module.rank, lora_wrapped_module.out_features, bias=False)
                            })
                    elif isinstance(module, nn.MultiheadAttention):
                        # Replace the original base layer with lora compatiable layer
                        lora_wrapped_module = LoRAMultiheadAttention(module, self.module_config.func_adapter_cfg)
                        name, self.base_layer = general_set_module(self.base_layer, name, lora_wrapped_module)

                        # record name of lora wrapped module
                        self.lora_module_name_list.append(name)

                        if num_adapters > 0:
                            lora_func_adapter_template.layer_wise_lora_adapters[name.replace(".", "_")] = nn.ModuleDict({
                                "lora_a" : nn.Linear(lora_wrapped_module.original_layer.out_proj.in_features, lora_wrapped_module.original_layer.out_proj.rank, bias=False),
                                "lora_b" : nn.Linear(lora_wrapped_module.original_layer.out_proj.rank, lora_wrapped_module.original_layer.out_proj.out_features, bias=False)
                            })
                            lora_func_adapter_template.layer_wise_lora_parameters[name.replace(".", "_")] = nn.ParameterDict({
                                "lora_a" : nn.Linear(lora_wrapped_module.in_features, lora_wrapped_module.rank, bias=False),
                                "lora_b" : nn.Linear(lora_wrapped_module.rank, lora_wrapped_module.out_features, bias=False)
                            })


            lora_func_adapter_template.to(dtype=self._base_layer_dtype, device=self._base_layer_device)

            for _ in range(num_adapters):
                new_func_adapters_list.append(copy.deepcopy(lora_func_adapter_template))

            del lora_func_adapter_template
                    
            self.clare_func_adapters = nn.ModuleDict({self.adapter_name:new_func_adapters_list})
        else:
            new_func_adapters_list = nn.ModuleList([self._create_adapter() for _ in range(num_adapters)])
            self.clare_func_adapters: nn.ModuleDict[str, nn.ModuleList[FuncAdapter]] = \
            nn.ModuleDict({self.adapter_name:new_func_adapters_list})


        # create discriminators
        new_discriminators_list = nn.ModuleList([self._create_discriminator() for _ in range(num_discriminators)])
        self.clare_discriminators: nn.ModuleDict[str, nn.ModuleList[Discriminator]] = \
            nn.ModuleDict({self.adapter_name:new_discriminators_list})

        self._info_dicts: dict = {}
        self._active_task: int = -1
        self._forwarded_adapter_id: int = -1
        self._forwarded_discriminator_id: int = -1
        self._train_discriminator: bool = False
        self._previous_forwarded_adapter_id:int = -1
        self._stack_discriminator_once_in_eval: bool = True
        self._stacked_discriminator = {}

    def _create_adapter(self):
        if self.module_config.use_trainable_copy:
            adapter = copy.deepcopy(self.base_layer)
        else:
            adapter = FuncAdapter(
                self.module_config.func_adapter_cfg, 
                self.module_config.feature_dim, 
                self.module_config.out_feature_dim
            )
        for p in adapter.parameters():
            p.requires_grad = True
        return adapter
    
    def _create_discriminator(self):
        disc_cls = get_discriminaor_class(self.module_config.discriminator_cfg.type)
        new_dis = disc_cls(self.module_config.discriminator_cfg, self.module_config.feature_dim)
        new_dis.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        return new_dis

    def _forward_discriminators(self, x: torch.Tensor):

        # if self._stack_discriminator_once_in_eval:
        #     new_batched_discriminator = BatchedAutoEncoderSmall(self.module_config.discriminator_cfg, self.clare_discriminators[self.adapter_name])
        #     new_batched_discriminator.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        #     self._stacked_discriminator[self.adapter_name] = new_batched_discriminator
        #     self._stack_discriminator_once_in_eval = False

        # losses, info_dicts = self._stacked_discriminator[self.adapter_name](x)

        losses = []
        info_dicts = []

        for discriminator in self.clare_discriminators[self.adapter_name]:
            loss, info_dict = discriminator(x)
            losses.append(loss)
            info_dicts.append(info_dict)

        losses = torch.stack(losses, dim=0)

        return losses, info_dicts

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.training:
            # reset the flag
            if not self._stack_discriminator_once_in_eval:
                self._stack_discriminator_once_in_eval = True
                # release previous BatchedAutoEncoderSmall
                self._stacked_discriminator.clear()
                # load discriminators into GPU again
                for discriminator in self.clare_discriminators[self.adapter_name]:
                    discriminator.to(self._base_layer_device)

            # forward specific discriminator
            if self._train_discriminator:
                _, info_dict = self.clare_discriminators[self.adapter_name][self._forwarded_discriminator_id](x)

                if self._forwarded_discriminator_id == -1:
                    discriminator_id = len(self.clare_discriminators[self.adapter_name]) - 1  
                else:
                    discriminator_id = self._forwarded_discriminator_id
                self._info_dicts[f"discriminator_{discriminator_id}"] = info_dict

                for indice, discriminator in enumerate(self.clare_discriminators[self.adapter_name]):
                    if indice != discriminator_id:
                        info_dict = {
                            "running_mean" : discriminator.running_mean,
                            "running_std" : discriminator.running_std,
                            "num_batches_tracked" : discriminator.num_batches_tracked,
                        }
                        self._info_dicts[f"discriminator_{indice}"] = info_dict

            # forward specific adapter
            if self.use_lora:
                self._activate_lora_adapter(self._forwarded_adapter_id)
                result = self.base_layer(x, **kwargs)
            else:
                adapter_result = self.clare_func_adapters[self.adapter_name][self._forwarded_adapter_id](x)
                base_result = self.base_layer(x, **kwargs)
                result = base_result + adapter_result
        else:

            losses, info_dicts = self._forward_discriminators(x)

            for indice, info_dict in enumerate(info_dicts):
                self._info_dicts[f"discriminator_{indice}"] = info_dict

            top_1_idx_list = torch.argmin(losses, dim=0).tolist()

            self._info_dicts["losses"] = losses.transpose(0, 1) # (n_discriminators, n_envs) -> (n_envs, n_discriminators)
            self._info_dicts["top_1_idx_list"] = top_1_idx_list

            if not self.module_config.batch_first and x.ndim == 3:
                adapter_input = einops.rearrange(x, "t b d ... -> b t d ... ")
            else:
                adapter_input = x

            if adapter_input.ndim == 2:
                B = adapter_input.shape[0]
                adapter_output_shape = (B, self.module_config.out_feature_dim)
            else:
                B, T = adapter_input.shape[:2]
                adapter_output_shape = (B, T, self.module_config.out_feature_dim)

            # Process each sample individually
            adapter_result = torch.zeros(adapter_output_shape, device=adapter_input.device, dtype=adapter_input.dtype)

            for idx, top_1_idx in enumerate(top_1_idx_list):
                _forwarded_adapter_id = self.clare_discriminators[self.adapter_name][top_1_idx].connected_adapter_indices.item()
                
                # Select single sample while preserving dims
                current_input = adapter_input[idx]
                
                # Process this sample with its best adapter
                if self.use_lora:
                    self._activate_lora_adapter(_forwarded_adapter_id)
                    if self.module_config.batch_first:
                        current_input = current_input.unsqueeze(dim=0) # (T, D) -> (1, T, D) / (D) -> (1, D)
                    else:
                        current_input = current_input.unsqueeze(dim=-2) # (T, D) -> (T, 1, D) / (D) -> (1, D)
                    current_output = self.base_layer(current_input, **kwargs)
                    if self.module_config.batch_first:
                        current_output = current_output.squeeze(dim=0) # (1, T, D) -> (T, D) / (1, D) -> (D)
                    else:
                        current_output = current_output.squeeze(dim=1) # (T, 1, D) -> (T, D) / (1, D) -> (D)
                    adapter_result[idx]= current_output
                else:
                    adapter_result[idx] = self.clare_func_adapters[self.adapter_name][_forwarded_adapter_id](current_input)

            if not self.module_config.batch_first and adapter_result.ndim == 3:
                adapter_result = einops.rearrange(adapter_result, "b t d ... -> t b d ... ")
        
            if self.use_lora:
                result = adapter_result
            else:
                base_result = self.base_layer(x, **kwargs)
                result = base_result + adapter_result

        return result

    def _activate_lora_adapter(self, _forwarded_adapter_id: int):
        # only reload adapters when switching adapters
        if self._previous_forwarded_adapter_id != _forwarded_adapter_id:
            # reload adapters
            for sub_module_name in self.lora_module_name_list:
                sub_module = general_get_module(self.base_layer, sub_module_name)
                sub_module.set_lora_adapter(self.clare_func_adapters[self.adapter_name][_forwarded_adapter_id], sub_module_name.replace(".", "_"))

            # update _previous_forwarded_adapter_id
            self._previous_forwarded_adapter_id = _forwarded_adapter_id

    def add_adapter_and_discriminator(self, new_task_id:int):
        if self.use_lora:
            new_adapter = LoRAFuncAdapter(self.module_config.func_adapter_cfg)
            new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
            for sub_module_name in self.lora_module_name_list:
                sub_module = general_get_module(self.base_layer, sub_module_name)
                in_features = sub_module.in_features
                out_features = sub_module.out_features
                rank = self.module_config.func_adapter_cfg.lora_rank

                if isinstance(sub_module, LoRALinear):
                    new_adapter.layer_wise_lora_adapters[sub_module_name.replace(".", "_")] = nn.ModuleDict({
                        "lora_a" : nn.Linear(in_features, rank, bias=False),
                        "lora_b" : nn.Linear(rank, out_features, bias=False)
                    })
                elif isinstance(sub_module, LoRAMultiheadAttention):
                    new_adapter.layer_wise_lora_adapters[sub_module_name.replace(".", "_")] = nn.ModuleDict({
                        "lora_a" : nn.Linear(sub_module.original_layer.out_proj.in_features, sub_module.original_layer.out_proj.rank, bias=False),
                        "lora_b" : nn.Linear(sub_module.original_layer.out_proj.rank, sub_module.original_layer.out_proj.out_features, bias=False)
                    })
                    new_adapter.layer_wise_lora_parameters[sub_module_name.replace(".", "_")] = nn.ParameterDict({
                        "lora_a" : nn.Linear(in_features, rank, bias=False),
                        "lora_b" : nn.Linear(rank, out_features, bias=False)
                    })
        else:
            new_adapter = self._create_adapter()
            new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        
        new_adapter.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self.clare_func_adapters[self.adapter_name].append(new_adapter)
        self.num_adapters += 1
        adapter_parameter = list(new_adapter.parameters())
            
        discriminator_parameter = self.add_discriminator(self.num_adapters - 1, new_task_id)

        return adapter_parameter, discriminator_parameter

    def add_discriminator(self, connected_adapter_indices:int, new_task_id:int):
        new_discriminator = self._create_discriminator()
        new_discriminator.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        new_discriminator.connected_adapter_indices = torch.tensor(connected_adapter_indices, dtype=torch.int64)
        new_discriminator.connected_adapter_task_id = self.clare_func_adapters[self.adapter_name][connected_adapter_indices].task_id
        new_discriminator.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self.clare_discriminators[self.adapter_name].append(new_discriminator)
        self.num_discriminators += 1

        discriminator_parameter = list(new_discriminator.parameters())

        return discriminator_parameter

    def train_discriminator(self, train_discriminator:bool):
        self._train_discriminator = train_discriminator

    def track_z_score(self, require_z_score:bool):
        for discriminator in self.clare_discriminators[self.adapter_name]:
            discriminator.require_z_score = require_z_score

    def update_stats(self, require_update_stats:bool):
        self.clare_discriminators[self.adapter_name][self._forwarded_discriminator_id].require_update_stats = require_update_stats
        # for discriminator in self.clare_discriminators[self.adapter_name]:
        #     discriminator.require_update_stats = require_update_stats

    def get_adapter_id_by_discriminator_id(self, discriminator_id):
        return self.clare_discriminators[self.adapter_name][discriminator_id].connected_adapter_indices.item()
    
    @property
    def info_dicts(self):
        return self._info_dicts
    
    def __getattr__(self, name):
        # First, try normal behavior (important!)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # Then search inside base_layer
        if hasattr(self.base_layer, name):
            return getattr(self.base_layer, name)

        # Attribute not found
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

