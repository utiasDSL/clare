# peft/tuners/clare/model.py

from __future__ import annotations

import re
from typing import Any, List, Union, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists, onload_layer
from peft.utils import _get_submodules

from .config import CLAREConfig
from .layer import CLARELayer

def extract_layer(current_key: str, key_pattern: str) -> Optional[Tuple[str, int]]:
    """
    Returns (layer_name, layer_id) if `key_pattern` matches `current_key`,
    else returns None.

    key_pattern should contain:
      - (?P<layer_name>...)   -> e.g. (layers) or (encoders|decoders)
      - (?P<layer_id>\d+)     -> the numeric id
    """
    m = re.search(key_pattern, current_key)
    if not m:
        return None
    if "layer_name" in m.re.groupindex and "layer_id" in m.re.groupindex:
        layer_name = m.group("layer_name")
        if bool(m.group("layer_id")):
            layer_id = int(m.group("layer_id"))
        else:
            layer_id = 0
    else:
        layer_name = m.group(0)
        layer_id = 0
    return layer_name, layer_id


class CLAREModel(BaseTuner):
    """
    PEFT-compatible tuner that injects OurAdapterLayer into target modules.
    """
    prefix = "clare_"
    _clare_layers: List[CLARELayer] = []

    @staticmethod
    def _check_target_module_exists(peft_config: CLAREConfig, key: str) -> bool:
        # Check if any pattern in module_configs matches the key
        module_config = peft_config.get_module_config(key)
        return module_config is not None
    
    def _create_and_replace(
        self,
        peft_config: CLAREConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
            
        # Get the specific config for this module
        module_config = peft_config.get_module_config(current_key)
        if not module_config:
            raise ValueError(f"No configuration found for module {current_key}")
            
        # Extract layer info using the module's pattern
        layer_name, layer_id = extract_layer(current_key, module_config.pattern)

        # normal situation
        device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
        new_module = self._create_new_module(peft_config, adapter_name, target, layer_name, layer_id, target_name, original_key=current_key, device_map=device_map)
        self._replace_module(parent, target_name, new_module, target)

        self._clare_layers.append(new_module)



    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if isinstance(module, CLARELayer):
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(device=weight.device, dtype=weight.dtype)

    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, layer_name, layer_id, base_layer_name, original_key, **kwargs):
        key = f"{layer_name}.{layer_id}"
        current_key = f"{key}.{base_layer_name}"

        if key not in peft_config.structure:
            peft_config.structure[key] = [0, 0]

        num_adapters, num_discriminators = peft_config.structure[key]
        
        # Get module-specific config
        module_config = peft_config.get_module_config(original_key)
        if not module_config:
            raise ValueError(f"No configuration found for module {original_key}")
            
        # Create CLARELayer with module-specific config
        new_module = CLARELayer(
            base_layer=target,
            peft_config=peft_config,
            module_config=module_config,  # Use module-specific config
            adapter_name=adapter_name,
            layer_name=layer_name, 
            layer_id=layer_id,
            base_layer_name=base_layer_name,
            num_adapters=num_adapters,
            num_discriminators=num_discriminators,
        )

        return new_module

    def disable_adapter_layers(self):
        pass

    def enable_adapter_layers(self):
        pass

    def set_adapter(self, adapter_name: str | list[str]):
        self.active_adapter = adapter_name

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "unload_and_optionally_merge_module"):
                    # if layers have special unloading method, like MultiheadAttention, use that
                    unloaded_module = target.unload_and_optionally_merge_module(
                        merge=merge, safe_merge=safe_merge, adapter_names=adapter_names
                    )
                    self._replace_module(parent, target_name, unloaded_module, target)
                elif hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
        

    @property
    def adapter_layers(self):
        return self._clare_layers

    # Add this static method: return the config unchanged
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """
        Prepare and return the adapter config.
        For OurAdapter, we simply return the config unchanged.
        """
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Mark only the adapter parameters as trainable.
        Freeze everything else (already done in __init__).
        This satisfies the BaseTuner abstract method contract.
        """

        # no need to enable gradient here
        # only enbale gradient when adding new adapters
        for n, p in model.named_parameters():
            p.requires_grad = False

        # # Explicitly enable gradients on adapter params
        # for layer in self._clare_layers:
        #     for n, p in layer.named_parameters():
        #         if "base_layer" not in n:
        #             p.requires_grad = True

    # ------- Convenience controls --------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    