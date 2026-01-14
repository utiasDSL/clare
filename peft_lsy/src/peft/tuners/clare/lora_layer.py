import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from .config import FuncAdapterConfig
from .func_adapter import LoRAFuncAdapter

class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear modules"""
    def __init__(self, original_layer: nn.Module, config: FuncAdapterConfig):
        super().__init__()
        self.original_layer = original_layer
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.lora_dropout = nn.Dropout(p=config.dropout_p) if hasattr(config, "dropout_p") else nn.Identity()
        self.scaling = self.alpha / self.rank
        
        if isinstance(original_layer, nn.Linear):
            in_features, out_features = original_layer.in_features, original_layer.out_features
        else:
            raise NotImplementedError("The LoRA adapter only supports nn.Linear for now.")

        self.lora = {
            "A": None,
            "B": None
        }

        self.in_features = in_features
        self.out_features = out_features

    def set_lora_adapter(self, adapters_handler: LoRAFuncAdapter, sub_module_name: str):
        self.lora["A"] = adapters_handler.layer_wise_lora_adapters[sub_module_name]["lora_a"]
        self.lora["B"] = adapters_handler.layer_wise_lora_adapters[sub_module_name]["lora_b"]

    # highly inspried by LoRA implementation
    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight
        """
        device = self.lora["B"].weight.device
        dtype = self.lora["B"].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora["A"].weight
        weight_B = self.lora["B"].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora["A"].weight.data = weight_A.to(dtype)
            self.lora["B"].weight.data = weight_B.to(dtype)

        return output_tensor

    # highly inspried by LoRA implementation
    def merge(self):
        # Note that safe_merge will be slower than the normal merge
        # because of the copy operation.
        orig_weight = self.original_layer.weight.data.clone()
        orig_dtype = orig_weight.dtype
        
        delta_weight = self.get_delta_weight()
        orig_weight += delta_weight.to(orig_dtype)

        if not torch.isfinite(orig_weight).all():
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter seems to be broken"
            )

        self.original_layer.weight.data = orig_weight

        # No lora bias used for now

        # if self.lora_bias:
        #     if getattr(self.original_layer, "bias", None) is None:
        #         raise RuntimeError(
        #             "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
        #         )
        #     new_bias = self.original_layer.bias + self.lora["B"].bias * self.scaling[active_adapter]
        #     if not torch.isfinite(new_bias).all():
        #         raise ValueError(
        #             f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
        #         )
        #     self.original_layer.bias.data = new_bias.to(orig_dtype)

    # highly inspried by LoRA implementation
    def unmerge(self):
        weight = self.original_layer.weight

        orig_dtype = weight.dtype
        delta_weight = self.get_delta_weight()
        weight.data -= delta_weight.to(orig_dtype)

        # No lora bias used for now

        # if self.lora_bias[active_adapter]:
        #     self.get_base_layer().bias.data -= self.lora["B"].bias * self.scaling[active_adapter]
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_layer(x)

        if self.lora["A"] is None:
            raise ValueError("Uninitialized lora_a.")
        else:
            lora_a = self.lora["A"]

        if self.lora["B"] is None:
            raise ValueError("Uninitialized lora_b.")
        else:
            lora_b = self.lora["B"]
    
        lora_output = lora_b(lora_a(self.lora_dropout(x))) * self.scaling

        return base_output + lora_output
    
    @property
    def weight(self) -> torch.Tensor:
        return self.original_layer.weight
    
    @property
    def bias(self) -> torch.Tensor:
        return self.original_layer.bias


class LoRAMultiheadAttention(nn.Module):
    """LoRA wrapper for nn.MultiheadAttention modules"""
    def __init__(self, original_layer: nn.Module, config: FuncAdapterConfig):
        super().__init__()
        if not getattr(original_layer, "_qkv_same_embed_dim", True):
            # default for this value appears to be True:
            # https://github.com/pytorch/pytorch/blob/701ba5203fe68d55d655bd4d6c008be94cf34ea5/torch/nn/modules/activation.py#L1128-L1130
            raise ValueError(
                f"Only same embed for query/key/value is supported as of now for {self.__class__.__name__}."
            )
        
        self.original_layer = original_layer
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.lora_dropout = nn.Dropout(p=config.dropout_p) if hasattr(config, "dropout_p") else nn.Identity()
        self.scaling = self.alpha / self.rank
        
        if isinstance(original_layer, nn.MultiheadAttention):
            in_features, out_features = original_layer.embed_dim, 3 * original_layer.embed_dim
        else:
            raise NotImplementedError("The LoRA adapter only supports nn.MultiheadAttention for now.")

        self.lora = {
            "A": None,
            "B": None
        }

        # Replace the original out_proj layer with LoRALinear
        self.original_layer.out_proj = LoRALinear(self.original_layer.out_proj, config)

        self.in_features = in_features
        self.out_features = out_features

    def set_lora_adapter(self, adapters_handler: LoRAFuncAdapter, sub_module_name: str):
        self.lora["A"] = adapters_handler.layer_wise_lora_parameters[sub_module_name]["lora_a"]
        self.lora["B"] = adapters_handler.layer_wise_lora_parameters[sub_module_name]["lora_b"]

        self.original_layer.out_proj.set_lora_adapter(adapters_handler, sub_module_name)

    # highly inspried by LoRA implementation
    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight
        """
        device = self.lora["B"].weight.device
        dtype = self.lora["B"].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora["A"].weight
        weight_B = self.lora["B"].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora["A"].weight.data = weight_A.to(dtype)
            self.lora["B"].weight.data = weight_B.to(dtype)

        return output_tensor

    # highly inspried by LoRA implementation
    def merge(self):
        orig_dtype = self.original_layer.out_proj.original_layer.weight.dtype

        # merging in_proj (nn.Parameter)
        orig_weight_in = self.original_layer.in_proj_weight.data.detach().clone()
        orig_weight_in += self.get_delta_weight().to(orig_dtype)
        if not torch.isfinite(orig_weight_in).all():
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter seems to be broken"
            )

        # merging out_proj (subclass of nn.Linear)
        orig_weight_out = self.original_layer.out_proj.original_layer.weight.data.detach().clone()
        orig_weight_out += self.original_layer.out_proj.get_delta_weight().to(orig_dtype)
        if not torch.isfinite(orig_weight_out).all():
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter seems to be broken"
            )

        # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
        # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
        del self.original_layer.in_proj_weight
        self.original_layer.in_proj_weight = orig_weight_in

        del self.original_layer.out_proj.original_layer.weight
        self.original_layer.out_proj.original_layer.weight = orig_weight_out
        self.original_layer.out_proj.merge()

    # highly inspried by LoRA implementation
    def unmerge(self):
        orig_dtype = self.original_layer.out_proj.original_layer.weight.dtype

        # in_proj
        delta_weight = self.get_delta_weight().to(orig_dtype)
        old_weight = self.original_layer.in_proj_weight.data - delta_weight
        del self.original_layer.in_proj_weight
        self.original_layer.register_parameter("in_proj_weight", nn.Parameter(old_weight, requires_grad=False))

        # out_proj
        delta_weight = self.original_layer.out_proj.get_delta_weight().to(orig_dtype)
        old_weight = self.original_layer.out_proj.original_layer.weight.data - delta_weight
        del self.original_layer.out_proj.original_layer.weight
        self.original_layer.out_proj.original_layer.register_parameter(
            "weight", nn.Parameter(old_weight, requires_grad=False)
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.lora["A"] is None:
            raise ValueError("Uninitialized lora_a.")

        if self.lora["B"] is None:
            raise ValueError("Uninitialized lora_b.")

        try:
            self.merge()
            result = self.original_layer(x, *args, **kwargs)
        finally:
            # it's safe to call unmerge(), which unmerges all adapters, because we checked that not self.merged,
            # i.e. there is was no merged layer before
            self.unmerge()

        return result