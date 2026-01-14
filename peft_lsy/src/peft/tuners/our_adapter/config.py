from __future__ import annotations
from dataclasses import dataclass, field
import draccus
from typing import Any, List, Optional, Union, Dict

import draccus.parsers

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType

ModuleSelector = Union[str, "re.Pattern[str]"]

@dataclass
class FuncAdapterConfig:
    """
    This is the sub-configuration class to store the configuration of a [`OurAdapterModel`].

    Args:
        hidden_dim (`int`):
            The dimension of the hidden feature of the bottleneck adapter.
        use_lora (`bool`):
            whether to use lora on functional adapter or not
        lora_rank (`int`):
            Lora attention dimension (the "rank").
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
    """
    hidden_dim: int = field(default=0)
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=32)
    lora_alpha: int = field(default=32)



@dataclass
class DiscriminatorConfig(draccus.ChoiceRegistry):
    """
    This is the sub-configuration class to store the configuration of a [`OurAdapterModel`].
    
    Args:
        
        max_batches_tracked (`int`):
            How many batches will be tracked to calculate the statistic.
    """
    feature_dim: int = None
    batch_first: bool = True
    feature_fusion: bool = False
    num_tokens: int = None
    fused_feature_dim: int = None
    max_batches_tracked: int = 2000
    use_momentum: bool = True
    momentum: float = 0.1

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "autoencoder"

@dataclass
class OurAdapterConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`OurAdapterModel`].

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually. To avoid targeting any modules (because you want
            to apply `target_parameters`), set `target_modules=[]`.
        feature_dim (`int`):
            The dimension of the input feature. Given an input of shape (B, T, D), D is feature_dim
        out_feature_dim (`int`):
            The dimension of the output feature.
        discriminator_cfg (`DiscriminatorConfig`):
            The configuration of Discriminator
        use_trainable_copy (`bool`):
            whether to copy the module from base model as adapter or not
        func_adapter_cfg (`FuncAdapterConfig`):
             The configuration of FuncAdapter
    """
    target_modules: Union[list[str], str] = field(default="(?P<layer_name>.+)\.(?P<layer_id>\d+)(?:\.[^.]+)*\.mlp")
    feature_dim: int = None
    out_feature_dim: int = None
    batch_first: bool = True
    discriminator_cfg: DiscriminatorConfig = None
    use_trainable_copy: bool = False
    add_zero_init_conv_layer:bool = False
    func_adapter_cfg: FuncAdapterConfig = None
    num_learned_task: int = 0
    structure: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Assign to a valid PEFT type so load/save works. It does not alter the tuner name.
        self.peft_type = PeftType.OUR_ADAPTER

        # out_feature_dim default the same as feature_dim
        self.out_feature_dim = self.out_feature_dim or self.feature_dim

        if isinstance(self.func_adapter_cfg, dict):
            self.func_adapter_cfg = FuncAdapterConfig(**self.func_adapter_cfg)

        if isinstance(self.discriminator_cfg, dict):
            discriminator_cfg = self.discriminator_cfg
            discriminator_type = discriminator_cfg.pop("type")

            self.discriminator_cfg = DiscriminatorConfig.get_choice_class(discriminator_type)(**discriminator_cfg)

