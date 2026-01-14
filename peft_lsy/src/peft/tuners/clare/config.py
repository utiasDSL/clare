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
    This is the sub-configuration class to store the configuration of a [`CLAREModel`].

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
    This is the sub-configuration class to store the configuration of a [`CLAREModel`].
    
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
class CLAREModuleConfig:
    """Configuration for a specific target module pattern.
    
    Args:
        pattern (`str`):
            Regex pattern to match module names
        feature_dim (`int`):
            The dimension of the input feature
        out_feature_dim (`Optional[int]`):
            The dimension of the output feature. If None, defaults to feature_dim
        discriminator_cfg (`Optional[DiscriminatorConfig]`):
            Configuration for the discriminator
        batch_first (`bool`):
            Whether the input tensor has batch dimension first (B, T, D) vs (T, B, D)
        use_trainable_copy (`bool`):
            Whether to copy the module from base model as adapter
        add_zero_init_conv_layer (`bool`):
            Whether to add a zero-initialized conv layer
        func_adapter_cfg (`Optional[FuncAdapterConfig]`):
            Configuration for the functional adapter
    """
    pattern: str
    feature_dim: int
    out_feature_dim: Optional[int] = None
    discriminator_cfg: Optional[DiscriminatorConfig] = None
    batch_first: bool = True
    use_trainable_copy: bool = False
    add_zero_init_conv_layer: bool = False
    func_adapter_cfg: Optional[FuncAdapterConfig] = None

@dataclass
class CLAREConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CLAREModel`].

    Args:
        target_modules (`Union[List[CLAREModuleConfig], Dict[str, CLAREModuleConfig], str]`):
            The configuration for different module patterns. Can be specified in several ways:
            - A string (legacy mode): Single regex pattern with default config
            - A list of CLAREModuleConfig: Each config specifies its own pattern and settings
            - A dict mapping regex patterns to CLAREModuleConfig objects
            Legacy behavior is preserved when passing a single string or using default values.
        feature_dim (`Optional[int]`):
            Default dimension of the input feature. Used if not specified in module config.
            Given an input of shape (B, T, D), D is feature_dim
        out_feature_dim (`Optional[int]`):
            Default dimension of the output feature. Used if not specified in module config.
        discriminator_cfg (`Optional[DiscriminatorConfig]`):
            Default discriminator configuration. Used if not specified in module config.
        use_trainable_copy (`bool`):
            Default setting for whether to copy the module from base model as adapter.
        func_adapter_cfg (`Optional[FuncAdapterConfig]`):
            Default adapter configuration. Used if not specified in module config.
    """
    target_modules: Union[List[CLAREModuleConfig], Dict[str, CLAREModuleConfig], str] = \
        field(default="(?P<layer_name>.+)\.(?P<layer_id>\d+)(?:\.[^.]+)*\.mlp")
    feature_dim: Optional[int] = None
    out_feature_dim: Optional[int] = None
    # Default values for module configs when not specified
    batch_first: bool = True  # Default batch_first for new module configs
    discriminator_cfg: Optional[DiscriminatorConfig] = None
    use_trainable_copy: bool = False  # Default use_trainable_copy for new module configs
    add_zero_init_conv_layer: bool = False
    func_adapter_cfg: Optional[FuncAdapterConfig] = None
    num_learned_task: int = 0
    structure: Dict = field(default_factory=dict)
    
    # Internal state to store processed module configs
    _module_configs: Dict[str, CLAREModuleConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Assign to a valid PEFT type so load/save works. It does not alter the tuner name.
        self.peft_type = PeftType.CLARE

        # Process default configurations first
        self.out_feature_dim = self.out_feature_dim or self.feature_dim
        if isinstance(self.func_adapter_cfg, dict):
            self.func_adapter_cfg = FuncAdapterConfig(**self.func_adapter_cfg)
        if isinstance(self.discriminator_cfg, dict):
            discriminator_cfg = self.discriminator_cfg
            discriminator_type = discriminator_cfg.pop("type")
            self.discriminator_cfg = DiscriminatorConfig.get_choice_class(discriminator_type)(**discriminator_cfg)

        # Process module configurations using processed defaults
        self._process_module_configs()

    def _process_module_configs(self) -> None:
        """Process and validate module configurations while maintaining backward compatibility."""
        if isinstance(self.target_modules, str):
            # Legacy mode - single pattern with default config
            self._module_configs[self.target_modules] = CLAREModuleConfig(
                pattern=self.target_modules,
                feature_dim=self.feature_dim,
                out_feature_dim=self.out_feature_dim,
                discriminator_cfg=self.discriminator_cfg,
                batch_first=self.batch_first,
                use_trainable_copy=self.use_trainable_copy,
                add_zero_init_conv_layer=self.add_zero_init_conv_layer,
                func_adapter_cfg=self.func_adapter_cfg
            )
        elif isinstance(self.target_modules, list):
            # List of module configs
            for config in self.target_modules:
                if isinstance(config, dict):
                    config = CLAREModuleConfig(**config)
                # Apply default values if not specified
                if config.out_feature_dim is None:
                    config.out_feature_dim = config.feature_dim
                if config.batch_first is None:
                    config.batch_first = self.batch_first
                if config.use_trainable_copy is None:
                    config.use_trainable_copy = self.use_trainable_copy
                if config.func_adapter_cfg and isinstance(config.func_adapter_cfg, dict):
                    config.func_adapter_cfg = FuncAdapterConfig(**config.func_adapter_cfg)
                if config.discriminator_cfg and isinstance(config.discriminator_cfg, dict):
                    disc_cfg = config.discriminator_cfg
                    disc_type = disc_cfg.pop("type")
                    config.discriminator_cfg = DiscriminatorConfig.get_choice_class(disc_type)(**disc_cfg)
                self._module_configs[config.pattern] = config
        elif isinstance(self.target_modules, dict):
            # Dict mapping patterns to configs
            for pattern, config in self.target_modules.items():
                if isinstance(config, dict):
                    config = CLAREModuleConfig(pattern=pattern, **config)
                # Apply default values if not specified
                if config.out_feature_dim is None:
                    config.out_feature_dim = config.feature_dim
                if config.batch_first is None:
                    config.batch_first = self.batch_first
                if config.use_trainable_copy is None:
                    config.use_trainable_copy = self.use_trainable_copy
                if config.func_adapter_cfg and isinstance(config.func_adapter_cfg, dict):
                    config.func_adapter_cfg = FuncAdapterConfig(**config.func_adapter_cfg)
                if config.discriminator_cfg and isinstance(config.discriminator_cfg, dict):
                    disc_cfg = config.discriminator_cfg
                    disc_type = disc_cfg.pop("type")
                    config.discriminator_cfg = DiscriminatorConfig.get_choice_class(disc_type)(**disc_cfg)
                self._module_configs[pattern] = config
                
    def get_module_config(self, module_name: str) -> Optional[CLAREModuleConfig]:
        """
        Get the configuration for a given module name by matching against patterns.
        Uses PEFT's check_target_module_exists for consistent pattern matching behavior.
        """
        from peft.tuners.tuners_utils import check_target_module_exists

        for pattern, config in self._module_configs.items():
            # Create temporary config to use check_target_module_exists
            temp_config = type('TempConfig', (), {'target_modules': pattern})()
            if check_target_module_exists(temp_config, module_name):
                return config
        return None

