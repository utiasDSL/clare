import torch
import torch.nn as nn
from .config import FuncAdapterConfig

class FuncAdapter(nn.Module):
    config: FuncAdapterConfig

    def __init__(self, config: FuncAdapterConfig, feature_dim: int, out_feature_dim: int):
        super().__init__()

        self.config  = config

        self.use_lora = config.use_lora
        
        self.down_proj = nn.Linear(feature_dim, config.hidden_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(config.hidden_dim, out_feature_dim)

        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))

    def forward(self, x):
        
        x = self.down_proj(x)
        x = self.activation(x)
        output = self.up_proj(x)

        return output
    
class LoRAFuncAdapter(nn.Module):
    config: FuncAdapterConfig

    def __init__(self, config: FuncAdapterConfig):
        super().__init__()

        self.config  = config

        self.layer_wise_lora_adapters = nn.ModuleDict({})

        self.layer_wise_lora_parameters = nn.ParameterDict({})

        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))

