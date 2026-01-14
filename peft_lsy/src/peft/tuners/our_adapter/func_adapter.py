import torch
import torch.nn as nn
from .config import FuncAdapterConfig

class FuncAdapter(nn.Module):
    config: FuncAdapterConfig

    def __init__(self, config: FuncAdapterConfig, feature_dim: int, out_feature_dim: int):
        super().__init__()

        self.use_lora = config.use_lora

        if config.use_lora:
            rank = config.lora_rank
            alpha = config.lora_alpha
            self.scaling = alpha / rank
            self.down_lora_A = nn.Linear(feature_dim, rank, bias=False)
            self.down_lora_B = nn.Linear(rank, config.hidden_dim, bias=True)
            self.activation = nn.ReLU()
            self.up_lora_A = nn.Linear(config.hidden_dim, rank, bias=False)
            self.up_lora_B = nn.Linear(rank, out_feature_dim, bias=True)    
        else:
            self.down_proj = nn.Linear(feature_dim, config.hidden_dim)
            self.activation = nn.ReLU()
            self.up_proj = nn.Linear(config.hidden_dim, out_feature_dim)

        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))

    def forward(self, x):
        if self.use_lora:
            # Down projection: Wx ≈ BAx
            x_down = self.down_lora_B(self.down_lora_A(x)) * self.scaling
            x_down = self.activation(x_down)
            # Up projection: Wx ≈ BAx
            output = self.up_lora_B(self.up_lora_A(x_down)) * self.scaling
        else:
            x = self.down_proj(x)
            x = self.activation(x)
            output = self.up_proj(x)

        return output