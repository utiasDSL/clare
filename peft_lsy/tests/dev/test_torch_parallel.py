# torch >= 2.1 recommended
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

# ----- Configs -----
torch.manual_seed(0)
device = "cuda"  # change to "cuda" if available
B, T, D = 5, 3, 6     # input shape
H = 2                 # MLP hidden dim
O = 4                 # output dim (can be != D)
N = 5                  # number of separate MLPs in the "ensemble"

# ----- Model -----
class MLP(nn.Module):
    def __init__(self, id, d_in, d_hidden, d_out):
        super().__init__()
        self.id = id 
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        # nn.Linear broadcasts on the leading dims, so (B,T,D) -> (B,T,O) works directly.
        return self.net(x)

# Create N independent MLPs (same architecture, different params)
mlps = [MLP(i, D, H, O).to(device) for i in range(N)]
ids = [0, 3, 2, 0, 2]
used_mlps = [mlps[id] for id in ids]

prototype = mlps[0]

# Stack their states for batched functional calls
params, buffers = stack_module_state(used_mlps)  # each leaf tensor now has a leading dim N

# ----- Input -----
x = torch.Tensor(
    [
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        [
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        [
            [0, 0, 0, 0, 0, 0], # [3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0], # [3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0], # [3, 3, 3, 3, 3, 3],
        ],
        [
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            # [4, 4, 4, 4, 4, 4],
            # [4, 4, 4, 4, 4, 4],
            # [4, 4, 4, 4, 4, 4],
        ],
    ],
).to(device)

# ----- Vectorized forward over models -----
def forward_one_model(params_i, buffers_i, x_i):
    # functional_call runs 'prototype' with the provided params/buffers
    return functional_call(prototype, (params_i, buffers_i), (x_i,))

# adapter_result has shape (N, B, T, O)
adapter_result = vmap(forward_one_model)(params, buffers, x)

# ----- Reference (loop) -----
with torch.no_grad():
    ref_list = [m(x) for m in mlps]          # list of (B,T,O)
    ref = torch.stack(ref_list, dim=0)       # (N,B,T,O)

# ----- Tests -----
max_abs_diff = (adapter_result - ref).abs().max().item()
print(f"adapter_result shape: {adapter_result.shape}  (expected {(N, B, T, O)})")
print(f"Max |diff| vs. Python loop: {max_abs_diff:.3e}")
assert torch.allclose(adapter_result, ref, atol=1e-6, rtol=0), "Vectorized and loop outputs differ!"
print("✅ Vectorized vmap forward matches loop over individual MLPs.")
