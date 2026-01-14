from safetensors import safe_open
from safetensors.torch import save_file
import torch

input_path = "./outputs/smolvla_image_only_lerobot_base/model_original.safetensors"
output_path = "./outputs/smolvla_image_only_lerobot_base/model.safetensors"

keys_to_remove = {
    "normalize_targets.buffer_action.mean",
    "normalize_targets.buffer_action.std",
    "unnormalize_outputs.buffer_action.mean",
    "unnormalize_outputs.buffer_action.std",
}

# Load all tensors
tensors = {}
with safe_open(input_path, framework="torch") as f:
    for key in f.keys():
        if key not in keys_to_remove:
            tensors[key] = f.get_tensor(key)

# Save new safetensors file without the unwanted tensors
save_file(tensors, output_path)

print("Done. Saved cleaned file to:", output_path)
