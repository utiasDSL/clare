from safetensors.torch import load_file, save_file

# Load the safetensors file
state = load_file("data/checkpoints/pi0_libero_finetuned/renamed/model.safetensors")

# List keys you want to delete
keys_to_delete = [
    "normalize_inputs.buffer_observation_state_joint.mean",
    "normalize_inputs.buffer_observation_state_joint.std",
]

# Remove them
for k in keys_to_delete:
    if k in state:
        del state[k]

# Save
save_file(state, "data/checkpoints/pi0_libero_finetuned/model.safetensors")
print("Saved cleaned model.")
