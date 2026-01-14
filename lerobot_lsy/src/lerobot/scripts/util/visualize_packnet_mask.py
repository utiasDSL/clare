import argparse
from safetensors import safe_open
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main():
    parser = argparse.ArgumentParser(description="Visualize a tensor from a safetensors file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to safetensors checkpoint")
    parser.add_argument("--target_key", type=str, required=True, help="Exact tensor key to visualize")

    args = parser.parse_args()

    ckpt_path = args.checkpoint_path
    target_key = args.target_key
    output_path = f"packnet_mask_{target_key}.png"

    print(f"Loading {ckpt_path} ...\n")
    with safe_open(ckpt_path, framework="numpy") as f:
        tensor = f.get_tensor(target_key)

    print(f"{target_key}: shape={tensor.shape}, dtype={tensor.dtype}")

    # define 10 distinct colors
    cmap = ListedColormap([
        "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
    ])

    if tensor.ndim == 2:
        # Simple case (H, W)
        fig, ax = plt.subplots()
        im = ax.imshow(tensor, cmap=cmap, vmin=0, vmax=9)
        fig.colorbar(im, ticks=range(10), label="Value")
        ax.set_title(target_key)

    elif tensor.ndim == 4 and tensor.shape[-1] == 3:
        # Case (H, W, C, 3) → show last dim as 3 subplots
        H, W, C, K = tensor.shape
        fig, axes = plt.subplots(1, K, figsize=(12, 4))
        for i in range(K):
            im = axes[i].imshow(tensor[:, :, 0, i], cmap=cmap, vmin=0, vmax=9)
            axes[i].set_title(f"{target_key} - slice {i}")
            axes[i].axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), ticks=range(10), label="Value")

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved image to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
