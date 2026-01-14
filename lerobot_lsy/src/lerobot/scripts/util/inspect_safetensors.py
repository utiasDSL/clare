import argparse
from safetensors import safe_open

def main():
    parser = argparse.ArgumentParser(description="Print tensors from a safetensors file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to safetensors checkpoint")
    parser.add_argument("--target_keys", type=str, default="", help="Comma-separated list of substrings to match in tensor names")
    args = parser.parse_args()

    ckpt_path = args.checkpoint_path
    target_keys = [k.strip() for k in args.target_keys.split(",") if k.strip()] if args.target_keys else None

    print(f"Loading {ckpt_path} ...\n")
    with safe_open(ckpt_path, framework="torch") as f:
        print(f"{'Name':80} {'Values':10} {'Shape':10} {'Dtype':10}")
        print("-" * 120)
        for name in f.keys():
            if target_keys and not any(key in name for key in target_keys):
                continue
            tensor = f.get_tensor(name)

            # convert to string, truncate if very long
            val_str = str(tensor)
            if len(val_str) > 30:
                val_str = val_str[:27] + "..."

            if len(name) > 80:
                name = "..." + name[-77:]

            print(f"{name:80} {val_str:10} {str(tensor.shape):10} {str(tensor.dtype):10}")

if __name__ == "__main__":
    main()
