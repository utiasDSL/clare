from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Optional

from peft import LoraConfig

VALID_BIAS = {"none", "all", "lora_only"}

def comma_separated_list(arg: str) -> str | list[str]:
    items = [item.strip() for item in arg.split(",") if item.strip()]
    if len(items) == 1:
        return items[0]   # return plain string
    return items          # return list of strings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a PEFT LoRA config JSON.")
    p.add_argument("--save_path", required=True, type=Path, help="Output JSON file path.")
    p.add_argument("--r", type=int, default=16, help="LoRA rank (r).")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling).")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout in [0,1].")
    p.add_argument("--bias", type=bool, default=False, choices=sorted(VALID_BIAS),
                   help="Bias setting for LoRA.")
    p.add_argument("--init_lora_weights", type=str, default="gaussian")
    p.add_argument("--target_modules", type=comma_separated_list, default=[],
                   help="Module names to apply LoRA to. Space-separated list.")
    p.add_argument("--modules_to_save", nargs="*", default=[],
                   help="Extra modules to keep in full precision (optional).")
    p.add_argument("--task_type", type=str, default=None,
                   help="Optional PEFT task type, e.g., 'FEATURE_EXTRACTION', 'SEQ_CLS', etc.")
    p.add_argument("--fan_in_fan_out", action="store_true",
                   help="Set fan_in_fan_out=True (default False).")
    p.add_argument("--inference_mode", action="store_true",
                   help="Set inference_mode=True (default False).")
    p.add_argument("--dtype", type=str, default=None,
                   help="Optional dtype override (e.g., 'float16', 'bfloat16').")
    return p.parse_args()

def validate(args: argparse.Namespace) -> None:
    if args.r <= 0:
        raise ValueError("--r must be > 0")
    if args.lora_alpha <= 0:
        raise ValueError("--lora_alpha must be > 0")
    if not (0.0 <= args.lora_dropout <= 1.0):
        raise ValueError("--lora_dropout must be between 0 and 1")

def build_config(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    bias: str,
    init_lora_weights:str,
    target_modules: List[str],
    modules_to_save: List[str],
    task_type: Optional[str],
    fan_in_fan_out: bool,
    inference_mode: bool,
) -> dict:
    cfg = LoraConfig(
        r = r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=bias,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type=task_type,
        fan_in_fan_out=fan_in_fan_out,
        inference_mode=inference_mode
    )
    return cfg

def main():
    args = parse_args()
    validate(args)

    out_path: Path = args.save_path
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = build_config(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        init_lora_weights=args.init_lora_weights,
        target_modules=args.target_modules,
        modules_to_save=args.modules_to_save,
        task_type=args.task_type,
        fan_in_fan_out=args.fan_in_fan_out,
        inference_mode=args.inference_mode,
    )

    cfg.save_pretrained(out_path)

    print(f"Wrote LoRA config to {out_path.resolve()}")

if __name__ == "__main__":
    main()