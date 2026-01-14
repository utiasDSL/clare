#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import List, Tuple, Dict

import torch  # required for safetensors.torch
from safetensors.torch import load_file, save_file


def parse_csv_list(s: str) -> List[str]:
    # Split by comma and strip whitespace
    # (simple CSV since names are unlikely to contain commas)
    return [part.strip() for part in s.split(",") if part.strip()]


def build_rules(patterns: List[str], replacements: List[str]) -> List[Tuple[re.Pattern, str]]:
    if len(patterns) != len(replacements):
        raise ValueError(
            f"--target_full_name and --new_name must have the same length "
            f"(got {len(patterns)} vs {len(replacements)})"
        )
    rules: List[Tuple[re.Pattern, str]] = []
    for i, (pat, rep) in enumerate(zip(patterns, replacements)):
        try:
            compiled = re.compile(pat)
        except re.error as e:
            raise ValueError(f"Invalid regex at position {i} ('{pat}'): {e}") from e
        rules.append((compiled, rep))
    return rules


def rename_keys(state: Dict[str, torch.Tensor], rules: List[Tuple[re.Pattern, str]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Apply the first matching regex rule to each key. Keys that don't match any rule are kept as-is.
    Returns (new_state, mapping_original_to_new).
    """
    new_state: Dict[str, torch.Tensor] = {}
    mapping: Dict[str, str] = {}

    for key, tensor in state.items():
        new_key = key
        for pat, repl in rules:
            if pat.search(key):
                # Use re.sub to allow capture groups: \1, \g<name>, etc.
                new_key = pat.sub(repl, key)
                break  # apply only the first matching rule

        # Collision check: if a different original key already produced the same new_key
        if new_key in new_state and key != mapping.get(new_key, key):
            raise RuntimeError(
                f"Key collision after renaming: '{key}' -> '{new_key}', "
                f"but '{mapping_inv(new_state, mapping).get(new_key, 'UNKNOWN')}' already mapped to that name."
            )

        new_state[new_key] = tensor
        mapping[key] = new_key

    return new_state, mapping


def mapping_inv(new_state: Dict[str, torch.Tensor], mapping: Dict[str, str]) -> Dict[str, str]:
    # Helper to invert mapping (new -> old) for error messages
    inv = {}
    for old, new in mapping.items():
        inv[new] = old
    return inv


def main():
    parser = argparse.ArgumentParser(description="Rename parameters in a safetensors checkpoint using regex pairs.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the input .safetensors file.")
    parser.add_argument("--target_full_name", required=True,
                        help="Comma-separated list of regex patterns to match parameter names (e.g., 'layer\\.(\\d+),^embeddings\\.')")
    parser.add_argument("--new_name", required=True,
                        help="Comma-separated list of replacements; same length as target_full_name (e.g., 'block.\\1,embed.')")
    parser.add_argument("--saved_path", default=None,
                        help="Output path for the renamed checkpoint (.safetensors). Defaults to '<checkpoint_path>.renamed.safetensors'.")

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint_path):
        print(f"Error: checkpoint not found: {args.checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    patterns = parse_csv_list(args.target_full_name)
    replacements = parse_csv_list(args.new_name)
    try:
        rules = build_rules(patterns, replacements)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.saved_path:
        out_path = args.saved_path + "/model.safetensors"
    else:
        out_path = args.checkpoint_path + "/model_renamed.safetensors"

    # Load tensors
    print(f"Loading: {args.checkpoint_path}")
    state = load_file(args.checkpoint_path)

    # Rename
    new_state, mapping = rename_keys(state, rules)

    # Report summary
    changed = {k: v for k, v in mapping.items() if k != v}
    unchanged = [k for k, v in mapping.items() if k == v]
    print(f"Total params: {len(mapping)} | Renamed: {len(changed)} | Unchanged: {len(unchanged)}")

    if changed:
        print("Renamed examples (up to 20):")
        shown = 0
        for old, new in list(changed.items())[:20]:
            print(f"  {old}  ->  {new}")
            shown += 1
        if len(changed) > shown:
            print(f"  ... ({len(changed) - shown} more)")

    # Save
    print(f"Saving to: {out_path}")
    # Keep minimal metadata; you may copy original metadata if needed.
    save_file(new_state, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
