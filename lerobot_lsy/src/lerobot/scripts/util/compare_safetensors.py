#!/usr/bin/env python3
"""
Compare tensors (by name) between two .safetensors checkpoints.

Usage:
  python compare_safetensors.py \
    --checkpoint_0_path=path/to/first.safetensors \
    --checkpoint_1_path=path/to/second.safetensors \
    [--rtol=1e-7 --atol=0]

Notes:
- Uses safetensors' NumPy backend to avoid requiring PyTorch.
- For float tensors, values are compared with np.allclose(rtol, atol).
- For non-float tensors, values are compared with exact equality.
"""

import argparse
import sys
from typing import Dict, Tuple, List

import numpy as np
from safetensors.torch import load_file as load_safetensors


def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    try:
        tensors = load_safetensors(path)
        # load_file returns a dict-like object mapping names -> np.ndarray (mmap-backed)
        return dict(tensors)
    except Exception as e:
        print(f"[ERROR] Failed to load '{path}': {e}", file=sys.stderr)
        sys.exit(2)


def classify_name_sets(keys0: set, keys1: set) -> Tuple[set, set, set]:
    common = keys0 & keys1
    only0 = keys0 - keys1
    only1 = keys1 - keys0
    return common, only0, only1


def is_float_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float,
    atol: float
) -> Tuple[bool, dict]:
    """Return (equal, stats). For floats, use allclose; otherwise exact."""
    stats = {
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "reason": "",
    }

    if a.shape != b.shape:
        stats["reason"] = "shape_mismatch"
        return False, stats

    if a.dtype != b.dtype:
        stats["reason"] = "dtype_mismatch"
        return False, stats

    # Compute some difference stats regardless, they help when mismatching.
    diff_stats = {}
    try:
        # Avoid copies: operate on views when possible
        if a.size > 0:
            # Differences only meaningful when shapes match (checked above)
            diff = a.astype(np.float64) - b.astype(np.float64)
            diff_stats = {
                "max_abs_diff": float(np.max(np.abs(diff))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "max_rel_diff": float(np.max(np.abs(diff) / (np.abs(b.astype(np.float64)) + 1e-45))),
            }
        else:
            diff_stats = {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "max_rel_diff": 0.0}
    except Exception:
        # If casting fails (e.g., object dtypes), leave empty
        diff_stats = {}

    stats.update(diff_stats)

    if is_float_dtype(a.dtype):
        equal = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not equal and stats.get("reason", "") == "":
            stats["reason"] = "value_mismatch_allclose"
        return equal, stats
    else:
        equal = np.array_equal(a, b)
        if not equal and stats.get("reason", "") == "":
            stats["reason"] = "value_mismatch_exact"
        return equal, stats


def format_name_list(title: str, names: List[str]) -> str:
    if not names:
        return f"{title}: (none)"
    # Keep a deterministic order for readability
    names_sorted = sorted(names)
    return f"{title} ({len(names_sorted)}):\n  - " + "\n  - ".join(names_sorted)


def main():
    parser = argparse.ArgumentParser(description="Compare two .safetensors checkpoints by tensor name and values.")
    parser.add_argument("--checkpoint_0_path", required=True, help="Path to the first .safetensors file")
    parser.add_argument("--checkpoint_1_path", required=True, help="Path to the second .safetensors file")
    parser.add_argument("--rtol", type=float, default=1e-7, help="Relative tolerance for float comparisons (np.allclose)")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for float comparisons (np.allclose)")
    args = parser.parse_args()

    ck0 = load_checkpoint(args.checkpoint_0_path)
    ck1 = load_checkpoint(args.checkpoint_1_path)

    keys0 = set(ck0.keys())
    keys1 = set(ck1.keys())

    common, only0, only1 = classify_name_sets(keys0, keys1)

    # Report name mismatches
    print("=" * 80)
    print("NAME MISMATCH REPORT")
    print(format_name_list("Only in checkpoint_0", list(only0)))
    print()
    print(format_name_list("Only in checkpoint_1", list(only1)))
    print("=" * 80)
    print()

    # Compare common tensors
    mismatches = []
    matches = 0

    for name in sorted(common):
        a = ck0[name]
        b = ck1[name]
        equal, stats = compare_tensors(a, b, args.rtol, args.atol)
        if equal:
            matches += 1
        else:
            mismatches.append((name, stats))

    # Summary
    total = len(common)
    print("COMPARISON SUMMARY")
    print(f"  Common tensors: {total}")
    print(f"  Matching tensors: {matches}")
    print(f"  Mismatching tensors: {len(mismatches)}")
    print()

    if mismatches:
        print("MISMATCH DETAILS")
        for name, st in mismatches:
            reason = st.get("reason", "value_mismatch")
            shape_a = st.get("shape_a")
            shape_b = st.get("shape_b")
            dtype_a = st.get("dtype_a")
            dtype_b = st.get("dtype_b")
            mad = st.get("max_abs_diff", None)
            mead = st.get("mean_abs_diff", None)
            mrd = st.get("max_rel_diff", None)

            print(f"- {name}: {reason}")
            print(f"    shape: {shape_a} vs {shape_b}")
            print(f"    dtype: {dtype_a} vs {dtype_b}")
            if mad is not None:
                print(f"    max_abs_diff: {mad:.6g}")
            if mead is not None:
                print(f"    mean_abs_diff: {mead:.6g}")
            if mrd is not None:
                print(f"    max_rel_diff: {mrd:.6g}")


if __name__ == "__main__":
    main()
