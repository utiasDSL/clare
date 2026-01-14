import re
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# set global font sizes
plt.rcParams.update({
    "font.size": 14,            # base font size for text
    "axes.titlesize": 16,      # axes title
    "axes.labelsize": 14,      # x/y labels
    "xtick.labelsize": 12,     # x tick labels
    "ytick.labelsize": 12,     # y tick labels
    "legend.fontsize": 14,     # legend text
    # optional: increase line width / marker size for visibility
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})

SRC = Path(__file__).parent
TABLE_FILE = SRC / "threshold_ablation.txt"
OUT_FILE = f"outputs/figs/threshold_ablation_plots.png"

# baseline reference values (value, uncertainty)
BASELINE = {
    "auc": (55.87, 1.47),
    "fwt": (67.67, 1.65),
    "nbt": (15.79, 0.48),
}


def parse_table(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # collect lines inside tabular (between \midrule and \bottomrule)
    data_lines = []
    in_block = False
    for L in lines:
        if r"\midrule" in L:
            in_block = True
            continue
        if r"\bottomrule" in L:
            in_block = False
            continue
        if in_block:
            # only keep table rows (those that end with \\)
            if r"\\" in L:
                data_lines.append(L.strip())

    rows = []
    for L in data_lines:
        # remove trailing \\ and possible LaTeX whitespace
        L = re.sub(r"\\\\.*$", "", L).strip()
        # split columns on '&'
        cols = [c.strip() for c in L.split("&")]
        if len(cols) < 5:
            continue
        rows.append(cols[:5])  # Threshold, #Adapters, AUC, FWT, NBT

    def parse_val_unc(s):
        # find all numbers (integers or floats, positive or negative)
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if not nums:
            return np.nan, 0.0
        val = float(nums[0])
        err = float(nums[1]) if len(nums) > 1 else 0.0
        return val, err

    thresholds = []
    adapters = []
    adapters_err = []
    auc = []
    auc_err = []
    fwt = []
    fwt_err = []
    nbt = []
    nbt_err = []

    for cols in rows:
        th_s, ad_s, auc_s, fwt_s, nbt_s = cols
        # threshold may be like "0" or "2.5"
        th_val, _ = parse_val_unc(th_s)
        a_val, a_err = parse_val_unc(ad_s)
        auc_val, auc_e = parse_val_unc(auc_s)
        fwt_val, fwt_e = parse_val_unc(fwt_s)
        nbt_val, nbt_e = parse_val_unc(nbt_s)

        thresholds.append(th_val)
        adapters.append(int(round(a_val)))
        adapters_err.append(a_err)
        auc.append(auc_val)
        auc_err.append(auc_e)
        fwt.append(fwt_val)
        fwt_err.append(fwt_e)
        nbt.append(nbt_val)
        nbt_err.append(nbt_e)

    # convert to numpy arrays sorted by threshold
    arr = np.argsort(thresholds)
    return {
        "threshold": np.array(thresholds)[arr],
        "adapters": np.array(adapters)[arr],
        "adapters_err": np.array(adapters_err)[arr],
        "auc": np.array(auc)[arr],
        "auc_err": np.array(auc_err)[arr],
        "fwt": np.array(fwt)[arr],
        "fwt_err": np.array(fwt_err)[arr],
        "nbt": np.array(nbt)[arr],
        "nbt_err": np.array(nbt_err)[arr],
    }
# ...existing code...


def plot(data, out_path):
    x = data["threshold"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes = axes.flatten()

    # format x ticks: show "5" instead of "5.0", keep decimals for non-integers (e.g. 2.5)
    fmt = FuncFormatter(lambda v, pos: f"{int(v)}" if float(v).is_integer() else f"{v:g}")
    for a in axes:
        a.xaxis.set_major_formatter(fmt)

    # AUC
    ax = axes[0]
    ax.errorbar(x, data["auc"], yerr=data["auc_err"], marker="o", capsize=3, color="#377eb8", label="CLARE (ours)")
    # ax.fill_between([x.min() - 1, x.max() + 1], auc_base - auc_err, auc_base + auc_err, color="gray", alpha=0.15)
    # baseline for AUC
    auc_base, auc_err = BASELINE["auc"]
    ax.axhline(auc_base, color="gray", linestyle="--", linewidth=1.5, label="ER")
    ax.set_title("AUC $\\uparrow$ [%]")
    ax.set_xlabel("Threshold $\\gamma$")
    # ax.set_ylabel("AUC [%]")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0, top=70)
    ax.legend()

    # FWT
    ax = axes[1]
    ax.errorbar(x, data["fwt"], yerr=data["fwt_err"], marker="o", capsize=3, color="#377eb8", label="CLARE (ours)")
    # ax.fill_between([x.min() - 1, x.max() + 1], fwt_base - fwt_err, fwt_base + fwt_err, color="gray", alpha=0.15)
    # baseline for FWT
    fwt_base, fwt_err = BASELINE["fwt"]
    ax.axhline(fwt_base, color="gray", linestyle="--", linewidth=1.5, label="ER")
    ax.set_title("FWT $\\uparrow$ [%]")
    ax.set_xlabel("Threshold $\\gamma$")
    # ax.set_ylabel("FWT [%]")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0, top=70)
    ax.legend()

    # NBT
    ax = axes[2]
    ax.errorbar(x, data["nbt"], yerr=data["nbt_err"], marker="o", capsize=3, color="#377eb8", label="CLARE (ours)")
    # ax.fill_between([x.min() - 1, x.max() + 1], nbt_base - nbt_err, nbt_base + nbt_err, color="gray", alpha=0.15)
    # baseline for NBT
    nbt_base, nbt_err = BASELINE["nbt"]
    ax.axhline(nbt_base, color="gray", linestyle="--", linewidth=1.5, label="ER")
    ax.set_title("NBT $\\downarrow$ [%]")
    ax.set_xlabel("Threshold $\\gamma$")
    # ax.set_ylabel("NBT [%]")
    ax.set_xticks(x)
    ax.set_ylim(bottom=-2)
    ax.legend(loc="center left")

    # Adapters
    ax = axes[3]
    ax.bar(x, data["adapters"], yerr=data["adapters_err"], capsize=4, width=1.5, align="center", color="#377eb8")
    ax.set_title("Number of Adapters")
    ax.set_xlabel("Threshold $\\gamma$")
    # ax.set_ylabel("# Adapters")
    ax.set_xticks(x)


    


    # plt.suptitle("Expansion Threshold Ablation", fontsize=14)
    plt.tight_layout(rect=[-0.01, -0.01, 1.01, 1.01])
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

    for ax in axes:
        ax.set_xlim([-0.5, 20.5])


if __name__ == "__main__":
    data = parse_table(TABLE_FILE)
    plot(data, OUT_FILE)