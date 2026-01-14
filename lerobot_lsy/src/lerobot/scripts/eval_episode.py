from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy

SUPERVISION_PREFIXES = ("action", "next.", "target", "labels")


def to_device_batch(batch: dict, device: torch.device, non_blocking: bool = True) -> dict:  # noqa: D103
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def _make_inference_batch(batch: dict) -> dict:
    cleaned = {}
    for k, v in batch.items():
        # keep only observation/metadata; drop action-like keys
        if any(k == p or k.startswith(p) for p in SUPERVISION_PREFIXES):
            continue
        cleaned[k] = v
    return cleaned

def batchify_torch_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            # Scalar tensor -> [1]
            if v.ndim == 0:
                out[k] = v.unsqueeze(0)
            else:
                out[k] = v.unsqueeze(0)
        else:
            # strings or other metadata: leave as-is
            out[k] = v
    return out

def eval_episode(
    dataset: LeRobotDataset | MultiLeRobotDataset,
    policy: PreTrainedPolicy,
    num_workers: int = 0,
    device: torch.device | None = None,
    episode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a policy on a specific episode from a dataset.

    Args:
        dataset (LeRobotDataset): The dataset to evaluate on.
        policy (PreTrainedPolicy): The policy to evaluate.
        num_workers (int, optional): Number of workers for the data loader. Defaults to 4.
        device (torch.device | None, optional): Device to run the evaluation on. If None, uses CUDA if available, else CPU. Defaults to None.
        episode (int, optional): Episode index to evaluate. Defaults to 0.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - targets (torch.Tensor): The ground truth actions from the dataset of shape [T, D].
            - preds (torch.Tensor): The predicted actions from the policy of shape [T, D].
            - times (torch.Tensor): The timestamps or frame indices corresponding to each action of shape [T].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    targets: List[torch.Tensor] = []
    preds: List[torch.Tensor] = []
    times: List[float] = []
    found_any = False
    last_batch: Optional[Dict[str, Any]] = None

    for batch in dataloader:
        # Try to find the correct batch
        b_ep = batch.get("episode_index")
        if b_ep is None:
            raise KeyError("Expected key 'episode_index' in batch.")
        b_ep = int(b_ep.view(-1)[0].item())

        if b_ep < episode:
            continue
        if b_ep > episode:
            break

        found_any = True
        last_batch = batch

        batch = to_device_batch(batch, device, non_blocking=True)

        tgt = batch["action"].detach().float().view(-1)

        # Predicted action from policy
        cleaned_batch = _make_inference_batch(batch)
        pred = policy.select_action(cleaned_batch)
        pred = pred.detach().float().view(-1)

        # Collect
        targets.append(tgt.cpu())
        preds.append(pred.cpu())

        # Time (fallback to step index if timestamp is absent)
        if "timestamp" in batch:
            t = float(batch["timestamp"].view(-1)[0].detach().cpu().item())
        else:
            # reconstruct pseudo-time from frame_index if available; else just enumerate
            if "frame_index" in batch:
                t = float(batch["frame_index"].view(-1)[0].detach().cpu().item())
            else:
                t = float(len(times))
        times.append(t)

        if not found_any:
            raise ValueError(
                f"No frames found for episode_index={episode}. Check that the dataset contains this episode."
            )
    if not found_any or last_batch is None:
        raise ValueError(
            f"No frames found for episode_index={episode}. Check that the dataset contains this episode."
        )

    # Stack
    assert targets and preds, "No data collected"

    targets_t = torch.stack(targets, dim=0)  # [T, D]
    preds_t = torch.stack(preds, dim=0)  # [T, D]
    times_t = torch.tensor(times)  # [T]

    return targets_t, preds_t, times_t

def eval_episode_no_loader(
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy,
    device: torch.device | None = None,
    episode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    times: list[float] = []
    found_any = False


    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]

            b_ep = batch.get("episode_index")
            if b_ep is None:
                raise KeyError("Expected key 'episode_index' in batch.")
            b_ep = int(torch.as_tensor(b_ep).view(-1)[0].item())

            if b_ep != episode:
                continue

            found_any = True

            batch = to_device_batch(batch, device, non_blocking=(device.type == "cuda"))

            tgt = torch.as_tensor(batch["action"]).detach().float().view(-1)

            cleaned_batch = _make_inference_batch(batch)
            # Include a function that goes through the dictionary of the cleaned_batch and includes a batch dimension 
            # For pictures it is obvious 3,256,256 -> 1,3,256,256
            # For keys like timestamp and episode_index i think it can be [1] as a torch tensor dont know if they also need additional help
            cleaned_batch = batchify_torch_dict(cleaned_batch)
            pred = policy.select_action(cleaned_batch).detach().float().view(-1)

            targets.append(tgt.cpu())
            preds.append(pred.cpu())

            if "timestamp" in batch:
                t = float(torch.as_tensor(batch["timestamp"]).view(-1)[0].detach().cpu().item())
            elif "frame_index" in batch:
                t = float(torch.as_tensor(batch["frame_index"]).view(-1)[0].detach().cpu().item())
            else:
                t = float(len(times))
            times.append(t)

    if not found_any:
        raise ValueError(
            f"No frames found for episode_index={episode}. Check that the dataset contains this episode."
        )

    targets_t = torch.stack(targets, dim=0)  # [T, D]
    preds_t = torch.stack(preds, dim=0)      # [T, D]
    times_t = torch.tensor(times, dtype=torch.float32)  # [T]

    return targets_t, preds_t, times_t

def create_episode_plot(targets_t, preds_t, times_t, save_path: Path):
    """Create and save a plot comparing target and predicted actions over time."""
    import matplotlib.pyplot as plt

    T, D = preds_t.shape

    # ----- Plot -----
    fig, axes = plt.subplots(D, 1, figsize=(9, 2.3 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.plot(times_t.numpy(), targets_t[:, d].numpy(), label="Target")
        ax.plot(times_t.numpy(), preds_t[:, d].numpy(), label="Pred")
        ax.set_ylabel(f"dim {d}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(
            [
                float(torch.min(targets_t[:, d]).item()) * 1.01,
                float(torch.max(targets_t[:, d]).item()) * 1.01,
            ]
        )

    # single legend outside if many dims
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.98, 0.98])

    fig.savefig(save_path, dpi=150)
    plt.close(fig)