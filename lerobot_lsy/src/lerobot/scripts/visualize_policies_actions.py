import argparse  # noqa: D100
from pathlib import Path

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.eval_episode import create_episode_plot, eval_episode

# Anything that the policy should not read
SUPERVISION_PREFIXES = ("action", "next.", "target", "labels")


parser = argparse.ArgumentParser(description="Visualize policy vs. dataset actions")
parser.add_argument(
    "--policy",
    type=str,
    required=True,
    help="Path to the policy policy directory (e.g., outputs/train/.../pretrained_model).",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to the dataset directory OR a Hub repo id (e.g., lerobot/aloha_static_coffee).",
)
parser.add_argument(
    "--episode",
    type=int,
    default=0,
    help="Episode index to compare (default: 0).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional path for saving the matplotlib figure. "
    "If omitted, saved as actions_episode<EP>.png in cwd",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Dataloader workers (default: 8).",
)
args = parser.parse_args()

# Find out the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the policy
train_config = TrainPipelineConfig.from_pretrained(args.policy)
policy_cls = get_policy_class(train_config.policy.type)

policy = policy_cls.from_pretrained(args.policy)
policy.to(device)
policy.eval()

# Load the dataset
dataset = LeRobotDataset(repo_id=args.dataset)
episode = args.episode

# Evaluate the episode
targets_t, preds_t, times_t = eval_episode(
    dataset=dataset, policy=policy, num_workers=args.num_workers, device=device, episode=episode
)

# Create and save the plot
policy_name = args.policy.replace("/", "_").replace("\\", "_")
dataset_name = args.dataset.replace("/", "_").replace("\\", "_")
save_path = args.output or Path(f"outputs/actions_episode{episode}_{policy_name}_{dataset_name}.png")
create_episode_plot(targets_t=targets_t, preds_t=preds_t, times_t=times_t, save_path=save_path)

print(f"Saved continuous plot to {save_path}")
