import torch
import random
import einops
import argparse
from pathlib import Path
from typing import Sequence
from datasets import concatenate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import EpisodeSampler
from lerobot.datasets.utils import DEFAULT_FEATURES


def sample_episodes(dataset: LeRobotDataset, sample_size: int, seed: int = 42) -> list[int]:
    """
    Randomly sample a given number of episodes from a dataset.

    Args:
        dataset (LeRobotDataset): The dataset to sample from.
        sample_size (int): Number of episodes to sample.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list[int]: List of sampled episode indices.
    """
    rng = random.Random(seed)
    all_episodes = list(dataset.meta.episodes.keys())
    if sample_size > len(all_episodes):
        raise ValueError(
            f"Requested {sample_size} episodes, but dataset only has {len(all_episodes)}"
        )
    return rng.sample(all_episodes, sample_size)


def merge_datasets(
    datasets: Sequence[LeRobotDataset],
    sample_sizes: Sequence[int],
    merged_repo_id: str,
    root: str | Path | None = None,
    seed: int = 42,
) -> LeRobotDataset:
    """
    Merge random episodes from multiple datasets.

    Args:
        datasets (Sequence[LeRobotDataset]): List of datasets to merge.
        sample_sizes (Sequence[int]): Number of episodes to sample from each dataset.
        merged_repo_id (str): New repo_id for the merged dataset.
        root (str | Path | None, optional): Root directory to save merged dataset. Defaults to None.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        LeRobotDataset: The merged dataset.
    """
    if len(datasets) != len(sample_sizes):
        raise ValueError("datasets and sample_sizes must have the same length.")

    # --- Step 1: Create merged dataset metadata ---
    merged = LeRobotDataset.create(
        repo_id=merged_repo_id,
        fps=datasets[0].fps,
        features=datasets[0].features,
        robot_type=datasets[0].meta.robot_type,
        use_videos=any(len(ds.meta.video_keys) > 0 for ds in datasets),
    )
    expected_features = set(merged.features) - set(DEFAULT_FEATURES)

    # --- Step 2: Process datasets ---
    for i, (dataset, sample_size) in enumerate(zip(datasets, sample_sizes)):
        episode_index_list = sample_episodes(dataset, sample_size, seed + i)

        for episode_index in episode_index_list:
            episode_sampler = EpisodeSampler(dataset, episode_index)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=1,
                batch_size=1,
                sampler=episode_sampler,
            )

            for sample in dataloader:
                task = sample["task"][0]
                frame = {}
                for key in expected_features:
                    if isinstance(sample[key], torch.Tensor):
                        frame[key] = sample[key].squeeze(0)
                        if frame[key].ndim == 3:  # CHW -> HWC
                            frame[key] = einops.rearrange(frame[key], "c h w -> h w c")
                merged.add_frame(frame, task)

            merged.save_episode()
            merged.clear_episode_buffer()

    return merged

def parse_args():
    parser = argparse.ArgumentParser(description="Merge episodes from multiple LeRobot datasets.")
    parser.add_argument(
        "--repo_ids",
        type=str,
        required=True,
        help="Comma-separated list of dataset repo ids",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        required=True,
        help="Number of episodes to sample from each dataset",
    )
    parser.add_argument(
        "--merged_repo_id",
        type=str,
        required=True,
        help="Repo id for the merged dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    repo_ids = [r.strip() for r in args.repo_ids.split(",")]
    datasets = [LeRobotDataset(repo_id) for repo_id in repo_ids]
    sample_sizes = [args.num_episodes] * len(datasets)

    merged_dataset = merge_datasets(
        datasets,
        sample_sizes,  # number of episodes sampled from each
        merged_repo_id=args.merged_repo_id,
        seed=args.seed,
    )

    print(merged_dataset)
