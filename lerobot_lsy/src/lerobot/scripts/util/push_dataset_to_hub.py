import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str,
                        help="Repo_id", required=True)
    args = parser.parse_args()
    lerobot_dataset = LeRobotDataset(
        repo_id=args.repo_id
    )

    lerobot_dataset.push_to_hub(push_videos=False, private=False)