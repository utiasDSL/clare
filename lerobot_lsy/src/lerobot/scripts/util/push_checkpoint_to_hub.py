#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
from huggingface_hub import Repository, create_repo

def push_policy(repo_id: str, policy_dir: str, commit_message: str):
    # Create the repo on the Hub if it doesn't already exist.
    print(f"Ensuring repository {repo_id} exists on Hugging Face Hub...")
    create_repo(repo_id=repo_id, private=True, exist_ok=True)

    # Define a temporary local directory for the cloned repo.
    repo_local_path = "./temp_repo"
    
    # Remove any existing temporary directory.
    if os.path.exists(repo_local_path):
        shutil.rmtree(repo_local_path)

    # Clone the repository locally.
    print("Cloning repository locally...")
    repo = Repository(local_dir=repo_local_path, clone_from=repo_id)

    # Copy the pretrained policy files from the specified directory into the repository.
    print(f"Copying policy files from {policy_dir} to {repo_local_path}...")
    if os.path.isdir(policy_dir):
        for item in os.listdir(policy_dir):
            src_path = os.path.join(policy_dir, item)
            dst_path = os.path.join(repo_local_path, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
    elif os.path.isfile(policy_dir):
        # In case policy_dir is a file rather than a folder.
        shutil.copy2(policy_dir, os.path.join(repo_local_path, os.path.basename(policy_dir)))
    else:
        raise FileNotFoundError(f"The specified policy directory or file {policy_dir} was not found.")

    # Add, commit, and push the changes.
    print("Adding changes, committing, and pushing to the Hub...")
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(commit_message)
    repo.git_push()
    print("Push complete!")

    # Remove temporary directory after push
    if os.path.exists(repo_local_path):
        shutil.rmtree(repo_local_path)

def main():
    parser = argparse.ArgumentParser(
        description="Push a pretrained policy checkpoint to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repository id on Hugging Face Hub (e.g., 'username/model_name')."
    )
    parser.add_argument(
        "--policy_dir",
        type=str,
        help="Local directory (or file) containing the pretrained policy checkpoint.."
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Initial commit of pretrained policy",
        help="Commit message to use when pushing to the Hub."
    )
    args = parser.parse_args()

    push_policy(args.repo_id, args.policy_dir, args.commit_message)

if __name__ == "__main__":
    main()
