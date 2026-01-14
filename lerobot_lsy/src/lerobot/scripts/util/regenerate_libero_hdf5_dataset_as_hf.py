"""
Inspired by paper "OpenVLA: An Open-Source Vision-Language-Action Model"
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
import imageio
import gymnasium as gym
import robosuite as suite
import gym_libero # type: ignore # noqa: F401
import gym_libero.libero.benchmark as benchmark
from gym_libero.libero.envs.bddl_utils import get_problem_info
from gym_libero.libero.envs import TASK_MAPPING

from lerobot.constants import OBS_IMAGES, OBS_ROBOT
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.datasets.utils import create_branch, create_lerobot_dataset_card, flatten_dict

IMAGE_RESOLUTION = 256 # same as OpenVLA and other libero dataset in hugging face


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    # Create new Hugging face dataset for regenerated demos

    features = {
        f"{OBS_IMAGES}.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": [
                "height",
                "width",
                "channel"
            ]
        },
        f"{OBS_IMAGES}.wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": [
                "height",
                "width",
                "channel"
            ]
        },
        OBS_ROBOT: {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper",
                    "gripper"
                ]
            }
        },
        f"{OBS_ROBOT}.joint": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "motors": [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "joint_7"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper"
                ]
            }
        },
    }

    lerobot_dataset = LeRobotDataset.create(
        repo_id = args.repo_id,
        fps = 20, # default in libero
        features = features,
        use_videos = False,
    )

    for task_id in range(num_tasks_in_suite):

        if args.task_ids:
            if task_id not in args.task_ids:
                continue

        # Get task in suite
        task = task_suite.get_task(task_id)
        bddl_file_path = task_suite.get_task_bddl_file_path(task_id)
        problem_info = get_problem_info(bddl_file_path)
        problem_name = problem_info["problem_name"]

        env = TASK_MAPPING[problem_name](
            bddl_file_path,
            robots=["Panda"],
            controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
            render_camera="frontview",
            has_renderer=True,
            camera_names=[
                "frontview",
                "agentview",
                "robot0_eye_in_hand",
            ],
            control_freq=20,
            camera_heights=IMAGE_RESOLUTION,
            camera_widths=IMAGE_RESOLUTION,
            horizon=1000, # make sure not time out
        )
        task_description = task.language

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        num_successful_demo = 0

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]
            init_state = demo_data.attrs["init_state"]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.sim.set_state_from_flattened(init_state)
            env.sim.forward()
            env._check_success()
            env._post_process()
            env._update_observables(force=True)

            # for _ in range(10):
            #     action = np.zeros((7,))
            #     obs, _, _, _ = env.step(action)

            # Set up new data lists
            actions = []
            ee_states = []
            joint_states = []
            agentview_images = []
            eye_in_hand_images = []

            frames = []

            # Replay original demo actions in environment and record observations
            for indices, (action, state) in enumerate(zip(orig_actions, orig_states)):
                # Skip transitions with no-op actions
                # prev_action = actions[-1] if len(actions) > 0 else None
                # if is_noop(action, prev_action):
                #     print(f"\tSkipping no-op action: {action}")
                #     num_noops += 1
                #     continue

                # Record original action (from demo)
                actions.append(action)

                # Execute demo action in environment
                # obs, reward, done, info = env.step(action)
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                done = env._check_success()
                env._post_process()
                env._update_observables(force=True)
                obs = env._get_observations()
                image = env.sim.render(camera_name="agentview", height=IMAGE_RESOLUTION, width=IMAGE_RESOLUTION)
                image = np.flip(image, axis=0)
                frames.append(image)

                # Record data returned by environment
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]
                        )
                    )
                )
                agentview_images.append(np.flip(obs["agentview_image"], axis=0).copy())
                eye_in_hand_images.append(np.flip(obs["robot0_eye_in_hand_image"], axis=0).copy())

            # temporal fix, make it always success
            done = True

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                # write episode into lerobot dataset
                timestamp = 0
                
                for ee_state, joint_state, agentview_image, eye_in_hand_image, action \
                    in zip(ee_states, joint_states, agentview_images, eye_in_hand_images, actions):
                    frame = {
                        "timestamp": np.array([timestamp], dtype=np.float32),
                        "task": task_description,
                        OBS_ROBOT: ee_state.astype(np.float32),
                        f"{OBS_ROBOT}.joint": joint_state.astype(np.float32),
                        f"{OBS_IMAGES}.image": agentview_image,
                        f"{OBS_IMAGES}.wrist_image": eye_in_hand_image,
                        "action": action.astype(np.float32),
                    }
                    lerobot_dataset.add_frame(frame)
                    timestamp += 1 / 20
                lerobot_dataset.save_episode()
                lerobot_dataset.clear_episode_buffer()

                num_success += 1
                num_successful_demo += 1
                # num_total_samples += ep_data_grp.attrs["num_samples"]
            else:
                imageio.mimsave(f"./data/debug/{args.libero_task_suite}_task_{task_id}_ep_{i}.mp4", np.stack(frames), fps=20)

            num_replays += 1

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )
            
            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        env.close()

        # Close HDF5 files
        orig_data_file.close()
        print(f"Total # successful demos: {num_successful_demo}")

    if args.push_to_hub:
        lerobot_dataset.push_to_hub(push_videos=False, private=False)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./data/datasets/libero/original/libero_10", required=True)
    parser.add_argument("--task_ids", type=int, nargs='+',
                        help="Only use these tasks to build dataset", default=None)
    parser.add_argument("--hugging_face_root", type=str,
                        help="Root path to regenerated dataset directory. Example: ./data/datasets/huggingface/", default="./data/datasets/huggingface/")
    parser.add_argument("--repo_id", type=str,
                        help="Repo_id", required=True)
    parser.add_argument("--push_to_hub", type=bool,
                        help="if push it to hub", default=False)
    args = parser.parse_args()

    # Start data regeneration
    main(args)
