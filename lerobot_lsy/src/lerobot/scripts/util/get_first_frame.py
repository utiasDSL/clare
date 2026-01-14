import os
import json
import subprocess
import re

def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

def load_first_episodes(jsonl_path):
    first_occurrence = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            task = data["tasks"][0]
            idx = data["episode_index"]
            if task not in first_occurrence:
                first_occurrence[task] = idx
    return first_occurrence

def extract_first_frame_ffmpeg(video_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",             # overwrite output
        "-i", video_path, # input file
        "-vframes", "1",  # only 1 frame
        output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False

def main(video_dir, jsonl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    task_map = load_first_episodes(jsonl_path)

    for task, ep_index in task_map.items():
        video_name = f"episode_{ep_index:06d}.mp4"
        video_path = os.path.join(video_dir, video_name)

        if not os.path.exists(video_path):
            print(f"⚠️ Missing video: {video_name}")
            continue

        img_name = sanitize_filename(task) + ".jpg"
        img_path = os.path.join(out_dir, img_name)

        if extract_first_frame_ffmpeg(video_path, img_path):
            print(f"✅ Saved {img_path}")
        else:
            print(f"❌ Failed to extract from {video_name}")


if __name__ == "__main__":
    VIDEO_DIR = "./data/datasets_all/huggingface/lerobot/EfreetSultan/real_world/videos/chunk-000/observation.images.right_third_person_camera_top_right"        # directory with episode_00000.mp4 ...
    JSONL_PATH = "./data/datasets_all/huggingface/lerobot/EfreetSultan/real_world/meta/episodes.jsonl"
    OUTPUT_DIR = "./data/realworld"

    main(VIDEO_DIR, JSONL_PATH, OUTPUT_DIR)
