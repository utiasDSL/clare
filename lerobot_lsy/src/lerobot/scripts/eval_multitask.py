import copy
import json
import logging
from pathlib import Path
from typing import Any

from lerobot.scripts.eval import eval_main as eval_single_task_main
from lerobot.configs import parser
from lerobot.scripts.eval import EvalMultiPipelineConfig


def get_task_specific_cfg(base_cfg: EvalMultiPipelineConfig, task: str) -> EvalMultiPipelineConfig:
    task_cfg = copy.deepcopy(base_cfg)
    task_cfg.env.task = task
    task_cfg.job_name = f"{base_cfg.job_name}_{task}"
    task_cfg.output_dir = Path(base_cfg.output_dir) / task
    return task_cfg


@parser.wrap()
def eval_multitask_main(base_cfg: EvalMultiPipelineConfig):
    if not base_cfg.env.task:
        raise ValueError("env.task must be provided as a comma-separated string, e.g., \"Libero_10_Task_1,Libero_10_Task_2\"")

    task_list = [task.strip() for task in base_cfg.env.task.split(",") if task.strip()]
    if not task_list:
        raise ValueError("No valid tasks found in env.task")

    results = {}
    success_rates = []

    for task in task_list:
        logging.info(f"\n========== Evaluating task: {task} ==========")
        task_cfg = get_task_specific_cfg(base_cfg, task)

        # Run single-task evaluation using eval_main from eval.py
        eval_single_task_main(task_cfg)

        # Re-load metrics from output dir
        with open(Path(task_cfg.output_dir) / "eval_info.json") as f:
            metrics = json.load(f)
        results[task] = metrics["aggregated"]
        success_rates.append(metrics["aggregated"]["pc_success"])

    avg_success = sum(success_rates) / len(success_rates)
    print("\n========== Multi-task Evaluation Summary ==========")
    for task, metric in results.items():
        print(f"{task}: success = {metric['pc_success']:.2f}%, avg_reward = {metric['avg_sum_reward']:.2f}")
    print(f"Average Success Rate across {len(task_list)} tasks: {avg_success:.2f}%")

    # Save all results
    output_dir = Path(base_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "multitask_eval_info.json", "w") as f:
        json.dump({"per_task": results, "avg_success": avg_success}, f, indent=2)


if __name__ == "__main__":
    from lerobot.utils.utils import init_logging

    init_logging()
    eval_multitask_main()
