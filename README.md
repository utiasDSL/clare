# CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion

This is the official anonymous code repository for the paper **"CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion"**.

## Abstract

To teach robots complex manipulation tasks, it is now a common practice to fine-tune a pre-trained vision-language-action model (VLA) on task-specific data. However, since this recipe updates existing representations, it is unsuitable for long-term operation in the real world, where robots must continually adapt to new tasks and environments while retaining the knowledge they have already acquired. Existing continual learning methods for robotics commonly require storing previous data (exemplars), struggle with long task sequences, or rely on task identifiers for deployment.

To address these limitations, we propose **CLARE**, a general, parameter-efficient framework for non-exemplar continual learning with VLAs. CLARE introduces lightweight modular adapters into selected feedforward layers and autonomously expands the model only where necessary when learning a new task, guided by layer-wise feature similarity. During deployment, an autoencoder-based routing mechanism dynamically activates the most relevant adapters without requiring task labels. Through extensive experiments on the LIBERO benchmark, we show that CLARE achieves high performance on new tasks without catastrophic forgetting of earlier tasks, significantly outperforming even exemplar-based methods.

## Project Structure

This codebase is built on top of two open-source frameworks from Hugging Face:

- **`lerobot_lsy/`**: Modified version of [LeRobot](https://github.com/huggingface/lerobot) for designing, training, and fine-tuning Vision-Language-Action (VLA) models
- **`peft_lsy/`**: Modified version of [PEFT](https://github.com/huggingface/peft) implementing the CLARE algorithm as a LoRA-compatible adapter

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd clare
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n clare python=3.10
   conda activate clare
   ```

3. **Install PEFT-LSY in editable mode**
   ```bash
   cd peft_lsy
   pip install -e .
   cd ..
   ```

4. **Install LeRobot-LSY in editable mode**
   ```bash
   cd lerobot_lsy
   pip install -e .
   cd ..
   ```

5. **Install additional dependencies** (if needed)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## PEFT Configuration

CLARE uses a custom PEFT adapter configuration. Below is an example configuration for the CLARE adapter:

```json
{
  "peft_type": "CLARE",
  "task_type": null,
  "auto_mapping": {
    "base_model_class": "PeftWrapperPolicy",
    "parent_library": "__main__"
  },
  "base_model_name_or_path": null,
  "revision": null,
  "target_modules": ".*velocity_net.cond_proj",
  "inference_mode": true,
  "batch_first": true,
  "num_learned_task": 1,
  "feature_dim": 2574,
  "out_feature_dim": 512,
  "use_trainable_copy": false,
  "add_zero_init_conv_layer": false,
  "structure": {
    "_policy.dit_flow.velocity_net.cond_proj.0": [1, 1]
  },
  "discriminator_cfg": {
    "type": "autoencoder",
    "batch_first": true,
    "feature_dim": 2574,
    "feature_fusion": false,
    "fused_feature_dim": null,
    "hidden_dim": 256,
    "latent_dim": 128,
    "num_tokens": 16,
    "lora_rank": 32,
    "lora_alpha": 32,
    "use_lora": false,
    "use_momentum": true,
    "momentum": 0.1,
    "max_batches_tracked": 2000
  },
  "func_adapter_cfg": {
    "hidden_dim": 1024,
    "lora_rank": 32,
    "lora_alpha": 32,
    "use_lora": false
  }
}
```

### Key Configuration Parameters

- **`target_modules`**: Regular expression pattern matching the model layers where adapters will be applied
- **`feature_dim`**: Dimension of input features for the adapter
- **`out_feature_dim`**: Output dimension after adapter transformation
- **`discriminator_cfg`**: Configuration for the autoencoder-based routing mechanism
  - `type`: Type of discriminator (autoencoder or other)
  - `hidden_dim`: Hidden layer dimension
  - `latent_dim`: Latent space dimension for the autoencoder
  - `use_momentum`: Whether to use momentum for feature statistics
- **`func_adapter_cfg`**: Configuration for functional adapter modules
- **`num_learned_task`**: Number of tasks learned so far

## Usage

### Training with CLARE on LIBERO Benchmark

The main training script is located at `lerobot_lsy/src/lerobot/scripts/clare.py`. Below is an example command for continual learning on the LIBERO-10 benchmark:

```bash
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=1000 \
    --job_name=clare_libero_10_task_0 \
    --output_dir=./outputs/clare_libero_10_task_0 \
    --dataset.repo_id=libero_10_image_task_0 \
    --dataset.root=$DATASET_ROOT/libero_10_image_task_0 \
    --policy.path=$POLICY_ROOT/dit_mt_libero_90_pretrain \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=20000 \
    --env.type=libero \
    --env.task=Libero_10_Task_0 \
    --eval.batch_size=20 \
    --eval.n_episodes=100 \
    --eval.max_episodes_rendered=100 \
    --eval_freq=200000 \
    --save_freq=20000 \
    --log_freq=100 \
    --peft_cfg_path=$DATASET_ROOT/peft_config/clare_config.json \
    --expand_threshold=10.00 \
    --detect_distribution_shift_steps=200 \
    --detect_distribution_shift_batch_size=32 \
    --detect_distribution_shift_num_workers=16 \
    --detect_distribution_shift_log_freq=10 \
    --train_discriminators_steps=2000 \
    --train_discriminators_batch_size=32 \
    --train_discriminators_num_workers=16 \
    --train_discriminators_log_freq=50 \
    --train_discriminators_eval_freq=2000 \
    --train_discriminators_save_freq=2000 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=<your-wandb-entity>
```

### Key Training Arguments

#### Dataset & Model
- `--dataset.repo_id`: Hugging Face dataset repository ID
- `--dataset.root`: Local path to the dataset
- `--policy.path`: Path to the pre-trained VLA model

#### Training Configuration
- `--batch_size`: Training batch size
- `--num_workers`: Number of data loading workers
- `--steps`: Total training steps
- `--seed`: Random seed for reproducibility

#### CLARE-Specific Parameters
- `--peft_cfg_path`: Path to the PEFT configuration JSON file
- `--expand_threshold`: Threshold for autonomous adapter expansion (based on feature similarity)
- `--detect_distribution_shift_steps`: Steps for distribution shift detection
- `--train_discriminators_steps`: Training steps for the autoencoder-based routing mechanism

#### Evaluation
- `--env.type`: Environment type (e.g., libero, aloha)
- `--env.task`: Specific task name
- `--eval.n_episodes`: Number of evaluation episodes
- `--eval_freq`: Frequency of evaluation (in training steps)

#### Logging
- `--wandb.enable`: Enable Weights & Biases logging
- `--wandb.project`: W&B project name
- `--log_freq`: Logging frequency

## Environment Variables

Set the following environment variables before running experiments:

```bash
export DATASET_ROOT=/path/to/your/datasets
export POLICY_ROOT=/path/to/your/pretrained/policy
```

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{anonymous2025clare,
  title={CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

## License

This project is released under the same license as the original LeRobot and PEFT frameworks.

## Acknowledgments

This work builds upon:
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
- [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/) benchmark

We thank the authors of these projects for their open-source contributions.
