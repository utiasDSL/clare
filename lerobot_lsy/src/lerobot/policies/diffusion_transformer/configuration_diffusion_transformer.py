from dataclasses import dataclass, field

from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

@PreTrainedConfig.register_subclass("diffusion_transformer")
@dataclass
class DiffusionTransformerConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 5
    horizon: int = 16 # action steps that the model predict
    n_action_steps: int = 8 # action steps that the model 

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone_size: int = 18
    pretrained_backbone_weights: str | None = None
    avg_pool: bool = True
    vision_backbone_norm_name: str = "group_norm"
    vision_backbone_norm_num_groups: int = 16
    use_separate_rgb_encoder_per_camera = True

    # for spatial softmax resnet encoder
    use_spatial_softmax:bool = True
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = False
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32

    # Transformer layers.
    dim_model: int = 256
    n_heads: int = 4
    dropout_emb: float = 0.0
    dropout_attn: float = 0.3
    causal_attn = True
    n_encoder_layers: int = 0 # set n_encoder_layers as 0 to use MLP encoder
    n_decoder_layers: int = 8

    # Noise scheduler.
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    set_alpha_to_one: bool=True
    steps_offset: int = 0

    # Inference.
    num_inference_steps: int | None = 16

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-3
    optimizer_grad_clip_norm: float = 10.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.vision_backbone_size not in [18, 34, 50]:
            raise ValueError(
                f"`vision_backbone_size` must be one of the ResNet variants. Got {self.vision_backbone_size}."
            )
        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.horizon} for `chunk_size`."
            )


    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> None:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None