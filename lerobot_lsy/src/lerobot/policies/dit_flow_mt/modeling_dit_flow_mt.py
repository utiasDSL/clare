# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
# Heavy inspiration taken from
# * DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# * DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT
# * DiT Policy by Dasari et. al. : https://github.com/sudeepdasari/dit-policy

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel

from lerobot.constants import OBS_ENV_STATE, OBS_ROBOT, ACTION, OBS_IMAGES
from lerobot.policies.dit_flow_mt.configuration_dit_flow_mt import DiTFlowMTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

USE_STATE_PROJ = False
NAMING_AS_MLP = True

def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)



class LanguageEncoder(nn.Module):
    """
    Language Encoder using pretrained CLIP "Learning Transferable Visual Models From Natural Language Supervision"
    (paper: https://arxiv.org/pdf/2103.00020)
    """
    
    def __init__(self, config:DiTFlowMTConfig):
        super().__init__()
        
        self.config = config

        self.tokenizer = CLIPTokenizer.from_pretrained(config.language_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(config.language_model_name)
        self.cache = {}
        
        # Get the hidden size of CLIP model
        self.hidden_size = self.clip_model.config.hidden_size
        
        # Freeze the base model if specified
        if config.freeze_language_pretrained:
            self.clip_model.requires_grad_(False)
            # for param in self.clip_model.parameters():
            #     param.requires_grad = False
            self.clip_model.eval()

    def forward(self, texts):
        """
        Encodes input text into embeddings and projects to specified output dimension.
        
        Args:
            texts (list[str]): List of text strings to be encoded (batch size B).
        
        Returns:
            torch.Tensor: The projected text embeddings of shape (B, output_dim).
        """
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append(self.cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            # Tokenize the input texts
            inputs = self.tokenizer(
                uncached_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
            )
            
            # Move inputs to the same device as the model
            device = self.config.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings from CLIP text model
            with torch.set_grad_enabled(not self.clip_model.training):
                outputs = self.clip_model(**inputs)
            
            # Get the EOS token embeddings
            uncached_embeddings = outputs.pooler_output
            
            # Update cache
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = embedding.detach().cpu()  # Store in CPU to save GPU memory
        
        # Combine cached and uncached embeddings in the original order
        all_embeddings = [None] * len(texts)
        # Process uncached texts
        if uncached_texts:
            for i, emb in zip(uncached_indices, uncached_embeddings):
                all_embeddings[i] = emb
        for i, text in enumerate(texts):
            if text in self.cache and all_embeddings[i] is None:
                # Move cached embedding to same device as model
                all_embeddings[i] = self.cache[text].to(self.config.device)
        
        # Stack all embeddings into a single tensor
        return torch.stack(all_embeddings)


class DINOv2Encoder(nn.Module):
    def __init__(self, config: DiTFlowMTConfig):
        super().__init__()
        self.config = config
        self._model = AutoModel.from_pretrained(config.vit_name)
        self._model.to(config.device)
        self._model.requires_grad_(False) # hack
        self._model.eval() # hack

        self.hidden_size = self._model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self._model(x)
        cls_token = outputs.pooler_output # (B, 768)

        return cls_token


class _TimeNetwork(nn.Module):
    def __init__(self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000):
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)
        super().__init__()

        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        assert len(t.shape) == 1, "assumes 1d input timestep array"
        t = t[:, None] * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class MLP(nn.Module):
    def __init__(self, d_model=256, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)

        return x


class _DiTDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        if NAMING_AS_MLP:
            self.mlp = MLP(
                d_model=d_model, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=activation
            )
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)

        # mlp built upon reference could meet issue with peft
        # create mlp
        # self.mlp = nn.Sequential(
        #     self.linear1,
        #     self.activation,
        #     self.dropout2,
        #     self.linear2,
        #     self.dropout3,
        # )

        # create modulation layers
        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        x3 = self.mlp_modulate(self.norm2(x), cond)

        if NAMING_AS_MLP:
            x3 = self.mlp(x3)
        else:
            x3 = self.activation(self.linear1(x3))
            x3 = self.dropout2(x3)
            x3 = self.linear2(x3)
            x3 = self.dropout3(x3)

        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, t, cond):
        x = src
        for layer in self.layers:
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        cond_dim,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        clip_sample=False,
        clip_sample_range=1.0,
    ):
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk

        # positional encoding blocks
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        # clip the output samples
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def forward(self, noisy_actions, time, global_cond):
        c = self.cond_proj(global_cond)
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noisy_actions)  # [B, T, adim] -> [B, T, hidden_dim]
        ac_tokens = ac_tokens.transpose(0, 1)  # [B, T, hidden_dim] -> [T, B, hidden_dim]

        # Allow variable length action chunks
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]  # [T, B, hidden_dim]

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, c)

        # apply final epsilon prediction layer
        eps_out = self.eps_out(dec_out, time_enc, c)  # [T, B, hidden_dim] -> [T, B, adim]
        return eps_out.transpose(0, 1)  # [T, B, adim] -> [B, T, adim]

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, timesteps: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        # Use Euler integration to solve the ODE.
        batch_size, device = condition.shape[0], condition.device
        x_0 = self.sample_noise(batch_size, device, generator)
        dt = 1.0 / timesteps
        t_all = (
            torch.arange(timesteps, device=device).float().unsqueeze(0).expand(batch_size, timesteps)
            / timesteps
        )

        for k in range(timesteps):
            t = t_all[:, k]
            x_0 = x_0 + dt * self.forward(x_0, t, condition)
            if self.clip_sample:
                x_0 = torch.clamp(x_0, -self.clip_sample_range, self.clip_sample_range)
        return x_0

    def sample_noise(self, batch_size: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(batch_size, self.ac_chunk, self.ac_dim, device=device, generator=generator)


class DiTFlowMTPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiTFlowMTConfig
    name = "DiTFlowMT"

    def __init__(
        self,
        config: DiTFlowMTConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.dit_flow = DiTFlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.dit_flow.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        for key in batch:
            if key in self._queues:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)

        # batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.dit_flow.generate_actions(batch)

        # TODO(rcadene): make above methods return output dictionary?
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        loss = self.dit_flow.compute_loss(batch)
        return loss, None


class DiTFlowModel(nn.Module):
    def __init__(self, config: DiTFlowMTConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        self.language_encoder = LanguageEncoder(config).to(self.config.device)
        self.language_embedding_projection = nn.Linear(self.language_encoder.hidden_size, config.hidden_dim)

        language_cond_dim = config.hidden_dim

        if USE_STATE_PROJ:
            global_cond_dim = config.hidden_dim
            self.state_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.hidden_dim)
        else:
            global_cond_dim = self.config.robot_state_feature.shape[0]

        if self.config.image_features:
            self.pretrained_rgb_encoder = DINOv2Encoder(config)
            self.rgb_embedding_projection = nn.Linear(self.pretrained_rgb_encoder.hidden_size, config.hidden_dim)
            global_cond_dim += config.hidden_dim * len(self.config.image_features)

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]


        self.velocity_net = _DiTNoiseNet(
            ac_dim=config.action_feature.shape[0],
            ac_chunk=config.horizon,
            cond_dim=language_cond_dim + global_cond_dim * config.n_obs_steps,
            time_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

        self.num_inference_steps = config.num_inference_steps or 100
        self.training_noise_sampling = config.training_noise_sampling
        if config.training_noise_sampling == "uniform":
            self.noise_distribution = torch.distributions.Uniform(
                low=0,
                high=1,
            )
        elif config.training_noise_sampling == "beta":
            # From the Pi0 paper, https://www.physicalintelligence.company/download/pi0.pdf Appendix B.
            # There, they say the PDF for the distribution they use is the following:
            # $p(t) = Beta((s-t) / s; 1.5, 1)$
            # So, we first figure out the distribution over $t'$ and then transform it to $t = s - s * t'$.
            s = 0.999  # constant from the paper
            beta_dist = torch.distributions.Beta(
                concentration1=1.5,  # alpha
                concentration0=1.0,  # beta
            )
            affine_transform = torch.distributions.transforms.AffineTransform(loc=s, scale=-s)
            self.noise_distribution = torch.distributions.TransformedDistribution(
                beta_dist, [affine_transform]
            )

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Expand global conditioning to the batch size.
        if global_cond is not None:
            global_cond = global_cond.expand(batch_size, -1).to(device=device, dtype=dtype)

        # Sample prior.
        sample = self.velocity_net.sample(
            global_cond, timesteps=self.num_inference_steps, generator=generator
        )
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]

        # encode text description
        with torch.no_grad():
            language_embedding = self.language_encoder(batch["task"])
        language_cond_feats = self.language_embedding_projection(language_embedding)
        # language embedding as the first token
        global_cond_feats = [language_cond_feats]

        if USE_STATE_PROJ:
            states = einops.rearrange(batch[OBS_ROBOT], "b s ... -> (b s) ...", b=batch_size, s=n_obs_steps)
            states_embedding = self.state_proj(states)
            states_feature = einops.rearrange(states_embedding, "(b s) ... -> b s ...", b=batch_size, s=n_obs_steps)
            global_cond_feats.append(states_feature)
        else:
            global_cond_feats.append(batch[OBS_ROBOT].flatten(start_dim=1))

        # Extract image features.
        if self.config.image_features:
            images = einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...", b=batch_size, s=n_obs_steps, n=len(self.config.image_features))
            with torch.no_grad():
                img_cls_tokens = self.pretrained_rgb_encoder(images)
            img_embeddings = self.rgb_embedding_projection(img_cls_tokens)
            img_features = einops.rearrange(
                img_embeddings, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps, n=len(self.config.image_features)
            )
            global_cond_feats.append(img_features.flatten(start_dim=1))

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE].flatten(start_dim=1))

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1)

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        noise = self.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = self.noise_distribution.sample((trajectory.shape[0],)).to(trajectory.device)
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * trajectory

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.velocity_net(noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond)
        target = trajectory - noise
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
