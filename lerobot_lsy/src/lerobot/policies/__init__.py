# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .act.configuration_act import ACTConfig as ACTConfig
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
from .diffusion_transformer.configuration_diffusion_transformer import DiffusionTransformerConfig as DiffusionTransformerConfig
from .dit.configuration_dit import DiTConfig as DiTConfig
from .dit_mt.configuration_dit_mt import DiTMTConfig as DiTMTConfig
# from .dit_update_mt.configuration_dit_update_mt import DiTUpdateMTConfig as DiTUpdateMTConfig
# from .dit_flow_update_mt.configuration_dit_flow_update_mt import DiTFlowUpdateMTConfig as DiTFlowUpdateMTConfig
from .dit_flow.configuration_dit_flow import DiTFlowConfig as DiTFlowConfig
from .dit_flow_mt.configuration_dit_flow_mt import DiTFlowMTConfig as DiTFlowMTConfig
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
# from .smolvla_image_only.configuration_smolvla_image_only import SmolVLAImageConfig as SmolVLAImageConfig
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
