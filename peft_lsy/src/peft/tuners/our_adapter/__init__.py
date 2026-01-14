# peft/tuners/our_adapter/__init__.py

from peft.utils import register_peft_method

from .config import OurAdapterConfig
from .model import OurAdapterModel

__all__ = ["OurAdapterConfig", "OurAdapterModel"]

# PEFT looks up tuners by the name here. It will upper-case internally.
register_peft_method(
    name="our_adapter",
    config_cls=OurAdapterConfig,
    model_cls=OurAdapterModel,
)
