# peft/tuners/clare/__init__.py

from peft.utils import register_peft_method

from .config import CLAREConfig
from .model import CLAREModel

__all__ = ["CLAREConfig", "CLAREModel"]

# PEFT looks up tuners by the name here. It will upper-case internally.
register_peft_method(
    name="clare",
    config_cls=CLAREConfig,
    model_cls=CLAREModel,
)
