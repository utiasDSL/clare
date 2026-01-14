import os
import json
import draccus
from dataclasses import dataclass
from pathlib import Path

from peft import OurAdapterConfig
from peft.utils import CONFIG_NAME

@dataclass
class CreateOurAdapterConfigPipeline:
    save_path: Path 
    peft_cfg: OurAdapterConfig


@draccus.wrap()
def main(cfg: CreateOurAdapterConfigPipeline):

    os.makedirs(cfg.save_path, exist_ok=True)

    with open(cfg.save_path / CONFIG_NAME, "w") as f, draccus.config_type("json"):
        draccus.dump(cfg.peft_cfg, f, indent=4)

if __name__ == "__main__":
    main()