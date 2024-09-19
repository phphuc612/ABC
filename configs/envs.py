import typing
from pathlib import Path

import torch
from attr import dataclass


@dataclass
class _GlobalConfig:
    wd = Path().resolve()
    data_dir = wd / "data"
    ckpt_dir = wd / "ckpts"
    config_dir = wd / "configs"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = wd / "cache"

    # DATA DIR **DEFAULT** CONFIG
    data_dir = wd / "data"
    deep_fashion_dir = data_dir / "DeepFashion2"

    # GLOBAL CONFIG VALIDATION
    def __call__(self, *args, **kwargs):
        # Data dir
        assert Path.is_dir(
            self.data_dir
        ), f"{self.data_dir} is not a directory"

        # Checkpopint dir
        assert Path.is_dir(
            self.ckpt_dir
        ), f"{self.ckpt_dir} is not a directory"

        # Config dir
        assert Path.is_dir(
            self.config_dir
        ), f"{self.config_dir} is not a directory"

        # Cache dir
        if not Path.is_dir(self.cache_dir):
            Path.mkdir(self.cache_dir, exist_ok=True, parents=True)
        assert Path.is_dir(
            self.cache_dir
        ), f"{self.cache_dir} is not a directory"


GLOBAL_CONFIG = _GlobalConfig()
