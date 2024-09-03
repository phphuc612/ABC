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

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
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
