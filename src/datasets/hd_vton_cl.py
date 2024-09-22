import json
import logging
from os.path import join
from typing import Any, List

import numpy as np
from PIL import Image
from pydantic import BaseModel
from torch.utils import data
from torchvision import transforms

from src.datasets import DatasetSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    metadata_path: List[str]
    img_dir: List[str]


class VTONConfig(DatasetConfig):
    pass


class VITON_CL(data.Dataset):
    def __init__(self, config: VTONConfig, split: DatasetSplit):
        super(VITON_CL, self).__init__()
        self.config = config
        self.split = split

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return 0
