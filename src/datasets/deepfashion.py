import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from configs.envs import GLOBAL_CONFIG
from src.datasets import DatasetSplit

logger = logging.getLogger(__name__)


class CustomRandAugment(transforms.RandAugment):
    def _augmentation_space(
        self,
        num_bins: int,
        image_size: Tuple[int, int],
    ) -> Dict[str, Tuple[Tensor, bool]]:

        ori_ops = super()._augmentation_space(num_bins, image_size)
        del ori_ops["Posterize"]
        del ori_ops["Solarize"]

        return ori_ops


class DeepFashionCL(Dataset):
    def __init__(
        self,
        train_pair_path: str,
        split: DatasetSplit,
        img_root_path: Union[str, Path] = GLOBAL_CONFIG.deep_fashion_dir,
        img_size: int = 512,
        num_aug_ops: int = 5,
    ) -> None:
        super().__init__()
        self._split = split
        self._img_size = img_size
        self._img_root_path = Path(img_root_path)
        self.train_pairs = self._create_train_pair(train_pair_path)

        self._transform = transforms.Compose(
            [
                transforms.Resize((self._img_size, self._img_size)),
                CustomRandAugment(num_ops=num_aug_ops),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _create_train_pair(
        self, train_pair_path: str
    ) -> List[Tuple[int, int]]:
        paired_ids = []
        with open(train_pair_path, "r") as f:
            for line in f:
                ids: List[int] = list(map(int, line.strip().split()))
                paired_ids.append(ids)

        train_pairs = []
        for ids in paired_ids:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    train_pairs.append((ids[i], ids[j]))

        return train_pairs

    def __len__(self) -> int:
        return len(self.train_pairs)

    def __getitem__(self, index: int):
        def _get_path(img_id) -> Path:
            return self._img_root_path / f"{img_id}.jpg"

        id1, id2 = self.train_pairs[index]
        img1, img2 = (
            self._load_img(_get_path(id1)),
            self._load_img(_get_path(id2)),
        )

        return img1, img2

    def _load_img(self, path: Path):
        """Read and transform images with RandAug"""
        img = Image.open(path)
        img_tensor = self._transform(img)

        return img_tensor
