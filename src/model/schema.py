import os
from pathlib import Path
from typing import List, Literal, Optional

import pydantic

from configs.envs import GLOBAL_CONFIG
from src.model.image_encoder import HFImageEncoder
from src.model.projection_head import MLPProjectionHead


class EncoderSchema(pydantic.BaseModel):
    source: str
    name: str
    pretrained: bool = True
    gradient_checkpointing: bool = False

    cache_dir: Path = GLOBAL_CONFIG.cache_dir
    local_files_only: Optional[bool] = pydantic.Field(None)

    max_length: int = 256

    freeze: bool = False


class ImageEncoderSchema(EncoderSchema):
    model_type: str = "swin"

    img_size: int = 224
    # Avoid warning
    model_config = pydantic.ConfigDict(
        protected_namespaces=("protect_me_", "also_protect_")
    )

    pooling: Literal["bos", "mean"] = "mean"


class ProjectionHeadSchema(pydantic.BaseModel):
    name: str
    projection_dim: int
    dropout_rate: float
    fully_connected_dim: List[int] = []
    activation: str = "gelu"


def load_image_encoder(config: ImageEncoderSchema):
    if config.source.lower() == "huggingface":
        local_file_exists = config.local_files_only
        if not local_file_exists:
            local_file_exists = os.path.exists(
                os.path.join(
                    config.cache_dir,
                    f'models--{config.name.replace("/", "--")}',
                )
            )
        _image_encoder = HFImageEncoder(
            **{
                **config.dict(exclude={"source"}),
                "local_files_only": local_file_exists,
            }
        )
    else:
        raise NotImplementedError(
            f"Image encoder source {config.source} not implemented"
        )

    return _image_encoder


def load_projection_head(config: ProjectionHeadSchema, embedding_dim: int):
    if config.name.lower() == MLPProjectionHead.NAME.lower():
        _projection_head = MLPProjectionHead(
            **{"embedding_dim": embedding_dim, **config.dict(exclude={"name"})}
        )

    else:
        raise NotImplementedError(
            f"Projection head {config.name} not implemented"
        )

    return _projection_head
