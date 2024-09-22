import logging
from pathlib import Path
from typing import Union

from torch import nn
from transformers import AutoConfig, AutoModel, SwinModel, ViTModel

from configs.envs import GLOBAL_CONFIG

logger = logging.getLogger(__name__)


class HFImageEncoder(nn.Module):
    def __init__(
        self,
        name: str = "facebook/deit-base-distilled-patch16-224",
        pretrained: bool = True,
        gradient_checkpointining: bool = False,
        cache_dir: Union[Path, str] = GLOBAL_CONFIG.cache_dir,
        model_type: str = "vit",
        local_files_only: bool = True,
        **kwargs,
    ):
        super().__init__()
        logger.info(
            f"Loading image encoder: {name}\n"
            f"Pretrained: {pretrained}\n"
            f"Gradient checkpointing: {gradient_checkpointining}\n"
            f"Cache dir: {cache_dir}\n"
            f"Model type: {model_type}\n"
            f"Local files only: {local_files_only}\n"
            f"Unprocessed kwargs: {kwargs}\n"
        )

        self.model_type = model_type
        if pretrained:
            if self.model_type == "swin":
                self.image_encoder = SwinModel.from_pretrained(name)
            else:
                self.image_encoder = AutoModel.from_pretrained(
                    pretrained_model_name_or_path=name,
                    add_pooling_layer=False,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
        else:
            model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            if type(model_config).__name__ == "ViTConfig":
                self.image_encoder = ViTModel(
                    config=model_config, add_pooling_layer=False
                )
            else:
                raise NotImplementedError(
                    f"Not support model type: {type(model_config).__name__}"
                )

        if (
            gradient_checkpointining
            and self.image_encoder.supports_gradient_checkpointing
        ):
            self.image_encoder.gradient_checkpointing_enable()  # type: ignore

        self.out_dim = self.image_encoder.config.hidden_size  # type: ignore

    def forward(self, image):
        if self.model_type == "vit":
            output = self.image_encoder(
                pixel_values=image, interpolate_pos_encoding=True
            )  # type: ignore
        elif self.model_type == "swin":
            output = self.image_encoder(pixel_values=image)  # type: ignore

        # (batch, seq_len, hidden_size)
        return output["last_hidden_state"]  # type: ignore

    def embed(self, image, pooling="mean"):
        raw = self.forward(image)

        if pooling == "mean":
            pooling_vector = raw.mean(dim=1)
        elif pooling == "bos":
            pooling_vector = raw[:, 0]
        else:
            raise ValueError(f"Pooling {pooling} not supported")

        return pooling_vector
