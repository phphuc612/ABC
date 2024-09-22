import datetime
import enum
import logging
import time
from typing import List, Optional

import pydantic
import torch
import torch.nn.functional as F
from torch import nn

from src import utils
from src.exceptions.CollapseException import CollapseException
from src.model import (
    ImageEncoderSchema,
    ProjectionHeadSchema,
    load_image_encoder,
    load_projection_head,
)
from src.utils.factory import create
from src.utils.logger import MetricLogger

logger = logging.getLogger(__name__)


@enum.unique
class MODALITY(enum.Enum):
    Image = "I"
    Text = "T"

    @classmethod
    def has_value(cls, value) -> bool:
        return value in cls._value2member_map_


class ModelConfig(pydantic.BaseModel, extra="allow"):
    model_name: str
    image_encoder: ImageEncoderSchema
    projection_head: Optional[ProjectionHeadSchema] = pydantic.Field(None)

    temperature: Optional[float] = pydantic.Field(None)
    queue_size: int = 65536
    momentum: float = 0.999
    alpha: float = 0.4
    sub_batch_size: int = 16

    # See #pydantic.config.ConfigDict.protected_namespaces
    model_config = pydantic.ConfigDict(
        protected_namespaces=("protect_me_", "also_protect_")
    )

    @pydantic.validator("model_name")
    def validate_model(cls, model):
        try:
            create(model)
        except Exception as e:
            raise ValueError(f"Model {model} not found. Error: {e}")

        return model


class MAC(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MAC, self).__init__()
        self.config = config

        # Initialization
        # Image encoder
        self.image_encoder = load_image_encoder(self.config.image_encoder)

        # Projection heads
        self.have_projection = False
        if self.config.projection_head is not None:
            self.have_projection = True
            self.image_projection = load_projection_head(
                config=self.config.projection_head,
                embedding_dim=self.image_encoder.out_dim,
            )

        self.sub_batch_size = self.config.sub_batch_size

        if self.config.temperature:
            logger.info(f"Using temperature: {self.temperature}")
            self.temperature = nn.Parameter(
                # scalar value only
                torch.ones([])
                * self.temperature
            )
        else:
            logger.info("No temperature provided")
            self.temperature = torch.ones([])

    def encode_image(self, img):
        img_embed = self.image_encoder.embed(
            img,
            pooling=self.config.image_encoder.pooling,
        )
        if self.have_projection:
            img_embed = self.image_projection(img_embed)

        return F.normalize(img_embed, dim=-1)

    def _assert_no_collapse(self, img, rtol=0, atol=1e-5):
        def _is_collapsed(embeddings):
            return torch.isclose(
                embeddings, embeddings[0], rtol=rtol, atol=atol
            ).all()

        msg = ""
        if _is_collapsed(img):
            msg = msg + "Latent space is (nearly) collapsed\n"

        if msg:
            raise CollapseException(msg)

    def forward(
        self,
        all_query_imgs,
        all_key_imgs,
        device,
        scaler,
    ):
        sub_query_imgs = torch.split(all_query_imgs, self.sub_batch_size)
        sub_key_imgs = torch.split(all_key_imgs, self.sub_batch_size)

        with torch.no_grad():
            self.temperature.clamp_(0.001, 1)

            key_img_embeds = []

            for key_imgs in sub_key_imgs:
                # Current sub-batch embeddings
                key_img_embeds.append(self.encode_image(key_imgs))

            # Stack embeddings
            key_img_embeds = torch.concat(key_img_embeds)
            self._assert_no_collapse(key_img_embeds)

            # Labels
            all_labels = torch.arange(all_query_imgs.size(0), device=device)

        # Sub-batch training
        avg_loss = 0
        for sub_id, query_imgs in enumerate(sub_query_imgs):
            st_id = sub_id * self.sub_batch_size
            # for drop_last = False or (train_bs, sub_bs) != sub_bs
            en_id = st_id + query_imgs.size(0)

            query_img_embeds = self.encode_image(query_imgs)

            # Compute similarity scores
            sim = query_img_embeds @ key_img_embeds.T / self.temperature

            # Losses
            # sum all cross entropy losses with gradients
            loss = F.cross_entropy(
                sim, all_labels[st_id:en_id], reduction="mean"
            )

            # Gradient accumulation step
            loss /= len(sub_query_imgs)
            avg_loss += loss.item()

            # Backward for training
            if loss.grad_fn is not None:
                assert (
                    scaler is not None
                ), "Scaler must be provided for training"
                scaler.scale(loss).backward()

        return avg_loss

    @staticmethod
    def train_step(
        model,
        data_loader,
        optimizer,
        scaler,
        epoch,
        device,
        args,
    ):
        # train
        model.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
        )
        metric_logger.add_meter(
            "temp", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
        )
        metric_logger.add_meter(
            "avg_loss", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
        )

        header = "Train Epoch: [{}]".format(epoch)
        print_freq = 20
        # step_size = 50
        # warmup_iterations = warmup_steps * step_size

        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        for _, (query_img, key_img) in enumerate(
            metric_logger.log_every(data_loader, print_freq, logger, header)
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(  # type: ignore
                device_type="cuda", dtype=torch.float16
            ):
                query_img = query_img.to(device)
                key_img = key_img.to(device)
                avg_loss = model(query_img, key_img, device, scaler)

            scaler.step(optimizer)
            scaler.update()

            metric_logger.update(avg_loss=avg_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(temp=model.temperature.item())

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info(f"Averaged stats: {metric_logger.global_avg()}")
        return {
            k: "{:.9f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    @torch.no_grad()
    def evaluate_step(
        model,
        data_loader,
        device,
        topk: List[int] = [1, 5, 10],
    ):
        raise NotImplementedError
