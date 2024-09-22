import logging
from typing import List

from torch import nn

logger = logging.getLogger(__name__)


class AbstractProjectionHead(nn.Module):
    NAME: str = "abstract"


class MLPProjectionHead(AbstractProjectionHead):
    NAME: str = "mlp"

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int,
        dropout_rate: float,
        fully_connected_dim: List[int] = [],
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        logger.info(
            f"Loading MLP projection head\n"
            f"Embedding dim: {embedding_dim}\n"
            f"Projection dim: {projection_dim}\n"
            f"Dropout rate: {dropout_rate}\n"
            f"Fully connected dim: {fully_connected_dim}\n"
            f"Activation: {activation}\n"
            f"Unprocessed kwargs: {kwargs}\n"
        )

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} not supported")

        self.fc_layers = []
        fully_connected_dim = (
            [embedding_dim] + fully_connected_dim + [projection_dim]
        )
        for cur, nxt in zip(fully_connected_dim[:-1], fully_connected_dim[1:]):
            self.fc_layers.append(nn.Linear(cur, nxt))
        self.fc_layers = nn.ModuleList(self.fc_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(projection_dim)

        self._init_params()

    def _init_params(self):
        for fc in self.fc_layers:
            nn.init.xavier_normal_(fc.weight)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def forward(self, x):
        for fc in self.fc_layers:
            x_ = fc(x)
            x_ = self.dropout(x_)
            x_ = self.activation(x_)
            if x.shape == x_.shape:
                x = x + x_
            else:
                x = x_

        x = self.layer_norm(x)
        return x
