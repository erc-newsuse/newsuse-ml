from typing import Any, Literal

import torch
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN, get_activation

from newsuse.utils import validate_call

_NormT = Literal["layer"]
_NormStrategyT = Literal["standard", "ic"]
_ActivationT = Literal[*list(ACT2FN)]


class FeedForwardNetworkConfig(PretrainedConfig):
    """Config for :class:`FeedForwardNetwork`.

    Attributes
    ----------
    dim
        Number of input and output features.
    num_layers
        Number of layers.
        Zero layers are equivalent to identitity transformation of inputs.
    bias
        Should linear layers use bias weights (aka intercept).
    activation
        Activation function.
    dropout
        Dropout value to apply.
    norm
        Output normalization function.
    norm_strategy
        Strategy for combining dropout and normalization.
    """

    model_type = "feed-forward-network"

    @validate_call
    def __init__(
        self,
        dim: PositiveInt = 1,
        num_layers: NonNegativeInt = 0,
        bias: bool = True,
        activation: _ActivationT = "relu",
        dropout: NonNegativeFloat = 0.1,
        norm: _NormT = "layer",
        norm_strategy: _NormStrategyT = "ic",
        **kwargs: Any,
    ) -> None:
        self.num_layers = num_layers
        self.dim = dim
        self.bias = bias
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.norm_strategy = norm_strategy
        super().__init__(**{**kwargs, "num_labels": 1})


class FeedForwardNetwork(PreTrainedModel):
    """Feed forward network model."""

    config_class = FeedForwardNetworkConfig

    def __init__(self, config: FeedForwardNetworkConfig) -> None:
        super().__init__(config)
        self.activation = get_activation(self.config.activation)
        self.dropout = (
            torch.nn.Dropout(self.config.dropout) if self.config.dropout else None
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "linear": torch.nn.Linear(
                            in_features=self.config.dim,
                            out_features=self.config.dim,
                            bias=self.config.bias,
                        ),
                        "norm": torch.nn.LayerNorm(self.config.dim),
                    }
                )
                for _ in range(self.config.num_layers)
            ]
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        for layer in self.layers:
            output = layer.linear(output)
            if self.config.norm_strategy == "ic":
                output = self.activation(output)
                output = layer.norm(output)
                if self.dropout and self.dropout.p:
                    output = self.dropout(output)
            else:
                output = layer.norm(output)
                output = self.activation(output)
                if self.dropout and self.dropout.p:
                    output = self.dropout(output)
        return output
