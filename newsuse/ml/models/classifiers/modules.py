from collections.abc import Mapping
from typing import Any

import torch
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import get_activation

from newsuse.utils import validate_call

from ..feedforward import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
    _ActivationT,
    _NormStrategyT,
    _NormT,
)

__all__ = ("SequenceClassificationHead", "SequenceClassificationHeadConfig")


class SequenceClassificationHeadConfig(PretrainedConfig):
    """Config for :class:`SequenceClassificationHeadConfig`.

    Attributes
    ----------
    dim
        Embedding dimension.
    ffn
        Config of feed forward network for transforming embeddings.
    activation
        Activation function to apply to embeddings.
    dropout
        Dropout to apply to embeddings.
    norm
        Normalization function to apply to embeddings.
    norm_strategy
        Strategy for combining dropout and normalization.
    initializer_range
        Gaussian standard deviation for initializing linear weights.
    """

    model_type = "sequence-classification-head"

    @validate_call
    def __init__(
        self,
        dim: PositiveInt = 1,
        activation: _ActivationT = "relu",
        dropout: NonNegativeFloat = 0.2,
        norm: _NormT | None = None,
        norm_strategy: _NormStrategyT = "ic",
        ffn: FeedForwardNetworkConfig | Mapping | None = None,
        initializer_range: PositiveFloat = 0.02,
        **kwargs: Any,
    ) -> None:
        self.dim = dim
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.norm_strategy = norm_strategy
        if ffn is None:
            ffn = {}
        if isinstance(ffn, FeedForwardNetworkConfig):
            ffn.dim = dim
        else:
            ffn = dict(ffn)
            ffn["dim"] = dim
            ffn["activation"] = ffn.get("activation", activation)
            if dropout:
                ffn["dropout"] = ffn.get("dropout", dropout)
            if norm:
                ffn["norm"] = ffn.get("norm", norm)
            ffn["norm_strategy"] = ffn.get("norm_strategy", norm_strategy)
            ffn = FeedForwardNetworkConfig(**ffn)
        if not isinstance(ffn, FeedForwardNetworkConfig):
            ffn = FeedForwardNetworkConfig(**(ffn or {}))
        self.ffn = ffn
        super().__init__(**kwargs)
        self.initializer_range = initializer_range


class EmbeddingFineTuner(torch.nn.Module):
    """Embedding fine-tuner."""

    def __init__(self, config: SequenceClassificationHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = torch.nn.LayerNorm(self.config.dim) if self.config.norm else None
        self.ffn = FeedForwardNetwork(self.config.ffn)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        output = embeddings
        if self.norm:
            output = self.norm(output)
        output = self.ffn(output)
        return output


class SequenceClassificationHead(PreTrainedModel):
    """Sequence classification head model."""

    config_class = SequenceClassificationHeadConfig

    def __init__(self, config: SequenceClassificationHeadConfig) -> None:
        super().__init__(config)
        self.finetuner = EmbeddingFineTuner(self.config)
        dropout = torch.nn.Dropout(self.config.dropout) if self.config.dropout else None
        norm = torch.nn.LayerNorm(self.config.dim) if self.config.norm else None
        self.preclassifier = torch.nn.Linear(self.config.dim, self.config.dim)
        if self.config.norm_strategy != "ic":
            self.norm = norm
        self.activation = get_activation(self.config.activation)
        if self.config.norm_strategy == "ic":
            self.norm = norm
        self.dropout = dropout
        self.classifier = torch.nn.Linear(self.config.dim, self.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        output = self.finetuner(embedding)
        output = self.preclassifier(output)
        if self.config.norm_strategy == "ic":
            output = self.activation(output)
            if self.norm:
                output = self.norm(output)
        else:
            if self.norm:
                output = self.norm(output)
            output = self.activation(output)
        if self.dropout and self.dropout.p:
            output = self.dropout(output)
        logits = self.classifier(output)
        return logits

    def _init_weights(self, module: torch.nn.Module) -> None:
        # Follows 'DistilBertPreTrainedModel._init_weights()'
        if self.config.initializer_range is None:
            self.config.initializer_range = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
