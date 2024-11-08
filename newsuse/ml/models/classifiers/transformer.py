from collections.abc import Callable, Mapping
from typing import Any, Literal, NoReturn, Self

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from newsuse.utils import validate_call

from .modules import SequenceClassificationHead, SequenceClassificationHeadConfig

_ProblemT = Literal[
    "regression", "single_label_classification", "multi_label_classification"
]
_LossFuncT = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def _raise_unsupported_model(model: str) -> NoReturn:
    errmsg = f"'{model}' model type is not supported"
    raise ValueError(errmsg)


class SequenceClassifierTransformerConfig(PretrainedConfig):
    """Config for :class:`NewsuseSequenceClassifier`.

    Attributes
    ----------
    base
        Name or path of a pretrained base language model.
    base_config
        Config of the base model.
    head
        Config of the classification head.
    """

    model_type = "sequence-classifier-transformer"

    @validate_call
    def __init__(
        self,
        base_name_or_path: str | None = None,
        base: PretrainedConfig | Mapping | None = None,
        head: SequenceClassificationHeadConfig | Mapping | None = None,
        **kwargs: Any,
    ) -> None:
        if not base_name_or_path and isinstance(base, Mapping):
            base_name_or_path = base.get("_name_or_path")
        if isinstance(base, PretrainedConfig):
            if base_name_or_path and base._name_or_path != base_name_or_path:
                errmsg = (
                    f"'base={base_name_or_path}' but 'base_config' "
                    f"is for '{base._name_or_path}'"
                )
                raise ValueError(errmsg)
        elif base_name_or_path:
            base = AutoConfig.from_pretrained(base_name_or_path, **(base or {}))
        self.base = base

        if self.base:
            if self.base.model_type == "distilbert":
                base_dim = self.base.dim
                dropout = self.base.seq_classif_dropout
            elif self.base.model_type == "bert":
                base_dim = self.base.hidden_size
                dropout = self.base.classifier_dropout or 0.2
            else:
                _raise_unsupported_model(self.base.model_type)

            if isinstance(head, SequenceClassificationHeadConfig):
                head.dim = base_dim
            else:
                head = dict(head or {})
                head["dim"] = base_dim
                head["dropout"] = head.get("dropout", dropout)
                head["initializer_range"] = head.get(
                    "initializer_range", self.base.initializer_range
                )
        head = SequenceClassificationHeadConfig(**(head or {}))
        self.head = head
        super().__init__(**kwargs)
        self.head.num_labels = self.num_labels

    @property
    def base_name_or_path(self) -> str:
        return self.base._name_or_path


class SequenceClassifierTransformer(PreTrainedModel):
    config_class = SequenceClassifierTransformerConfig

    def __init__(self, config: config_class) -> None:
        super().__init__(config)
        self.base = AutoModel.from_pretrained(self.config.base_name_or_path)
        self.head = SequenceClassificationHead(self.config.head)
        self.loss_func = None
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        weight: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        weights = weight
        output = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self._get_pooled_output(output)
        logits = self.head(pooled_output)

        loss = None
        if labels is not None:
            self._set_problem_type(labels)
            if self.loss_func is None:
                self.loss_func = self._get_loss_function()
            loss = self.loss_func(logits, labels, weights)

        if not return_dict:
            output = (logits,) + output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def _get_pooled_output(self, output: BaseModelOutput) -> torch.Tensor:
        if self.config.base.model_type == "distilbert":
            hidden_states = output[0]
            pooled_output = hidden_states[:, 0]
        elif self.config.base.model_type == "bert":
            pooled_output = output.pooler_output
        else:
            _raise_unsupported_model(self.config.base.model_type)
        return pooled_output

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(self.config.base_name_or_path)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any], **kwargs: Any) -> Self:
        """Construct from ``config_dict`` with optional additional ``**kwargs``."""
        config = cls.config_class(**config_dict, **kwargs)
        return cls(config)

    def _set_problem_type(self, labels: torch.Tensor) -> _ProblemT:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

    def _get_loss_function(self) -> _LossFuncT:
        # ruff: noqa: C901
        if self.config.problem_type == "regression":

            def loss_fct_regression(
                logits: torch.Tensor,
                labels: torch.Tensor,
                weights: torch.Tensor | None = None,
            ) -> torch.Tensor:
                reduction = "mean" if weights is None else "none"
                loss_fct = torch.nn.MSELoss(reduction=reduction)
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
                if weights is not None:
                    loss = (loss * weights).mean()
                return loss

            return loss_fct_regression

        if self.config.problem_type == "single_label_classification":

            def loss_fct_single_label(
                logits: torch.Tensor,
                labels: torch.Tensor,
                weights: torch.Tensor | None = None,
            ) -> torch.Tensor:
                reduction = "mean" if weights is None else "none"
                loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if weights is not None:
                    loss = (loss * weights).mean()
                return loss

            return loss_fct_single_label

        if self.config.problem_type == "multi_label_classification":

            def loss_fct_multi_label(
                logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
            ) -> torch.Tensor:
                reduction = "mean" if weights is None else "none"
                loss_fct = torch.nn.BCEWithLogitsLoss(reduction=reduction)
                loss = loss_fct(logits, labels)
                if weights is not None:
                    loss = (loss * weights).mean()
                return loss

            return loss_fct_multi_label

        errmsg = f"cannot define loss function for problem type {self.config.problem_type}"
        raise ValueError(errmsg)


# Register -------------------------------------------------------------------------------

AutoConfig.register(
    SequenceClassifierTransformerConfig.model_type, SequenceClassifierTransformerConfig
)
AutoModelForSequenceClassification.register(
    SequenceClassifierTransformerConfig, SequenceClassifierTransformer
)
