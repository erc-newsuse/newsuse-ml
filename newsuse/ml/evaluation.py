from collections.abc import Callable, Iterable
from functools import singledispatch, singledispatchmethod
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    EvalPrediction,
    Pipeline,
    PreTrainedModel,
    TextClassificationPipeline,
)

from .datasets import Dataset

__all__ = ("Evaluation", "Evaluator")


_EvalFuncT = Callable[[Iterable[float]], float]


@singledispatch
def get_evaluation_function(
    num_labels: int,
    metric: Literal["precision", "recall", "f1"] = "f1",
    *,
    reduction: str | _EvalFuncT = "hmean",
    **kwargs: Any,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Get performance function for model.

    Arguments
    ---------
    num_labels
        Number of labels.
        Determined from config if a :class:`transformers.PreTrainedModel` is passed.
    metric
        Performance metric to use.
        It is calculated for all labels used as targets and averaged using ``reduction``.
    reduction
        Method for averaging label-specific metrics.
    **kwargs
        Passed to metric function.
    """
    if isinstance(reduction, str):
        try:
            reduction = {
                "min": min,
                "max": max,
                "mean": np.mean,
                "gmean": sp.stats.gmean,
                "hmean": sp.stats.hmean,
            }[reduction]
        except KeyError as exc:
            errmsg = f"'{reduction}' reduction is not supported"
            raise ValueError(errmsg) from exc
    if isinstance(metric, str):
        try:
            func = {
                "f1": f1_score,
                "precision": precision_score,
                "recall": recall_score,
            }[metric]
        except KeyError as exc:
            errmsg = f"'{metric}' metric is not supported"
            raise ValueError(errmsg) from exc

    @singledispatch
    def evaluation_function(labels, predictions) -> dict[str, float]:
        scores = {
            f"{metric}-{label}": func(labels, predictions, pos_label=label, **kwargs)
            for label in range(num_labels)
        }
        main_score = reduction(list(scores.values()))
        scores = {metric: main_score, **scores}
        return {k: float(v) for k, v in scores.items()}

    @evaluation_function.register
    def _(eval_pred: EvalPrediction) -> dict[str, float]:
        labels = eval_pred.label_ids
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return evaluation_function(labels, predictions)

    return evaluation_function


@get_evaluation_function.register
def _(
    model: PreTrainedModel, *args: Any, **kwargs: Any
) -> Callable[[EvalPrediction], dict[str, float]]:
    return get_evaluation_function(model.num_labels, *args, **kwargs)


class Evaluation:
    """Evaluation class for configuring model performance assessment.

    Attributes
    ----------
    metric
        Name of the evaluation metric or evaluation function.
    target_name
        Name of the data field storing target values.
    **kwargs
        Additional arguments passed to :func:`get_evaluation_function`
        when only metric name is provided. Otherwise, ``**kwargs`` will
        be passed to the metric function at call time.
    """

    def __init__(
        self,
        metric: str | Callable[[EvalPrediction], dict[str, float]] = "f1",
        **kwargs: Any,
    ) -> None:
        self.metric = metric
        self.metric_kwargs = kwargs

    def get_evaluation_function(
        self, model_or_pipeline: PreTrainedModel | Pipeline
    ) -> _EvalFuncT:
        if isinstance(self.metric, Callable):
            return self.metric
        if isinstance(model_or_pipeline, Pipeline):
            model_or_pipeline = model_or_pipeline.model
        return get_evaluation_function(model_or_pipeline, self.metric, **self.metric_kwargs)


class Evaluator:
    """Evaluator class for running model performance assessment.

    Attributes
    ----------
    pipeline
        Model pipeline for generating predictions.
    evaluation
        Evaluation configuration or name of evaluation metric.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        evaluation: Evaluation | str | None = None,
        *,
        target_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialization method.

        Arguments
        ---------
        **kwargs
            Are passed to :class:`Evaluation` when ``evaluation is None``.
            Ignored otherwise.
        """
        self.pipeline = pipeline
        if evaluation is None:
            evaluation = Evaluation(**kwargs)
        elif isinstance(evaluation, str):
            evaluation = Evaluation(evaluation, **kwargs)
        self.evaluation = evaluation
        self.evaluate = self.evaluation.get_evaluation_function(self.pipeline)
        if not target_name:
            if isinstance(self.pipeline, TextClassificationPipeline):
                target_name = "label"
            else:
                errmsg = f"cannot determine 'target_name' for ;{type(self.pipeline)}'"
                raise ValueError(errmsg)
        self.target_name = target_name

    @property
    def num_labels(self) -> int:
        return self.pipeline.model.config.num_labels

    @singledispatchmethod
    def __call__(self, dataset, **kwargs: Any) -> pd.Series:  # noqa
        self._raise_not_implemented(dataset)

    @__call__.register
    def _(self, dataset: Dataset, **kwargs: Any) -> pd.Series:
        return self._compute(self.pipeline, dataset, **kwargs)

    @__call__.register
    def _(self, dataset: pd.DataFrame, **kwargs: Any) -> pd.Series:
        return self(Dataset.from_pandas(dataset), **kwargs)

    @staticmethod
    def _raise_not_implemented(obj):
        errmsg = f"evaluation for '{type(obj)}' is not yet implemented"
        raise NotImplementedError(errmsg)

    @singledispatchmethod
    def _compute(self, pipeline, dataset: Dataset, **kwargs: Any) -> pd.Series:  # noqa
        self._raise_not_implemented(pipeline)

    @_compute.register
    def _(
        self, pipeline: TextClassificationPipeline, dataset: Dataset, **kwargs: Any
    ) -> pd.Series:
        predictions = list(pipeline(dataset, **kwargs))
        predictions = [p[self.target_name] for p in predictions]
        labels = list(dataset[self.target_name])
        if getattr(pipeline.model.config, "id2label", None):
            predictions = [pipeline.model.config.label2id[p] for p in predictions]
            labels = [pipeline.model.config.label2id.get(label, label) for label in labels]
        return pd.Series(self.evaluate(labels, predictions), name="metrics")
