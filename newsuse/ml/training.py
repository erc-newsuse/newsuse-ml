import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path
from shutil import rmtree
from typing import Any

import optuna
import torch
import transformers
from transformers import PreTrainedModel
from transformers.trainer import TrainOutput

from newsuse.config import Config, Paths
from newsuse.types import PathLike

from . import AutoModelForSequenceClassification
from .datasets import DatasetDict
from .evaluation import Evaluation

__all__ = ("TrainingArguments", "Trainer", "define_objective")


_LabelsT = int | Iterable[Hashable]
_LossOutputT = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def label_config(labels: _LabelsT) -> dict[str, Any]:
    if isinstance(labels, int):
        num_labels = labels
        labels = list(range(num_labels))
    else:
        labels = list(labels)
        num_labels = len(labels)
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in enumerate(labels)}
    config = {
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
    }
    return config


class TrainingArguments(transformers.TrainingArguments):
    def __init__(
        self,
        *args: Any,
        train_use_sample_weights: bool | None = False,
        eval_use_sample_weights: bool | None = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_use_sample_weights = train_use_sample_weights
        self.eval_use_sample_weights = eval_use_sample_weights


class Trainer(transformers.Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.compute_metrics, Evaluation):
            self.compute_metrics = self.compute_metrics.get_evaluation_function(self.model)
        self.problem_type = None
        self.loss_func = None

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,  # noqa
    ) -> _LossOutputT:
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)
        self._set_problem_type(labels)
        self._set_loss_func()
        outputs = model(labels=None, **inputs)
        loss = self.loss_func(outputs.logits, labels, weights)
        return (loss, outputs) if return_outputs else loss

    def train(self, *args: Any, ignore_warnings: bool = True, **kwargs: Any) -> TrainOutput:
        outdir = Path(self.args.output_dir)
        if (key := "resume_from_checkpoint") in kwargs:
            resume_from_checkpoint = kwargs[key]
        elif resume_from_checkpoint := self.args.resume_from_checkpoint:
            kwargs[key] = self.args.resume_from_checkpoint
        if not resume_from_checkpoint and outdir.exists():
            self.remove_checkpoints()
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
            return super().train(*args, **kwargs)

    def save_model(
        self,
        output_dir: PathLike | None = None,
        *args: Any,
        remove_checkpoints: bool = False,
        **kwargs: Any,
    ) -> None:
        if output_dir:
            output_dir = Path(output_dir)
            if output_dir.exists():
                for path in output_dir.glob("*"):
                    if not path.name.startswith("checkpoint-"):
                        if path.is_file():
                            path.unlink()
                        else:
                            rmtree(path)
        super().save_model(output_dir, *args, **kwargs)
        if output_dir and remove_checkpoints:
            self.remove_checkpoints()

    def remove_checkpoints(self) -> None:
        outdir = Path(self.args.output_dir)
        for checkpoint in outdir.glob("checkpoint-*"):
            if checkpoint.is_file():
                checkpoint.unlink()
            else:
                rmtree(checkpoint)

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        if self._train_use_sample_weights or self._eval_use_sample_weights:
            self._signature_columns = [*self._signature_columns, "weight"]

    @property
    def _train_use_sample_weights(self) -> bool:
        return self._should_use_weights("train_use_sample_weights")

    @property
    def _eval_use_sample_weights(self) -> bool:
        return self._should_use_weights("eval_train_sample_weights")

    def _should_use_weights(self, attr: str) -> bool:
        use_weights = getattr(self.args, attr, False)
        try:
            data_has_weights = "weight" in self.train_dataset.column_names
            if use_weights is None:
                use_weights = data_has_weights
        except AttributeError:
            use_weights = use_weights or False
        return use_weights

    def _set_problem_type(self, labels: torch.Tensor) -> None:
        if self.problem_type is None:
            if self.model.config.num_labels == 1:
                self.problem_type = "regression"
            elif self.model.config.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

    def _set_loss_func(self) -> None:
        if self.loss_func is None:
            if self.problem_type == "regression":
                self.loss_func = self._regression_loss
            elif self.problem_type == "single_label_classification":
                self.loss_func = self._single_label_classification_loss
            elif self.problem_type == "multi_label_classification":
                self.loss_func = self._multi_label_classification_loss

    def _regression_loss(
        self,
        yhat: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reduction = "mean" if weights is None else "none"
        loss_fct = torch.nn.MSELoss(reduction=reduction)
        if self.model.config.num_labels == 1:
            loss = loss_fct(yhat.squeeze(), y.squeeze())
        else:
            loss = loss_fct(yhat, y)
        if weights is not None:
            loss = (loss * weights).mean()
        return loss

    def _single_label_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reduction = "mean" if weights is None else "none"
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if weights is not None:
            loss = (loss * weights).mean()
        return loss

    def _multi_label_classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        reduction = "mean" if weights is None else "none"
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        loss = loss_fct(logits, labels)
        if weights is not None:
            loss = (loss * weights).mean()
        return loss


class Trial:
    """Experiment trial spawned from :class:`optuna.Trial`.

    Attributes
    ----------
    experiment
        Parent experiment instance.
    model
        Name of the model section in the config.
    trial
        :class:`optuna.Trial` instance.
    """

    def __init__(
        self,
        model: str,
        trial: optuna.Trial,
        config: Config,
        paths: Paths,
        dataset: DatasetDict,
        *,
        train_split: str = "train",
        eval_split: str = "test",
    ) -> None:
        self.model = model
        self.trial = trial
        self.config = config
        self.paths = paths
        self.tokenizer = self.get_tokenizer()
        self.dataset = self.preprocess_dataset(dataset)
        self.train_split = train_split
        self.eval_split = eval_split

    @property
    def model_config(self) -> Config:
        return self.config.models[self.model]

    def get_tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(self.model_config.base)

    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        return dataset.tokenize(self.tokenizer, **self.model_config.tokenize)

    def get_model_init(self) -> Callable[[], transformers.PreTrainedModel]:
        id2label = dict(enumerate(self.config.annotations.labels))
        label2id = {v: k for k, v in id2label.items()}

        def model_init() -> transformers.PreTrainedModel:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_config.base,
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
            )

        return model_init

    def get_training_args(self) -> TrainingArguments:
        hyper = {}
        for param, spec in self.model_config.training.hyper.space.items():
            spec = spec.copy()
            type = spec.pop("type")
            hyper[param] = getattr(self.trial, f"suggest_{type}")(param, **spec)
        return self.model_config.training.arguments(self.paths.models / self.model, **hyper)

    def get_training_callbacks(self) -> list[transformers.trainer_callback.TrainerCallback]:
        return [cb.make() for cb in self.model_config.training.callbacks]

    def get_compute_metrics(self) -> Evaluation:
        return self.model_config.training.evaluation()

    def get_trainer(self) -> Trainer:
        return Trainer(
            args=self.get_training_args(),
            model_init=self.get_model_init(),
            train_dataset=self.dataset[self.train_split],
            eval_dataset=self.dataset[self.eval_split],
            tokenizer=self.tokenizer,
            compute_metrics=self.get_compute_metrics(),
            callbacks=self.get_training_callbacks(),
        )


def define_objective(
    config: Config,
    paths: Paths,
    dataset: DatasetDict,
    models: Sequence[str],
    *,
    trial_class: type[Trial] = Trial,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical("model", models)
        trainer = trial_class(model, trial, config, paths, dataset).get_trainer()
        trainer.train()
        return _get_objective_value(trainer)

    return objective


def _get_objective_value(trainer: Trainer) -> str:
    metric = trainer.args.metric_for_best_model
    lowered = metric.lower()
    if lowered == "loss" or lowered.endswith("_loss"):
        field = "eval_loss"
        return min(log[field] for log in trainer.state.log_history if field in log)
    errmsg = f"'{metric}' selection metric is not supported"
    raise ValueError(errmsg)
