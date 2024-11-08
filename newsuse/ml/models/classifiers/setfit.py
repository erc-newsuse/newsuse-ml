from collections.abc import Callable, Iterable
from pathlib import Path
from shutil import rmtree
from typing import Any, Self

import setfit
from datasets import ClassLabel

from newsuse.types import PathLike

from ...datasets import Dataset


class TrainingArguments(setfit.TrainingArguments):
    pass


class Trainer(setfit.Trainer):
    def train(self, *args: Any, **kwargs: Any) -> None:
        outdir = Path(self.args.output_dir)
        if not kwargs.get("resume_from_checkpoint") and outdir.exists():
            for checkpoint in outdir.glob("checkpoint-*"):
                if checkpoint.is_file():
                    checkpoint.unlink()
                else:
                    rmtree(checkpoint)

        super().train(*args, **kwargs)

        tmp = Path.cwd() / "tmp_trainer"
        if tmp.exists():
            rmtree(tmp)

    def save_model(
        self, output_dir: PathLike | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Save model."""
        if output_dir is None:
            output_dir = self.args.output_dir
        for path in Path(output_dir).glob("*"):
            if not path.name.startswith("checkpoint-"):
                if path.is_file():
                    path.unlink()
                else:
                    rmtree(path)
        self.model.save_pretrained(output_dir, *args, **kwargs)


class SetFitModel(setfit.SetFitModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike,
        *args: Any,
        labels: ClassLabel | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if labels is not None:
            if isinstance(labels, ClassLabel):
                labels = labels.names
            model.labels = list(labels)
        return model

    @classmethod
    def factory(
        cls,
        name_or_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[..., Self]:
        """Get ``model_init()`` factory function."""

        def model_init():
            return cls.from_pretrained(name_or_path, *args, **kwargs)

        return model_init

    @classmethod
    def get_tokenizer(
        cls,
        name_or_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """SetFit models do not use external tokenizer."""

    @classmethod
    def get_training_arguments(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> TrainingArguments:
        """Get :class:`TrainingArguments` instance."""
        return TrainingArguments(*args, **kwargs)

    @classmethod
    def get_trainer(
        cls,
        *,
        args: TrainingArguments,
        model: Self | None = None,
        model_init: Callable[..., Self] | None = None,
        **kwargs: Any,
    ) -> Trainer:
        """Get :class:`Trainer` instance."""
        kwargs.pop("tokenizer", None)
        return Trainer(
            args=args,
            model=model,
            model_init=model_init,
            **kwargs,
        )

    @classmethod
    def preprocess_dataset(
        cls,
        dataset: Dataset,
        *args: Any,  # noqa
        **kwargs: Any,  # noqa
    ) -> Dataset:
        """Preprocess ``dataset`` before training."""
        return dataset
