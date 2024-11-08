from collections.abc import Iterable, Mapping, Sequence
from functools import singledispatchmethod, wraps
from typing import Any

import pandas as pd
import torch
import torch.utils
import torch.utils.data
import transformers
from tqdm.auto import tqdm
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.pt_utils import KeyDataset as _KeyDataset

from . import datasets

__all__ = (
    "pipeline",
    "TextClassificationPipeline",
)


_ExampleT = dict[str, Any]
_OutputT = dict[str, float]


class SimpleDataset(torch.utils.data.Dataset):
    """Simple indexable :mod:`torch` dataset
    fetching examples from elements of ``self.data``.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1,2,3], "b": [1,2,3]})
    >>> dataset = SimpleDataset(df)
    >>> dataset[0]
    {'a': 1, 'b': 1}
    >>> dataset[:2]
    {'a': [1, 2], 'b': [1, 2]}

    It is compatible with :class:`newsuse.ml.KeyDaset`.

    >>> keyed = KeyDataset(dataset, "a")
    >>> keyed[0]
    1
    >>> keyed[:2]
    [1, 2]
    """

    def __init__(
        self, data: Sequence[_ExampleT] | Iterable[_ExampleT] | pd.DataFrame
    ) -> None:
        if not isinstance(data, Sequence | pd.DataFrame):
            data = list(data)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> _ExampleT:
        return self._get(self.data, idx)

    @singledispatchmethod
    def _get(self, data, idx: int) -> _ExampleT:
        return data[idx]

    @_get.register
    def _(self, data: pd.Series, idx: int | slice) -> _ExampleT:
        out = data.iloc[idx]
        if isinstance(idx, slice):
            out = out.tolist()
        return out

    @_get.register
    def _(self, data: pd.DataFrame, idx: int | slice) -> _ExampleT:
        out = data.iloc[idx]
        if isinstance(idx, slice):
            return out.to_dict(orient="list")
        return out.to_dict()


class KeyDataset(_KeyDataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset | Sequence | pd.DataFrame, key: str
    ) -> None:
        if not isinstance(dataset, torch.utils.data.Dataset):
            dataset = SimpleDataset(dataset)
        super().__init__(dataset, key)


@wraps(transformers.pipeline)
def pipeline(*args: Any, **kwargs: Any) -> transformers.Pipeline:
    if kwargs.get("device") is None and torch.cuda.is_available():
        kwargs["device"] = "cuda"
    pipe = transformers.pipeline(*args, **kwargs)
    if pipe.tokenizer is None and hasattr(pipe.model, "get_tokenizer"):
        pipe.tokenizer = pipe.model.get_tokenizer()
    return pipe


class TextClassificationPipeline(transformers.TextClassificationPipeline):
    """:class:`transformers.pipelines.TextClassificationPipeline`
    with automatic batch processing and support for :class:`pandas.DataFrame`s
    and single-dictionary outputs with probabilities for all labels.

    See also
    --------
    transformers.pipelines.TextClassificationPipeline : Parent pipeline class.
    """

    @singledispatchmethod
    def __call__(
        self,
        inputs,
        key: str | None = None,
        *,
        progress: bool | Mapping[str, Any] = False,
        **kwargs: Any,
    ) -> _OutputT | list[_OutputT]:
        if isinstance(progress, bool):
            tqdm_kwargs = {"disable": not progress}
        else:
            tqdm_kwargs = {"disable": not progress, **progress}
        try:
            tqdm_kwargs["total"] = len(inputs)
        except TypeError:
            pass
        if not isinstance(inputs, torch.utils.data.Dataset):
            inputs = KeyDataset(inputs, key) if key else SimpleDataset(inputs)
        outputs = tqdm(super().__call__(inputs, **kwargs), **tqdm_kwargs)
        return list(outputs)

    @__call__.register
    def _(
        self, inputs: datasets.Dataset, key: str = "text", *args: Any, **kwargs: Any
    ) -> _OutputT | list[_OutputT]:
        dataset = KeyDataset(inputs, key)
        return self.__call__(dataset, *args, **kwargs)

    @__call__.register
    def _(
        self,
        inputs: pd.DataFrame,
        key: str = "text",
        *args: Any,
        **kwargs: Any,
    ) -> _OutputT | list[_OutputT]:
        dataset = KeyDataset(inputs, key)
        return self.__call__(dataset, *args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        outputs = super().postprocess(*args, **kwargs)
        if not isinstance(outputs, Mapping):
            if kwargs["top_k"] == 1:
                outputs = outputs[0]
            else:
                outputs = {o["label"]: o["score"] for o in outputs}
        return outputs

    def _sanitize_parameters(self, *args: Any, **kwargs: Any) -> tuple[dict, dict, dict]:
        preprocess, forward, postprocess = super()._sanitize_parameters(*args, **kwargs)
        preprocess = {"padding": True, "truncation": True}
        postprocess = {"top_k": 1, **postprocess}
        return preprocess, forward, postprocess


# Register pipelines ---------------------------------------------------------------------

PIPELINE_REGISTRY.register_pipeline(
    "text-classification",
    pipeline_class=TextClassificationPipeline,
    pt_model=transformers.AutoModelForSequenceClassification,
)

# ----------------------------------------------------------------------------------------
