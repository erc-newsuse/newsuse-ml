from collections.abc import Callable, Iterable, Mapping
from functools import singledispatchmethod
from math import ceil
from pathlib import Path
from shutil import rmtree
from typing import Any, Self

import datasets
import numpy as np
import pandas as pd
from datasets.features import Features

from newsuse.annotations import Annotations
from newsuse.types import PathLike
from newsuse.utils import hashseed, inthash

__all__ = (
    "Dataset",
    "DatasetDict",
)


class DatasetDict(datasets.DatasetDict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name, split in self.items():
            self[name] = split

    def __setitem__(self, name: str, value: "Dataset") -> None:
        if isinstance(value, datasets.Dataset) and not isinstance(value, Dataset):
            value = Dataset.from_dataset(value)
        super().__setitem__(name, value)

    @property
    def features(self) -> Features:
        try:
            first = list(self)[0]
            return self[first].features
        except IndexError:
            return Features()

    def tokenize(self, *args: Any, **kwargs: Any) -> Self:
        """See :meth:`Dataset.tokenize`."""

        def _tokenize(dataset):
            if isinstance(dataset, datasets.Dataset):
                dataset = Dataset.from_dataset(dataset)
            return dataset.tokenize(*args, **kwargs)

        return self.__class__({s: _tokenize(d) for s, d in self.items()})

    def select_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().select_columns(*args, **kwargs))

    def filter(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().filter(*args, **kwargs))

    def select(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().select(*args, **kwargs))

    def map(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().map(*args, **kwargs))

    def class_encode_column(self, *args: Any, **kwargs) -> Self:
        return self.__class__(
            {
                name: split.class_encode_column(*args, **kwargs)
                for name, split in self.items()
            }
        )

    def rename_column(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().rename_column(*args, **kwargs))

    def rename_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().rename_columns(*args, **kwargs))

    def remove_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.__class__(super().remove_columns(*args, **kwargs))

    def sample(
        self,
        size: int | float | Mapping[str, int | float],  # noqa
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(size, Mapping):
            size = {n: size for n in self}
        dsets = {self[n].sample(s, seed=seed, **kwargs) for n, s in size.items()}
        return self.__class__(dsets)

    def stack(self, *splits: str) -> "Dataset":
        dataset = datasets.concatenate_datasets(
            [v for k, v in self.items() if not splits or k in splits]
        )
        return Dataset.from_dataset(dataset)

    def update_data(
        self,
        data: pd.DataFrame | datasets.Dataset | datasets.DatasetDict,
        splits: Mapping[str, float | int],
        *,
        key: str = "key",
        allow_data_loss: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Update dataset splits from ``data``."""
        keys = []
        dct = {}
        for name, split in self.items():
            _data = data[name] if isinstance(data, datasets.DatasetDict) else data
            dct[name] = split.update_data(_data, key=key, allow_data_loss=allow_data_loss)
            keys.extend(self[name][key])
        if isinstance(data, pd.DataFrame):
            data = Dataset.from_pandas(data)
        elif isinstance(data, datasets.DatasetDict):
            data = datasets.concatenate_datasets(list(data.values()))
        idx = np.where(~pd.Series(data["key"]).isin(keys))[0]
        new_splits = data.select(idx).make_splits(splits, **kwargs)
        for name, split in new_splits.items():
            if name in dct:
                dct[name] = dct[name].concat(split) if name in self else split
        return self.__class__(dct)

    def add_balancing_weights(self, *args: Any, **kwargs: Any) -> Self:
        data = self.stack().add_balancing_weights(*args, **kwargs)
        start = 0
        dct = {}
        for split, shape in self.shape.items():
            nrows, *_ = shape
            dct[split] = data.select(range(start, start + nrows))
            start += nrows
        return self.__class__(dct)


class Dataset(datasets.Dataset):
    def get_seed(self, seed: int | None = None) -> int:
        return hashseed(tuple(self["key"]), seed)

    @classmethod
    def from_annotations(
        cls,
        annotations: Annotations,
        *,
        top_n: int | None = None,
        seed: int = 0,
        metadata: str | Iterable[str] = (),
    ) -> Self:
        """Construct from instances of :class:`newsuse.annotations.Annotations`.

        Parameters
        ----------
        top_n
            Use only ``top_n`` observations with greatest number of annotations
            per sheet (randomly shuffled).
        metadata
            Column with additional metadata to retain.
        """
        data = annotations.data

        if top_n:
            seed = hashseed(data["key"], seed)
            n_annotations = data[annotations.annotator_cols].notnull().sum(axis=1)
            data = (
                data.assign(n_annotations=n_annotations)
                .groupby(annotations.config.sheet_index_name)
                .sample(frac=1, replace=False, random_state=seed)
                .sort_values("n_annotations", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
                .drop(columns="n_annotations")
            )

        if isinstance(metadata, str):
            metadata = [metadata]
        data = annotations.data[["key", *metadata, "human", "text"]].dropna(
            ignore_index=True
        )
        dataset = (
            cls.from_pandas(data)
            .rename_column("human", "label")
            .class_encode_column("label", names=annotations.config.labels)
        )
        return cls.from_dataset(dataset)

    def train_test_split(self, *args: Any, **kwargs: Any) -> Self:
        kwargs["seed"] = self.get_seed(kwargs.get("seed"))
        return super().train_test_split(*args, **kwargs)

    @singledispatchmethod
    def make_splits(
        self,
        splits: Mapping[str, float | int],
        *,
        seed: int | None = None,
        main_split: str = "train",
        **kwargs: Any,
    ) -> DatasetDict[str, Self]:
        """Split dataset."""
        splits = dict(splits)
        n_examples = len(self)

        for k, v in splits.items():
            if isinstance(v, float):
                splits[k] = int(ceil(v * n_examples))

        n_in_split = sum(splits.values())
        if n_in_split > n_examples:
            errmsg = "cannot define splits with more examples than the size of the dataset"
            raise ValueError(errmsg)
        if n_in_split < n_examples:
            if main_split in splits:
                errmsg = f"default split name '{main_split}' is already defined"
                raise ValueError(errmsg)
            splits[main_split] = n_examples - n_in_split

        seed = self.get_seed(seed)
        kwargs = {"seed": seed, "keep_in_memory": True, **kwargs}
        data = self.shuffle(**kwargs)
        dset = {}
        start = 0
        for name, n in splits.items():
            dset[name] = data.select(range(start, start + n), keep_in_memory=True)
            start += n
        return DatasetDict(dset)

    @make_splits.register
    def _(
        self, split: tuple, *splits: tuple[str, float | int], **kwargs: Any
    ) -> DatasetDict[str, Self]:
        dct = dict([split, *splits])
        return self.split(dct, **kwargs)

    def shuffle(self, *args: Any, **kwargs: Any) -> Self:
        kwargs["seed"] = self.get_seed(kwargs.get("seed"))
        return super().shuffle(*args, **kwargs)

    def concat(self, *others: datasets.Dataset | pd.DataFrame, **kwargs: Any) -> Self:
        _others = [
            o if isinstance(o, datasets.Dataset) else self.from_pandas(o) for o in others
        ]
        dataset = datasets.concatenate_datasets([self, *_others], **kwargs)
        return self.from_dataset(dataset)

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        *args: Any,
        clear_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        path = Path(dataset_dict_path)
        if path.exists():
            rmtree(path)
        super().save_to_disk(dataset_dict_path, *args, **kwargs)
        if clear_cache:
            for cached in path.rglob("cache-*.arrow"):
                cached.unlink()

    @classmethod
    def from_dataset(cls, dataset: datasets.Dataset) -> Self | DatasetDict:
        return cls(
            dataset.data,
            dataset.info,
            dataset.split,
            dataset._indices,
            dataset._fingerprint,
        )

    @classmethod
    def from_disk(cls, *args: Any, **kwargs: Any) -> Self | DatasetDict:
        """Load using :func:`datasets.load_from_disk`."""
        dataset = datasets.load_from_disk(*args, **kwargs)
        if isinstance(dataset, datasets.DatasetDict):
            return DatasetDict(dataset)
        return cls.from_dataset(dataset)

    def tokenize(
        self,
        tokenizer: Callable[[str, ...], list[int]],
        text_field: str = "text",
        *,
        padding: str = "max_length",
        truncation: bool = True,
        batched: bool = True,
        **kwargs: Any,
    ) -> Self:
        def tokenize(example):
            return tokenizer(
                example[text_field], padding=padding, truncation=truncation, **kwargs
            )

        dataset = self.map(tokenize, batched=batched)
        return self.from_dataset(dataset)

    def filter(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().filter(*args, **kwargs))

    def select(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().select(*args, **kwargs))

    def map(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().map(*args, **kwargs))

    def select_columns(self, *args: Any, **kwargs: Any) -> Self:
        dataset = super().select_columns(*args, **kwargs)
        return self.from_dataset(dataset)

    def rename_column(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().rename_column(*args, **kwargs))

    def rename_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().rename_columns(*args, **kwargs))

    def remove_columns(self, *args: Any, **kwargs: Any) -> Self:
        return self.from_dataset(super().remove_columns(*args, **kwargs))

    def class_encode_column(
        self, column: str, names: Iterable[str] | None = None, *args: Any, **kwargs: Any
    ) -> Self:
        names = list(names or [])

        if isinstance(self.features[column], datasets.ClassLabel):
            dataset = self.from_dataset(self)
            if names:
                dataset.info.features[column].names = names  # type: ignore
            return dataset

        dataset = super().class_encode_column(column, *args, **kwargs)
        schema = dataset.info.features[column]  # type: ignore
        if names:
            if schema.names == [str(i) for i in range(len(names))]:
                schema.names = names
            else:
                label2id = {n: i for i, n in enumerate(dataset.features[column].names)}
                remap = {label2id[n]: i for i, n in enumerate(names)}

                def do_remap(d):
                    d[column] = remap[d[column]]
                    return d

                dataset = dataset.map(do_remap)
                dataset.info.features[column].names = names  # type: ignore
        return self.from_dataset(dataset)

    @singledispatchmethod
    def sample(
        self,
        size,
        *,
        seed: int | np.random.Generator | None = None,
        hash_key: str | None = "key",
        **kwargs: Any,
    ) -> Self:
        if size % 1 != 0:
            errmsg = f"'size' must be an integer, not {size}"
            raise ValueError(errmsg)
        size = int(size)
        if size <= 0:
            errmsg = f"size has to be positive, not {size}"
            raise ValueError(errmsg)
        if not isinstance(seed, np.random.Generator):
            if hash_key and hash_key in self.column_names:
                seed = seed or 0
                seed += inthash(tuple(self[hash_key]))
            rng = np.random.default_rng(seed)
        else:
            rng = seed
        idx = np.arange(len(self))
        rng.shuffle(idx)
        idx = idx[:size]
        return self.select(idx, **kwargs)

    @sample.register
    def _(self, size: float, **kwargs: Any) -> Self:
        if size > 1:
            errmsg = "'size' cannot exceed '1.0' fraction of the examples in the dataset"
            raise ValueError(errmsg)
        size = int(ceil(len(self)))
        return self.sample(size, **kwargs)

    def add_balancing_weights(
        self, *fields: str, weight_field_name: str = "weight"
    ) -> Self:
        """Add weights balancing examples in groups given by ``*field``."""
        df = self.to_pandas()
        if not isinstance(df, pd.DataFrame):
            df = pd.concat(list(df), axis=0, ignore_index=True)
        n = df.groupby(list(fields)).size()
        target = len(self) / len(n)
        w = target / n
        idx = df[list(fields)]
        idx = idx.apply(tuple, axis=1) if len(fields) > 1 else idx[fields[0]]
        idx.reset_index(drop=True, inplace=True)
        df[weight_field_name] = w.loc[idx].to_numpy()
        df[weight_field_name] *= len(df) / df[weight_field_name].sum()
        dataset = self.__class__.from_pandas(df)
        info = self.info.copy()
        info.features[weight_field_name] = dataset.features[weight_field_name]  # type: ignore
        return dataset.update_info(info)

    def update_data(
        self,
        data: pd.DataFrame | datasets.Dataset,
        *,
        key: str = "key",
        allow_data_loss: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Update current data with new values from ``data``."""
        if isinstance(data, datasets.Dataset):
            _data = data.to_pandas()
            if not isinstance(_data, pd.DataFrame):
                kwargs = {"axis": 0, "ignore_index": True, **kwargs}
                _data = pd.concat(list(_data), **kwargs)
            data = _data
        mask = data[key].isin(self[key])
        data = data.loc[mask, self.column_names].reset_index(drop=True)
        if not allow_data_loss and (n_missing := len(self) - len(data)) > 0:
            errmsg = f"incoming data is missing {n_missing} examples"
            raise ValueError(errmsg)
        dataset = self.from_pandas(data)
        return dataset.update_info(self.info)

    def update_info(self, info: datasets.DatasetInfo) -> Self:
        """Update metadata based on ``dataset``."""
        return self.__class__(self.data, info, self.split, self._indices, self._fingerprint)
