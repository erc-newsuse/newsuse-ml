# Newsuse ML: Machine learning tools for the NEWSUSE project.

The purpose of this package is to facilitate some of the basic machine learning tasks
typical for the NEWSUSE project. This includes data preprocessing, model training and evaluation.

Moreover, `newsuse.ml` comes with some extra tools
(implemented in the `newususe.data` package) for easy handling of data files
in various formats through a customized `DataFrame` class, which inherits from
the well known `pandas.DataFrame`.

> Currently `.csv`, `.tsv`, `.jsonl`, `.parquet` and excel formats are supported.

In general methods and classes provided by `newsuse.ml` are simple wrappers around
tools from _HuggingFace_ packages such as `transformers`, `datasets`, `evaluate` or `pandas`.
Thus, dependency on `newsuse.ml` can be quite easily droppped in favor of direct use
of the standard packages. However, the core purpose of `newsuse.ml` is to reduce the
boilerplate and free the user from having to remember about all package dependencies
and proper versioning.

## Installation

```bash
pip install git+ssh://git@github.com/erc-newsuse/newsuse-ml.git
# From a specific branch, e.g. 'dev'
pip install git+ssh://git@github.com/erc-newsuse/newsuse-ml.git@dev
```

## Usage examples

All usage examples below assume that the `newsuse.ml` package is already installed.

### Text classification

Below is an example for applying a
[`text-classification` pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines)
to a textual dataset represented as a `pandas.DataFrame`
(or any object that behaves like one)

```python
from newsuse.data import DataFrame
from newsuse.ml import pipeline

# Path to a datafile which should be interpretable as a DataFrame
# with at least one column called 'text' (other columns are ignored)
datapath = <path-to-data-file>
# Path to a local model directory with a `transformers` model
modelpath = <path-to-model-directory>

data = DataFrame.from_(datapath)
# GPU device is selected automatically if available
# But it can be controlled to - see the docstring
classifier = pipeline("text-classification", modelpath)
# Set `batch_size` to value appropriate for the available RAM/GPU
results = DataFrame(classifier(data, progress=True, batch_size=16))
```

## Development

### Environment setup and dev installation

```bash
# Clone the repo
git clone git+ssh://git@github.com/erc-newsuse/newsuse-ml.git
# Or from a specific branch, e.g. 'dev'
git clone git+ssh://git@github.com/erc-newsuse/newsuse-ml.git@dev
```

Configure the environment and install the package after cloning.
The make commands below also set up and configure version control (GIT).

```bash
cd ml
conda env create -f environment.yaml
conda activate newsuse-ml
make init
```

`Makefile` defines several commands that are useful during development.

```bash
# Clean up auxiliary files
make clean
# List explicitly imported dependencies
make list-deps
```

### Testing

```bash
pytest
## With automatic debugger session
pytest --pdb
```

### Unit test coverage statistics

```bash
# Calculate and display
make coverage
# Only calculate
make cov-run
# Only display (based on previous calculations)
make cov-report
```
