from importlib.metadata import PackageNotFoundError, version

__version__: str | None
try:
    __version__ = version(__package__)
except PackageNotFoundError:
    # package is not installed
    __version__ = None
