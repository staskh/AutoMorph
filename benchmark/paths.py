# ABOUTME: Resolves the benchmark dataset root from an environment variable or the repo default.
# ABOUTME: Only fetch scripts create these directories; readers raise a clear error when data is absent.

"""Benchmark data path resolution.

One gitignored root, resolved as: explicit argument -> environment override -> the default
directory at the repository root.

==================  ================================  ====================
Root                Environment override              Default
==================  ================================  ====================
datasets            ``$AUTOMORPH_BENCHMARK_DATA``     ``.benchmark_data/``
==================  ================================  ====================

Kept separate from ``$AUTOMORPH_DATA``, which is where a pipeline *run* writes its images and
Results. This root holds downloaded reference datasets, which outlive any single run.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ENV = "AUTOMORPH_BENCHMARK_DATA"
DATA_DEFAULT = ".benchmark_data"


def data_root(explicit=None):
    """Root holding downloaded benchmark datasets.

    :param explicit: overrides both the environment variable and the default.
    :return: the datasets root, which is not guaranteed to exist.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(DATA_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return REPO_ROOT / DATA_DEFAULT


def dataset_dir(name, explicit=None):
    """Directory for one dataset under the data root.

    :param name: dataset identifier, e.g. ``"fives"``.
    :param explicit: overrides the data-root resolution.
    :return: the dataset directory.
    :raises FileNotFoundError: if it does not exist, naming the fetch script to run.
    """
    path = data_root(explicit) / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"benchmark dataset {name!r} not found at {path} — "
            f"run `python -m benchmark.fetch_{name}` (see benchmark/docs/dataset.md)"
        )
    return path
