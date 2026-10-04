"""Dataset versioning and reproducibility helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import BenchmarkCase, BenchmarkDataset, DatasetVersion
from .serialization import dataset_hash


def build_dataset_version(
    *,
    benchmark_version: str = "0.1.0",
    seed: int = 0,
    generator_version: str = "0.1.0",
    config: Mapping[str, Any] | None = None,
    cases: Sequence[BenchmarkCase] | None = None,
) -> DatasetVersion:
    """Build a manifest object before cases are attached."""

    return DatasetVersion(
        benchmark_version=str(benchmark_version),
        seed=int(seed),
        generator_version=str(generator_version),
        config=dict(config or {}),
        case_count=len(cases or []),
    )


def compute_dataset_hash(dataset: BenchmarkDataset | Mapping[str, Any]) -> str:
    return dataset_hash(dataset)


def finalize_dataset(
    dataset: BenchmarkDataset,
    *,
    benchmark_version: str | None = None,
    seed: int | None = None,
    generator_version: str | None = None,
    config: Mapping[str, Any] | None = None,
    compute_statistics: bool = True,
) -> BenchmarkDataset:
    """Fill case count/hash (and optionally statistics) in-place and return it.

    Mutating and returning the same object is convenient for generator code;
    callers that need immutability can pass ``BenchmarkDataset.from_dict``
    first.  The hash excludes timestamps, statistics, and the previous hash.
    """

    if benchmark_version is not None:
        dataset.version.benchmark_version = str(benchmark_version)
    if seed is not None:
        dataset.version.seed = int(seed)
    if generator_version is not None:
        dataset.version.generator_version = str(generator_version)
    if config is not None:
        dataset.version.config = dict(config)
    dataset.version.case_count = len(dataset.cases)
    if compute_statistics:
        dataset.compute_statistics()
    dataset.version.dataset_hash = dataset_hash(dataset)
    return dataset


def verify_dataset_hash(dataset: BenchmarkDataset) -> bool:
    """Return whether the stored hash matches current content."""

    stored = str(dataset.version.dataset_hash or "")
    if stored.lower().startswith("sha256:"):
        stored = stored.split(":", 1)[1]
    return bool(stored) and stored == dataset_hash(dataset)


# Public alias used in some integrations.
finalize = finalize_dataset
