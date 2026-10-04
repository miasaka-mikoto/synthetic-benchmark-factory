"""Deterministic, dependency-free serializers for benchmark artifacts."""

from __future__ import annotations

import csv
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, get_args, get_origin, get_type_hints

from .models import BenchmarkCase, BenchmarkDataset, DatasetStatistics

T = TypeVar("T")


def _normalize(value: Any, *, sort_sets: bool = True) -> Any:
    """Convert arbitrary model values into JSON-compatible primitives."""

    if isinstance(value, Enum):
        return _normalize(value.value, sort_sets=sort_sets)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _normalize(getattr(value, field.name), sort_sets=sort_sets) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _normalize(item, sort_sets=sort_sets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item, sort_sets=sort_sets) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_normalize(item, sort_sets=sort_sets) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)) if sort_sets else values
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # Numpy/Pandas and similar optional values often expose ``item``; using it
    # opportunistically keeps this module dependency-free.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _normalize(item(), sort_sets=sort_sets)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def to_dict(value: Any) -> Any:
    """Recursively convert a model or arbitrary object into JSON data."""

    return _normalize(value)


def canonical_json(value: Any) -> str:
    """Return canonical UTF-8 JSON used for hashes and reproducibility."""

    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any, *, algorithm: str = "sha256") -> str:
    """Hash canonical JSON; SHA-256 is the default dataset hash algorithm."""

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc
    digest.update(canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def _parse_datetime(value: Any) -> Any:
    if isinstance(value, datetime) or value is None:
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def from_dict(value: Any, target_type: type[T] | None = None) -> T | Any:
    """Rehydrate one of the public dataclasses from a JSON dictionary.

    ``from_dict(data)`` is intentionally also useful for generic JSON: without
    a target type it recursively copies values and leaves primitive types
    untouched.  For a model use ``from_dict(data, BenchmarkCase)``.
    """

    if target_type is None:
        if isinstance(value, Mapping):
            return {str(k): from_dict(v) for k, v in value.items()}
        if isinstance(value, list):
            return [from_dict(item) for item in value]
        return value
    if value is None:
        return None
    if target_type is datetime:
        return _parse_datetime(value)
    if target_type is date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)
    origin = get_origin(target_type)
    if origin in (list, Sequence):
        subtype = get_args(target_type)[0] if get_args(target_type) else Any
        return [from_dict(item, subtype if isinstance(subtype, type) else None) for item in value]
    if origin in (dict, Mapping):
        return dict(value)
    try:
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            return target_type.coerce(value) if hasattr(target_type, "coerce") else target_type(value)
    except TypeError:
        pass
    if isinstance(target_type, type) and is_dataclass(target_type):
        # Models perform nested coercion in __post_init__, so passing the
        # dictionary through is both robust and compatible with old exports.
        return target_type(**dict(value))
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return value


def from_json(text: str, target_type: type[T] | None = None) -> T | Any:
    return from_dict(json.loads(text), target_type)


def to_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, indent=indent, allow_nan=False)


def write_json(value: Any, path: str | Path, *, indent: int | None = 2) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(to_json(value, indent=indent) + "\n", encoding="utf-8")
    return destination


def read_json(path: str | Path, target_type: type[T] | None = None) -> T | Any:
    return from_json(Path(path).read_text(encoding="utf-8"), target_type)


def dataset_hash(value: BenchmarkDataset | Mapping[str, Any] | Sequence[Any]) -> str:
    """Compute a content hash independent of timestamps and run results.

    A generator run at two different wall-clock times must produce the same
    hash when its seed/config/content are identical.  Manifest timestamps and
    attached model outputs are therefore excluded from the content identity;
    the latter are evaluation artifacts rather than dataset cases.
    """

    def content_only(item: Any, key: str | None = None) -> Any:
        if isinstance(item, Mapping):
            # Validation and run bookkeeping are derived metadata.  They may
            # be added after generation (or refreshed by a newer validator)
            # and therefore must not change the identity of the generated
            # benchmark content.
            omitted = {
                "created_at",
                "updated_at",
                "evaluated_at",
                "dataset_hash",
                "statistics",
                "model_outputs",
                "evaluation_results",
                "validation_status",
                "validation_errors",
            }
            return {str(k): content_only(v, str(k)) for k, v in item.items() if str(k) not in omitted}
        if isinstance(item, list):
            return [content_only(v, key) for v in item]
        if isinstance(item, tuple):
            return [content_only(v, key) for v in item]
        return item

    return stable_hash(content_only(_normalize(value)))


def _case_records(value: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if isinstance(value, BenchmarkDataset):
        dataset_data = _normalize(value)
        case_records = dataset_data.pop("cases", [])
        return {"record_type": "manifest", "dataset": dataset_data}, case_records
    if isinstance(value, Mapping) and "cases" in value:
        data = dict(value)
        cases = data.pop("cases", [])
        return {"record_type": "manifest", "dataset": data}, [_normalize(case) for case in cases]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return None, [_normalize(item) for item in value]
    return None, [_normalize(value)]


def write_jsonl(value: Any, path: str | Path, *, include_manifest: bool = True) -> Path:
    """Write one JSON object per line.

    Dataset exports include a first manifest record by default.  Set
    ``include_manifest=False`` for a strict case-only benchmark JSONL file.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest, cases = _case_records(value)
    lines: list[str] = []
    if include_manifest and manifest is not None:
        lines.append(canonical_json(manifest))
    lines.extend(canonical_json(case) for case in cases)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def read_jsonl(path: str | Path, *, as_dataset: bool = True) -> BenchmarkDataset | list[dict[str, Any]]:
    manifest: Mapping[str, Any] | None = None
    cases: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        # The canonical writer emits ``record_type=manifest``.  The first
        # release's run_demo emitted a compact ``{"dataset": ...}`` record;
        # accept both so old exports remain readable.
        if "dataset" in item and (item.get("record_type") == "manifest" or set(item) == {"dataset"}):
            manifest = item["dataset"]
        else:
            cases.append(item)
    if not as_dataset or manifest is None:
        return cases
    data = dict(manifest)
    data["cases"] = cases
    return BenchmarkDataset.from_dict(data)


def _flatten(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return canonical_json(value)
    return value


def write_csv(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BenchmarkDataset):
        records = [_normalize(case) for case in value.cases]
    elif isinstance(value, Mapping) and "cases" in value:
        records = [_normalize(case) for case in value["cases"]]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records = [_normalize(item) for item in value]
    else:
        records = [_normalize(value)]
    keys: list[str] = []
    for record in records:
        if isinstance(record, Mapping):
            for key in record:
                if key not in keys:
                    keys.append(str(key))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["value"])
        writer.writeheader()
        for record in records:
            if isinstance(record, Mapping):
                writer.writerow({key: _flatten(record.get(key)) for key in keys})
            else:
                writer.writerow({"value": _flatten(record)})
    return destination


def write_markdown(value: Any, path: str | Path, *, title: str | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = value if isinstance(value, BenchmarkDataset) else None
    cases = dataset.cases if dataset is not None else list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []
    heading = title or (dataset.name if dataset is not None else "Synthetic Benchmark Dataset")
    lines = [f"# {heading}", ""]
    if dataset is not None:
        version = _normalize(dataset.version)
        lines += [f"- Benchmark version: `{version.get('benchmark_version', '')}`", f"- Dataset hash: `{version.get('dataset_hash', '')}`", f"- Case count: `{len(cases)}`", ""]
    lines += ["| Case ID | Split | Type | Difficulty | Validation | Prompt | Expected answer |", "|---|---|---|---:|---|---|---|"]
    for case in cases:
        item = _normalize(case)
        prompt = str(item.get("prompt", "")).replace("|", "\\|").replace("\n", " ")
        answer = json.dumps(item.get("ground_truth", item.get("expected_output")), ensure_ascii=False, default=str).replace("|", "\\|")
        lines.append(
            f"| {item.get('case_id', '')} | {item.get('split', '')} | {item.get('task_type', '')} | {item.get('difficulty', '')} | {item.get('validation_status', '')} | {prompt} | {answer} |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_dataset(dataset: BenchmarkDataset, path: str | Path, *, format: str | None = None, include_manifest: bool = True) -> Path:
    destination = Path(path)
    fmt = (format or destination.suffix.lstrip(".") or "json").lower()
    if fmt in {"json", "js"}:
        return write_json(dataset, destination)
    if fmt in {"jsonl", "ndjson"}:
        return write_jsonl(dataset, destination, include_manifest=include_manifest)
    if fmt == "csv":
        return write_csv(dataset, destination)
    if fmt in {"md", "markdown"}:
        return write_markdown(dataset, destination)
    raise ValueError(f"Unsupported dataset format: {fmt}")


def read_dataset(path: str | Path) -> BenchmarkDataset:
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix == ".jsonl" or suffix == ".ndjson":
        result = read_jsonl(destination, as_dataset=True)
        return result if isinstance(result, BenchmarkDataset) else BenchmarkDataset(cases=result)
    if suffix == ".json":
        data = json.loads(destination.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and "cases" in data:
            return BenchmarkDataset.from_dict(data)
        if isinstance(data, list):
            return BenchmarkDataset(cases=data)
        raise ValueError("JSON dataset must be an object with 'cases' or a case array")
    raise ValueError(f"Cannot infer dataset format from extension: {destination.suffix}")
