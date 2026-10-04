"""Deterministic synthetic benchmark case generation.

The generator in this module deliberately does *not* call a model provider.  It
uses small procedural grammars and a local pseudo random generator so that a
dataset can be recreated exactly from its definition, seed and generator
version.  The generated cases are useful for exercising a runner while still
making the source of every answer explicit.

The module is kept loosely coupled to :mod:`synthetic_benchmark_factory.models`.
The core models module is created by the application bootstrap, but the helper
``_construct`` below also makes this module usable with older model versions or
plain dictionaries (handy for command line scripts and tests).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
import re
import string
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # The canonical models are supplied by the core agent.
    from . import models as _models  # type: ignore

    BenchmarkCase = getattr(_models, "BenchmarkCase", None)
    BenchmarkDataset = getattr(_models, "BenchmarkDataset", None)
    BenchmarkDefinition = getattr(_models, "BenchmarkDefinition", None)
    # Older revisions called this BenchmarkType; current models use TaskType.
    BenchmarkType = getattr(_models, "BenchmarkType", getattr(_models, "TaskType", None))
    Distractor = getattr(_models, "Distractor", None)
    DifficultyParameters = getattr(_models, "DifficultyParameters", None)
    GeneratorMetadata = getattr(_models, "GeneratorMetadata", None)
    DatasetVersion = getattr(_models, "DatasetVersion", None)
except Exception:  # pragma: no cover - fallback keeps the generator standalone
    BenchmarkCase = None  # type: ignore
    BenchmarkDataset = None  # type: ignore
    BenchmarkDefinition = None  # type: ignore
    BenchmarkType = None  # type: ignore
    Distractor = None  # type: ignore
    DifficultyParameters = None  # type: ignore
    GeneratorMetadata = None  # type: ignore
    DatasetVersion = None  # type: ignore


GENERATOR_VERSION = "1.0.0"
SUPPORTED_TYPES = (
    "extraction",
    "classification",
    "memory",
    "planning",
    "tool_selection",
    "arithmetic",
    "structured_output",
    "instruction_following",
)
SPLITS = ("train", "validation", "test", "challenge")
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.10,
    "challenge": 0.05,
}


# These values are intentionally real generation controls, rather than merely
# labels.  They are public so a UI can display the knobs used to create a case.
DIFFICULTY_PROFILES: dict[int, dict[str, int]] = {
    1: {
        "context_length": 120,
        "distractor_count": 0,
        "number_of_facts": 2,
        "conflicting_facts": 0,
        "time_distance": 0,
        "operation_depth": 1,
        "label_count": 2,
        "constraint_count": 1,
        "required_fields": 2,
        "tool_count": 3,
    },
    2: {
        "context_length": 260,
        "distractor_count": 1,
        "number_of_facts": 3,
        "conflicting_facts": 0,
        "time_distance": 1,
        "operation_depth": 2,
        "label_count": 3,
        "constraint_count": 2,
        "required_fields": 3,
        "tool_count": 4,
    },
    3: {
        "context_length": 520,
        "distractor_count": 3,
        "number_of_facts": 5,
        "conflicting_facts": 1,
        "time_distance": 3,
        "operation_depth": 3,
        "label_count": 4,
        "constraint_count": 3,
        "required_fields": 4,
        "tool_count": 5,
    },
    4: {
        "context_length": 900,
        "distractor_count": 6,
        "number_of_facts": 7,
        "conflicting_facts": 2,
        "time_distance": 7,
        "operation_depth": 4,
        "label_count": 5,
        "constraint_count": 4,
        "required_fields": 6,
        "tool_count": 7,
    },
    5: {
        "context_length": 1500,
        "distractor_count": 10,
        "number_of_facts": 10,
        "conflicting_facts": 3,
        "time_distance": 30,
        "operation_depth": 6,
        "label_count": 6,
        "constraint_count": 6,
        "required_fields": 8,
        "tool_count": 9,
    },
}


@dataclass
class _FallbackDefinition:
    """Used only when the core models module is not available yet."""

    name: str = "Synthetic Benchmark"
    capability: str = "general"
    description: str = "Procedurally generated benchmark"
    difficulty: int = 3
    task_type: str = "arithmetic"
    input_schema: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    scoring: dict[str, Any] = field(default_factory=lambda: {"method": "exact_match"})
    constraints: list[str] = field(default_factory=list)


@dataclass
class _FallbackCase:
    case_id: str
    benchmark_type: str
    split: str
    prompt: str
    context: str | dict[str, Any] | list[Any]
    expected_output: Any
    ground_truth: Any
    difficulty: int
    difficulty_parameters: dict[str, Any]
    distractors: list[Any]
    generator: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "unvalidated"
    validation_errors: list[str] = field(default_factory=list)
    model_outputs: list[Any] = field(default_factory=list)


@dataclass
class _FallbackDataset:
    cases: list[Any]
    definition: Any
    seed: int
    generator_version: str
    splits: dict[str, int]
    dataset_hash: str = ""
    version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    """Return a stable, JSON-compatible representation of a model value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "value") and not isinstance(value, (bytes, bytearray)):
        return _jsonable(value.value)
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reproducible_value(value: Any) -> Any:
    """Strip wall-clock fields before hashing generated artifacts.

    Core dataclasses timestamp their creation for auditability.  Those
    timestamps must not make the same seed produce a different dataset hash.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _reproducible_value(item)
            for key, item in value.items()
            if str(key).lower() not in {"created_at", "updated_at", "evaluated_at"}
        }
    if isinstance(value, (list, tuple)):
        return [_reproducible_value(item) for item in value]
    if is_dataclass(value):
        return _reproducible_value(asdict(value))
    return _jsonable(value)


def _construct(cls: Any, values: Mapping[str, Any], fallback: Any = None) -> Any:
    """Construct a core dataclass while tolerating model-version differences."""

    if cls is None:
        return fallback if fallback is not None else dict(values)
    data = dict(values)
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_kwargs:
            data = {k: v for k, v in data.items() if k in params}
        return cls(**data)
    except Exception:
        # Some pydantic models expose model_validate rather than a permissive
        # constructor.  Try those before falling back to an ordinary mapping.
        for method_name in ("model_validate", "from_dict"):
            method = getattr(cls, method_name, None)
            if method:
                try:
                    return method(dict(values))
                except Exception:
                    pass
        try:
            obj = cls()
            for key, value in values.items():
                try:
                    setattr(obj, key, value)
                except Exception:
                    continue
            return obj
        except Exception:
            return fallback if fallback is not None else dict(values)


def _set_attr(obj: Any, key: str, value: Any) -> None:
    try:
        setattr(obj, key, value)
    except Exception:
        if isinstance(obj, MutableMapping):
            obj[key] = value


def _get_attr(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(obj, Mapping) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def _normalise_type(value: Any) -> str:
    if value is None:
        return "arithmetic"
    value = getattr(value, "value", value)
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "extract": "extraction",
        "information_extraction": "extraction",
        "classify": "classification",
        "tool": "tool_selection",
        "tools": "tool_selection",
        "structured": "structured_output",
        "instruction": "instruction_following",
        "instructionfollowing": "instruction_following",
    }
    return aliases.get(text, text)


def _definition_value(definition: Any, *keys: str, default: Any = None) -> Any:
    value = _get_attr(definition, *keys, default=default)
    return value


def _make_definition(value: Any) -> Any:
    if value is None:
        # Prefer the canonical core model when it is available.  Returning a
        # private fallback dataclass here would make BenchmarkDataset reject
        # the definition during JSON round-trips.
        value = {
            "name": "Synthetic Benchmark",
            "capability": "general",
            "description": "Procedurally generated benchmark",
            "difficulty": 3,
            "task_type": "arithmetic",
            "input_schema": {},
            "expected_output": None,
            "scoring": {"method": "exact_match"},
            "constraints": {},
        }
    if isinstance(value, str):
        value = {"name": value}
    if isinstance(value, Mapping):
        data = dict(value)
        data.setdefault("name", data.get("Name", "Synthetic Benchmark"))
        data.setdefault("capability", data.get("Capability", "general"))
        data.setdefault("description", data.get("Description", "Procedurally generated benchmark"))
        data.setdefault("difficulty", data.get("Difficulty", 3))
        data.setdefault("task_type", data.get("Task Type", data.get("taskType", "arithmetic")))
        data.setdefault("input_schema", data.get("Input Schema", {}))
        data.setdefault("expected_output", data.get("Expected Output"))
        data.setdefault("scoring", data.get("Scoring", {"method": "exact_match"}))
        constraints = data.get("constraints", data.get("Constraints", {}))
        # Core models use a mapping for constraints; accept the concise list
        # form used by the editor and preserve it under ``items``.
        if isinstance(constraints, (list, tuple, set)):
            constraints = {"items": list(constraints)}
        data["constraints"] = constraints or {}
        return _construct(BenchmarkDefinition, data, _FallbackDefinition(**{
            k: data[k]
            for k in ("name", "capability", "description", "difficulty", "task_type", "input_schema", "expected_output", "scoring", "constraints")
            if k in data
        }))
    return value


def _difficulty(level: Any) -> tuple[int, dict[str, int]]:
    try:
        level_int = int(getattr(level, "value", level))
    except Exception:
        level_int = 3
    level_int = max(1, min(5, level_int))
    return level_int, dict(DIFFICULTY_PROFILES[level_int])


def _difficulty_object(level: int, params: Mapping[str, Any]) -> Any:
    if DifficultyParameters is None:
        return dict(params, level=level)
    values = dict(params)
    # DifficultyParameters intentionally keeps task-specific knobs in ``extra``.
    known = {"context_length", "distractor_count", "number_of_facts", "conflicting_facts", "time_distance", "missing_data", "tool_failure", "extra"}
    values["extra"] = {"level": level, **{k: v for k, v in values.items() if k not in known}}
    values = {k: v for k, v in values.items() if k in known}
    return _construct(DifficultyParameters, values, dict(params, level=level))


def _distractor_object(kind: str, content: Any, *, severity: int = 1, source: str = "procedural") -> Any:
    values = {"kind": kind, "type": kind, "content": content, "severity": severity, "source": source}
    if Distractor is None:
        return values
    return _construct(Distractor, values, values)


def _safe_identifier(rng: random.Random, prefix: str = "item") -> str:
    return f"{prefix}-{rng.randrange(100000, 999999)}"


FIRST_NAMES = ["Aiko", "Ren", "Mina", "Sora", "Kai", "Yuna", "Taro", "Nami", "Leo", "Hana"]
DOMAINS = ["archive", "garden", "station", "library", "laboratory", "market", "observatory"]
COLORS = ["amber", "blue", "coral", "green", "indigo", "violet"]
LABELS = ["policy", "science", "finance", "sports", "travel", "health"]


class BenchmarkGenerator:
    """Generate reproducible, ground-truthed benchmark datasets.

    Parameters
    ----------
    definition:
        A ``BenchmarkDefinition`` or a dictionary containing its fields.
    seed:
        Master seed.  Every case records a derived case seed as well.
    generator_version:
        Included in metadata and dataset hashes for reproducibility.
    """

    def __init__(self, definition: Any = None, seed: int = 20261004, generator_version: str = GENERATOR_VERSION):
        self.definition = _make_definition(definition)
        self.seed = int(seed)
        self.generator_version = str(generator_version)

    def generate(
        self,
        definition: Any = None,
        count: int = 100,
        seed: int | None = None,
        split_ratios: Mapping[str, float] | None = None,
        splits: Mapping[str, float] | None = None,
        split_config: Mapping[str, float] | None = None,
        distractors: Sequence[Any] | None = None,
        difficulty_schedule: Sequence[int] | None = None,
    ) -> Any:
        """Generate ``count`` cases and assign deterministic dataset splits."""

        # A convenient positional form ``generate(20, seed=...)`` is common
        # in small scripts.  The full signature keeps ``definition`` first so
        # callers can pass a custom definition, but an integer in that slot is
        # unambiguously a requested case count when ``count`` is untouched.
        if isinstance(definition, (int, float)) and not isinstance(definition, bool) and count == 100:
            count = int(definition)
            definition = None

        if definition is not None:
            self.definition = _make_definition(definition)
        definition_obj = self.definition
        try:
            count = int(count)
        except Exception as exc:
            raise ValueError("count must be an integer") from exc
        if count < 0:
            raise ValueError("count must be non-negative")
        master_seed = self.seed if seed is None else int(seed)
        # Generated artifacts use a deterministic timestamp so serializing the
        # same seed twice produces byte-identical manifests.  (Interactive
        # edits may still use the normal model defaults.)
        deterministic_epoch = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=abs(master_seed) % 31536000)
        if hasattr(definition_obj, "created_at"):
            _set_attr(definition_obj, "created_at", deterministic_epoch)
        ratios = split_ratios if split_ratios is not None else (splits if splits is not None else split_config)
        ratio_map = self._normalise_splits(ratios)
        allocation = self._allocate_splits(count, ratio_map)
        type_name = _normalise_type(_definition_value(definition_obj, "task_type", "benchmark_type", "type", default="arithmetic"))
        if type_name not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported benchmark type: {type_name}. Supported: {', '.join(SUPPORTED_TYPES)}")
        level, params = _difficulty(_definition_value(definition_obj, "difficulty", default=3))
        cases: list[Any] = []
        index = 0
        for split in SPLITS:
            for _ in range(allocation.get(split, 0)):
                case_seed = master_seed + index * 1000003 + (hash(split) & 0xFFFF)
                # Python's hash is process-randomized.  Replace the split term
                # with a stable checksum so datasets are identical across runs.
                case_seed = master_seed + index * 1000003 + int(_stable_hash(split)[:8], 16)
                rng = random.Random(case_seed)
                case_level = level
                if difficulty_schedule:
                    try:
                        scheduled = difficulty_schedule[index % len(difficulty_schedule)]
                        case_level, case_params = _difficulty(scheduled)
                    except (TypeError, ValueError, IndexError):
                        case_level, case_params = level, params
                else:
                    case_params = params
                case_payload = self._generate_case(
                    type_name,
                    case_level,
                    case_params,
                    rng,
                    index,
                    split,
                    case_seed,
                    definition_obj,
                    master_seed=master_seed,
                )
                if distractors:
                    self._append_requested_distractors(case_payload, distractors, rng)
                case = self._make_case(case_payload)
                if hasattr(case, "created_at"):
                    _set_attr(case, "created_at", deterministic_epoch + timedelta(seconds=index))
                cases.append(case)
                index += 1
        dataset_version_hash_placeholder = ""
        version_values = {
            "benchmark_version": self.generator_version,
            "seed": master_seed,
            "generator_version": self.generator_version,
            "config": {
                "split_ratios": ratio_map,
                "benchmark_type": type_name,
                "difficulty": level,
                **({"difficulty_schedule": list(difficulty_schedule)} if difficulty_schedule else {}),
            },
            "dataset_hash": dataset_version_hash_placeholder,
            "case_count": len(cases),
        }
        version_obj = _construct(DatasetVersion, version_values, version_values) if DatasetVersion is not None else version_values
        if hasattr(version_obj, "created_at"):
            _set_attr(version_obj, "created_at", deterministic_epoch)
        dataset_values = {
            "name": str(_definition_value(definition_obj, "name", default="Synthetic Benchmark Dataset")),
            "cases": cases,
            "definitions": [definition_obj],
            "items": cases,
            "definition": definition_obj,
            "benchmark_definition": definition_obj,
            "seed": master_seed,
            "generator_version": self.generator_version,
            "version": version_obj,
            "splits": allocation,
            "split_counts": allocation,
            "metadata": {
                "synthetic": True,
                "provider": "procedural_generator",
                "generator": self.generator_version,
                "seed": master_seed,
                "benchmark_type": type_name,
                "difficulty": level,
                "case_count": len(cases),
            },
        }
        dataset = _construct(BenchmarkDataset, dataset_values, _FallbackDataset(
            cases=cases,
            definition=definition_obj,
            seed=master_seed,
            generator_version=self.generator_version,
            splits=allocation,
        ))
        # Add all non-derived metadata before hashing.  ``dataset_hash`` itself
        # is excluded by the canonical serializer, so it can be filled after
        # the digest is known without creating a circular hash.
        metadata = dict(dataset_values["metadata"])
        if difficulty_schedule:
            metadata["difficulty_schedule"] = list(difficulty_schedule)
        _set_attr(dataset, "metadata", metadata)
        try:
            # Populate display-ready statistics immediately; the serializer and
            # hash intentionally exclude this derived field.
            compute_statistics = getattr(dataset, "compute_statistics", None)
            if callable(compute_statistics):
                compute_statistics()
        except Exception:
            pass
        # Use the package serializer's canonical content hash as the single
        # source of truth.  It excludes timestamps, attached run outputs and
        # the previous digest, so regeneration with the same seed is stable
        # and ``verify_dataset_hash(dataset)`` agrees with this value.
        try:
            from .serialization import dataset_hash as _canonical_dataset_hash

            dataset_hash = _canonical_dataset_hash(dataset)
        except Exception:
            dataset_hash = _stable_hash({
                "definition": _reproducible_value(definition_obj),
                "seed": master_seed,
                "generator_version": self.generator_version,
                "cases": [_reproducible_value(c) for c in cases],
            })
        _set_attr(dataset, "dataset_hash", dataset_hash)
        _set_attr(dataset, "hash", dataset_hash)
        # Keep the version label human-readable and independent of the hash;
        # the digest is stored separately in DatasetVersion.dataset_hash.
        version_label = self.generator_version
        current_version = _get_attr(dataset, "version", default=None)
        if current_version is not None and not isinstance(current_version, str):
            _set_attr(current_version, "benchmark_version", version_label)
            _set_attr(current_version, "dataset_hash", dataset_hash)
            _set_attr(current_version, "case_count", len(cases))
            _set_attr(current_version, "seed", master_seed)
            _set_attr(current_version, "generator_version", self.generator_version)
        else:
            _set_attr(dataset, "version", version_label)
        _set_attr(dataset, "benchmark_version", version_label)
        metadata["dataset_hash"] = dataset_hash
        _set_attr(dataset, "metadata", metadata)
        return dataset

    @staticmethod
    def _append_requested_distractors(payload: MutableMapping[str, Any], requested: Sequence[Any], rng: random.Random) -> None:
        """Apply editor-specified distractor kinds without losing provenance."""

        existing = list(payload.get("distractors") or [])
        context = str(payload.get("context") or "")
        for item in requested:
            if isinstance(item, Mapping):
                kind = item.get("kind", item.get("type", "custom"))
                content = item.get("content", f"{kind} note")
            else:
                label = str(item)
                normalized = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
                kind = {
                    "irrelevant_info": "irrelevant_info",
                    "contradictory_info": "contradictory_info",
                    "near_match": "near_match",
                    "long_context": "long_context",
                    "missing_data": "missing_data",
                    "tool_failure": "tool_failure",
                }.get(normalized, normalized or "custom")
                content = f"{label} distractor {rng.randrange(1000, 9999)}."
            distractor = _distractor_object(str(kind), content, severity=1, source="editor")
            existing.append(distractor)
            context += ("\n" if context else "") + str(content)
        payload["distractors"] = existing
        payload["context"] = context

    def generate_demo(self, count: int = 500, seed: int | None = None) -> Any:
        """Generate the first-party Long-Term Memory demonstration dataset.

        The demo intentionally uses all four standard splits and includes
        writing, temporal distance, distractors, conflicts, ordering and
        multi-entity recall.  The default is 500 cases as requested by the
        product specification; callers can request a smaller smoke dataset.
        """

        definition = _make_definition({
            "name": "Long-Term Memory Benchmark",
            "capability": "long-term memory",
            "description": "Recall stable and updated facts after temporal and irrelevant context.",
            "difficulty": 3,
            "task_type": "memory",
            "input_schema": {"context": "string", "question": "string"},
            "expected_output": {"answer": "string", "evidence_fact_id": "string"},
            "scoring": {"method": "exact_match", "fields": ["answer", "evidence_fact_id"]},
            "constraints": ["Use the most recent non-conflicting fact.", "Return JSON only."],
        })
        return self.generate(definition=definition, count=count, seed=seed, difficulty_schedule=(1, 2, 3, 4, 5))

    # Common aliases used by command line clients.
    generate_dataset = generate
    generate_demo_dataset = generate_demo

    @staticmethod
    def _normalise_splits(ratios: Mapping[str, float] | None) -> dict[str, float]:
        source = dict(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
        unknown = set(source) - set(SPLITS)
        if unknown:
            raise ValueError(f"Unknown split(s): {', '.join(sorted(unknown))}")
        result = {name: float(source.get(name, 0.0)) for name in SPLITS}
        if any(value < 0 for value in result.values()):
            raise ValueError("Split ratios cannot be negative")
        total = sum(result.values())
        if total <= 0:
            raise ValueError("At least one split ratio must be positive")
        return {name: value / total for name, value in result.items()}

    @staticmethod
    def _allocate_splits(count: int, ratios: Mapping[str, float]) -> dict[str, int]:
        raw = {name: count * float(ratios.get(name, 0.0)) for name in SPLITS}
        allocation = {name: int(math.floor(value)) for name, value in raw.items()}
        remainder = count - sum(allocation.values())
        order = sorted(SPLITS, key=lambda name: (raw[name] - allocation[name], -SPLITS.index(name)), reverse=True)
        for name in order[:remainder]:
            allocation[name] += 1
        return allocation

    def _make_case(self, payload: Mapping[str, Any]) -> Any:
        fallback = _FallbackCase(**{
            key: payload[key]
            for key in _FallbackCase.__dataclass_fields__
            if key in payload
        })
        case = _construct(BenchmarkCase, payload, fallback)
        # Preserve fields even when a newer model omitted a convenience field.
        for key, value in payload.items():
            if not hasattr(case, key):
                _set_attr(case, key, value)
        return case

    def _base_payload(
        self,
        type_name: str,
        level: int,
        params: Mapping[str, Any],
        rng: random.Random,
        index: int,
        split: str,
        case_seed: int,
        definition: Any,
        master_seed: int | None = None,
    ) -> dict[str, Any]:
        case_id = f"{type_name[:4].upper()}-{index + 1:05d}-{_stable_hash((case_seed, type_name))[:6]}"
        generator_values = {
            "template": f"{type_name}.v1",
            "rules": ["ground_truth_determined", "seeded_random", "no_paid_model"],
            "procedural_generator": f"BenchmarkGenerator._gen_{type_name}",
            "seed": int(master_seed if master_seed is not None else case_seed),
            "generator_version": self.generator_version,
            "config": dict(params),
        }
        generator_obj = _construct(GeneratorMetadata, generator_values, generator_values) if GeneratorMetadata is not None else generator_values
        difficulty_obj = _difficulty_object(level, params)
        definition_id = str(_definition_value(definition, "definition_id", "id", default=""))
        return {
            "case_id": case_id,
            "id": case_id,
            "definition_id": definition_id,
            "benchmark_type": type_name,
            "task_type": type_name,
            "split": split,
            "prompt": "",
            "context": "",
            "expected_output": None,
            "ground_truth": None,
            "difficulty": level,
            "difficulty_level": level,
            "difficulty_parameters": difficulty_obj,
            "difficulty_params": difficulty_obj,
            "distractors": [],
            "generator": generator_obj,
            "generator_version": self.generator_version,
            "seed": case_seed,
            "metadata": {
            "synthetic": True,
                "provider": "procedural_generator",
                "definition_name": _definition_value(definition, "name", default="Synthetic Benchmark"),
                "source": "template+rules+procedural",
                "source_split": split,
                "ground_truth_determined": True,
                "case_seed": case_seed,
                "master_seed": int(master_seed if master_seed is not None else case_seed),
            },
            "validation_status": "unvalidated",
            "validation_errors": [],
            "model_outputs": [],
        }

    def _generate_case(
        self,
        type_name: str,
        level: int,
        params: Mapping[str, Any],
        rng: random.Random,
        index: int,
        split: str,
        case_seed: int,
        definition: Any,
        master_seed: int | None = None,
    ) -> dict[str, Any]:
        base = self._base_payload(type_name, level, params, rng, index, split, case_seed, definition, master_seed=master_seed)
        method = getattr(self, f"_gen_{type_name}")
        payload = method(base, level, params, rng)
        # Every case has an answer and an explicit, serialisable ground truth.
        if payload.get("ground_truth") is None:
            raise ValueError(f"Generator produced no ground truth for case {payload['case_id']}")
        payload["expected_output"] = payload.get("expected_output", payload["ground_truth"])
        payload["metadata"].setdefault("answer_hash", _stable_hash(payload["ground_truth"]))
        payload["metadata"].setdefault("difficulty_parameters", dict(params))
        return payload

    # ------------------------------------------------------------------
    # Individual benchmark grammars
    # ------------------------------------------------------------------
    def _gen_extraction(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        record_count = max(1, min(1 + level, 5))
        records: list[dict[str, Any]] = []
        for i in range(record_count):
            records.append({
                "record_id": f"R{i + 1:02d}",
                "name": rng.choice(FIRST_NAMES),
                "domain": rng.choice(DOMAINS),
                "amount": rng.randrange(10, 90) * 5,
                "status": rng.choice(["open", "closed", "pending"]),
            })
        target = records[-1]
        lines = [
            f"Record {r['record_id']}: name={r['name']}; domain={r['domain']}; amount={r['amount']}; status={r['status']}."
            for r in records
        ]
        distractors: list[Any] = []
        for i in range(p["distractor_count"]):
            text = f"Note {i + 1}: archive shelf {rng.randrange(1, 30)} is reserved for maintenance."
            lines.insert(rng.randrange(len(lines) + 1), text)
            distractors.append(_distractor_object("irrelevant_info", text))
        if level >= 3:
            near = f"Near match: record {target['record_id']} was mentioned in a draft, but its status was not updated."
            lines.insert(0, near)
            distractors.append(_distractor_object("near_match", near, severity=2))
        if level >= 5:
            long_note = "Long archive note: " + "irrelevant maintenance detail; " * 18
            lines.insert(0, long_note)
            distractors.append(_distractor_object("long_context", long_note, severity=1))
        answer = {"record_id": target["record_id"], "name": target["name"], "domain": target["domain"], "amount": target["amount"], "status": target["status"]}
        b["context"] = "\n".join(lines)
        b["prompt"] = "Extract the fields of the final complete record (the record with the highest ID) as JSON. Ignore notes and draft mentions."
        b["ground_truth"] = answer
        b["expected_output"] = answer
        b["distractors"] = distractors
        b["scoring"] = {"method": "exact_match", "normalise": "json_object", "required_fields": list(answer)}
        b["metadata"].update({"target_record_id": target["record_id"], "source_spans": [target["record_id"]], "answer_not_in_instruction": True})
        return b

    def _gen_classification(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        labels = LABELS[: p["label_count"]]
        label = rng.choice(labels)
        cues = {
            "policy": "The memo discusses a new compliance rule and review deadline.",
            "science": "The report describes a controlled trial, samples, and measured variance.",
            "finance": "The ledger records a quarterly budget, invoice, and revenue forecast.",
            "sports": "The match summary lists the score, coach, and training session.",
            "travel": "The itinerary contains a departure gate, hotel, and return ticket.",
            "health": "The clinic note mentions symptoms, dosage, and follow-up care.",
        }
        text = cues[label]
        if level >= 3:
            text += " " + rng.choice(["The team also reviewed a general schedule.", "A neutral index was attached.", "The document uses a standard template."])
        distractors = []
        for i in range(p["distractor_count"]):
            d = f"Unrelated footer {i + 1}: document ID {rng.randrange(1000, 9999)}."
            text += " " + d
            distractors.append(_distractor_object("irrelevant_info", d))
        if level >= 5:
            contradiction = "Contradictory footer: a generic template mentions a different domain; classify by the dominant document cue."
            text += " " + contradiction
            distractors.append(_distractor_object("contradictory_info", contradiction, severity=1))
        b["context"] = text
        b["prompt"] = f"Classify the document into exactly one of: {', '.join(labels)}. Return the label only."
        b["ground_truth"] = label
        b["expected_output"] = label
        b["distractors"] = distractors
        b["scoring"] = {"method": "exact_match", "labels": labels}
        b["metadata"].update({"labels": labels, "classification_rule": f"dominant domain cue => {label}"})
        return b

    def _gen_memory(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        entity_count = max(2, min(4, 2 + level // 2))
        entities = [_safe_identifier(rng, "person") for _ in range(entity_count)]
        facts: list[dict[str, Any]] = []
        # ``number_of_facts`` is a real knob: higher levels add more distinct
        # dated facts rather than merely changing a label.  Entities repeat so
        # the benchmark also exercises multi-entity recall.
        for i in range(max(1, int(p.get("number_of_facts", len(entities))))):
            entity = entities[i % len(entities)]
            facts.append({"fact_id": f"F{i + 1:02d}", "entity": entity, "attribute": "base", "value": rng.choice(COLORS), "day": 0})
        conflict_count = min(p["conflicting_facts"], len(facts))
        for i in range(conflict_count):
            original = facts[i]
            facts.append({
                "fact_id": f"F{len(facts) + 1:02d}",
                "entity": original["entity"],
                "attribute": "base",
                "value": rng.choice([c for c in COLORS if c != original["value"]]),
                "day": p["time_distance"] + i + 1,
            })
        # Stable ordering makes the conflict-resolution rule explicit.
        facts.sort(key=lambda f: (f["day"], f["fact_id"]))
        target = facts[-1]
        lines = [f"Day {f['day']}: FACT {f['fact_id']} — {f['entity']} has base color {f['value']}." for f in facts]
        distractors: list[Any] = []
        if conflict_count:
            for old in facts[:conflict_count]:
                contradictory = f"Contradictory earlier report: {old['entity']} was once listed as {old['value']}; use the latest confirmed fact."
                lines.insert(0, contradictory)
                distractors.append(_distractor_object("contradictory_info", contradictory, severity=2))
        for i in range(p["distractor_count"]):
            d = f"Day {rng.randrange(0, max(1, p['time_distance'] + 2))}: unrelated weather report {i + 1}."
            lines.insert(rng.randrange(len(lines) + 1), d)
            distractors.append(_distractor_object("irrelevant_info", d))
        if level >= 4:
            near = f"A draft says {target['entity']} might use a different color; drafts are not facts."
            lines.insert(0, near)
            distractors.append(_distractor_object("near_match", near, severity=2))
        if level >= 5:
            long_note = "Long archive note: " + "unrelated historical detail; " * 18
            lines.insert(0, long_note)
            distractors.append(_distractor_object("long_context", long_note, severity=1))
        b["context"] = "\n".join(lines)
        b["prompt"] = f"What is the most recently confirmed base color of {target['entity']}? Return JSON with answer and fact_id. Ignore drafts and unrelated reports."
        answer = {"answer": target["value"], "fact_id": target["fact_id"], "entity": target["entity"]}
        b["ground_truth"] = answer
        b["expected_output"] = answer
        b["distractors"] = distractors
        b["scoring"] = {"method": "exact_match", "fields": ["answer", "fact_id", "entity"]}
        b["metadata"].update({"facts": facts, "target_entity": target["entity"], "conflict_resolution": "latest_confirmed_fact", "time_distance": p["time_distance"]})
        return b

    def _gen_planning(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        # Include a deterministic batch tag in action text so two planning
        # cases with the same difficulty still exercise different concrete
        # plans. Without this, all cases shared an identical context/answer
        # and the duplicate validator correctly flagged them.
        batch_tag = f"batch-{rng.randrange(100000, 999999)}"
        steps = [
            f"collect materials for {batch_tag}",
            f"inspect inputs for {batch_tag}",
            f"run preparation for {batch_tag}",
            f"perform operation for {batch_tag}",
            f"check result for {batch_tag}",
            f"archive report for {batch_tag}",
        ][: 3 + level]
        dependencies = [[steps[i], steps[i + 1]] for i in range(len(steps) - 1)]
        constraints = [f"{a} must happen before {c}" for a, c in dependencies[: p["constraint_count"]]]
        # One independent step provides a meaningful but deterministic choice.
        if level >= 3:
            steps.insert(1, "record identifier")
            dependencies.insert(0, [steps[0], steps[1]])
            constraints.insert(0, f"{steps[0]} must happen before {steps[1]}")
        b["context"] = "Goal: complete a reproducible lab handoff. Available actions: " + ", ".join(steps) + "."
        b["prompt"] = "Return an ordered JSON list of actions satisfying every constraint: " + "; ".join(constraints) + ". Use each action exactly once."
        b["ground_truth"] = steps
        b["expected_output"] = steps
        b["scoring"] = {"method": "task_specific", "metric": "dependency_satisfaction", "constraints": constraints}
        b["metadata"].update({"constraints": constraints, "dependencies": dependencies, "valid_order_count": 1})
        return b

    def _gen_tool_selection(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        tools = [
            {"name": "search", "purpose": "find documents", "required": ["query"]},
            {"name": "calculator", "purpose": "compute arithmetic", "required": ["expression"]},
            {"name": "calendar", "purpose": "look up events", "required": ["date"]},
            {"name": "file_reader", "purpose": "read a local file", "required": ["path"]},
            {"name": "table_filter", "purpose": "filter tabular records", "required": ["column", "value"]},
            {"name": "summarizer", "purpose": "summarize provided text", "required": ["text"]},
            {"name": "translator", "purpose": "translate text", "required": ["text", "target_language"]},
            {"name": "sorter", "purpose": "sort records", "required": ["records", "key"]},
            {"name": "validator", "purpose": "validate a schema", "required": ["data", "schema"]},
        ][: p["tool_count"]]
        target = rng.choice(tools)
        query = f"Please {target['purpose']} for the supplied item."
        if target["name"] == "calculator":
            query = "Compute the total of the supplied expression exactly."
        failures: list[Any] = []
        failed_tool_names: list[str] = []
        if level >= 3:
            # The failed tool must be different from the ground-truth target.
            # Earlier versions marked ``target`` as unavailable while still
            # asking the model to choose it, creating an impossible case.
            alternatives = [tool for tool in tools if tool["name"] != target["name"]]
            failed_tool = rng.choice(alternatives) if alternatives else None
            if failed_tool is not None:
                failure = {
                    "name": failed_tool["name"],
                    "status": "temporarily unavailable",
                    "instruction": "Do not choose a failed tool.",
                }
                failures.append(_distractor_object("tool_failure", failure, severity=2))
                failed_tool_names.append(failed_tool["name"])
            # Add a healthy duplicate-looking tool with a different purpose.
            query += " Choose a healthy tool from the catalog."
        b["context"] = "Tool catalog:\n" + "\n".join(f"- {t['name']}: {t['purpose']} (args: {', '.join(t['required'])})" for t in tools)
        b["prompt"] = query + " Return JSON {tool, arguments} and no explanation."
        # Keep arguments deterministic but unique enough that repeated choices
        # of the same tool remain distinct benchmark cases.
        request_tag = rng.randrange(100000, 999999)
        args = {key: f"sample_{key}_{request_tag}" for key in target["required"]}
        answer = {"tool": target["name"], "arguments": args}
        b["ground_truth"] = answer
        b["expected_output"] = answer
        b["distractors"] = failures
        b["scoring"] = {"method": "schema", "required_fields": ["tool", "arguments"], "valid_tools": [t["name"] for t in tools]}
        b["metadata"].update({"tools": tools, "failed_tools": failed_tool_names, "target_tool": target["name"]})
        return b

    def _gen_arithmetic(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        depth = p["operation_depth"]
        # Use integer-safe operations only.  Each generated expression is
        # evaluated locally and the exact result is stored as ground truth.
        numbers = [rng.randrange(2, 10 + level * 5) for _ in range(depth + 1)]
        ops = [rng.choice(["+", "-", "*"]) for _ in range(depth)]
        expression = str(numbers[0])
        for op, number in zip(ops, numbers[1:]):
            expression = f"({expression} {op} {number})"
        def eval_node(node: ast.AST) -> int:
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return int(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = eval_node(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                left, right = eval_node(node.left), eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                return left * right
            raise ValueError(f"unsupported arithmetic node: {ast.dump(node)}")
        try:
            result = eval_node(ast.parse(expression, mode="eval"))
        except Exception as exc:  # defensive; expression is generated above
            raise ValueError(f"Unable to evaluate generated arithmetic expression: {expression}") from exc
        b["context"] = f"Expression: {expression}"
        b["prompt"] = "Evaluate the expression exactly. Return the integer only."
        b["ground_truth"] = result
        b["expected_output"] = result
        b["scoring"] = {"method": "exact_match", "numeric": True}
        b["metadata"].update({"expression": expression, "operators": ops, "operand_count": len(numbers)})
        return b

    def _gen_structured_output(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        fields = ["id", "title", "priority", "tags", "owner", "active", "score", "region"][: p["required_fields"]]
        source = {
            "id": _safe_identifier(rng, "task"),
            "title": rng.choice(["Audit", "Index", "Migrate", "Review"]) + " dataset",
            "priority": rng.choice(["low", "medium", "high"]),
            "tags": [rng.choice(LABELS), rng.choice(COLORS)],
            "owner": rng.choice(FIRST_NAMES),
            "active": bool(rng.randrange(2)),
            "score": rng.randrange(10, 100),
            "region": rng.choice(["east", "west", "north", "south"]),
        }
        answer = {field: source[field] for field in fields}
        schema = {field: type(source[field]).__name__ for field in fields}
        b["context"] = "Source record: " + json.dumps(source, ensure_ascii=False, sort_keys=True)
        b["prompt"] = "Return only a JSON object with exactly these fields and types: " + json.dumps(schema, sort_keys=True) + ". Do not add fields."
        b["ground_truth"] = answer
        b["expected_output"] = answer
        b["scoring"] = {"method": "schema", "required_fields": fields, "additional_properties": False, "types": schema}
        b["metadata"].update({"schema": schema, "additional_properties": False, "source_record": source})
        if level >= 4:
            missing = "Optional source note omitted intentionally; do not invent a field."
            b["context"] += " " + missing
            b["distractors"] = [_distractor_object("missing_data", missing, severity=1)]
        return b

    def _gen_instruction_following(self, b: dict[str, Any], level: int, p: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        # A per-case numeric suffix prevents a finite five-token vocabulary
        # from producing duplicate formal benchmark items at larger counts.
        text = f"{rng.choice(['alpha', 'bravo', 'charlie', 'delta', 'echo'])}-{rng.randrange(100000, 999999)}"
        repeat = 1 + level // 2
        prefix = rng.choice(["NOTE", "RESULT", "ITEM"])
        instructions = [f"Repeat the token '{text}' exactly {repeat} time(s).", f"Prefix the result with '{prefix}:'.", "Return one line and no punctuation after the final token."]
        answer = f"{prefix}: " + " ".join([text] * repeat)
        b["context"] = f"Input token: {text}"
        b["prompt"] = "Follow every instruction exactly:\n- " + "\n- ".join(instructions)
        b["ground_truth"] = answer
        b["expected_output"] = answer
        b["scoring"] = {"method": "exact_match", "normalise": "trim_outer_whitespace"}
        b["metadata"].update({"instructions": instructions, "token": text, "repeat": repeat, "prefix": prefix})
        return b


def generate_dataset(definition: Any = None, count: int = 100, seed: int = 20261004, **kwargs: Any) -> Any:
    """Functional convenience wrapper around :class:`BenchmarkGenerator`."""

    return BenchmarkGenerator(definition=definition, seed=seed).generate(count=count, **kwargs)


def generate_demo_dataset(count: int = 500, seed: int = 20261004) -> Any:
    """Functional convenience wrapper for the long-term-memory demo."""

    return BenchmarkGenerator(seed=seed).generate_demo(count=count, seed=seed)


DatasetGenerator = BenchmarkGenerator
ProceduralGenerator = BenchmarkGenerator
Generator = BenchmarkGenerator


__all__ = [
    "BenchmarkGenerator",
    "DatasetGenerator",
    "ProceduralGenerator",
    "Generator",
    "DIFFICULTY_PROFILES",
    "SUPPORTED_TYPES",
    "SPLITS",
    "DEFAULT_SPLIT_RATIOS",
    "generate_dataset",
    "generate_demo_dataset",
]
