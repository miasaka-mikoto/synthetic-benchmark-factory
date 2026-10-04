"""Dependency-free domain models used by every Benchmark Factory component.

The models are intentionally plain dataclasses.  This keeps them convenient
for generators and tests while still giving the editor, validator, runner and
exporters one stable contract.  JSON-friendly conversion is implemented in
``serialization.py``; each model exposes ``to_dict``/``from_dict`` as a small
ergonomic convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
import hashlib
import json
import re
from typing import Any, ClassVar, Mapping, Sequence


def utc_now() -> datetime:
    """Return an aware UTC timestamp.

    A function (rather than a module-level value) is used for dataclass
    defaults so importing the package never freezes timestamps.
    """

    return datetime.now(timezone.utc)


def _coerce_datetime(value: Any) -> Any:
    """Parse ISO timestamps from JSON while preserving unknown values."""

    if isinstance(value, datetime) or value is None:
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text or "benchmark"


def _config_dict(value: Any, *, scalar_key: str = "value", sequence_key: str = "items") -> dict[str, Any]:
    """Normalize concise config forms without requiring a schema library."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple, set)):
        return {sequence_key: list(value)}
    return {scalar_key: value}


class _CoercibleEnum(Enum):
    @classmethod
    def coerce(cls, value: Any):
        if isinstance(value, cls):
            return value
        if value is None:
            return None
        text = str(value).strip()
        # Exact value/name first.
        for item in cls:
            if text == str(item.value) or text.lower() == item.name.lower():
                return item
        normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        for item in cls:
            if normalized in {
                re.sub(r"[^a-z0-9]+", "_", str(item.value).lower()).strip("_"),
                item.name.lower(),
            }:
                return item
        raise ValueError(f"Unknown {cls.__name__}: {value!r}")


class DifficultyLevel(IntEnum):
    """The five supported difficulty levels.

    The integer values are meaningful and should be backed by actual
    ``DifficultyParameters`` changes; the label is never treated as the sole
    source of difficulty.
    """

    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5

    @classmethod
    def coerce(cls, value: Any) -> "DifficultyLevel":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            match = re.search(r"[1-5]", value)
            if match:
                value = int(match.group(0))
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Difficulty must be an integer from 1 to 5: {value!r}") from exc


class DatasetSplit(str, _CoercibleEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CHALLENGE = "challenge"


class TaskType(str, _CoercibleEnum):
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    MEMORY = "memory"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    ARITHMETIC = "arithmetic"
    STRUCTURED_OUTPUT = "structured_output"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CUSTOM = "custom"


class DistractorType(str, _CoercibleEnum):
    IRRELEVANT_INFO = "irrelevant_info"
    CONTRADICTORY_INFO = "contradictory_info"
    NEAR_MATCH = "near_match"
    LONG_CONTEXT = "long_context"
    MISSING_DATA = "missing_data"
    TOOL_FAILURE = "tool_failure"
    CUSTOM = "custom"


class ValidationStatus(str, _CoercibleEnum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class DifficultyParameters:
    """Concrete knobs that make a case easier or harder.

    The common memory-benchmark knobs are first-class fields so the UI can
    display and filter them.  ``extra`` keeps the model open for task-specific
    parameters without silently discarding unknown configuration.
    """

    context_length: int | None = None
    distractor_count: int = 0
    number_of_facts: int | None = None
    conflicting_facts: int = 0
    time_distance: int | float | str | None = None
    missing_data: int = 0
    tool_failure: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("context_length", "distractor_count", "number_of_facts", "conflicting_facts", "missing_data"):
            value = getattr(self, name)
            if value is not None:
                try:
                    value = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be an integer or null") from exc
                if value < 0:
                    raise ValueError(f"{name} cannot be negative")
                setattr(self, name, value)
        self.tool_failure = bool(self.tool_failure)
        if self.extra is None:
            self.extra = {}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "DifficultyParameters":
        if isinstance(value, cls):
            return value
        value = dict(value or {})
        known = {
            "context_length",
            "distractor_count",
            "number_of_facts",
            "conflicting_facts",
            "time_distance",
            "missing_data",
            "tool_failure",
            "extra",
        }
        extra = dict(value.get("extra") or {})
        extra.update({k: v for k, v in value.items() if k not in known})
        value["extra"] = extra
        return cls(**{k: value[k] for k in known if k in value})

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class Distractor:
    """A deliberately inserted distractor and its provenance."""

    kind: DistractorType | str = DistractorType.IRRELEVANT_INFO
    content: Any = ""
    severity: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.kind = DistractorType.coerce(self.kind)
        except ValueError:
            # Custom generator kinds are allowed while retaining a known enum
            # for standard cases.
            self.kind = str(self.kind)
        if self.severity is not None:
            self.severity = float(self.severity)

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class GeneratorMetadata:
    """Reproducibility information for the case generator."""

    template: str = ""
    rules: list[str] = field(default_factory=list)
    procedural_generator: str = ""
    seed: int = 0
    generator_version: str = "0.1.0"
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rules = list(self.rules or [])
        try:
            self.seed = int(self.seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("generator seed must be an integer") from exc
        if self.config is None:
            self.config = {}

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class BenchmarkDefinition:
    """The user-authored definition of a benchmark family."""

    name: str
    capability: str
    description: str = ""
    difficulty: DifficultyLevel | int | str = DifficultyLevel.LEVEL_1
    task_type: TaskType | str = TaskType.CUSTOM
    input_schema: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    scoring: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    definition_id: str = ""
    definition_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.capability = str(self.capability).strip()
        if not self.name:
            raise ValueError("benchmark definition name cannot be empty")
        if not self.capability:
            raise ValueError("benchmark capability cannot be empty")
        self.difficulty = DifficultyLevel.coerce(self.difficulty)
        try:
            self.task_type = TaskType.coerce(self.task_type)
        except ValueError:
            self.task_type = str(self.task_type)
        if not self.definition_id:
            self.definition_id = _slug(self.name)
        self.input_schema = _config_dict(self.input_schema)
        self.scoring = _config_dict(self.scoring, scalar_key="method")
        self.constraints = _config_dict(self.constraints)
        self.metadata = dict(self.metadata or {})
        self.created_at = _coerce_datetime(self.created_at)

    @property
    def id(self) -> str:
        """Compatibility alias used by a few integrations."""

        return self.definition_id

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkDefinition":
        raw = dict(value)
        aliases = {
            "Name": "name",
            "Capability": "capability",
            "Description": "description",
            "Difficulty": "difficulty",
            "Task Type": "task_type",
            "TaskType": "task_type",
            "Input Schema": "input_schema",
            "InputSchema": "input_schema",
            "Expected Output": "expected_output",
            "ExpectedOutput": "expected_output",
            "Scoring": "scoring",
            "Constraints": "constraints",
            "ID": "definition_id",
            "Id": "definition_id",
            "Definition ID": "definition_id",
            "Definition Version": "definition_version",
        }
        data = {aliases.get(str(key), key): item for key, item in raw.items()}
        if "id" in data and "definition_id" not in data:
            data["definition_id"] = data.pop("id")
        return cls(**data)


@dataclass
class ModelOutput:
    """One model/provider response attached to a benchmark case."""

    provider: str
    output: Any = None
    model: str | None = None
    latency_ms: float | None = None
    tokens: int | dict[str, int] | None = None
    error: str | None = None
    run_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = str(self.provider)
        if self.latency_ms is not None:
            self.latency_ms = float(self.latency_ms)
        if self.metadata is None:
            self.metadata = {}
        self.metadata.setdefault("provider", self.provider)
        self.created_at = _coerce_datetime(self.created_at)

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class EvaluationResult:
    """Evaluation metrics for one provider's output on one case."""

    case_id: str = ""
    provider: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    passed: bool | None = None
    latency_ms: float | None = None
    tokens: int | dict[str, int] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=utc_now)
    # These optional fields mirror the richer evaluator record while keeping
    # the core model usable by lightweight providers and external scripts.
    prediction: Any = None
    expected: Any = None
    errors: list[str] = field(default_factory=list)
    task_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_id = str(self.case_id)
        self.provider = str(self.provider)
        self.metrics = {str(k): float(v) for k, v in (self.metrics or {}).items()}
        if self.score is not None:
            self.score = float(self.score)
        if self.latency_ms is not None:
            self.latency_ms = float(self.latency_ms)
        if self.details is None:
            self.details = {}
        self.errors = [str(item) for item in (self.errors or [])]
        if self.metadata is None:
            self.metadata = {}
        self.evaluated_at = _coerce_datetime(self.evaluated_at)

    @property
    def pass_fail(self) -> bool | None:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class BenchmarkCase:
    """A single deterministic benchmark item with explicit ground truth."""

    case_id: str = ""
    definition_id: str = ""
    split: DatasetSplit | str = DatasetSplit.TEST
    difficulty: DifficultyLevel | int | str = DifficultyLevel.LEVEL_1
    difficulty_params: DifficultyParameters | Mapping[str, Any] = field(default_factory=DifficultyParameters)
    task_type: TaskType | str = TaskType.CUSTOM
    prompt: str = ""
    context: Any = None
    input_data: Any = None
    expected_output: Any = None
    ground_truth: Any = None
    scoring: dict[str, Any] = field(default_factory=dict)
    distractors: list[Distractor | Mapping[str, Any] | Any] = field(default_factory=list)
    generator: GeneratorMetadata | Mapping[str, Any] = field(default_factory=GeneratorMetadata)
    validation_status: ValidationStatus | str = ValidationStatus.PENDING
    validation_errors: list[str] = field(default_factory=list)
    synthetic: bool = True
    model_outputs: list[ModelOutput | Mapping[str, Any]] = field(default_factory=list)
    evaluation_results: list[EvaluationResult | Mapping[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.definition_id = str(self.definition_id or "")
        try:
            self.split = DatasetSplit.coerce(self.split)
        except ValueError:
            self.split = str(self.split)
        self.difficulty = DifficultyLevel.coerce(self.difficulty)
        self.difficulty_params = DifficultyParameters.from_dict(self.difficulty_params)
        try:
            self.task_type = TaskType.coerce(self.task_type)
        except ValueError:
            self.task_type = str(self.task_type)
        self.distractors = [
            item if isinstance(item, Distractor) else Distractor(**item) if isinstance(item, Mapping) else Distractor(content=item)
            for item in (self.distractors or [])
        ]
        self.generator = self.generator if isinstance(self.generator, GeneratorMetadata) else GeneratorMetadata(**dict(self.generator or {}))
        # Early generator revisions used ``unvalidated``; retain that input
        # while exposing the canonical pending/valid/invalid/warning enum.
        status_aliases = {
            "unvalidated": ValidationStatus.PENDING,
            "not_validated": ValidationStatus.PENDING,
            "validated": ValidationStatus.VALID,
        }
        raw_status = str(getattr(self.validation_status, "value", self.validation_status)).strip().lower()
        if raw_status in status_aliases:
            self.validation_status = status_aliases[raw_status]
        else:
            try:
                self.validation_status = ValidationStatus.coerce(self.validation_status)
            except ValueError:
                self.validation_status = str(self.validation_status)
        self.validation_errors = [str(item) for item in (self.validation_errors or [])]
        self.synthetic = bool(self.synthetic)
        self.model_outputs = [
            item if isinstance(item, ModelOutput) else ModelOutput(**dict(item)) for item in (self.model_outputs or [])
        ]
        self.evaluation_results = [
            item if isinstance(item, EvaluationResult) else EvaluationResult(**dict(item))
            for item in (self.evaluation_results or [])
        ]
        self.scoring = _config_dict(self.scoring, scalar_key="method")
        self.metadata = dict(self.metadata or {})
        if self.synthetic:
            self.metadata.setdefault("data_origin", "Synthetic")
        self.created_at = _coerce_datetime(self.created_at)
        if not self.case_id:
            # Stable enough for generated cases and independent of process
            # randomisation; generators should still pass an explicit index
            # when they need human-readable IDs.
            payload = json.dumps(
                {
                    "definition_id": self.definition_id,
                    "split": str(getattr(self.split, "value", self.split)),
                    "prompt": self.prompt,
                    "context": self.context,
                    "input_data": self.input_data,
                    "ground_truth": self.ground_truth,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            self.case_id = f"case-{hashlib.sha256(payload).hexdigest()[:12]}"

    @property
    def expected_answer(self) -> Any:
        """Compatibility alias; ``ground_truth`` is the canonical answer."""

        return self.ground_truth if self.ground_truth is not None else self.expected_output

    # Compatibility aliases used by early generator/UI prototypes.  They are
    # properties (rather than duplicate dataclass fields), so canonical JSON
    # remains unambiguous and hashes do not count the same value twice.
    @property
    def id(self) -> str:
        return self.case_id

    @id.setter
    def id(self, value: str) -> None:
        self.case_id = str(value)

    @property
    def benchmark_type(self) -> TaskType | str:
        return self.task_type

    @benchmark_type.setter
    def benchmark_type(self, value: TaskType | str) -> None:
        try:
            self.task_type = TaskType.coerce(value)
        except ValueError:
            self.task_type = str(value)

    @property
    def difficulty_parameters(self) -> DifficultyParameters:
        return self.difficulty_params

    @difficulty_parameters.setter
    def difficulty_parameters(self, value: DifficultyParameters | Mapping[str, Any]) -> None:
        self.difficulty_params = DifficultyParameters.from_dict(value)

    @property
    def seed(self) -> int:
        return self.generator.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.generator.seed = int(value)

    @property
    def generator_version(self) -> str:
        return self.generator.generator_version

    @generator_version.setter
    def generator_version(self, value: str) -> None:
        self.generator.generator_version = str(value)

    @property
    def input(self) -> Any:
        return self.input_data

    @input.setter
    def input(self, value: Any) -> None:
        self.input_data = value

    @property
    def expected(self) -> Any:
        return self.expected_answer

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkCase":
        raw = dict(value)
        aliases = {
            "Case ID": "case_id",
            "ID": "case_id",
            "Definition ID": "definition_id",
            "Split": "split",
            "Difficulty": "difficulty",
            "Task Type": "task_type",
            "Prompt": "prompt",
            "Context": "context",
            "Input": "input_data",
            "Input Data": "input_data",
            "Difficulty Parameters": "difficulty_params",
            "Distractors": "distractors",
            "Generator": "generator",
            "Expected Output": "expected_output",
            "Ground Truth": "ground_truth",
            "Expected Answer": "ground_truth",
            "Validation Status": "validation_status",
            "Validation Errors": "validation_errors",
            "Synthetic": "synthetic",
        }
        data = {aliases.get(str(key), key): item for key, item in raw.items()}
        if "expected_answer" in data and "ground_truth" not in data:
            data["ground_truth"] = data.pop("expected_answer")
        return cls(**data)


@dataclass
class DatasetVersion:
    """Manifest-level reproducibility information."""

    benchmark_version: str = "0.1.0"
    seed: int = 0
    generator_version: str = "0.1.0"
    config: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    case_count: int = 0
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self.case_count = int(self.case_count)
        self.config = dict(self.config or {})
        self.created_at = _coerce_datetime(self.created_at)

    @property
    def version(self) -> str:
        return self.benchmark_version

    @version.setter
    def version(self, value: str) -> None:
        self.benchmark_version = str(value)

    @property
    def hash(self) -> str:
        return self.dataset_hash

    @hash.setter
    def hash(self, value: str) -> None:
        self.dataset_hash = str(value)

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class DatasetStatistics:
    """Computed, display-ready statistics for a dataset."""

    case_count: int = 0
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    type_distribution: dict[str, int] = field(default_factory=dict)
    split_distribution: dict[str, int] = field(default_factory=dict)
    context_length: dict[str, float] = field(default_factory=dict)
    answer_length: dict[str, float] = field(default_factory=dict)
    duplicate_rate: float = 0.0
    validation_failure_rate: float = 0.0
    synthetic_case_count: int = 0
    valid_case_count: int = 0
    invalid_case_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _length(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
            try:
                return len(value)
            except TypeError:
                pass
        return len(str(value))

    @classmethod
    def from_cases(cls, cases: Sequence[BenchmarkCase]) -> "DatasetStatistics":
        items = list(cases or [])
        n = len(items)
        difficulty: dict[str, int] = {}
        types: dict[str, int] = {}
        splits: dict[str, int] = {}
        contexts: list[int] = []
        answers: list[int] = []
        fingerprints: list[str] = []
        synthetic = valid = invalid = 0
        for case in items:
            level = str(int(case.difficulty))
            task = str(getattr(case.task_type, "value", case.task_type))
            split = str(getattr(case.split, "value", case.split))
            difficulty[level] = difficulty.get(level, 0) + 1
            types[task] = types.get(task, 0) + 1
            splits[split] = splits.get(split, 0) + 1
            contexts.append(cls._length(case.context))
            answers.append(cls._length(case.ground_truth if case.ground_truth is not None else case.expected_output))
            fp_payload = json.dumps(
                {"prompt": case.prompt, "context": case.context, "input": case.input_data, "answer": case.ground_truth},
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            fingerprints.append(fp_payload)
            synthetic += int(case.synthetic)
            status = str(getattr(case.validation_status, "value", case.validation_status))
            valid += int(status == ValidationStatus.VALID.value)
            invalid += int(status == ValidationStatus.INVALID.value)

        def summary(values: list[int]) -> dict[str, float]:
            if not values:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
            ordered = sorted(values)
            mid = len(ordered) // 2
            median = float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
            return {"min": float(ordered[0]), "max": float(ordered[-1]), "mean": sum(values) / len(values), "median": median}

        unique = len(set(fingerprints))
        return cls(
            case_count=n,
            difficulty_distribution=difficulty,
            type_distribution=types,
            split_distribution=splits,
            context_length=summary(contexts),
            answer_length=summary(answers),
            duplicate_rate=(n - unique) / n if n else 0.0,
            validation_failure_rate=invalid / n if n else 0.0,
            synthetic_case_count=synthetic,
            valid_case_count=valid,
            invalid_case_count=invalid,
        )

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)


@dataclass
class BenchmarkDataset:
    """A versioned collection of benchmark cases and their definitions."""

    name: str = "Synthetic Benchmark Dataset"
    cases: list[BenchmarkCase | Mapping[str, Any]] = field(default_factory=list)
    definitions: list[BenchmarkDefinition | Mapping[str, Any]] = field(default_factory=list)
    version: DatasetVersion | Mapping[str, Any] = field(default_factory=DatasetVersion)
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: DatasetStatistics | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.cases = [item if isinstance(item, BenchmarkCase) else BenchmarkCase.from_dict(item) for item in (self.cases or [])]
        self.definitions = [
            item if isinstance(item, BenchmarkDefinition) else BenchmarkDefinition.from_dict(item)
            for item in (self.definitions or [])
        ]
        self.version = self.version if isinstance(self.version, DatasetVersion) else DatasetVersion(**dict(self.version or {}))
        self.metadata = dict(self.metadata or {})
        if self.statistics is not None and not isinstance(self.statistics, DatasetStatistics):
            self.statistics = DatasetStatistics(**dict(self.statistics))

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def definition(self) -> BenchmarkDefinition | None:
        """Return the first definition for single-definition integrations."""

        return self.definitions[0] if self.definitions else None

    @definition.setter
    def definition(self, value: BenchmarkDefinition | Mapping[str, Any] | None) -> None:
        if value is None:
            self.definitions = []
        else:
            item = value if isinstance(value, BenchmarkDefinition) else BenchmarkDefinition.from_dict(value)
            self.definitions = [item]

    @property
    def items(self) -> list[BenchmarkCase]:
        return self.cases

    @items.setter
    def items(self, value: Sequence[BenchmarkCase | Mapping[str, Any]]) -> None:
        self.cases = [item if isinstance(item, BenchmarkCase) else BenchmarkCase.from_dict(item) for item in value]

    @property
    def seed(self) -> int:
        return self.version.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.version.seed = int(value)

    @property
    def generator_version(self) -> str:
        return self.version.generator_version

    @generator_version.setter
    def generator_version(self, value: str) -> None:
        self.version.generator_version = str(value)

    @property
    def dataset_hash(self) -> str:
        return self.version.dataset_hash

    @dataset_hash.setter
    def dataset_hash(self, value: str) -> None:
        self.version.dataset_hash = str(value)

    @property
    def hash(self) -> str:
        return self.dataset_hash

    @hash.setter
    def hash(self, value: str) -> None:
        self.dataset_hash = value

    @property
    def benchmark_version(self) -> str:
        return self.version.benchmark_version

    @benchmark_version.setter
    def benchmark_version(self, value: str) -> None:
        self.version.benchmark_version = str(value)

    @property
    def splits(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            key = str(getattr(case.split, "value", case.split))
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def split_counts(self) -> dict[str, int]:
        return self.splits

    def compute_statistics(self) -> DatasetStatistics:
        self.statistics = DatasetStatistics.from_cases(self.cases)
        return self.statistics

    def to_dict(self) -> dict[str, Any]:
        from .serialization import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkDataset":
        raw = dict(value)
        aliases = {
            "Name": "name",
            "Cases": "cases",
            "Items": "cases",
            "Definitions": "definitions",
            "Version": "version",
            "Dataset Version": "version",
            "Metadata": "metadata",
            "Statistics": "statistics",
        }
        data = {aliases.get(str(key), key): item for key, item in raw.items()}
        # Older exports may call the manifest ``dataset_version``.
        if "dataset_version" in data and "version" not in data:
            data["version"] = data.pop("dataset_version")
        return cls(**data)

    @classmethod
    def from_definition(
        cls,
        definition: BenchmarkDefinition | Mapping[str, Any],
        cases: Sequence[BenchmarkCase | Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "BenchmarkDataset":
        """Convenience constructor for the common single-definition case."""

        item = definition if isinstance(definition, BenchmarkDefinition) else BenchmarkDefinition.from_dict(definition)
        kwargs.setdefault("name", item.name)
        return cls(definitions=[item], cases=list(cases or []), **kwargs)


# Friendly aliases used by early prototypes and external scripts.
Definition = BenchmarkDefinition
Case = BenchmarkCase
Dataset = BenchmarkDataset
# Historical name retained for integrations that predate ``TaskType``.
BenchmarkType = TaskType
Split = DatasetSplit
