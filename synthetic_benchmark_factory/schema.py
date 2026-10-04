"""Lightweight schema and integrity validation for benchmark artifacts.

The project intentionally avoids a mandatory JSON-schema dependency.  These
checks cover the invariants that matter for an executable benchmark factory:
explicit ground truth, valid difficulty/split values, unique identifiers,
definition references, and deterministic duplicate detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDefinition,
    DatasetSplit,
    DifficultyLevel,
    ValidationStatus,
)


@dataclass(frozen=True)
class SchemaIssue:
    path: str
    message: str
    severity: str = "error"
    code: str = "schema_error"

    @property
    def is_error(self) -> bool:
        return self.severity.lower() == "error"

    def __str__(self) -> str:
        return f"{self.severity.upper()} {self.path}: {self.message}"


class SchemaValidationError(ValueError):
    def __init__(self, issues: Sequence[SchemaIssue], message: str | None = None):
        self.issues = list(issues)
        super().__init__(message or "\n".join(str(issue) for issue in self.issues))

    def to_dict(self) -> dict[str, Any]:
        return {"valid": False, "issues": [
            {"path": issue.path, "message": issue.message, "severity": issue.severity, "code": issue.code}
            for issue in self.issues
        ]}


class ValidationIssues(list[SchemaIssue]):
    """List-compatible validation result with convenient status properties.

    Keeping this as a ``list`` subclass preserves the original lightweight API
    (``for issue in validate_dataset(...)``) while allowing UI/tests to use
    ``result.valid``, ``result.errors`` and ``result.warnings`` directly.
    """

    @property
    def errors(self) -> list[SchemaIssue]:
        return [issue for issue in self if issue.is_error]

    @property
    def warnings(self) -> list[SchemaIssue]:
        return [issue for issue in self if not issue.is_error]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def invalid(self) -> bool:
        return not self.valid

    def to_dict(self) -> list[dict[str, str]]:
        return [
            {"path": issue.path, "message": issue.message, "severity": issue.severity, "code": issue.code}
            for issue in self
        ]


def _coerce_definition(value: BenchmarkDefinition | Mapping[str, Any]) -> BenchmarkDefinition:
    return value if isinstance(value, BenchmarkDefinition) else BenchmarkDefinition.from_dict(value)


def _coerce_case(value: BenchmarkCase | Mapping[str, Any]) -> BenchmarkCase:
    return value if isinstance(value, BenchmarkCase) else BenchmarkCase.from_dict(value)


def _issue(path: str, message: str, *, severity: str = "error", code: str = "schema_error") -> SchemaIssue:
    return SchemaIssue(path=path, message=message, severity=severity, code=code)


def validate_benchmark_definition(
    definition: BenchmarkDefinition | Mapping[str, Any], *, raise_on_error: bool = False
) -> ValidationIssues:
    """Validate a benchmark definition and return structured issues."""

    issues = ValidationIssues()
    try:
        item = _coerce_definition(definition)
    except Exception as exc:
        issues.append(_issue("definition", f"cannot parse definition: {exc}", code="invalid_definition"))
        if raise_on_error:
            raise SchemaValidationError(issues)
        return issues
    if not item.name.strip():
        issues.append(_issue("name", "name is required", code="required"))
    if not item.capability.strip():
        issues.append(_issue("capability", "capability is required", code="required"))
    if not isinstance(item.input_schema, Mapping):
        issues.append(_issue("input_schema", "must be an object", code="invalid_schema"))
    if not isinstance(item.scoring, Mapping) or not item.scoring:
        issues.append(_issue("scoring", "must define at least one scoring rule", code="invalid_scoring"))
    if not isinstance(item.constraints, Mapping):
        issues.append(_issue("constraints", "must be an object", code="invalid_constraints"))
    try:
        DifficultyLevel.coerce(item.difficulty)
    except ValueError:
        issues.append(_issue("difficulty", "must be Level 1 through Level 5", code="invalid_difficulty"))
    if item.expected_output is None:
        issues.append(
            _issue(
                "expected_output",
                "definition should describe the expected output schema/value",
                severity="warning",
                code="missing_expected_output_schema",
            )
        )
    if raise_on_error and any(issue.is_error for issue in issues):
        raise SchemaValidationError(issues)
    return issues


def _answer_fingerprint(case: BenchmarkCase) -> str:
    payload = {
        "prompt": case.prompt,
        "context": case.context,
        "input_data": case.input_data,
        "ground_truth": case.ground_truth,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def validate_benchmark_case(
    case: BenchmarkCase | Mapping[str, Any],
    *,
    definition: BenchmarkDefinition | Mapping[str, Any] | None = None,
    raise_on_error: bool = False,
) -> ValidationIssues:
    """Validate one case, including the invariant that ground truth is known."""

    issues = ValidationIssues()
    try:
        item = _coerce_case(case)
    except Exception as exc:
        issues.append(_issue("case", f"cannot parse case: {exc}", code="invalid_case"))
        if raise_on_error:
            raise SchemaValidationError(issues)
        return issues
    if not item.case_id.strip():
        issues.append(_issue("case_id", "case_id is required", code="required"))
    if not item.definition_id.strip():
        issues.append(_issue("definition_id", "definition_id is required", code="required"))
    if not item.prompt.strip() and item.input_data is None and item.context is None:
        issues.append(_issue("prompt", "case has no prompt, input_data, or context", code="empty_case"))
    if item.ground_truth is None:
        issues.append(
            _issue(
                "ground_truth",
                "formal benchmark cases must have deterministic ground truth",
                code="missing_ground_truth",
            )
        )
    # Empty expected output is valid when ground_truth is explicit (for
    # example a task whose answer is an empty list), so do not reject it.
    try:
        level = DifficultyLevel.coerce(item.difficulty)
        if int(level) not in range(1, 6):
            raise ValueError
    except ValueError:
        issues.append(_issue("difficulty", "must be Level 1 through Level 5", code="invalid_difficulty"))
    try:
        DatasetSplit.coerce(item.split)
    except ValueError:
        issues.append(_issue("split", "must be train, validation, test, or challenge", code="invalid_split"))
    if item.generator is None:
        issues.append(_issue("generator", "generator metadata is required", code="missing_generator"))
    elif not isinstance(item.generator.seed, int):
        issues.append(_issue("generator.seed", "seed must be an integer", code="invalid_seed"))
    if not isinstance(item.distractors, list):
        issues.append(_issue("distractors", "must be an array", code="invalid_distractors"))
    status = str(getattr(item.validation_status, "value", item.validation_status))
    if status == ValidationStatus.VALID.value and item.validation_errors:
        issues.append(
            _issue(
                "validation_errors",
                "a valid case cannot retain validation errors",
                severity="warning",
                code="stale_validation_errors",
            )
        )
    if definition is not None:
        definition_item = _coerce_definition(definition)
        if item.definition_id != definition_item.definition_id:
            issues.append(
                _issue(
                    "definition_id",
                    f"does not match supplied definition {definition_item.definition_id!r}",
                    code="definition_mismatch",
                )
            )
        # Only enforce explicit JSON-Schema ``required`` fields.  A compact
        # schema such as ``{"context": "string"}`` often describes the
        # context rather than the separate ``input_data`` object, so treating
        # every key as required would reject valid generated cases.
        required = definition_item.input_schema.get("required", []) if isinstance(definition_item.input_schema, Mapping) else []
        if required and isinstance(item.input_data, Mapping):
            missing = [str(key) for key in required if key not in item.input_data]
            if missing:
                issues.append(
                    _issue(
                        "input_data",
                        f"missing required field(s): {', '.join(missing)}",
                        code="invalid_input_schema",
                    )
                )
        elif required and item.input_data is None:
            issues.append(
                _issue(
                    "input_data",
                    "input_data is required by the definition schema",
                    code="invalid_input_schema",
                )
            )
    # Generator metadata can flag known conditions even if the raw case shape
    # looks valid.  This is useful for procedural generators and keeps the
    # validator transparent rather than trying to infer semantics.
    flags = item.metadata or {}
    if flags.get("ambiguous") or flags.get("ambiguous_answer"):
        issues.append(_issue("metadata.ambiguous", "answer is marked ambiguous", code="ambiguous_answer"))
    if flags.get("impossible") or flags.get("impossible_case"):
        issues.append(_issue("metadata.impossible", "case is marked impossible", code="impossible_case"))
    if flags.get("data_leakage") or flags.get("leakage"):
        issues.append(_issue("metadata.data_leakage", "case is marked as data leakage", code="data_leakage"))
    if raise_on_error and any(issue.is_error for issue in issues):
        raise SchemaValidationError(issues)
    return issues


def validate_dataset(
    dataset: BenchmarkDataset | Mapping[str, Any], *, raise_on_error: bool = False
) -> ValidationIssues:
    """Validate definitions, cases, references, IDs, duplicates, and version metadata."""

    issues = ValidationIssues()
    try:
        item = dataset if isinstance(dataset, BenchmarkDataset) else BenchmarkDataset.from_dict(dataset)
    except Exception as exc:
        issues.append(_issue("dataset", f"cannot parse dataset: {exc}", code="invalid_dataset"))
        if raise_on_error:
            raise SchemaValidationError(issues)
        return issues
    definitions_by_id: dict[str, BenchmarkDefinition] = {}
    for index, definition in enumerate(item.definitions):
        path = f"definitions[{index}]"
        for issue in validate_benchmark_definition(definition):
            issues.append(SchemaIssue(f"{path}.{issue.path}", issue.message, issue.severity, issue.code))
        if definition.definition_id in definitions_by_id:
            issues.append(_issue(f"{path}.definition_id", "duplicate definition_id", code="duplicate_definition"))
        definitions_by_id[definition.definition_id] = definition
    ids: set[str] = set()
    fingerprints: dict[str, str] = {}
    for index, case in enumerate(item.cases):
        path = f"cases[{index}]"
        for issue in validate_benchmark_case(case, definition=definitions_by_id.get(case.definition_id)):
            issues.append(SchemaIssue(f"{path}.{issue.path}", issue.message, issue.severity, issue.code))
        if case.case_id in ids:
            issues.append(_issue(f"{path}.case_id", "duplicate case_id", code="duplicate_case_id"))
        ids.add(case.case_id)
        fingerprint = _answer_fingerprint(case)
        if fingerprint in fingerprints:
            issues.append(
                _issue(
                    path,
                    f"duplicate content of {fingerprints[fingerprint]}",
                    code="duplicate_case",
                )
            )
        else:
            fingerprints[fingerprint] = case.case_id
        if definitions_by_id and case.definition_id not in definitions_by_id:
            issues.append(
                _issue(
                    f"{path}.definition_id",
                    f"unknown definition_id {case.definition_id!r}",
                    code="broken_reference",
                )
            )
    if not item.version.benchmark_version:
        issues.append(_issue("version.benchmark_version", "is required", code="required"))
    if item.version.case_count and item.version.case_count != len(item.cases):
        issues.append(
            _issue(
                "version.case_count",
                f"declares {item.version.case_count} but dataset has {len(item.cases)} cases",
                severity="warning",
                code="case_count_mismatch",
            )
        )
    if raise_on_error and any(issue.is_error for issue in issues):
        raise SchemaValidationError(issues)
    return issues


def is_valid(issues: Iterable[SchemaIssue]) -> bool:
    if isinstance(issues, bool):
        return issues
    if hasattr(issues, "valid"):
        try:
            return bool(getattr(issues, "valid"))
        except Exception:
            pass
    return not any(getattr(issue, "is_error", True) for issue in issues)


def assert_valid(value: Any, *, kind: str = "dataset") -> None:
    if kind == "definition":
        issues = validate_benchmark_definition(value)
    elif kind == "case":
        issues = validate_benchmark_case(value)
    else:
        issues = validate_dataset(value)
    if not is_valid(issues):
        raise SchemaValidationError(issues)


# Concise aliases for code that uses the shorter names.
validate_definition = validate_benchmark_definition
validate_case = validate_benchmark_case
