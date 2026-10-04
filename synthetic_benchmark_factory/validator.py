"""Semantic validation for generated benchmark datasets.

The lightweight :mod:`schema` module checks dataclass shape.  This module adds
the checks that make a benchmark scientifically useful: deterministic answers,
duplicate and split leakage detection, ambiguity/impossibility flags, and
task-specific invariants.  It never invokes a model provider.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .generator import SUPPORTED_TYPES, _jsonable, _normalise_type, _stable_hash
from .models import BenchmarkCase, BenchmarkDataset, BenchmarkDefinition, ValidationStatus
from .schema import SchemaIssue, validate_benchmark_case, validate_benchmark_definition, validate_dataset


@dataclass(frozen=True)
class ValidationIssue:
    """One semantic or schema validation finding."""

    path: str
    code: str
    message: str
    severity: str = "error"

    @property
    def is_error(self) -> bool:
        return self.severity.lower() == "error"

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message, "severity": self.severity}

    def __str__(self) -> str:
        return f"{self.severity.upper()} {self.path}: {self.message} ({self.code})"


@dataclass
class ValidationReport:
    """Result returned by :class:`Validator`.

    ``issues`` is the canonical collection.  ``errors`` and ``warnings`` are
    exposed as lists for the UI and CLI, while ``valid`` is convenient in tests.
    The report is iterable to remain compatible with the schema validator's
    list-like return value.
    """

    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_cases: int = 0
    invalid_cases: int = 0
    warning_cases: int = 0
    duplicate_count: int = 0
    leakage_count: int = 0
    ambiguous_count: int = 0
    impossible_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    case_reports: dict[str, "ValidationReport"] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.is_error]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if not issue.is_error]

    @property
    def invalid(self) -> bool:
        return not self.valid

    @property
    def ok(self) -> bool:
        return self.valid

    def __iter__(self):
        return iter(self.issues)

    def __len__(self) -> int:
        return len(self.issues)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.is_error:
            self.valid = False

    def extend(self, issues: Iterable[ValidationIssue]) -> None:
        for issue in issues:
            self.add(issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "checked_cases": self.checked_cases,
            "invalid_cases": self.invalid_cases,
            "warning_cases": self.warning_cases,
            "duplicate_count": self.duplicate_count,
            "leakage_count": self.leakage_count,
            "ambiguous_count": self.ambiguous_count,
            "impossible_count": self.impossible_count,
            "metrics": dict(self.metrics),
        }


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(obj, Mapping) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def _set(obj: Any, key: str, value: Any) -> None:
    try:
        setattr(obj, key, value)
    except Exception:
        if isinstance(obj, MutableMapping):
            obj[key] = value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _canonical(value: Any) -> Any:
    """Canonicalize values for duplicate and leakage comparisons."""

    if isinstance(value, Mapping):
        return {str(k).casefold(): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]).casefold())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, set):
        return sorted((_canonical(v) for v in value), key=lambda item: repr(item))
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip()).casefold()
    if is_dataclass(value):
        return _canonical(asdict(value))
    return value


def _fingerprint(case: Any, *, include_answer: bool = True) -> str:
    payload = {
        "prompt": _canonical(_get(case, "prompt", default="")),
        "context": _canonical(_get(case, "context", default=None)),
        "input": _canonical(_get(case, "input_data", "input", default=None)),
    }
    if include_answer:
        payload["answer"] = _canonical(_get(case, "ground_truth", "expected_output", default=None))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _issue(path: str, code: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(path=path, code=code, message=message, severity=severity)


def _schema_to_semantic(issue: SchemaIssue, prefix: str = "") -> ValidationIssue:
    return ValidationIssue(path=f"{prefix}{issue.path}", code=issue.code, message=issue.message, severity=issue.severity)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _answer_key(value: Any) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_json(value: Any) -> bool:
    try:
        json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


class Validator:
    """Validate benchmark definitions, cases and complete datasets."""

    def __init__(self, *, strict: bool = False, mark_cases: bool = True):
        self.strict = bool(strict)
        self.mark_cases = bool(mark_cases)

    def validate_definition(self, definition: Any) -> ValidationReport:
        report = ValidationReport()
        for issue in validate_benchmark_definition(definition):
            report.add(_schema_to_semantic(issue))
        task_type = _normalise_type(_get(definition, "task_type", "benchmark_type", default=""))
        if task_type and task_type not in SUPPORTED_TYPES:
            report.add(_issue("task_type", "unsupported_task_type", f"unsupported task type {task_type!r}"))
        scoring = _get(definition, "scoring", default={})
        if not isinstance(scoring, Mapping) or not scoring:
            report.add(_issue("scoring", "invalid_schema", "scoring must be a non-empty object"))
        return report

    # Alias matching the model/schema naming convention.
    validate_benchmark_definition = validate_definition

    def validate_case(self, case: Any, definition: Any | None = None, *, path: str = "case") -> ValidationReport:
        report = ValidationReport(checked_cases=1)
        for issue in validate_benchmark_case(case, definition=definition):
            report.add(_schema_to_semantic(issue, prefix=f"{path}."))
        task_type = _normalise_type(_get(case, "task_type", "benchmark_type", default=""))
        if task_type not in SUPPORTED_TYPES:
            report.add(_issue(f"{path}.task_type", "unsupported_task_type", f"unsupported task type {task_type!r}"))
        ground_truth = _get(case, "ground_truth", default=None)
        expected = _get(case, "expected_output", default=None)
        if ground_truth is None:
            report.add(_issue(f"{path}.ground_truth", "missing_ground_truth", "formal benchmark cases require deterministic ground truth"))
        if ground_truth is not None and expected is not None and _answer_key(ground_truth) != _answer_key(expected):
            report.add(_issue(f"{path}.expected_output", "ground_truth_mismatch", "expected_output differs from ground_truth"))
        if not _safe_json(ground_truth):
            report.add(_issue(f"{path}.ground_truth", "invalid_schema", "ground_truth is not JSON serialisable"))
        self._check_metadata_flags(case, report, path)
        self._check_ambiguity(case, report, path)
        self._check_impossible(case, report, path)
        self._check_task_schema(case, report, path)
        self._check_leakage(case, report, path)
        if self.mark_cases:
            self._mark_case(case, report)
        report.invalid_cases = int(not report.valid)
        report.warning_cases = int(bool(report.warnings))
        return report

    validate_benchmark_case = validate_case

    def validate_dataset(self, dataset: Any) -> ValidationReport:
        report = ValidationReport()
        # Parse through the core schema first.  If a mapping is malformed, the
        # schema helper returns a useful issue and semantic checks can stop.
        try:
            item = dataset if isinstance(dataset, BenchmarkDataset) else BenchmarkDataset.from_dict(dataset)
        except Exception as exc:
            report.add(_issue("dataset", "invalid_dataset", f"cannot parse dataset: {exc}"))
            return report
        definitions = list(_get(item, "definitions", default=[] or []))
        if not definitions:
            one = _get(item, "definition", default=None)
            if one is not None:
                definitions = [one]
        definitions_by_id: dict[str, Any] = {}
        for i, definition in enumerate(definitions):
            d_report = self.validate_definition(definition)
            for issue in d_report.issues:
                report.add(_issue(f"definitions[{i}].{issue.path}", issue.code, issue.message, issue.severity))
            definition_id = str(_get(definition, "definition_id", "id", default=""))
            if definition_id in definitions_by_id and definition_id:
                report.add(_issue(f"definitions[{i}].definition_id", "duplicate_definition", "duplicate definition_id"))
            definitions_by_id[definition_id] = definition
        cases = list(_get(item, "cases", "items", default=[] or []))
        report.checked_cases = len(cases)
        fingerprints: dict[str, tuple[str, str]] = {}
        answer_fingerprints: dict[str, str] = {}
        cross_split_leakage = 0
        for i, case in enumerate(cases):
            case_id = str(_get(case, "case_id", "id", default=f"case-{i}"))
            case_report = self.validate_case(case, path=f"cases[{i}]")
            report.case_reports[case_id] = case_report
            for issue in case_report.issues:
                report.add(issue)
            definition_id = str(_get(case, "definition_id", default=""))
            if definitions_by_id and definition_id not in definitions_by_id:
                report.add(_issue(f"cases[{i}].definition_id", "broken_reference", f"unknown definition_id {definition_id!r}"))
            if not definition_id:
                report.add(_issue(f"cases[{i}].definition_id", "broken_case", "case is not linked to a definition"))
            split = str(_enum_value(_get(case, "split", default="")))
            if split not in {"train", "validation", "test", "challenge"}:
                report.add(_issue(f"cases[{i}].split", "invalid_schema", f"invalid split {split!r}"))
            fingerprint = _fingerprint(case, include_answer=True)
            if fingerprint in fingerprints:
                previous_id, previous_split = fingerprints[fingerprint]
                # A duplicate is invalid even inside one split: duplicate
                # items waste budget and distort item-analysis statistics.
                duplicate_issue = _issue(f"cases[{i}]", "duplicate_case", f"duplicate content of {previous_id}")
                report.add(duplicate_issue)
                case_report.add(duplicate_issue)
                report.duplicate_count += 1
                if previous_split != split:
                    cross_split_leakage += 1
                    leakage_issue = _issue(f"cases[{i}]", "data_leakage", f"content appears in both {previous_split} and {split}")
                    report.add(leakage_issue)
                    case_report.add(leakage_issue)
            else:
                fingerprints[fingerprint] = (case_id, split)
            answer_fp = _answer_key(_get(case, "ground_truth", default=None))
            # Same answer is not itself an error; this map is used for a
            # diagnostic metric and for detecting accidental clone prompts.
            answer_fingerprints[answer_fp] = answer_fingerprints.get(answer_fp, "") + "|" + case_id
        report.invalid_cases = sum(not r.valid for r in report.case_reports.values())
        report.warning_cases = sum(bool(r.warnings) for r in report.case_reports.values())
        report.ambiguous_count += sum(r.ambiguous_count for r in report.case_reports.values())
        report.impossible_count += sum(r.impossible_count for r in report.case_reports.values())
        report.leakage_count = cross_split_leakage + sum(r.leakage_count for r in report.case_reports.values())
        version = _get(item, "version", default=None)
        if version is not None:
            declared = int(_get(version, "case_count", default=0) or 0)
            if declared and declared != len(cases):
                report.add(_issue("version.case_count", "case_count_mismatch", f"declares {declared}, dataset has {len(cases)}", "warning"))
            if not str(_get(version, "benchmark_version", default="")):
                report.add(_issue("version.benchmark_version", "required", "benchmark version is required"))
        n = max(1, len(cases))
        report.metrics.update({
            "case_count": len(cases),
            "duplicate_rate": report.duplicate_count / n,
            "validation_failure_rate": report.invalid_cases / n,
            "data_leakage_rate": report.leakage_count / n,
            "ambiguous_rate": report.ambiguous_count / n,
            "impossible_rate": report.impossible_count / n,
            "valid_case_count": len(cases) - report.invalid_cases,
        })
        # Keep case status synchronized even when an issue came from a
        # cross-case check (duplicate/leakage).
        for case in cases:
            case_id = str(_get(case, "case_id", "id", default=""))
            if case_id in report.case_reports and self.mark_cases:
                self._mark_case(case, report.case_reports[case_id])
        return report

    validate = validate_dataset

    def _mark_case(self, case: Any, report: ValidationReport) -> None:
        if not self.mark_cases:
            return
        errors = [str(issue) for issue in report.errors]
        warnings = [str(issue) for issue in report.warnings]
        if errors:
            status: Any = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
        _set(case, "validation_status", status)
        _set(case, "validation_errors", errors + warnings)

    def _check_metadata_flags(self, case: Any, report: ValidationReport, path: str) -> None:
        metadata = _get(case, "metadata", default={}) or {}
        if metadata.get("ambiguous") or metadata.get("ambiguous_answer"):
            report.add(_issue(f"{path}.metadata", "ambiguous_answer", "case is marked ambiguous"))
            report.ambiguous_count += 1
        if metadata.get("impossible") or metadata.get("impossible_case"):
            report.add(_issue(f"{path}.metadata", "impossible_case", "case is marked impossible"))
            report.impossible_count += 1
        if metadata.get("data_leakage") or metadata.get("leakage"):
            report.add(_issue(f"{path}.metadata", "data_leakage", "case is marked as data leakage"))
            report.leakage_count += 1

    def _check_ambiguity(self, case: Any, report: ValidationReport, path: str) -> None:
        metadata = _get(case, "metadata", default={}) or {}
        candidates = metadata.get("candidate_answers", metadata.get("valid_answers"))
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and len(candidates) > 1:
            report.add(_issue(f"{path}.ground_truth", "ambiguous_answer", "more than one valid answer is declared"))
            report.ambiguous_count += 1
        if metadata.get("valid_order_count") == 0:
            report.add(_issue(f"{path}.metadata", "impossible_case", "planning constraints have no valid order"))
            report.impossible_count += 1
        scoring = _get(case, "scoring", default={}) or {}
        labels = scoring.get("labels") if isinstance(scoring, Mapping) else None
        if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
            normalized = [str(label).casefold() for label in labels]
            if len(normalized) != len(set(normalized)):
                report.add(_issue(f"{path}.scoring.labels", "ambiguous_answer", "classification labels are not unique"))
                report.ambiguous_count += 1

    def _check_impossible(self, case: Any, report: ValidationReport, path: str) -> None:
        metadata = _get(case, "metadata", default={}) or {}
        task_type = _normalise_type(_get(case, "task_type", default=""))
        answer = _get(case, "ground_truth", default=None)
        if task_type == "arithmetic":
            expression = metadata.get("expression")
            if expression:
                try:
                    expected = _safe_eval_arithmetic(expression)
                    if expected != answer:
                        report.add(_issue(f"{path}.ground_truth", "impossible_case", "arithmetic ground truth does not evaluate from expression"))
                        report.impossible_count += 1
                except Exception as exc:
                    report.add(_issue(f"{path}.context", "broken_case", f"arithmetic expression cannot be evaluated: {exc}"))
                    report.impossible_count += 1
        elif task_type == "tool_selection" and isinstance(answer, Mapping):
            failed = set(metadata.get("failed_tools") or [])
            if answer.get("tool") in failed:
                report.add(_issue(f"{path}.ground_truth", "impossible_case", "ground truth selects a failed tool"))
                report.impossible_count += 1
            raw_valid_tools = (_get(case, "scoring", default={}) or {}).get("valid_tools")
            if raw_valid_tools is None:
                raw_valid_tools = metadata.get("tools", [])
            if isinstance(raw_valid_tools, Mapping):
                raw_valid_tools = list(raw_valid_tools.values())
            valid_tools = {
                str(item.get("name")) if isinstance(item, Mapping) else str(item)
                for item in (raw_valid_tools or [])
            }
            if valid_tools and answer.get("tool") not in valid_tools:
                report.add(_issue(f"{path}.ground_truth", "impossible_case", "ground truth selects a tool absent from catalog"))
                report.impossible_count += 1
        elif task_type == "planning":
            steps = list(answer or []) if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)) else []
            if len(steps) != len(set(steps)):
                report.add(_issue(f"{path}.ground_truth", "impossible_case", "plan repeats an action"))
                report.impossible_count += 1
            constraints = metadata.get("constraints") or []
            positions = {step: i for i, step in enumerate(steps)}
            for constraint in constraints:
                match = re.match(r"(.+?) must happen before (.+)$", str(constraint))
                if match and (match.group(1) not in positions or match.group(2) not in positions or positions[match.group(1)] >= positions[match.group(2)]):
                    report.add(_issue(f"{path}.ground_truth", "impossible_case", f"plan violates constraint {constraint!r}"))
                    report.impossible_count += 1

    def _check_task_schema(self, case: Any, report: ValidationReport, path: str) -> None:
        task_type = _normalise_type(_get(case, "task_type", default=""))
        answer = _get(case, "ground_truth", default=None)
        scoring = _get(case, "scoring", default={}) or {}
        if task_type in {"extraction", "memory", "tool_selection", "structured_output"} and not isinstance(answer, Mapping):
            report.add(_issue(f"{path}.ground_truth", "invalid_schema", f"{task_type} ground truth should be a JSON object"))
        required = scoring.get("required_fields") if isinstance(scoring, Mapping) else None
        if required and isinstance(answer, Mapping):
            missing = [field for field in required if field not in answer]
            if missing:
                report.add(_issue(f"{path}.ground_truth", "invalid_schema", f"missing required fields: {', '.join(map(str, missing))}"))
            if scoring.get("additional_properties") is False:
                extras = [key for key in answer if key not in required]
                if extras:
                    report.add(_issue(f"{path}.ground_truth", "invalid_schema", f"unexpected fields: {', '.join(map(str, extras))}"))
        metadata = _get(case, "metadata", default={}) or {}
        schema = metadata.get("schema")
        if schema and isinstance(answer, Mapping):
            for key, type_name in schema.items():
                if key not in answer:
                    continue
                if not _matches_type(answer[key], type_name):
                    report.add(_issue(f"{path}.ground_truth.{key}", "invalid_schema", f"expected {type_name}"))

    def _check_leakage(self, case: Any, report: ValidationReport, path: str) -> None:
        metadata = _get(case, "metadata", default={}) or {}
        if metadata.get("answer_not_in_instruction") is False:
            return
        prompt = _text(_get(case, "prompt", default=""))
        # A formal answer marker is almost always an accidental gold-label
        # leak.  Natural source facts in context are intentionally allowed.
        if re.search(r"(?:ground[_ ]?truth|expected[_ ]?(?:answer|output)|correct answer)\s*[:=]", prompt, re.I):
            report.add(_issue(f"{path}.prompt", "data_leakage", "prompt contains an explicit answer marker"))
            report.leakage_count += 1
        answer = _get(case, "ground_truth", default=None)
        if isinstance(answer, (str, int, float, bool)) and answer is not None:
            answer_text = str(answer).strip()
            if answer_text and re.search(rf"(?:answer|solution)\s+(?:is|equals?)\s*{re.escape(answer_text)}\b", prompt, re.I):
                report.add(_issue(f"{path}.prompt", "data_leakage", "prompt reveals the ground-truth answer"))
                report.leakage_count += 1


def _matches_type(value: Any, type_name: Any) -> bool:
    name = str(type_name).lower()
    if name in {"str", "string"}:
        return isinstance(value, str)
    if name in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if name in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name in {"bool", "boolean"}:
        return isinstance(value, bool)
    if name in {"list", "array"}:
        return isinstance(value, list)
    if name in {"dict", "object", "mapping"}:
        return isinstance(value, Mapping)
    return True


def _safe_eval_arithmetic(expression: str) -> int:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            a, b = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            return a * b
        raise ValueError("unsupported expression")

    return evaluate(tree)


def validate_case(case: Any, definition: Any | None = None, **kwargs: Any) -> ValidationReport:
    return Validator(**{key: kwargs.pop(key) for key in list(kwargs) if key in {"strict", "mark_cases"}}).validate_case(case, definition=definition, **kwargs)


def validate_dataset(dataset: Any, **kwargs: Any) -> ValidationReport:
    return Validator(**{key: kwargs.pop(key) for key in list(kwargs) if key in {"strict", "mark_cases"}}).validate_dataset(dataset)


def validate_definition(definition: Any, **kwargs: Any) -> ValidationReport:
    return Validator(**{key: kwargs.pop(key) for key in list(kwargs) if key in {"strict", "mark_cases"}}).validate_definition(definition)


DatasetValidator = Validator
BenchmarkValidator = Validator


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "Validator",
    "DatasetValidator",
    "BenchmarkValidator",
    "validate_case",
    "validate_dataset",
    "validate_definition",
]
