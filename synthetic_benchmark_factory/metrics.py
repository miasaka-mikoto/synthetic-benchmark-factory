"""Evaluation metrics for synthetic benchmark cases.

All functions are deterministic and dependency-free.  Scores are normalized to
``0..1`` and can therefore be aggregated across heterogeneous benchmark task
types.  The evaluator never calls a model; it only compares a model prediction
with the case's ground truth.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class EvaluationResult:
    """Portable per-case evaluation record."""

    case_id: str | None = None
    provider: str | None = None
    prediction: Any = None
    expected: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    task_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_fail(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "prediction": self.prediction,
            "expected": self.expected,
            "metrics": dict(self.metrics),
            "score": self.score,
            "passed": self.passed,
            "pass_fail": self.passed,
            "errors": list(self.errors),
            "task_type": self.task_type,
            "metadata": dict(self.metadata),
        }


def parse_json_like(value: Any) -> Any:
    """Parse JSON strings when possible, otherwise return the original value."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def canonical(value: Any, *, case_sensitive: bool = False, sort_keys: bool = True) -> Any:
    """Create a comparison-safe representation while preserving structure."""

    value = parse_json_like(value)
    if isinstance(value, Mapping):
        return {
            str(k) if case_sensitive else str(k).lower(): canonical(v, case_sensitive=case_sensitive, sort_keys=sort_keys)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        } if sort_keys else {
            str(k) if case_sensitive else str(k).lower(): canonical(v, case_sensitive=case_sensitive, sort_keys=sort_keys)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [canonical(item, case_sensitive=case_sensitive, sort_keys=sort_keys) for item in value]
    if isinstance(value, set):
        items = [canonical(item, case_sensitive=case_sensitive, sort_keys=sort_keys) for item in value]
        return sorted(items, key=_stable_repr)
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip())
        return text if case_sensitive else text.casefold()
    return value


def exact_match(prediction: Any, expected: Any, *, normalize: bool = True, case_sensitive: bool = False) -> float:
    """Return 1 for an exact match and 0 otherwise."""

    left = canonical(prediction, case_sensitive=case_sensitive) if normalize else prediction
    right = canonical(expected, case_sensitive=case_sensitive) if normalize else expected
    return 1.0 if left == right else 0.0


def _tokens(value: Any) -> list[str]:
    value = parse_json_like(value)
    if isinstance(value, Mapping):
        # Include key/value atoms so extraction and structured outputs receive
        # useful partial credit without requiring a special tokenizer.
        tokens: list[str] = []
        for key in sorted(value, key=str):
            tokens.extend(_tokens(str(key)))
            tokens.extend(_tokens(value[key]))
        return tokens
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_tokens(item))
        return result
    text = str(value if value is not None else "").casefold()
    return re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)


def f1_score(prediction: Any, expected: Any) -> float:
    """Token/multiset F1 score supporting strings, lists and mappings."""

    predicted = Counter(_tokens(prediction))
    truth = Counter(_tokens(expected))
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    overlap = sum((predicted & truth).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(truth.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def schema_score(prediction: Any, expected: Any = None, schema: Any = None) -> float:
    """Score shape, required keys and primitive types of structured output.

    A schema can use JSON-schema-like ``{"field": "string"}`` notation or a
    nested example value.  If no explicit schema is supplied, the expected
    output's shape acts as the schema.  Values are not required to match here;
    use :func:`exact_match` or :func:`f1_score` alongside this score for that.
    """

    prediction = parse_json_like(prediction)
    expected = parse_json_like(expected)
    schema = parse_json_like(schema)
    if schema is None:
        schema = expected
    if schema is None:
        return 1.0 if prediction is not None else 0.0
    if isinstance(schema, Mapping):
        if not isinstance(prediction, Mapping):
            return 0.0
        if not schema:
            return 1.0 if isinstance(prediction, Mapping) else 0.0
        checks = []
        for key, spec in schema.items():
            if key not in prediction:
                checks.append(0.0)
                continue
            checks.append(_value_schema_score(prediction[key], spec))
        # Penalize unexpected top-level keys gently; extra metadata is common,
        # but an output containing only extras should not pass.
        extras = set(prediction) - set(schema)
        base = sum(checks) / len(checks) if checks else 1.0
        if extras:
            base *= max(0.0, 1.0 - min(0.5, len(extras) / max(1, len(schema) + len(extras))))
        return max(0.0, min(1.0, base))
    if isinstance(schema, (list, tuple)):
        if not isinstance(prediction, (list, tuple)):
            return 0.0
        if not schema:
            return 1.0
        if not prediction:
            return 0.0
        # A one-element schema describes all list elements.
        specs = list(schema)
        scores = []
        for index, item in enumerate(prediction):
            spec = specs[index] if index < len(specs) else specs[-1]
            scores.append(_value_schema_score(item, spec))
        return sum(scores) / len(scores) if scores else 0.0
    return 1.0 if _type_matches(prediction, schema) else 0.0


def _value_schema_score(value: Any, spec: Any) -> float:
    if isinstance(spec, str) and spec.lower() in {
        "string", "str", "integer", "int", "number", "float", "boolean", "bool", "array", "list", "object", "dict",
    }:
        return 1.0 if _type_matches(value, spec) else 0.0
    if isinstance(spec, Mapping):
        return schema_score(value, spec, spec)
    if isinstance(spec, (list, tuple)):
        return schema_score(value, spec, spec)
    # For literal examples, schema score is type-only (not value equality).
    return 1.0 if _type_matches(value, spec) else 0.0


def _type_matches(value: Any, spec: Any) -> bool:
    if isinstance(spec, str):
        name = spec.lower()
        return {
            "string": isinstance(value, str),
            "str": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "int": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, Number) and not isinstance(value, bool),
            "float": isinstance(value, float),
            "boolean": isinstance(value, bool),
            "bool": isinstance(value, bool),
            "array": isinstance(value, list),
            "list": isinstance(value, list),
            "object": isinstance(value, Mapping),
            "dict": isinstance(value, Mapping),
        }.get(name, True)
    if spec is None:
        return value is None
    if isinstance(spec, bool):
        return isinstance(value, bool)
    if isinstance(spec, int) and not isinstance(spec, bool):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(spec, float):
        return isinstance(value, Number) and not isinstance(value, bool)
    if isinstance(spec, str):
        return isinstance(value, str)
    if isinstance(spec, Mapping):
        return isinstance(value, Mapping)
    if isinstance(spec, (list, tuple)):
        return isinstance(value, (list, tuple))
    return isinstance(value, type(spec))


def numeric_score(prediction: Any, expected: Any, *, tolerance: float = 0.0) -> float:
    try:
        left = float(parse_json_like(prediction))
        right = float(parse_json_like(expected))
    except (TypeError, ValueError):
        return exact_match(prediction, expected)
    if math.isclose(left, right, rel_tol=float(tolerance), abs_tol=float(tolerance)):
        return 1.0
    if tolerance > 0:
        distance = abs(left - right)
        return max(0.0, 1.0 - distance / max(abs(right), tolerance))
    return 0.0


def ordering_score(prediction: Any, expected: Any) -> float:
    """Order-sensitive score for plans/step sequences."""

    pred = list(parse_json_like(prediction) or []) if isinstance(parse_json_like(prediction), (list, tuple)) else _tokens(prediction)
    truth = list(parse_json_like(expected) or []) if isinstance(parse_json_like(expected), (list, tuple)) else _tokens(expected)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    # Longest common subsequence normalized by expected length.
    matrix = [[0] * (len(truth) + 1) for _ in range(len(pred) + 1)]
    for i, left in enumerate(pred, 1):
        for j, right in enumerate(truth, 1):
            matrix[i][j] = matrix[i - 1][j - 1] + 1 if canonical(left) == canonical(right) else max(matrix[i - 1][j], matrix[i][j - 1])
    return matrix[-1][-1] / max(len(truth), 1)


def task_specific_score(
    prediction: Any,
    expected: Any,
    task_type: str | None = None,
    *,
    scoring: Mapping[str, Any] | str | None = None,
    schema: Any = None,
) -> float:
    """Choose an appropriate score for one benchmark type."""

    kind = str(task_type or "").lower().replace("-", "_").replace(" ", "_")
    config = scoring if isinstance(scoring, Mapping) else {}
    custom = config.get("callable", config.get("function", config.get("scorer"))) if isinstance(config, Mapping) else None
    if callable(custom):
        try:
            return max(0.0, min(1.0, float(custom(prediction, expected))))
        except TypeError:
            try:
                return max(0.0, min(1.0, float(custom(prediction=prediction, expected=expected))))
            except Exception:
                pass
        except Exception:
            pass
    metric_name = str(config.get("metric", config.get("method", config.get("type", config.get("name", ""))))).lower().replace(" ", "_").replace("-", "_")
    if metric_name in {"exact", "exact_match", "em"}:
        return exact_match(prediction, expected)
    if metric_name in {"f1", "token_f1"}:
        return f1_score(prediction, expected)
    if metric_name in {"schema", "json_schema"}:
        return schema_score(prediction, expected, schema)
    if metric_name in {"pass", "pass_fail", "binary"}:
        return exact_match(prediction, expected)
    if metric_name in {"ordering", "sequence"}:
        return ordering_score(prediction, expected)
    if metric_name in {"numeric", "arithmetic"}:
        return numeric_score(prediction, expected, tolerance=float(config.get("tolerance", 0.0)))

    if kind in {"structured_output", "structured", "json", "schema"}:
        shape = schema_score(prediction, expected, schema)
        value = exact_match(prediction, expected)
        # A schema-only benchmark may intentionally award credit for a valid
        # shape (``metric: schema`` above).  The default structured-output
        # evaluator, however, must not pass a wrong value merely because its
        # keys/types are correct.
        return value if shape >= 1.0 else 0.5 * shape
    if kind in {"planning", "plan", "multi_step_planning"}:
        return ordering_score(prediction, expected)
    if kind in {"arithmetic", "math", "numeric"}:
        return numeric_score(prediction, expected, tolerance=float(config.get("tolerance", 0.0)))
    if kind in {"extraction", "information_extraction", "memory", "long_term_memory"}:
        return f1_score(prediction, expected)
    # Classification, tool selection and instruction following generally have
    # an unambiguous answer.
    return exact_match(prediction, expected)


def evaluate_output(
    prediction: Any,
    expected: Any,
    *,
    task_type: str | None = None,
    scoring: Mapping[str, Any] | str | None = None,
    schema: Any = None,
    threshold: float | None = None,
    case_id: str | None = None,
    provider: str | None = None,
) -> EvaluationResult:
    """Evaluate one prediction and return all requested metrics."""

    task_type = _enum_value(task_type)
    errors: list[str] = []
    try:
        em = exact_match(prediction, expected)
        f1 = f1_score(prediction, expected)
        sch = schema_score(prediction, expected, schema) if schema is not None or isinstance(parse_json_like(expected), (Mapping, list, tuple)) else 1.0
        task = task_specific_score(prediction, expected, task_type, scoring=scoring, schema=schema)
    except Exception as exc:  # evaluator should never crash a complete run
        errors.append(f"metric_error: {exc}")
        em = f1 = sch = task = 0.0
    metrics = {
        "exact_match": round(float(em), 6),
        "f1": round(float(f1), 6),
        "schema": round(float(sch), 6),
        "task_specific": round(float(task), 6),
    }
    config = scoring if isinstance(scoring, Mapping) else {}
    score = float(task)
    if "weights" in config and isinstance(config["weights"], Mapping):
        weights = config["weights"]
        selected = {name: metrics.get(name, 0.0) for name in weights}
        denominator = sum(float(weight) for weight in weights.values())
        if denominator:
            score = sum(selected[name] * float(weights[name]) for name in selected) / denominator
    cutoff = float(threshold if threshold is not None else config.get("threshold", 1.0 if config.get("pass_exact", False) else 0.5))
    passed = not errors and score >= cutoff
    metrics["score"] = round(score, 6)
    metrics["pass_fail"] = 1.0 if passed else 0.0
    return EvaluationResult(
        case_id=case_id,
        provider=provider,
        prediction=prediction,
        expected=expected,
        metrics=metrics,
        score=round(score, 6),
        passed=passed,
        errors=errors,
        task_type=task_type,
        metadata={"threshold": cutoff},
    )


def evaluate_case(
    case: Any,
    prediction: Any,
    *,
    provider: str | None = None,
    provider_name: str | None = None,
    threshold: float | None = None,
) -> EvaluationResult:
    """Evaluate a ``BenchmarkCase`` or case-shaped mapping/object."""

    expected = _case_field(case, "ground_truth", "expected_output", "expected", default=None)
    if expected is None:
        # Some lightweight case dictionaries carry only ``expected_output``;
        # core dataclasses expose both fields with a nullable ground_truth.
        expected = _case_field(case, "expected_output", "expected", default=None)
    task_type = _enum_value(_case_field(case, "benchmark_type", "task_type", "type", default=None))
    scoring = _case_field(case, "scoring", default=None)
    input_data = _case_field(case, "input_data", "input", default={})
    schema = _case_field(case, "schema", default=None)
    if schema is None and isinstance(input_data, Mapping):
        schema = input_data.get("schema")
    return evaluate_output(
        prediction,
        expected,
        task_type=task_type,
        scoring=scoring,
        schema=schema,
        threshold=threshold,
        case_id=str(_case_field(case, "case_id", "id", default="")) or None,
        provider=provider if provider is not None else provider_name,
    )


def _case_field(case: Any, *names: str, default: Any = None) -> Any:
    if isinstance(case, Mapping):
        for name in names:
            if name in case:
                return case[name]
    for name in names:
        try:
            value = getattr(case, name)
        except AttributeError:
            continue
        if value is not None:
            return value
    return default


def _enum_value(value: Any) -> Any:
    """Return a JSON-friendly enum value without importing core models."""

    return getattr(value, "value", value)


def _stable_repr(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


# Short aliases used by the CLI/editor and by older experiment notebooks.
exact_match_score = exact_match
f1 = f1_score
schema_validation_score = schema_score
pass_fail_score = lambda prediction, expected: exact_match(prediction, expected)
evaluate_prediction = evaluate_output


__all__ = [
    "EvaluationResult",
    "canonical",
    "evaluate_case",
    "evaluate_output",
    "exact_match",
    "exact_match_score",
    "f1_score",
    "f1",
    "numeric_score",
    "ordering_score",
    "parse_json_like",
    "schema_score",
    "schema_validation_score",
    "pass_fail_score",
    "evaluate_prediction",
    "task_specific_score",
]
