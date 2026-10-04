"""Per-item benchmark analysis.

The analysis pass helps a benchmark author find cases that are unusable or
uninformative: cases every model misses, cases all models solve trivially, and
cases that separate providers.  It accepts ``RunRecord`` objects, exported
record dictionaries, or core ``EvaluationResult`` dictionaries.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class ItemAnalysis:
    case_id: str
    provider_count: int = 0
    evaluated_count: int = 0
    scores: dict[str, float | None] = field(default_factory=dict)
    passes: dict[str, bool | None] = field(default_factory=dict)
    mean_score: float | None = None
    score_spread: float = 0.0
    pass_rate: float | None = None
    flags: list[str] = field(default_factory=list)
    task_type: str | None = None
    difficulty: int | None = None
    validation_status: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def all_models_wrong(self) -> bool:
        return "all_models_wrong" in self.flags

    @property
    def too_easy(self) -> bool:
        return "too_easy" in self.flags

    @property
    def discriminating(self) -> bool:
        return "discriminating" in self.flags

    @property
    def problematic(self) -> bool:
        return "problematic" in self.flags

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "provider_count": self.provider_count,
            "evaluated_count": self.evaluated_count,
            "scores": self.scores,
            "passes": self.passes,
            "mean_score": self.mean_score,
            "score_spread": self.score_spread,
            "pass_rate": self.pass_rate,
            "flags": list(self.flags),
            "all_models_wrong": self.all_models_wrong,
            "too_easy": self.too_easy,
            "discriminating": self.discriminating,
            "problematic": self.problematic,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "validation_status": self.validation_status,
            "notes": list(self.notes),
        }


@dataclass
class ItemAnalysisReport:
    items: list[ItemAnalysis] = field(default_factory=list)
    all_models_wrong: list[str] = field(default_factory=list)
    too_easy: list[str] = field(default_factory=list)
    discriminating: list[str] = field(default_factory=list)
    problematic: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "all_models_wrong": list(self.all_models_wrong),
            "too_easy": list(self.too_easy),
            "discriminating": list(self.discriminating),
            "problematic": list(self.problematic),
            "summary": self.summary,
            "thresholds": self.thresholds,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Item Analysis",
            "",
            f"Items: **{self.item_count}**",
            f"All models wrong: **{len(self.all_models_wrong)}**",
            f"Too easy: **{len(self.too_easy)}**",
            f"Discriminating: **{len(self.discriminating)}**",
            f"Problematic: **{len(self.problematic)}**",
            "",
            "| Case | Mean score | Spread | Pass rate | Flags |",
            "|---|---:|---:|---:|---|",
        ]
        for item in self.items:
            mean = "" if item.mean_score is None else f"{item.mean_score:.3f}"
            rate = "" if item.pass_rate is None else f"{item.pass_rate:.1%}"
            lines.append(f"| {item.case_id} | {mean} | {item.score_spread:.3f} | {rate} | {', '.join(item.flags)} |")
        return "\n".join(lines) + "\n"


def analyze_items(
    records: Iterable[Any],
    *,
    cases: Iterable[Any] | Mapping[str, Any] | None = None,
    easy_threshold: float = 0.95,
    wrong_threshold: float = 0.5,
    discrimination_spread: float = 0.25,
    min_providers: int = 2,
) -> ItemAnalysisReport:
    """Analyze model outcomes grouped by case ID.

    ``cases`` is optional.  Supplying it enables validation-status and
    ground-truth checks that cannot be inferred from run records alone.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in records or []:
        item = _record_dict(raw)
        case_id = str(item.get("case_id") or item.get("id") or "")
        if not case_id:
            continue
        grouped.setdefault(case_id, []).append(item)
    case_map = _case_map(cases)
    output: list[ItemAnalysis] = []
    for case_id, group in grouped.items():
        case = case_map.get(case_id)
        provider_scores: dict[str, float | None] = {}
        provider_passes: dict[str, bool | None] = {}
        task_type = _field(case, "task_type", "benchmark_type", default=None)
        difficulty = _difficulty(_field(case, "difficulty", default=None))
        validation = _field(case, "validation_status", default=None)
        for item in group:
            provider = str(item.get("provider") or _field(item.get("model_output"), "provider", default="unknown"))
            evaluation = item.get("evaluation")
            if evaluation is None:
                evaluation = item.get("evaluation_result")
            if isinstance(evaluation, Mapping):
                score = evaluation.get("score")
                passed = evaluation.get("passed", evaluation.get("pass_fail", evaluation.get("pass")))
                if task_type is None:
                    task_type = evaluation.get("task_type")
            else:
                score = _field(evaluation, "score", default=item.get("score"))
                passed = _field(evaluation, "passed", "pass_fail", "pass", default=item.get("passed", item.get("pass")))
            if score is None:
                score = item.get("score")
            if passed is None:
                passed = item.get("passed", item.get("pass"))
            provider_scores[provider] = _float_or_none(score)
            provider_passes[provider] = _bool_or_none(passed)
            if item.get("error"):
                # Attach to a synthetic provider-specific note below.
                pass
        scores = [float(value) for value in provider_scores.values() if value is not None]
        passes = [value for value in provider_passes.values() if value is not None]
        mean = statistics.fmean(scores) if scores else None
        spread = (max(scores) - min(scores)) if scores else 0.0
        pass_rate = (sum(bool(value) for value in passes) / len(passes)) if passes else None
        flags: list[str] = []
        notes: list[str] = []
        errors = [item.get("error") for item in group if item.get("error")]
        invalid_validation = str(getattr(validation, "value", validation) or "").lower() in {"invalid", "warning"}
        missing_eval = not scores
        expected = _field(case, "ground_truth", "expected_output", "expected", default="__missing__")
        if expected == "__missing__" and case is not None:
            notes.append("missing ground truth")
        if len(provider_scores) >= 1 and all(
            (value is not None and value < wrong_threshold) or passed is False
            for value, passed in zip(provider_scores.values(), provider_passes.values())
        ):
            flags.append("all_models_wrong")
        if passes and len(provider_scores) >= 1 and all(value is True for value in provider_passes.values() if value is not None) and mean is not None and mean >= easy_threshold:
            flags.append("too_easy")
        if len(provider_scores) >= max(2, min_providers) and spread >= discrimination_spread and any(value is True for value in passes) and any(value is False for value in passes):
            flags.append("discriminating")
        if invalid_validation or missing_eval or errors or (case is not None and expected == "__missing__"):
            flags.append("problematic")
        if errors:
            notes.append(f"{len(errors)} provider error(s)")
        if invalid_validation:
            notes.append(f"validation status: {validation}")
        output.append(
            ItemAnalysis(
                case_id=case_id,
                provider_count=len(provider_scores),
                evaluated_count=len(scores),
                scores=provider_scores,
                passes=provider_passes,
                mean_score=round(mean, 6) if mean is not None else None,
                score_spread=round(spread, 6),
                pass_rate=round(pass_rate, 6) if pass_rate is not None else None,
                flags=flags,
                task_type=_enum_value(task_type),
                difficulty=difficulty,
                validation_status=_enum_value(validation) if validation is not None else None,
                notes=notes,
            )
        )
    output.sort(key=lambda item: item.case_id)
    report = ItemAnalysisReport(
        items=output,
        all_models_wrong=[item.case_id for item in output if item.all_models_wrong],
        too_easy=[item.case_id for item in output if item.too_easy],
        discriminating=[item.case_id for item in output if item.discriminating],
        problematic=[item.case_id for item in output if item.problematic],
        thresholds={
            "easy_threshold": float(easy_threshold),
            "wrong_threshold": float(wrong_threshold),
            "discrimination_spread": float(discrimination_spread),
            "min_providers": int(min_providers),
        },
    )
    report.summary = {
        "item_count": len(output),
        "provider_count": len({provider for item in output for provider in item.scores}),
        "all_models_wrong_count": len(report.all_models_wrong),
        "too_easy_count": len(report.too_easy),
        "discriminating_count": len(report.discriminating),
        "problematic_count": len(report.problematic),
        "mean_score": statistics.fmean([item.mean_score for item in output if item.mean_score is not None]) if any(item.mean_score is not None for item in output) else None,
    }
    return report


# Alias matching the wording used in the editor and README.
analyze_results = analyze_items
item_analysis = analyze_items


def _record_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if hasattr(raw, "to_dict"):
        try:
            value = raw.to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    result: dict[str, Any] = {}
    for name in ("case_id", "provider", "score", "passed", "error", "evaluation", "evaluation_result", "model_output"):
        value = getattr(raw, name, None)
        if value is not None:
            result[name] = value
    return result


def _case_map(cases: Any) -> dict[str, Any]:
    if cases is None:
        return {}
    if isinstance(cases, Mapping):
        if "cases" in cases and isinstance(cases["cases"], (list, tuple)):
            cases = cases["cases"]
        elif all(isinstance(key, str) for key in cases):
            return {str(key): value for key, value in cases.items()}
    result = {}
    for case in cases or []:
        case_id = _field(case, "case_id", "id", default=None)
        if case_id is not None:
            result[str(case_id)] = case
    return result


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        try:
            candidate = getattr(value, name)
        except AttributeError:
            continue
        if candidate is not None:
            return candidate
    return default


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _difficulty(value: Any) -> int | None:
    try:
        return int(getattr(value, "value", value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "pass", "passed", "1"}:
            return True
        if value.lower() in {"false", "no", "fail", "failed", "0"}:
            return False
    return bool(value)


__all__ = ["ItemAnalysis", "ItemAnalysisReport", "analyze_items", "analyze_results", "item_analysis"]
