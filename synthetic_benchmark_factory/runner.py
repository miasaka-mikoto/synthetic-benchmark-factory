"""Execution engine for benchmark cases.

``BenchmarkRunner`` is deliberately provider-agnostic and local-first.  It can
run one case or an entire ``BenchmarkDataset`` against one or more providers,
capture timing/token metadata, evaluate outputs, and attach results back to the
case for browsing/export.  The default providers are deterministic
``MockModel`` and ``RuleModel``; no paid API is contacted automatically.
"""

from __future__ import annotations

import inspect
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .metrics import EvaluationResult, evaluate_case
from .providers import MockModel, ModelProvider, ProviderResponse, RuleModel

try:  # core models are available in the main package, but keep imports lazy-safe
    from .models import BenchmarkCase, BenchmarkDataset, ModelOutput
except Exception:  # pragma: no cover - allows isolated metrics/provider use
    BenchmarkCase = Any  # type: ignore
    BenchmarkDataset = Any  # type: ignore
    ModelOutput = Any  # type: ignore


@dataclass
class RunRecord:
    """One provider prediction and its evaluation."""

    case_id: str
    provider: str
    prediction: Any = None
    evaluation: EvaluationResult | None = None
    model_output: Any = None
    latency_ms: float = 0.0
    tokens: int | dict[str, int] = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    task_type: str | None = None
    split: str | None = None
    difficulty: int | None = None
    synthetic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float | None:
        return self.evaluation.score if self.evaluation else None

    @property
    def latency(self) -> float:
        return self.latency_ms

    @property
    def token_count(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def passed(self) -> bool | None:
        return self.evaluation.passed if self.evaluation else None

    def to_dict(self) -> dict[str, Any]:
        evaluation = self.evaluation.to_dict() if self.evaluation else None
        model_output = self.model_output.to_dict() if hasattr(self.model_output, "to_dict") else self.model_output
        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "prediction": self.prediction,
            "model_output": model_output,
            "evaluation": evaluation,
            "score": self.score,
            "passed": self.passed,
            "latency_ms": round(float(self.latency_ms), 4),
            "tokens": self.tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "task_type": self.task_type,
            "split": self.split,
            "difficulty": self.difficulty,
            "synthetic": self.synthetic,
            "metadata": self.metadata,
        }


@dataclass
class RunReport:
    """Collection-level output from :meth:`BenchmarkRunner.run_dataset`."""

    records: list[RunRecord] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def results(self) -> list[RunRecord]:
        return self.records

    @property
    def case_count(self) -> int:
        return len({record.case_id for record in self.records})

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "providers": list(self.providers),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_markdown(self) -> str:
        from .reports import report_to_markdown, build_report

        return build_report(run_report=self).to_markdown()


class BenchmarkRunner:
    """Run benchmark cases against local or user-supplied providers."""

    def __init__(
        self,
        providers: Sequence[ModelProvider | str] | ModelProvider | str | None = None,
        *,
        provider: ModelProvider | str | None = None,
        model_providers: Sequence[ModelProvider | str] | None = None,
        default_threshold: float | None = None,
        continue_on_error: bool = True,
    ) -> None:
        if providers is None:
            providers = model_providers if model_providers is not None else provider
        if providers is None:
            providers = [MockModel(), RuleModel()]
        if isinstance(providers, (str, bytes)) or hasattr(providers, "generate"):
            providers = [providers]  # type: ignore[list-item]
        self.providers = [self._resolve_provider(provider) for provider in providers]  # type: ignore[arg-type]
        self.default_threshold = default_threshold
        self.continue_on_error = bool(continue_on_error)

    @staticmethod
    def _resolve_provider(provider: ModelProvider | str) -> ModelProvider:
        if not isinstance(provider, str):
            return provider
        key = provider.strip().lower().replace("_", "-")
        if key in {"mock", "mockmodel", "mock-model"}:
            return MockModel()
        if key in {"rule", "rulemodel", "rule-model"}:
            return RuleModel()
        raise ValueError(f"Unknown provider {provider!r}; pass a provider object or mock/rule")

    def run_case(
        self,
        case: Any,
        provider: ModelProvider | str | None = None,
        *,
        provider_name: ModelProvider | str | None = None,
        evaluate: bool = True,
        threshold: float | None = None,
        attach: bool = True,
        run_id: str | None = None,
    ) -> RunRecord:
        """Run one case and return a :class:`RunRecord`.

        A provider can be omitted when the runner contains exactly one provider;
        with multiple providers, omission uses the first provider (convenient
        for interactive case browsing).
        """

        if provider is None:
            provider = provider_name
        model = self._resolve_provider(provider) if provider is not None else self.providers[0]
        case_id = str(_field(case, "case_id", "id", default=""))
        if not case_id:
            case_id = f"case-{uuid.uuid4().hex[:12]}"
        prompt = str(_field(case, "prompt", default=""))
        context = _field(case, "context", default=None)
        input_data = _field(case, "input_data", "input", default=None)
        task_type = _enum_value(_field(case, "task_type", "benchmark_type", "type", default=None))
        parameters = _field(case, "parameters", "config", default=None)
        seed = _field(case, "seed", default=None)
        provider_name = str(getattr(model, "name", getattr(model, "provider", model.__class__.__name__)))
        synthetic = bool(getattr(model, "synthetic", True))
        started = time.perf_counter()
        response: Any = None
        error: str | None = None
        try:
            response = _invoke_provider(
                model,
                prompt,
                context=context,
                input_data=input_data,
                task_type=task_type,
                parameters=parameters,
                seed=seed,
                case=case,
            )
            prediction, response_meta = _normalize_response(response)
        except Exception as exc:  # individual failures should be inspectable
            prediction = None
            response_meta = {}
            error = f"{type(exc).__name__}: {exc}"
            if not self.continue_on_error:
                raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        input_tokens, output_tokens = _token_counts(model, prompt, context, prediction, response_meta)
        tokens: int | dict[str, int] = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }
        evaluation: EvaluationResult | None = None
        if evaluate and error is None:
            evaluation = evaluate_case(
                case,
                prediction,
                provider=provider_name,
                threshold=threshold if threshold is not None else self.default_threshold,
            )
        elif evaluate:
            expected_on_error = _field(case, "ground_truth", default=None)
            if expected_on_error is None:
                expected_on_error = _field(case, "expected_output", "expected", default=None)
            evaluation = EvaluationResult(
                case_id=case_id,
                provider=provider_name,
                prediction=prediction,
                expected=expected_on_error,
                task_type=task_type,
                errors=[error or "provider_error"],
                metadata={"threshold": threshold if threshold is not None else self.default_threshold},
            )
        if evaluation is not None:
            # Operational metrics are kept alongside quality metrics so report
            # consumers can plot accuracy/latency/token trade-offs without
            # joining a second table.
            evaluation.metrics["latency_ms"] = round(float(latency_ms), 6)
            evaluation.metrics["tokens"] = float(input_tokens + output_tokens)
            evaluation.metadata.update({"latency_ms": latency_ms, "tokens": tokens})
        model_output = _make_model_output(
            provider_name,
            prediction,
            latency_ms,
            tokens,
            error,
            run_id=run_id,
            metadata={
                **response_meta,
                "synthetic": synthetic,
                "provider_type": getattr(model, "provider_type", "custom"),
            },
        )
        record = RunRecord(
            case_id=case_id,
            provider=provider_name,
            prediction=prediction,
            evaluation=evaluation,
            model_output=model_output,
            latency_ms=latency_ms,
            tokens=tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            task_type=task_type,
            split=_enum_value(_field(case, "split", default=None)),
            difficulty=_difficulty(_field(case, "difficulty", default=None)),
            synthetic=synthetic,
            metadata={
                "provider": provider_name,
                "provider_type": getattr(model, "provider_type", "custom"),
                "synthetic": synthetic,
                **response_meta,
            },
        )
        if attach:
            _attach_result(case, model_output, evaluation)
        return record

    def run_dataset(
        self,
        dataset: Any,
        providers: Sequence[ModelProvider | str] | ModelProvider | str | None = None,
        *,
        provider: ModelProvider | str | None = None,
        splits: Sequence[str] | str | None = None,
        case_ids: Sequence[str] | None = None,
        max_cases: int | None = None,
        evaluate: bool = True,
        threshold: float | None = None,
        attach: bool = True,
        continue_on_error: bool | None = None,
    ) -> RunReport:
        """Run all selected cases against each selected provider."""

        if providers is None and provider is not None:
            providers = provider
        selected_providers = self._coerce_provider_list(providers)
        cases = list(_dataset_cases(dataset))
        allowed_splits = None
        if splits is not None:
            allowed_splits = {str(s).lower() for s in ([splits] if isinstance(splits, str) else splits)}
        allowed_ids = {str(item) for item in case_ids} if case_ids is not None else None
        if allowed_splits is not None:
            cases = [case for case in cases if _enum_value(_field(case, "split", default="")) .lower() in allowed_splits]
        if allowed_ids is not None:
            cases = [case for case in cases if str(_field(case, "case_id", "id", default="")) in allowed_ids]
        if max_cases is not None:
            cases = cases[: max(0, int(max_cases))]

        original_continue = self.continue_on_error
        if continue_on_error is not None:
            self.continue_on_error = bool(continue_on_error)
        started_at = datetime.now(timezone.utc)
        records: list[RunRecord] = []
        try:
            for case in cases:
                for model in selected_providers:
                    records.append(self.run_case(case, model, evaluate=evaluate, threshold=threshold, attach=attach))
        finally:
            self.continue_on_error = original_continue
        finished_at = datetime.now(timezone.utc)
        report = RunReport(
            records=records,
            providers=[str(getattr(p, "name", p.__class__.__name__)) for p in selected_providers],
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "synthetic": all(bool(getattr(p, "synthetic", True)) for p in selected_providers),
                "case_count": len(cases),
                "record_count": len(records),
                "dataset_name": _field(dataset, "name", default=None),
            },
        )
        report.summary = summarize_records(records)
        return report

    # Friendly aliases used by CLI and early prototypes.
    def run(self, dataset: Any, **kwargs: Any) -> RunReport:
        return self.run_dataset(dataset, **kwargs)

    def execute(self, dataset: Any, **kwargs: Any) -> RunReport:
        return self.run_dataset(dataset, **kwargs)

    def _coerce_provider_list(self, providers: Any) -> list[ModelProvider]:
        if providers is None:
            return list(self.providers)
        if isinstance(providers, (str, bytes)) or hasattr(providers, "generate"):
            providers = [providers]
        return [self._resolve_provider(item) for item in providers]


def summarize_records(records: Sequence[RunRecord]) -> dict[str, Any]:
    """Aggregate metrics suitable for the report header/dashboard."""

    records = list(records or [])
    evaluated = [record for record in records if record.evaluation is not None]
    scores = [float(record.evaluation.score) for record in evaluated if record.evaluation and record.evaluation.score is not None]
    passed = [record.evaluation.passed for record in evaluated if record.evaluation and record.evaluation.passed is not None]
    latencies = [float(record.latency_ms) for record in records]
    totals = [record.input_tokens + record.output_tokens for record in records]
    by_provider: dict[str, dict[str, Any]] = {}
    for provider in sorted({record.provider for record in records}):
        subset = [record for record in records if record.provider == provider]
        sub_eval = [record for record in subset if record.evaluation is not None]
        sub_scores = [float(record.evaluation.score) for record in sub_eval if record.evaluation and record.evaluation.score is not None]
        sub_pass = [record.evaluation.passed for record in sub_eval if record.evaluation and record.evaluation.passed is not None]
        by_provider[provider] = {
            "case_count": len(subset),
            "mean_score": statistics.fmean(sub_scores) if sub_scores else None,
            "pass_rate": sum(bool(item) for item in sub_pass) / len(sub_pass) if sub_pass else None,
            "mean_latency_ms": statistics.fmean(record.latency_ms for record in subset) if subset else 0.0,
            "total_tokens": sum(record.input_tokens + record.output_tokens for record in subset),
        }
    return {
        "record_count": len(records),
        "case_count": len({record.case_id for record in records}),
        "evaluated_count": len(evaluated),
        "mean_score": statistics.fmean(scores) if scores else None,
        "pass_rate": sum(bool(item) for item in passed) / len(passed) if passed else None,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "total_tokens": sum(totals),
        "mean_tokens": statistics.fmean(totals) if totals else 0.0,
        "errors": sum(bool(record.error) for record in records),
        "by_provider": by_provider,
    }


def _invoke_provider(provider: Any, prompt: str, **kwargs: Any) -> Any:
    fn = getattr(provider, "generate")
    # First try the complete protocol.  For simple user adapters, progressively
    # fall back to smaller signatures without masking errors from the final call.
    try:
        return fn(prompt, **kwargs)
    except TypeError as first_error:
        candidates = [
            (prompt, kwargs.get("context")),
            (prompt,),
        ]
        for args in candidates:
            try:
                return fn(*args)
            except TypeError:
                continue
        raise first_error


def _normalize_response(response: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(response, ProviderResponse):
        value = response.value if response.value is not None else response.text
        return value, {
            "response_text": response.text,
            "response_model": response.model,
            "response_provider": response.provider,
            "response_input_tokens": response.input_tokens,
            "response_output_tokens": response.output_tokens,
            **dict(response.metadata or {}),
        }
    if isinstance(response, Mapping):
        # Core ModelOutput and common adapters use these names.
        for key in ("output", "prediction", "value", "answer", "text", "content"):
            if key in response:
                return response[key], dict(response.get("metadata") or {})
        return dict(response), dict(response.get("metadata") or {})
    if hasattr(response, "output"):
        return getattr(response, "output"), dict(getattr(response, "metadata", {}) or {})
    if hasattr(response, "value"):
        return getattr(response, "value"), dict(getattr(response, "metadata", {}) or {})
    return response, {}


def _token_counts(provider: Any, prompt: Any, context: Any, prediction: Any, metadata: Mapping[str, Any]) -> tuple[int, int]:
    response_in = metadata.get("response_input_tokens")
    response_out = metadata.get("response_output_tokens")
    if response_in is not None and response_out is not None:
        return int(response_in), int(response_out)
    counter = getattr(provider, "count_tokens", None)
    if callable(counter):
        try:
            input_count = int(counter(prompt)) + int(counter(context))
            output_count = int(counter(prediction))
            return input_count, output_count
        except Exception:
            pass
    return _rough_tokens(prompt) + _rough_tokens(context), _rough_tokens(prediction)


def _rough_tokens(value: Any) -> int:
    return len(str(value or "").split())


def _make_model_output(provider: str, prediction: Any, latency_ms: float, tokens: Any, error: str | None, *, run_id: str | None, metadata: dict[str, Any]) -> Any:
    try:
        return ModelOutput(
            provider=provider,
            output=prediction,
            latency_ms=latency_ms,
            tokens=tokens,
            error=error,
            run_id=run_id,
            metadata=metadata,
        )
    except Exception:
        return {
            "provider": provider,
            "output": prediction,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "error": error,
            "run_id": run_id,
            "metadata": metadata,
        }


def _attach_result(case: Any, model_output: Any, evaluation: EvaluationResult | None) -> None:
    if isinstance(case, Mapping):
        # A mapping cannot reliably be mutated into a dataclass, but retaining
        # outputs is still useful for callers that pass a mutable dict.
        if isinstance(case, dict):
            case.setdefault("model_outputs", []).append(model_output.to_dict() if hasattr(model_output, "to_dict") else model_output)
            if evaluation is not None:
                case.setdefault("evaluation_results", []).append(_core_evaluation_dict(evaluation))
        return
    try:
        if hasattr(case, "model_outputs"):
            case.model_outputs.append(model_output)
        if evaluation is not None and hasattr(case, "evaluation_results"):
            case.evaluation_results.append(_core_evaluation(evaluation, model_output=model_output))
    except Exception:
        # Attachment is convenience only; a frozen/custom case should still run.
        return


def _core_evaluation(evaluation: EvaluationResult, *, model_output: Any = None) -> Any:
    try:
        from .models import EvaluationResult as CoreEvaluationResult

        latency = _field(model_output, "latency_ms", default=None)
        tokens = _field(model_output, "tokens", default=None)
        metadata = dict(evaluation.metadata or {})
        if isinstance(_field(model_output, "metadata", default=None), Mapping):
            metadata.update(_field(model_output, "metadata", default={}) or {})
        return CoreEvaluationResult(
            case_id=evaluation.case_id or "",
            provider=evaluation.provider or "",
            metrics=evaluation.metrics,
            score=evaluation.score,
            passed=evaluation.passed,
            latency_ms=latency,
            tokens=tokens,
            details={
                "prediction": evaluation.prediction,
                "expected": evaluation.expected,
                "errors": evaluation.errors,
                "task_type": evaluation.task_type,
                **metadata,
            },
            prediction=evaluation.prediction,
            expected=evaluation.expected,
            errors=evaluation.errors,
            task_type=evaluation.task_type,
            metadata=metadata,
        )
    except Exception:
        return _core_evaluation_dict(evaluation)


def _core_evaluation_dict(evaluation: EvaluationResult) -> dict[str, Any]:
    return evaluation.to_dict()


def _dataset_cases(dataset: Any) -> Iterable[Any]:
    if isinstance(dataset, Mapping):
        return dataset.get("cases", [])
    cases = getattr(dataset, "cases", None)
    if cases is not None:
        return cases
    return dataset if dataset is not None else []


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


def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _difficulty(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(getattr(value, "value", value))
    except (TypeError, ValueError):
        return None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


Runner = BenchmarkRunner
ModelRunner = BenchmarkRunner

__all__ = ["BenchmarkRunner", "Runner", "ModelRunner", "RunRecord", "RunReport", "summarize_records"]
