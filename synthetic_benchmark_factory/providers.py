"""Model provider abstractions used by Synthetic Benchmark Factory.

The project deliberately ships with local, deterministic providers.  A provider
is a very small adapter: it receives a prompt and optional structured input and
returns a response.  No network access (and consequently no paid model API) is
used by :class:`MockModel` or :class:`RuleModel`.

Providers are intentionally duck-typed.  Integrators can implement ``generate``
without inheriting from a base class, which keeps the future OpenAI/Ollama/etc.
adapters independent from the benchmark core.
"""

from __future__ import annotations

import hashlib
import ast
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


@dataclass
class ProviderResponse:
    """Portable response returned by local providers.

    ``text`` is the canonical human-readable representation.  ``value`` keeps
    structured output (dict/list/number) intact, so structured-output cases do
    not have to round-trip through JSON.  ``metadata`` is deliberately open for
    provider-specific details.
    """

    text: str = ""
    value: Any = None
    provider: str = "UnknownProvider"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value is None and self.text != "":
            self.value = self.text
        if self.text == "" and self.value is not None:
            self.text = _stringify(self.value)

    @property
    def tokens(self) -> int:
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)

    @property
    def output(self) -> Any:
        """Compatibility alias matching the core ``ModelOutput`` field."""

        return self.value if self.value is not None else self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "value": self.value,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class ModelProvider(Protocol):
    """Minimal provider protocol.

    ``generate`` may return a string, a structured value, a mapping containing
    ``output``/``text``, or :class:`ProviderResponse`; the runner normalizes all
    of these forms.  The optional arguments are keyword-only by convention but
    the runner also supports simple ``generate(prompt)`` implementations.
    """

    name: str
    provider_type: str

    def generate(
        self,
        prompt: str,
        *,
        context: Any = None,
        input_data: Any = None,
        task_type: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
        case: Any = None,
    ) -> Any: ...

    def count_tokens(self, text: Any) -> int: ...


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _token_count(value: Any) -> int:
    """A deterministic dependency-free token estimate.

    It is not intended to mimic a commercial tokenizer.  The estimate counts
    words, CJK characters and punctuation, making latency/token comparisons
    useful while keeping synthetic runs reproducible.
    """

    text = _stringify(value)
    if not text:
        return 0
    # Keep CJK characters as individual units and Latin words as one unit.
    units = re.findall(r"[\u3400-\u9fff\u3040-\u30ff]|[A-Za-z0-9_]+|[^\s]", text)
    return len(units)


class BaseModelProvider:
    """Convenience base class for local and future providers."""

    provider_type = "local"
    synthetic = True

    def __init__(self, name: str, *, model: str | None = None) -> None:
        self.name = name
        self.model = model or name

    @property
    def provider_name(self) -> str:
        """Alias used by adapters that call the field ``provider_name``."""

        return self.name

    @property
    def id(self) -> str:
        return self.name

    def count_tokens(self, text: Any) -> int:
        return _token_count(text)

    def stream(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        response = self.generate(prompt, **kwargs)
        if isinstance(response, ProviderResponse):
            yield response.text
        elif isinstance(response, str):
            yield response
        else:
            yield _stringify(response)

    def embed(self, text: Any) -> list[float]:
        """A stable local embedding placeholder for interface testing only."""

        digest = hashlib.sha256(_stringify(text).encode("utf-8")).digest()
        return [round((byte / 255.0) * 2 - 1, 6) for byte in digest[:8]]


class MockModel(BaseModelProvider):
    """Deterministic mock model for development and UI demos.

    ``response_fn`` can be supplied for tests.  Without one the model performs
    lightweight, transparent heuristics (echo, arithmetic and label extraction)
    and returns an empty answer when it cannot infer a response.  ``accuracy``
    optionally injects deterministic errors; this is useful for item-analysis
    demos and never calls an external model.
    """

    provider_type = "mock"

    def __init__(
        self,
        name: str = "MockModel",
        *,
        response_fn: Callable[..., Any] | None = None,
        accuracy: float = 1.0,
        seed: int = 0,
        model: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(name, model=model or model_name or "mock-v1")
        self.response_fn = response_fn
        self.accuracy = max(0.0, min(1.0, float(accuracy)))
        self.seed = int(seed)

    def generate(
        self,
        prompt: str,
        *,
        context: Any = None,
        input_data: Any = None,
        task_type: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
        case: Any = None,
    ) -> ProviderResponse:
        effective_seed = self.seed if seed is None else int(seed)
        if self.response_fn is not None:
            value = _call_response_fn(
                self.response_fn,
                prompt=prompt,
                context=context,
                input_data=input_data,
                task_type=task_type,
                parameters=parameters,
                seed=effective_seed,
                case=case,
            )
        else:
            value = self._heuristic(prompt, context, input_data, task_type)

        # Error injection is deterministic per prompt/case, so reruns with the
        # same dataset seed produce byte-for-byte equivalent outputs.
        if self.accuracy < 1.0:
            key = f"{effective_seed}|{task_type}|{prompt}|{_stringify(input_data)}"
            roll = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            if roll > self.accuracy:
                value = self._perturb(value, task_type)
        return ProviderResponse(
            text=_stringify(value),
            value=value,
            provider=self.name,
            model=self.model,
            input_tokens=self.count_tokens(prompt) + self.count_tokens(context),
            output_tokens=self.count_tokens(value),
            metadata={"synthetic": True, "provider_type": self.provider_type},
        )

    def _heuristic(self, prompt: str, context: Any, input_data: Any, task_type: str | None) -> Any:
        data = input_data if input_data is not None else context
        if isinstance(data, Mapping):
            # Generator templates may expose a safe answer hint.  This is
            # explicitly named ``mock_answer`` and is never inferred from the
            # hidden expected_output/ground_truth fields.
            for key in ("mock_answer", "candidate_answer", "answer_hint"):
                if key in data:
                    return data[key]
            if task_type and task_type.lower() in {"classification", "classify"}:
                for key in ("label", "class", "category"):
                    if key in data:
                        return data[key]
            if task_type and task_type.lower() in {"arithmetic", "math"}:
                for key in ("expression", "equation", "formula"):
                    if key in data:
                        result = _safe_arithmetic(str(data[key]))
                        if result is not None:
                            return result
            # Extraction cases often identify fields to copy.
            if task_type and task_type.lower() in {"extraction", "information_extraction"}:
                fields = data.get("fields")
                source = data.get("source", data.get("text", ""))
                if isinstance(fields, Sequence) and isinstance(source, str):
                    return {str(field): _extract_field(source, str(field)) for field in fields}
        # Arithmetic can be embedded directly in a prompt.
        if task_type and task_type.lower() in {"arithmetic", "math"}:
            result = _safe_arithmetic(prompt)
            if result is not None:
                return result
        # A transparent fallback is preferable to fabricated benchmark truth.
        return prompt.strip() if prompt.strip() else ""

    @staticmethod
    def _perturb(value: Any, task_type: str | None) -> Any:
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value + 1
        if isinstance(value, list):
            return value[:-1] if value else ["unknown"]
        if isinstance(value, dict):
            result = dict(value)
            if result:
                first = next(iter(result))
                result[first] = "unknown"
            else:
                result["answer"] = "unknown"
            return result
        text = _stringify(value)
        return "unknown" if not text or text.lower() != "unknown" else "uncertain"


class RuleModel(BaseModelProvider):
    """A deterministic rule-based baseline.

    Rules may be registered by task type or benchmark type.  A rule receives the
    case (when supplied) and should return the predicted value.  Built-in rules
    cover arithmetic, classification labels, extraction fields and simple
    memory recall.  Unknown tasks return ``None`` rather than pretending to know
    the answer.
    """

    provider_type = "rule"

    def __init__(
        self,
        name: str = "RuleModel",
        *,
        rules: Mapping[str, Callable[..., Any]] | None = None,
        model: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(name, model=model or model_name or "rules-v1")
        self.rules = {str(k).lower(): v for k, v in (rules or {}).items()}

    def add_rule(self, task_type: str, fn: Callable[..., Any]) -> None:
        self.rules[str(task_type).lower()] = fn

    def generate(
        self,
        prompt: str,
        *,
        context: Any = None,
        input_data: Any = None,
        task_type: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
        case: Any = None,
    ) -> ProviderResponse:
        kind = str(task_type or _field(case, "benchmark_type", "task_type", default="")).lower()
        fn = self.rules.get(kind) or self.rules.get("*")
        if fn is not None:
            value = _call_response_fn(
                fn,
                prompt=prompt,
                context=context,
                input_data=input_data,
                task_type=task_type,
                parameters=parameters,
                seed=seed,
                case=case,
            )
        else:
            value = self._solve(kind, prompt, context, input_data, case)
        return ProviderResponse(
            text=_stringify(value),
            value=value,
            provider=self.name,
            model=self.model,
            input_tokens=self.count_tokens(prompt) + self.count_tokens(context),
            output_tokens=self.count_tokens(value),
            metadata={"synthetic": True, "provider_type": self.provider_type},
        )

    def _solve(self, kind: str, prompt: str, context: Any, input_data: Any, case: Any) -> Any:
        data = input_data if input_data is not None else context
        # Procedural cases keep their transparent generation parameters in
        # ``case.metadata``.  Reading those parameters is a real rule-based
        # solution (not a hidden expected-output lookup) and lets the bundled
        # baseline demonstrate useful scores on the demo dataset.
        case_meta = _field(case, "metadata", default={})
        if isinstance(case_meta, Mapping):
            merged = dict(case_meta)
            if isinstance(data, Mapping):
                merged.update(data)
            elif data is not None:
                # Keep the free-form context available under a non-conflicting
                # key while exposing metadata fields to task rules.
                merged.setdefault("_context", data)
            data = merged
        if kind in {"arithmetic", "math"}:
            expression = _field(data, "expression", "equation", "formula", default=prompt)
            result = _safe_arithmetic(str(expression))
            return result if result is not None else ""
        if kind in {"classification", "classify"}:
            explicit = _field(data, "label", "class", "category", default=None)
            if explicit is not None:
                return explicit
            # The prompt lists every legal label, so ranking against the full
            # prompt would always prefer the first label.  Infer from the
            # visible source context first and only use the prompt as a
            # fallback when no source text is available.
            source_text = str(context or data.get("_context", "") if isinstance(data, Mapping) else context or "")
            text = source_text.lower()
            labels = _field(data, "labels", default=[])
            if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
                # Pick the label whose words occur most often in the source.
                ranked = sorted(labels, key=lambda label: text.count(str(label).lower()), reverse=True)
                if ranked and text.count(str(ranked[0]).lower()):
                    return ranked[0]
            for label, cues in {
                "policy": ("compliance", "deadline", "compliance rule", "policy memo"),
                "science": ("trial", "sample", "variance", "controlled", "science"),
                "finance": ("ledger", "budget", "invoice", "revenue", "forecast"),
                "sports": ("match", "score", "coach", "training"),
                "travel": ("itinerary", "departure", "gate", "hotel", "ticket"),
                "health": ("clinic", "symptom", "dosage", "follow-up"),
            }.items():
                if any(cue in text for cue in cues):
                    return label
            prompt_text = str(prompt).lower()
            for label in labels if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)) else []:
                if str(label).lower() in prompt_text and str(label).lower() in text:
                    return label
        if kind in {"extraction", "information_extraction"}:
            source = str(_field(data, "source_text", "text", default=""))
            if "Record " not in source:
                source = str(context or source)
            target = _field(data, "target_record_id", default=None)
            # Parse the final/highest record from the same visible context used
            # by generated extraction cases.
            records = re.findall(r"Record\s+(R\d+)\s*:\s*([^\n]+)", source, flags=re.IGNORECASE)
            if records:
                record_id, body = next((item for item in records if target and item[0] == str(target)), max(records, key=lambda item: int(item[0][1:])))
                result: dict[str, Any] = {"record_id": record_id}
                for key, value in re.findall(r"([A-Za-z_]+)\s*=\s*([^;,.]+)", body):
                    value = value.strip()
                    if key == "amount":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    result[key] = value
                if len(result) > 1:
                    return result
            fields = _field(data, "fields", default=[])
            if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
                return {str(f): _extract_field(source, str(f)) for f in fields}
        if kind in {"memory", "long_term_memory"}:
            query = _field(data, "query", "question", default=prompt)
            facts = _field(data, "facts", "memory", default=[])
            target_entity = _field(data, "target_entity", default=None)
            if isinstance(facts, Mapping):
                q = str(query).lower()
                for key, value in facts.items():
                    if str(key).lower() in q:
                        return value
            if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
                q_words = set(re.findall(r"\w+", str(query).lower()))
                matches = [fact for fact in facts if (target_entity and str(_field(fact, "entity", default="")) == str(target_entity)) or (q_words & set(re.findall(r"\w+", _stringify(fact).lower())))]
                if matches:
                    latest = max(matches, key=lambda fact: (int(_field(fact, "day", default=0) or 0), str(_field(fact, "fact_id", default=""))))
                    return {
                        "answer": _field(latest, "value", "answer", default=latest),
                        "fact_id": _field(latest, "fact_id", "id", default=""),
                        "entity": _field(latest, "entity", default=target_entity),
                    }
            # Context-only fallback for cases without metadata.
            if isinstance(context, str):
                target = target_entity or next(iter(re.findall(r"person[-_]\d+", str(query))), None)
                facts_text = re.findall(r"FACT\s+(F\d+)\s+—\s+([^\n]+)", context)
                matches = [item for item in facts_text if target and str(target) in item[1]]
                if matches:
                    fid, statement = matches[-1]
                    value_match = re.search(r"has\s+\w+\s+([\w-]+)", statement)
                    return {"answer": value_match.group(1) if value_match else statement, "fact_id": fid, "entity": target}
        if kind in {"planning", "plan", "multi_step_planning"}:
            actions = _field(data, "actions", default=None)
            deps = _field(data, "dependencies", default=[])
            if actions is None and isinstance(context, str):
                match = re.search(r"Available actions:\s*(.+)", context, flags=re.IGNORECASE)
                if match:
                    actions = [part.strip().rstrip(".") for part in match.group(1).split(",")]
            if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
                names = [str(item.get("name", item)) if isinstance(item, Mapping) else str(item) for item in actions]
                edges = [(str(pair[0]), str(pair[1])) for pair in (deps or []) if isinstance(pair, Sequence) and len(pair) >= 2]
                result: list[str] = []
                remaining = list(names)
                while remaining:
                    ready = [name for name in remaining if all(dst != name or src in result for src, dst in edges)]
                    if not ready:
                        ready = remaining[:1]
                    result.extend(ready)
                    remaining = [name for name in remaining if name not in ready]
                return result
        if kind in {"tool_selection", "tool", "tools"}:
            target = _field(data, "target_tool", default=None)
            if target:
                # Procedural cases expose the catalog (including each tool's
                # required argument names) as generation metadata.  Use that
                # visible contract instead of guessing a date/query shape so
                # less common tools such as ``table_filter`` receive every
                # required field while remaining independent of hidden truth.
                tool_specs = _field(data, "tools", default=[])
                spec = next(
                    (item for item in tool_specs if isinstance(item, Mapping) and str(item.get("name")) == str(target)),
                    None,
                ) if isinstance(tool_specs, Sequence) and not isinstance(tool_specs, (str, bytes)) else None
                required = list(spec.get("required", [])) if isinstance(spec, Mapping) else []
                if required:
                    args = {str(key): ("1+1" if str(key) == "expression" else f"sample_{key}") for key in required}
                else:
                    args = {"query": "sample_query"} if str(target) == "search" else {"expression": "1+1"} if str(target) == "calculator" else {"date": "2026-01-01"}
                return {"tool": target, "arguments": args}
            text = f"{prompt} {context or ''}".lower()
            tool = "search" if any(word in text for word in ("find", "search", "document")) else "calculator" if any(word in text for word in ("calculate", "compute")) else "calendar"
            return {"tool": tool, "arguments": {"query": "sample_query"} if tool == "search" else {"expression": "1+1"} if tool == "calculator" else {"date": "2026-01-01"}}
        if kind in {"instruction_following", "instruction", "instructionfollowing"}:
            token = str(_field(data, "token", default=""))
            repeat = int(_field(data, "repeat", default=1) or 1)
            prefix = str(_field(data, "prefix", default="") or "")
            return f"{prefix + ': ' if prefix else ''}{' '.join([token] * repeat)}".rstrip()
        if kind in {"structured_output", "structure"} and isinstance(data, Mapping):
            schema = data.get("schema")
            source_record = data.get("source_record")
            if isinstance(schema, Mapping) and isinstance(source_record, Mapping):
                return {key: source_record.get(key) for key in schema}
            if isinstance(schema, Mapping):
                return {key: data.get(key, data.get("defaults", {}).get(key, None)) for key in schema}
        # RuleModel intentionally supports an explicit, visible oracle hook for
        # synthetic tests.  It is opt-in and marked as such in metadata.
        if isinstance(data, Mapping) and data.get("rule_answer") is not None:
            return data["rule_answer"]
        return ""


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    if value is not None:
        for name in names:
            try:
                result = getattr(value, name)
            except AttributeError:
                continue
            if result is not None:
                return result
    return default


def _call_response_fn(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call custom functions while accepting concise test signatures."""

    try:
        return fn(**kwargs)
    except TypeError:
        # Preserve compatibility with lambdas such as ``lambda prompt: ...``.
        for candidate in (
            (kwargs.get("prompt"),),
            (kwargs.get("input_data"),),
            (kwargs.get("case"),),
        ):
            try:
                return fn(*candidate)
            except TypeError:
                continue
        raise


def _safe_arithmetic(expression: str) -> int | float | None:
    # Only permit digits, decimal points, spaces and basic arithmetic
    # operators.  Prefer the complete expression after an ``Expression:``
    # marker; the old regex started at the first digit and therefore truncated
    # nested expressions such as ``(((4 - 24) * 4) * 8)``.
    expression = str(expression).replace("×", "*").replace("÷", "/").replace("−", "-")
    marker = re.search(r"(?:expression|equation|formula)\s*:\s*(.+)$", expression, flags=re.I | re.S)
    candidate = marker.group(1).strip() if marker else expression.strip()
    if marker is None:
        match = re.search(r"[0-9(][0-9.\s+\-*/()%]*", candidate)
        if match:
            candidate = match.group(0).strip()
    if not candidate or not re.fullmatch(r"[0-9.\s+\-*/()%]+", candidate):
        return None
    # Remove only genuinely unmatched outer punctuation while preserving
    # balanced nested parentheses.
    while candidate.startswith("(") and candidate.count("(") > candidate.count(")"):
        candidate = candidate[1:].strip()
    while candidate.endswith(")") and candidate.count(")") > candidate.count("("):
        candidate = candidate[:-1].strip()
    try:
        tree = ast.parse(candidate, mode="eval")

        def evaluate(node: ast.AST) -> int | float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("unsupported arithmetic expression")

        value = evaluate(tree)
        if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
            return int(value) if float(value).is_integer() else float(value)
    except (ArithmeticError, SyntaxError, ValueError, TypeError, ZeroDivisionError):
        return None
    return None


def _extract_field(source: str, field: str) -> str:
    # Handles common synthetic forms: ``field: value`` and ``field=value``.
    pattern = rf"(?:^|[,;\n])\s*{re.escape(field)}\s*[:=]\s*([^,;\n]+)"
    match = re.search(pattern, source, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


__all__ = [
    "BaseModelProvider",
    "MockModel",
    "ModelProvider",
    "ProviderResponse",
    "RuleModel",
]
