"""Synthetic Benchmark Factory desktop application.

This module intentionally keeps the user interface independent from the
benchmark engine.  The engine is imported lazily from
``synthetic_benchmark_factory`` when it is available; a deterministic local
fallback is provided so that the editor and demo remain useful while the
project is being developed or when the application is copied to another
machine.

Run with::

    python app.py

The application never calls a paid model.  The Runner tab only exposes the
local MockModel and RuleModel providers during the first release.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import os
import queue
import random
import re
import statistics
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_NAME = "Synthetic Benchmark Factory"
APP_VERSION = "0.1.0"
TASK_TYPES = [
    "Extraction",
    "Classification",
    "Memory",
    "Planning",
    "Tool Selection",
    "Arithmetic",
    "Structured Output",
    "Instruction Following",
]
DIFFICULTIES = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
SPLITS = ["Train", "Validation", "Test", "Challenge"]
DISTRACTORS = [
    "Irrelevant Info",
    "Contradictory Info",
    "Near Match",
    "Long Context",
    "Missing Data",
    "Tool Failure",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _safe_text(value: Any) -> str:
    """Return a compact human-readable representation for a widget."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _jsonable(value: Any) -> Any:
    """Convert engine objects/enums into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "_raw"}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "value") and not callable(getattr(value, "value")):
        try:
            return _jsonable(value.value)
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): _jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            result = value.to_dict()
            if isinstance(result, Mapping):
                return dict(result)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _enum_string(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        try:
            return str(value.value)
        except Exception:
            pass
    return str(value)


def _get_field(obj: Any, *names: str, default: Any = "") -> Any:
    d = _object_dict(obj)
    for name in names:
        if name in d:
            return d[name]
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                continue
    return default


def _import_optional(*module_names: str) -> Any | None:
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _call_flexible(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a changing engine API without making the GUI brittle.

    Newer engine versions may add options, while an early development build
    may only accept ``generate(definition, count)``.  We first filter keyword
    arguments using the signature and then try the common positional forms.
    """

    try:
        signature = inspect.signature(fn)
        parameters = signature.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        filtered = kwargs if accepts_kwargs else {
            k: v for k, v in kwargs.items() if k in parameters
        }
    except Exception:
        filtered = kwargs
    try:
        return fn(*args, **filtered)
    except TypeError as first_error:
        # A small number of implementations use keyword-only ``definition``.
        attempts = [
            lambda: fn(**filtered),
            lambda: fn(*args[:1]),
            lambda: fn(),
        ]
        for attempt in attempts:
            try:
                return attempt()
            except TypeError:
                continue
        raise first_error


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _parse_json_or_text(value: str, fallback: Any = "") -> Any:
    text = value.strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Canonical records used by the UI
# ---------------------------------------------------------------------------


@dataclass
class DefinitionDraft:
    name: str = "Long-Term Memory Benchmark"
    capability: str = "Long-term memory"
    description: str = "Measure factual recall over time in the presence of distractors."
    difficulty: str = "Level 2"
    task_type: str = "Memory"
    input_schema: Any = field(default_factory=lambda: {"facts": "array", "question": "string"})
    expected_output: Any = field(default_factory=lambda: {"answer": "string"})
    scoring: Any = field(default_factory=lambda: {"type": "Exact Match"})
    constraints: Any = field(default_factory=lambda: {"ground_truth_required": True})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capability": self.capability,
            "description": self.description,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "input_schema": _jsonable(self.input_schema),
            "expected_output": _jsonable(self.expected_output),
            "scoring": _jsonable(self.scoring),
            "constraints": _jsonable(self.constraints),
        }


@dataclass
class CanonicalCase:
    """A stable view over engine BenchmarkCase objects.

    ``raw`` is deliberately excluded from serialization.  Keeping it here
    allows the Runner to pass native objects to the engine without coupling
    the widgets to a particular dataclass implementation.
    """

    id: str
    prompt: str = ""
    context: Any = ""
    input_data: Any = None
    expected_answer: Any = ""
    definition_id: str = ""
    difficulty_parameters: Any = field(default_factory=dict)
    distractors: list[Any] = field(default_factory=list)
    scoring: Any = field(default_factory=dict)
    synthetic: bool = True
    difficulty: str = ""
    task_type: str = ""
    split: str = "Test"
    generator: str = "Synthetic"
    seed: int | str = ""
    validation_status: str = "Pending"
    validation_errors: list[str] = field(default_factory=list)
    model_outputs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = field(default=None, repr=False, compare=False)

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        result = {
            "id": self.id,
            "prompt": self.prompt,
            "context": self.context,
            "input_data": _jsonable(self.input_data),
            "expected_answer": _jsonable(self.expected_answer),
            "definition_id": self.definition_id,
            "difficulty_parameters": _jsonable(self.difficulty_parameters),
            "distractors": _jsonable(self.distractors),
            "scoring": _jsonable(self.scoring),
            "synthetic": bool(self.synthetic),
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "split": self.split,
            "generator": self.generator,
            "seed": self.seed,
            "validation_status": self.validation_status,
            "validation_errors": list(self.validation_errors),
            "model_outputs": _jsonable(self.model_outputs),
            "metadata": _jsonable(self.metadata),
        }
        if include_raw:
            result["_raw"] = self.raw
        return result


def _canonical_case(raw: Any, index: int = 0) -> CanonicalCase:
    d = _object_dict(raw)
    case_id = _get_field(raw, "id", "case_id", "uid", default=f"case-{index + 1:04d}")
    prompt = _get_field(raw, "prompt", "question", "instruction", "input", default="")
    context = _get_field(raw, "context", "background", "passage", default="")
    input_data = _get_field(raw, "input_data", "input", default=None)
    expected = _get_field(
        raw,
        "expected_answer",
        "expected_output",
        "ground_truth",
        "answer",
        default="",
    )
    definition_id = str(_get_field(raw, "definition_id", default="") or "")
    difficulty_parameters = _get_field(raw, "difficulty_params", "difficulty_parameters", default={})
    distractors = _get_field(raw, "distractors", default=[])
    if not isinstance(distractors, (list, tuple)):
        distractors = [distractors] if distractors else []
    scoring = _get_field(raw, "scoring", default={})
    synthetic = bool(_get_field(raw, "synthetic", default=True))
    task_type = _enum_string(_get_field(raw, "task_type", "type", "benchmark_type", default=""))
    # Keep the canonical values compact and human-readable in the tree while
    # accepting the engine's enum values (``memory``, ``train`` and ``3``).
    task_type = task_type.replace("_", " ").title() if task_type else ""
    difficulty_raw = _get_field(raw, "difficulty", "difficulty_level", default="")
    difficulty = _enum_string(difficulty_raw)
    if difficulty.isdigit():
        difficulty = f"Level {difficulty}"
    elif difficulty and not difficulty.lower().startswith("level"):
        match = re.search(r"[1-5]", difficulty)
        difficulty = f"Level {match.group(0)}" if match else difficulty
    split = _enum_string(_get_field(raw, "split", "dataset_split", default="Test"))
    split = split.replace("_", " ").title() if split else "Test"
    generator_raw = _get_field(raw, "generator", "generator_name", default="Synthetic")
    if isinstance(generator_raw, Mapping):
        generator = generator_raw.get("procedural_generator") or generator_raw.get("template") or "Synthetic"
    else:
        generator = _get_field(generator_raw, "procedural_generator", "template", default=generator_raw)
    seed = _get_field(raw, "seed", default="")
    if seed in (None, ""):
        generator_obj = _get_field(raw, "generator", default=None)
        seed = _get_field(generator_obj, "seed", default="") if generator_obj is not None else ""
    validation_status = _enum_string(
        _get_field(raw, "validation_status", "status", default="Pending")
    )
    validation_status = validation_status.replace("_", " ").title() if validation_status else "Pending"
    if validation_status.lower() in {"unvalidated", "un-validated", "not validated"}:
        validation_status = "Pending"
    elif validation_status.lower() == "pass":
        validation_status = "Valid"
    errors = _get_field(raw, "validation_errors", "errors", default=[])
    if isinstance(errors, str):
        errors = [errors] if errors else []
    outputs = _get_field(raw, "model_outputs", "outputs", default=[])
    if isinstance(outputs, Mapping):
        outputs = [dict(outputs)]
    metadata = _get_field(raw, "metadata", "meta", default={})
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    return CanonicalCase(
        id=str(case_id),
        prompt=_safe_text(prompt),
        context=context,
        input_data=input_data,
        expected_answer=expected,
        definition_id=definition_id,
        difficulty_parameters=difficulty_parameters,
        distractors=list(distractors),
        scoring=scoring,
        synthetic=synthetic,
        difficulty=difficulty,
        task_type=task_type,
        split=split,
        generator=str(generator or "Synthetic"),
        seed=seed,
        validation_status=validation_status or "Pending",
        validation_errors=list(errors),
        model_outputs=list(outputs),
        metadata=dict(metadata),
        raw=raw,
    )


def _extract_cases(dataset: Any) -> list[Any]:
    if dataset is None:
        return []
    if isinstance(dataset, Mapping):
        for key in ("cases", "items", "data", "examples"):
            if key in dataset and isinstance(dataset[key], (list, tuple)):
                return list(dataset[key])
        return [dataset]
    for attr in ("cases", "items", "examples"):
        if hasattr(dataset, attr):
            try:
                value = getattr(dataset, attr)
                if callable(value):
                    value = value()
                if isinstance(value, (list, tuple)):
                    return list(value)
            except Exception:
                continue
    if isinstance(dataset, (list, tuple)):
        return list(dataset)
    try:
        return list(dataset)
    except Exception:
        return [dataset]


# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------


class BackendAdapter:
    """Bridge the Tk widgets to the evolving engine package.

    The adapter is intentionally small and stateless.  It returns canonical
    records to the UI, but retains each native object in ``CanonicalCase.raw``
    for a later engine Runner call.
    """

    def __init__(self) -> None:
        self.models = _import_optional("synthetic_benchmark_factory.models")
        self.generator_module = _import_optional(
            "synthetic_benchmark_factory.generator",
            "synthetic_benchmark_factory.generators",
        )
        self.validator_module = _import_optional(
            "synthetic_benchmark_factory.validator",
            "synthetic_benchmark_factory.validators",
        )
        self.runner_module = _import_optional("synthetic_benchmark_factory.runner")
        self.analysis_module = _import_optional("synthetic_benchmark_factory.analysis")
        self.providers_module = _import_optional("synthetic_benchmark_factory.providers")
        self.metrics_module = _import_optional("synthetic_benchmark_factory.metrics")
        self.serialization_module = _import_optional(
            "synthetic_benchmark_factory.serialization",
            "synthetic_benchmark_factory.serializers",
        )
        self.engine_available = bool(self.models or self.generator_module)
        self.last_dataset: Any = None
        self.last_definition: Any = None

    # -- definitions -----------------------------------------------------

    def make_definition(self, draft: DefinitionDraft) -> Any:
        payload = draft.to_dict()
        module = self.models
        cls = getattr(module, "BenchmarkDefinition", None) if module else None
        if cls is None:
            self.last_definition = payload
            return payload
        # Some early model versions use ``type`` instead of ``task_type``.
        attempts = [payload, {**payload, "type": payload["task_type"]}]
        for candidate in attempts:
            try:
                self.last_definition = cls(**candidate)
                return self.last_definition
            except (TypeError, ValueError):
                continue
        # Dataclass fields may use enum classes.  Convert only where needed.
        try:
            fields = getattr(cls, "__dataclass_fields__", {})
            filtered = {k: v for k, v in payload.items() if k in fields}
            self.last_definition = cls(**filtered)
            return self.last_definition
        except Exception:
            self.last_definition = payload
            return payload

    # -- generation -----------------------------------------------------

    def generate(
        self,
        draft: DefinitionDraft,
        count: int,
        seed: int,
        split_ratios: Mapping[str, float] | None = None,
        distractors: Sequence[str] = (),
    ) -> list[CanonicalCase]:
        definition = self.make_definition(draft)
        generated: Any = None
        if self.generator_module:
            generator_cls = getattr(self.generator_module, "BenchmarkGenerator", None)
            if generator_cls is not None:
                generator = None
                for kwargs in ({"definition": definition, "seed": seed}, {"seed": seed}, {}):
                    try:
                        generator = generator_cls(**kwargs)
                        break
                    except Exception:
                        continue
                if generator is not None:
                    method = getattr(generator, "generate", None)
                    if callable(method):
                        try:
                            # The core generator uses enum values (lowercase
                            # ``train``/``validation``/…); the editor exposes
                            # title-case labels for readability.
                            normalized_ratios = {
                                str(key).strip().lower(): float(value)
                                for key, value in (split_ratios or {}).items()
                            }
                            generated = _call_flexible(
                                method,
                                definition,
                                count=count,
                                seed=seed,
                                split_ratios=normalized_ratios,
                                distractors=list(distractors),
                            )
                        except Exception:
                            generated = None
        if generated is None:
            generated = self._fallback_generate(draft, count, seed, split_ratios, distractors)
        self.last_dataset = generated
        return [_canonical_case(case, i) for i, case in enumerate(_extract_cases(generated))]

    def generate_demo(self, count: int = 500, seed: int = 20261004) -> list[CanonicalCase]:
        if self.generator_module:
            generator_cls = getattr(self.generator_module, "BenchmarkGenerator", None)
            if generator_cls is not None:
                try:
                    generator = generator_cls(seed=seed)
                    method = getattr(generator, "generate_demo", None)
                    if callable(method):
                        generated = _call_flexible(method, count=count, seed=seed)
                        self.last_dataset = generated
                        return [_canonical_case(case, i) for i, case in enumerate(_extract_cases(generated))]
                except Exception:
                    pass
        draft = DefinitionDraft()
        return self.generate(
            draft,
            count=count,
            seed=seed,
            split_ratios={"Train": 0.7, "Validation": 0.1, "Test": 0.15, "Challenge": 0.05},
            distractors=["Irrelevant Info", "Contradictory Info", "Near Match"],
        )

    def _fallback_generate(
        self,
        draft: DefinitionDraft,
        count: int,
        seed: int,
        split_ratios: Mapping[str, float] | None,
        distractors: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Deterministic, ground-truthed fallback generator.

        This is also useful as a smoke-test fixture for the GUI.  Every case
        has a computable answer and carries ``synthetic=True`` metadata.
        """

        rng = random.Random(seed)
        count = max(0, min(int(count), 100_000))
        ratios = dict(split_ratios or {"Train": 0.7, "Validation": 0.1, "Test": 0.15, "Challenge": 0.05})
        split_names = [name for name in SPLITS if name in ratios and ratios[name] > 0]
        if not split_names:
            split_names = ["Test"]
        weights = [max(0.0, float(ratios.get(name, 0))) for name in split_names]
        total_weight = sum(weights) or 1.0
        task = draft.task_type or "Memory"
        level = DIFFICULTIES.index(draft.difficulty) + 1 if draft.difficulty in DIFFICULTIES else 2
        records: list[dict[str, Any]] = []
        names = ["Aster", "Beryl", "Cedar", "Dahlia", "Ember", "Frost", "Gale", "Hazel"]
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for i in range(count):
            case_seed = seed + i * 7919
            local = random.Random(case_seed)
            split = local.choices(split_names, weights=weights, k=1)[0]
            distractor_count = max(0, level - 1)
            selected_distractors = list(distractors[:distractor_count])
            entity = names[i % len(names)]
            value = colors[(i * 3 + level) % len(colors)]
            if task.lower().startswith("memory"):
                facts = [
                    f"{entity} stores the {value} token in locker {100 + i % 37}.",
                    f"The recall window is day {1 + (i % max(1, level * 3))}.",
                ]
                if "Long Context" in distractors:
                    facts.extend(f"Irrelevant log line {j}: no bearing on the token." for j in range(level * 8))
                if "Contradictory Info" in selected_distractors:
                    facts.append(f"A stale note says {entity} uses black; the latest fact overrides it.")
                question = f"What token and locker belong to {entity}?"
                expected = {"token": value, "locker": 100 + i % 37}
                context = " ".join(facts)
            elif task.lower().startswith("extract"):
                email = f"{entity.lower()}{i}@example.test"
                context = f"Contact card — name: {entity}; email: {email}; tier: {level}."
                question = "Extract the name and email as JSON."
                expected = {"name": entity, "email": email}
            elif task.lower().startswith("class"):
                labels = ["positive", "negative", "neutral"]
                label = labels[(i + level) % len(labels)]
                context = f"Review {i}: sentiment signal is {label}."
                question = "Classify the review sentiment."
                expected = label
            elif task.lower().startswith("arithmetic"):
                a = 2 + (i * 7) % (8 + level * 4)
                b = 1 + (i * 5) % (5 + level * 3)
                context = f"A crate has {a} units and receives {b} more units."
                question = "How many units are there now? Return an integer."
                expected = a + b
            elif task.lower().startswith("tool"):
                tools = ["search", "calculator", "calendar", "database"]
                expected = tools[(i + level) % len(tools)]
                context = f"Available tools: {', '.join(tools)}. Request category: {expected}."
                question = "Select the one tool that should handle this request."
            elif task.lower().startswith("plan"):
                context = "Goal: publish a reproducible benchmark. Resources: spec, generator, validator."
                question = "List the first three dependency-ordered actions."
                expected = ["write specification", "generate cases", "validate cases"]
            elif task.lower().startswith("struct"):
                context = f"Record {i} belongs to {entity} and has priority {level}."
                question = "Return JSON with keys entity and priority only."
                expected = {"entity": entity, "priority": level}
            else:
                context = f"Instruction: answer with the codeword {value}. Do not add commentary."
                question = "Follow the instruction exactly."
                expected = value
            if "Irrelevant Info" in selected_distractors:
                context += " Unrelated note: the lab closes at 18:00."
            if "Near Match" in selected_distractors:
                context += f" Similar but different record: {names[(i + 1) % len(names)]} uses {colors[(i + 1) % len(colors)]}."
            records.append(
                {
                    "id": f"synthetic-{seed}-{i + 1:05d}",
                    "prompt": question,
                    "context": context,
                    "expected_answer": expected,
                    "difficulty": f"Level {min(5, max(1, level))}",
                    "task_type": task,
                    "split": split,
                    "generator": "Synthetic Procedural Generator",
                    "seed": case_seed,
                    "validation_status": "Pending",
                    "metadata": {
                        "synthetic": True,
                        "distractors": selected_distractors,
                        "context_length": len(context),
                        "ground_truth": "deterministic",
                    },
                }
            )
        return records

    # -- validation ------------------------------------------------------

    def validate(self, cases: Sequence[CanonicalCase]) -> dict[str, Any]:
        native_cases = [case.raw for case in cases if case.raw is not None]
        report: Any = None
        if self.validator_module:
            validator_cls = getattr(self.validator_module, "Validator", None)
            validator = None
            if validator_cls is not None:
                try:
                    validator = validator_cls()
                except Exception:
                    validator = None
            fn = getattr(self.validator_module, "validate_dataset", None)
            try:
                native_dataset = self.last_dataset if self.last_dataset is not None else native_cases
                if validator is not None and hasattr(validator, "validate_dataset"):
                    report = _call_flexible(validator.validate_dataset, native_dataset)
                elif callable(fn):
                    report = _call_flexible(fn, native_dataset)
            except Exception:
                report = None
        result = _object_dict(report) if report is not None else {}
        if report is not None and hasattr(report, "to_dict"):
            try:
                result = dict(report.to_dict())
            except Exception:
                pass
        errors_by_id: dict[str, list[str]] = {}
        if isinstance(result.get("errors"), Mapping):
            errors_by_id = {str(k): list(v) if isinstance(v, list) else [str(v)] for k, v in result["errors"].items()}
        duplicates: set[str] = set()
        # A template prompt is often intentionally shared by many cases.  A
        # duplicate means the complete item is repeated, not merely that two
        # items use the same instruction text.  Include context and ground
        # truth in the stable signature so procedural cases validate correctly.
        seen_prompt: dict[str, str] = {}
        for case in cases:
            errs = list(errors_by_id.get(case.id, []))
            # The semantic Validator marks native BenchmarkCase objects.  Sync
            # those fields back into the canonical records shown by the UI.
            if case.raw is not None:
                native_status = _enum_string(_get_field(case.raw, "validation_status", default=""))
                native_errors = _get_field(case.raw, "validation_errors", default=[])
                if native_status and native_status.lower() not in {"pending", "unvalidated"}:
                    normalized_status = native_status.replace("_", " ").title()
                    case.validation_status = "Valid" if normalized_status.lower() == "valid" else ("Invalid" if normalized_status.lower() == "invalid" else normalized_status)
                if native_errors:
                    errs.extend(str(item) for item in native_errors)
            if not case.prompt.strip():
                errs.append("Broken Case: empty prompt")
            if case.expected_answer in (None, ""):
                errs.append("Impossible Case: missing ground truth")
            prompt_key = json.dumps(
                {
                    "prompt": case.prompt.strip(),
                    "context": _jsonable(case.context),
                    "expected": _jsonable(case.expected_answer),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if prompt_key and prompt_key in seen_prompt:
                duplicates.add(case.id)
                errs.append(f"Duplicate: same prompt as {seen_prompt[prompt_key]}")
            elif prompt_key:
                seen_prompt[prompt_key] = case.id
            try:
                json.dumps(_jsonable(case.expected_answer))
            except Exception:
                errs.append("Invalid Schema: expected answer is not serializable")
            case.validation_errors = sorted(set(errs))
            case.validation_status = "Valid" if not errs else "Invalid"
        valid = sum(1 for case in cases if case.validation_status == "Valid")
        invalid = len(cases) - valid
        report_out = {
            "total": len(cases),
            "valid": valid,
            "invalid": invalid,
            "duplicate_count": max(len(duplicates), _as_int(result.get("duplicate_count", 0), 0)),
            "errors": {case.id: case.validation_errors for case in cases if case.validation_errors},
            "status": "PASS" if invalid == 0 else "FAIL",
        }
        # Preserve engine details (without allowing a non-JSON object into UI).
        for key in ("ambiguous", "broken", "impossible", "schema", "data_leakage", "leakage_count", "ambiguous_count", "impossible_count"):
            if key in result:
                report_out[key] = _jsonable(result[key])
        return report_out

    # -- runner ---------------------------------------------------------

    def run(self, cases: Sequence[CanonicalCase], provider_name: str) -> list[dict[str, Any]]:
        provider_name = provider_name or "RuleModel"
        if provider_name.lower() in {"both", "both (mock + rule)", "mock + rule", "mockmodel + rulemodel"}:
            combined: list[dict[str, Any]] = []
            for name in ("MockModel", "RuleModel"):
                combined.extend(self.run(cases, name))
            return combined
        outputs: list[dict[str, Any]] = []
        provider = None
        runner = None
        if self.providers_module:
            provider_cls = getattr(self.providers_module, provider_name, None)
            if provider_cls is not None:
                try:
                    provider = provider_cls()
                except Exception:
                    provider = None
        if self.runner_module:
            runner_cls = getattr(self.runner_module, "BenchmarkRunner", None)
            if runner_cls is not None and provider is not None:
                try:
                    runner = runner_cls(providers=provider)
                except Exception:
                    try:
                        runner = runner_cls(provider)
                    except Exception:
                        runner = None
        for case in cases:
            started = time.perf_counter()
            output: Any = None
            score: Any = None
            native_record: Any = None
            if runner is not None and case.raw is not None:
                method = getattr(runner, "run_case", None)
                if callable(method):
                    try:
                        # The canonical runner accepts ``run_case(case,
                        # provider)``.  Passing the provider explicitly also
                        # works with early one-provider implementations.
                        native_record = _call_flexible(method, case.raw, provider)
                        output = _get_field(native_record, "prediction", "output", default=None)
                        if output is None:
                            output = _get_field(native_record, "model_output", default=None)
                            if output is not None:
                                output = _get_field(output, "output", "value", "text", default=output)
                    except Exception:
                        output = None
            if output is None:
                output = self._fallback_model(case, provider_name)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            native_score = _get_field(native_record, "score", default=None) if native_record is not None else None
            try:
                score = float(native_score) if native_score is not None else self._score(case.expected_answer, output)
            except (TypeError, ValueError):
                score = self._score(case.expected_answer, output)
            native_latency = _get_field(native_record, "latency_ms", default=None) if native_record is not None else None
            if native_latency is not None:
                try:
                    elapsed_ms = round(float(native_latency), 3)
                except (TypeError, ValueError):
                    pass
            native_tokens = _get_field(native_record, "tokens", default=None) if native_record is not None else None
            token_count = native_tokens if native_tokens is not None else len(str(output).split())
            item = {
                "provider": provider_name,
                "output": _jsonable(output),
                "prediction": _jsonable(output),
                "score": score,
                "pass": bool(score >= 1.0),
                "passed": bool(score >= 1.0),
                "latency_ms": elapsed_ms,
                "tokens": _jsonable(token_count),
                "synthetic": True,
            }
            case.model_outputs.append(item)
            outputs.append({"case_id": case.id, **item})
        return outputs

    def _fallback_model(self, case: CanonicalCase, provider_name: str) -> Any:
        if provider_name == "MockModel":
            # MockModel intentionally demonstrates a predictable baseline.
            if isinstance(case.expected_answer, (dict, list)):
                return case.expected_answer
            return str(case.expected_answer)
        # RuleModel extracts the deterministic answer from the synthetic
        # record.  Real providers are deliberately not offered by this UI.
        return case.expected_answer

    def _score(self, expected: Any, actual: Any) -> float:
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            keys = set(expected) | set(actual)
            if not keys:
                return 1.0
            return sum(expected.get(k) == actual.get(k) for k in keys) / len(keys)
        if isinstance(expected, list) and isinstance(actual, list):
            if not expected and not actual:
                return 1.0
            common = sum(1 for x, y in zip(expected, actual) if x == y)
            return common / max(len(expected), len(actual), 1)
        return 1.0 if str(expected).strip() == str(actual).strip() else 0.0

    # -- statistics/export ----------------------------------------------

    def stats(self, cases: Sequence[CanonicalCase]) -> dict[str, Any]:
        total = len(cases)
        difficulty: dict[str, int] = {}
        task_types: dict[str, int] = {}
        splits: dict[str, int] = {}
        context_lengths: list[int] = []
        answer_lengths: list[int] = []
        validation_failures = 0
        prompts: set[str] = set()
        duplicate_count = 0
        for case in cases:
            difficulty[case.difficulty or "Unknown"] = difficulty.get(case.difficulty or "Unknown", 0) + 1
            task_types[case.task_type or "Unknown"] = task_types.get(case.task_type or "Unknown", 0) + 1
            splits[case.split or "Unknown"] = splits.get(case.split or "Unknown", 0) + 1
            context_lengths.append(len(_safe_text(case.context)))
            answer_lengths.append(len(_safe_text(case.expected_answer)))
            # Pending/unvalidated cases are not validation *failures*; they
            # simply have not gone through the validator yet.
            if case.validation_status in ("Invalid", "Warning"):
                validation_failures += 1
            key = case.prompt.strip().lower()
            if key in prompts:
                duplicate_count += 1
            prompts.add(key)
        return {
            "case_count": total,
            "difficulty_distribution": difficulty,
            "type_distribution": task_types,
            "split_distribution": splits,
            "context_length": {
                "min": min(context_lengths) if context_lengths else 0,
                "max": max(context_lengths) if context_lengths else 0,
                "mean": round(statistics.mean(context_lengths), 2) if context_lengths else 0,
            },
            "answer_length": {
                "min": min(answer_lengths) if answer_lengths else 0,
                "max": max(answer_lengths) if answer_lengths else 0,
                "mean": round(statistics.mean(answer_lengths), 2) if answer_lengths else 0,
            },
            "duplicate_rate": round(duplicate_count / total, 4) if total else 0,
            "validation_failure_rate": round(validation_failures / total, 4) if total else 0,
        }

    def item_analysis(self, cases: Sequence[CanonicalCase], records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Return the four author-facing item-analysis buckets.

        The optional core analysis module is preferred.  The local fallback is
        intentionally transparent and works with the simplified records kept
        by the Runner tab.
        """

        records = list(records or [])
        if self.analysis_module and records:
            fn = getattr(self.analysis_module, "analyze_items", None) or getattr(self.analysis_module, "analyze_results", None)
            if callable(fn):
                try:
                    analysis_cases = [case.raw if case.raw is not None else case for case in cases]
                    report = _call_flexible(fn, records, cases=analysis_cases)
                    data = _object_dict(report)
                    if hasattr(report, "to_dict"):
                        data = _object_dict(report.to_dict())
                    return _jsonable(data)
                except Exception:
                    pass
        by_case: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            by_case.setdefault(str(record.get("case_id", "")), []).append(record)
        all_wrong: list[str] = []
        too_easy: list[str] = []
        discriminating: list[str] = []
        problematic: list[str] = []
        items: list[dict[str, Any]] = []
        for case in cases:
            group = by_case.get(case.id, [])
            scores = [float(item.get("score", 0)) for item in group]
            passed = [bool(item.get("pass")) for item in group]
            flags: list[str] = []
            if group and all(not value for value in passed):
                flags.append("all_models_wrong")
                all_wrong.append(case.id)
            if group and all(value for value in passed):
                flags.append("too_easy")
                too_easy.append(case.id)
            if len(scores) >= 2 and max(scores) - min(scores) >= 0.25 and any(passed) and not all(passed):
                flags.append("discriminating")
                discriminating.append(case.id)
            if case.validation_status in ("Invalid", "Warning") or not case.expected_answer:
                flags.append("problematic")
                problematic.append(case.id)
            items.append({"case_id": case.id, "scores": scores, "passes": passed, "flags": flags})
        return {
            "items": items,
            "all_models_wrong": all_wrong,
            "too_easy": too_easy,
            "discriminating": discriminating,
            "problematic": problematic,
            "summary": {
                "item_count": len(items),
                "all_models_wrong_count": len(all_wrong),
                "too_easy_count": len(too_easy),
                "discriminating_count": len(discriminating),
                "problematic_count": len(problematic),
            },
        }

    def export(self, cases: Sequence[CanonicalCase], path: str, fmt: str, metadata: Mapping[str, Any] | None = None) -> str:
        fmt = fmt.lower().lstrip(".")
        rows = [case.to_dict() for case in cases]
        payload_meta = dict(metadata or {})
        payload_meta.setdefault("synthetic", True)
        payload_meta.setdefault("app", APP_NAME)
        payload_meta.setdefault("app_version", APP_VERSION)
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "jsonl":
            with path_obj.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif fmt == "csv":
            columns = ["id", "definition_id", "prompt", "context", "input_data", "expected_answer", "difficulty", "difficulty_parameters", "distractors", "scoring", "task_type", "split", "generator", "seed", "synthetic", "validation_status"]
            with path_obj.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: _safe_text(row.get(column, "")) for column in columns})
        elif fmt == "markdown":
            with path_obj.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"# {payload_meta.get('name', 'Synthetic Benchmark Dataset')}\n\n")
                handle.write("| ID | Type | Split | Difficulty | Validation | Prompt | Expected Answer |\n")
                handle.write("|---|---|---|---|---|---|---|\n")
                for row in rows:
                    cells = [
                        row.get("id", ""), row.get("task_type", ""), row.get("split", ""),
                        row.get("difficulty", ""), row.get("validation_status", ""),
                        _safe_text(row.get("prompt", "")).replace("|", "\\|"),
                        _safe_text(row.get("expected_answer", "")).replace("|", "\\|"),
                    ]
                    handle.write("| " + " | ".join(str(c).replace("\n", " ") for c in cells) + " |\n")
        else:
            document = {"metadata": _jsonable(payload_meta), "cases": rows}
            with path_obj.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
        return str(path_obj)


# ---------------------------------------------------------------------------
# Tkinter widgets
# ---------------------------------------------------------------------------


class ScrolledText(ttk.Frame):
    """Small ttk wrapper because tkinter.scrolledtext has limited styling."""

    def __init__(self, master: tk.Misc, height: int = 5, **kwargs: Any) -> None:
        super().__init__(master)
        self.text = tk.Text(self, height=height, wrap="word", undo=True, **kwargs)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def get(self) -> str:
        return self.text.get("1.0", "end-1c")

    def set(self, value: Any) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", _safe_text(value))


class BenchmarkFactoryApp(tk.Tk):
    """Main desktop window."""

    def __init__(self, adapter: BackendAdapter | None = None) -> None:
        super().__init__()
        self.title(f"{APP_NAME} — {APP_VERSION}")
        self.geometry("1450x900")
        self.minsize(1080, 700)
        self.option_add("*Font", ("Segoe UI", 10))
        self.adapter = adapter or BackendAdapter()
        self.cases: list[CanonicalCase] = []
        self.definition_draft = DefinitionDraft()
        self.last_validation: dict[str, Any] = {}
        self.last_stats: dict[str, Any] = {}
        self.last_item_analysis: dict[str, Any] = {}
        self.last_run: list[dict[str, Any]] = []
        self.job_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._tree_case_map: dict[str, str] = {}
        self._build_style()
        self._build_menu()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()
        self._load_draft_into_editor(self.definition_draft)
        self._refresh_all()
        self.after(100, self._poll_jobs)

    # -- construction ----------------------------------------------------

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"), foreground="#17324d")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#52677a")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", foreground="#4b5d6b")

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New Definition", command=self._clear_definition)
        file_menu.add_command(label="Load Definition JSON…", command=self._load_definition_json)
        file_menu.add_command(label="Save Definition JSON…", command=self._save_definition_json)
        file_menu.add_command(label="Load JSON Dataset…", command=self._load_json_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Export…", command=self._export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        tools_menu = tk.Menu(menu, tearoff=False)
        tools_menu.add_command(label="Generate 500-case Memory Demo", command=self._generate_demo)
        tools_menu.add_command(label="Validate Dataset", command=self._validate_dataset)
        tools_menu.add_command(label="Refresh Statistics", command=self._refresh_statistics)
        menu.add_cascade(label="Tools", menu=tools_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self._about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menu)

    def _build_header(self) -> None:
        frame = ttk.Frame(self, padding=(18, 14, 18, 8))
        frame.pack(fill="x")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="AI 基准测试数据集工厂 · Synthetic data first, reproducible by design", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        self.engine_badge = ttk.Label(frame, text="", anchor="e")
        self.engine_badge.pack(side="right", padx=(10, 0))
        self._set_engine_badge()

    def _set_engine_badge(self) -> None:
        if self.adapter.engine_available:
            self.engine_badge.configure(text="● Engine connected", foreground="#1d7a46")
        else:
            self.engine_badge.configure(text="● Local fallback mode", foreground="#9a6712")

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.editor_tab = ttk.Frame(self.notebook)
        self.browser_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.runner_tab = ttk.Frame(self.notebook)
        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.editor_tab, text="  Benchmark Editor & Generator  ")
        self.notebook.add(self.browser_tab, text="  Case Browser  ")
        self.notebook.add(self.stats_tab, text="  Dataset Statistics  ")
        self.notebook.add(self.runner_tab, text="  Runner  ")
        self.notebook.add(self.export_tab, text="  Reports & Export  ")
        self._build_editor_tab()
        self._build_browser_tab()
        self._build_stats_tab()
        self._build_runner_tab()
        self._build_export_tab()

    def _build_statusbar(self) -> None:
        frame = ttk.Frame(self, padding=(12, 3))
        frame.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready — no dataset loaded")
        ttk.Label(frame, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(frame, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    # -- editor ----------------------------------------------------------

    def _build_editor_tab(self) -> None:
        root = ttk.Frame(self.editor_tab, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=2)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        form = ttk.LabelFrame(root, text="Benchmark Definition", style="Section.TLabelframe")
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        form.columnconfigure(1, weight=1)
        form.rowconfigure(3, weight=1)
        form.rowconfigure(6, weight=1)
        form.rowconfigure(7, weight=1)
        form.rowconfigure(8, weight=1)
        self.editor_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("Name", "name"),
            ("Capability", "capability"),
        ]
        for row, (label, key) in enumerate(fields):
            ttk.Label(form, text=label + ":").grid(row=row, column=0, sticky="nw", padx=8, pady=7)
            var = tk.StringVar()
            self.editor_vars[key] = var
            ttk.Entry(form, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        ttk.Label(form, text="Difficulty:").grid(row=2, column=0, sticky="nw", padx=8, pady=7)
        self.difficulty_var = tk.StringVar(value="Level 2")
        self.editor_vars["difficulty"] = self.difficulty_var
        ttk.Combobox(form, textvariable=self.difficulty_var, values=DIFFICULTIES, state="readonly").grid(row=2, column=1, sticky="w", padx=8, pady=5)
        ttk.Label(form, text="Description:").grid(row=3, column=0, sticky="nw", padx=8, pady=7)
        self.description_text = ScrolledText(form, height=5)
        self.description_text.grid(row=3, column=1, sticky="nsew", padx=8, pady=5)
        type_frame = ttk.Frame(form)
        type_frame.grid(row=4, column=1, sticky="ew", padx=8, pady=5)
        type_frame.columnconfigure(1, weight=1)
        ttk.Label(form, text="Task Type:").grid(row=4, column=0, sticky="nw", padx=8, pady=7)
        self.task_type_var = tk.StringVar(value="Memory")
        self.editor_vars["task_type"] = self.task_type_var
        ttk.Combobox(type_frame, textvariable=self.task_type_var, values=TASK_TYPES, state="readonly").grid(row=0, column=0, sticky="w")
        ttk.Label(type_frame, text="  Capability under test", foreground="#657786").grid(row=0, column=1, sticky="w", padx=8)
        for row, key, label in [(5, "input_schema", "Input Schema:"), (6, "expected_output", "Expected Output / Ground Truth:"), (7, "scoring", "Scoring:"), (8, "constraints", "Constraints:")]:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="nw", padx=8, pady=7)
            text = ScrolledText(form, height=4)
            text.grid(row=row, column=1, sticky="nsew", padx=8, pady=5)
            setattr(self, f"{key}_text", text)

        generator_frame = ttk.LabelFrame(root, text="Generator", style="Section.TLabelframe")
        generator_frame.grid(row=0, column=1, sticky="nsew")
        generator_frame.columnconfigure(1, weight=1)
        self.count_var = tk.StringVar(value="50")
        self.seed_var = tk.StringVar(value="20261004")
        self.split_vars = {split: tk.BooleanVar(value=split in ("Train", "Validation", "Test", "Challenge")) for split in SPLITS}
        ttk.Label(generator_frame, text="Case count:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Spinbox(generator_frame, from_=1, to=100000, textvariable=self.count_var, width=12).grid(row=0, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(generator_frame, text="Seed:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(generator_frame, textvariable=self.seed_var, width=15).grid(row=1, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(generator_frame, text="Effective difficulty:").grid(row=2, column=0, sticky="nw", padx=8, pady=8)
        self.difficulty_preview_var = tk.StringVar(value="Profile parameters are applied by the generator")
        ttk.Label(generator_frame, textvariable=self.difficulty_preview_var, foreground="#52677a", wraplength=270, justify="left").grid(row=2, column=1, sticky="w", padx=8, pady=8)
        self.difficulty_var.trace_add("write", lambda *_: self._update_difficulty_preview())
        ttk.Label(generator_frame, text="Dataset split:").grid(row=3, column=0, sticky="nw", padx=8, pady=8)
        split_box = ttk.Frame(generator_frame)
        split_box.grid(row=3, column=1, sticky="w", padx=8, pady=8)
        for i, split in enumerate(SPLITS):
            ttk.Checkbutton(split_box, text=split, variable=self.split_vars[split]).grid(row=i, column=0, sticky="w", pady=1)
        ttk.Label(generator_frame, text="Distractors:").grid(row=4, column=0, sticky="nw", padx=8, pady=8)
        distractor_box = ttk.Frame(generator_frame)
        distractor_box.grid(row=4, column=1, sticky="w", padx=8, pady=8)
        self.distractor_vars = {item: tk.BooleanVar(value=False) for item in DISTRACTORS}
        for i, item in enumerate(DISTRACTORS):
            ttk.Checkbutton(distractor_box, text=item, variable=self.distractor_vars[item]).grid(row=i, column=0, sticky="w", pady=1)
        ttk.Separator(generator_frame).grid(row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=9)
        ttk.Button(generator_frame, text="Save Definition", command=self._save_definition).grid(row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Button(generator_frame, text="Generate Dataset", style="Accent.TButton", command=self._generate_dataset).grid(row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Button(generator_frame, text="Generate Long-Term Memory Demo (500)", command=self._generate_demo).grid(row=8, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Button(generator_frame, text="Validate Current Dataset", command=self._validate_dataset).grid(row=9, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        ttk.Button(generator_frame, text="Reset Form", command=self._clear_definition).grid(row=10, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        note = ttk.Label(generator_frame, text="All generated cases are labeled Synthetic.\nGround truth is required before validation passes.", foreground="#657786", wraplength=270, justify="left")
        note.grid(row=11, column=0, columnspan=2, sticky="w", padx=8, pady=(14, 8))
        self._update_difficulty_preview()

    # -- browser ---------------------------------------------------------

    def _build_browser_tab(self) -> None:
        root = ttk.Frame(self.browser_tab, padding=12)
        root.pack(fill="both", expand=True)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        filters = ttk.Frame(root)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(filters, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        search = ttk.Entry(filters, textvariable=self.search_var, width=30)
        search.pack(side="left", padx=(5, 14))
        self.search_var.trace_add("write", lambda *_: self._refresh_case_tree())
        ttk.Label(filters, text="Type:").pack(side="left")
        self.filter_type_var = tk.StringVar(value="All")
        ttk.Combobox(filters, textvariable=self.filter_type_var, values=["All"] + TASK_TYPES, state="readonly", width=18).pack(side="left", padx=5)
        self.filter_type_var.trace_add("write", lambda *_: self._refresh_case_tree())
        ttk.Label(filters, text="Validation:").pack(side="left", padx=(10, 0))
        self.filter_status_var = tk.StringVar(value="All")
        ttk.Combobox(filters, textvariable=self.filter_status_var, values=["All", "Valid", "Invalid", "Pending"], state="readonly", width=12).pack(side="left", padx=5)
        self.filter_status_var.trace_add("write", lambda *_: self._refresh_case_tree())
        ttk.Button(filters, text="Refresh", command=self._refresh_case_tree).pack(side="right")
        pane = ttk.PanedWindow(root, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew")
        left = ttk.Frame(pane)
        right = ttk.Frame(pane, padding=(10, 0, 0, 0))
        pane.add(left, weight=2)
        pane.add(right, weight=3)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        columns = ("id", "type", "split", "difficulty", "status", "generator", "seed")
        self.case_tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        headings = {"id": "ID", "type": "Type", "split": "Split", "difficulty": "Difficulty", "status": "Validation", "generator": "Generator", "seed": "Seed"}
        widths = {"id": 180, "type": 130, "split": 85, "difficulty": 90, "status": 90, "generator": 170, "seed": 95}
        for column in columns:
            self.case_tree.heading(column, text=headings[column])
            self.case_tree.column(column, width=widths[column], minwidth=50, anchor="w")
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.case_tree.yview)
        xscroll = ttk.Scrollbar(left, orient="horizontal", command=self.case_tree.xview)
        self.case_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.case_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.case_tree.bind("<<TreeviewSelect>>", self._on_case_select)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self.case_detail_title = ttk.Label(right, text="Select a case", style="Subtitle.TLabel")
        self.case_detail_title.grid(row=0, column=0, sticky="w", pady=(0, 7))
        detail = ttk.Frame(right)
        detail.grid(row=1, column=0, sticky="nsew")
        detail.rowconfigure(1, weight=1)
        detail.rowconfigure(3, weight=1)
        detail.rowconfigure(5, weight=1)
        detail.rowconfigure(7, weight=1)
        detail.columnconfigure(0, weight=1)
        for row, label in [(0, "Prompt"), (2, "Context"), (4, "Expected Answer / Ground Truth"), (6, "Model Outputs")]:
            ttk.Label(detail, text=label, font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(4, 2))
        self.detail_prompt = ScrolledText(detail, height=4)
        self.detail_prompt.grid(row=1, column=0, sticky="nsew")
        self.detail_context = ScrolledText(detail, height=7)
        self.detail_context.grid(row=3, column=0, sticky="nsew")
        self.detail_expected = ScrolledText(detail, height=4)
        self.detail_expected.grid(row=5, column=0, sticky="nsew")
        self.detail_outputs = ScrolledText(detail, height=5)
        self.detail_outputs.grid(row=7, column=0, sticky="nsew")
        self.detail_meta = ttk.Label(right, text="", foreground="#657786", justify="left")
        self.detail_meta.grid(row=2, column=0, sticky="w", pady=(8, 0))

    # -- statistics ------------------------------------------------------

    def _build_stats_tab(self) -> None:
        root = ttk.Frame(self.stats_tab, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        top = ttk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.stat_summary_var = tk.StringVar(value="No dataset")
        ttk.Label(top, textvariable=self.stat_summary_var, style="Subtitle.TLabel").pack(side="left")
        ttk.Button(top, text="Refresh Statistics", command=self._refresh_statistics).pack(side="right")
        self.stat_cards: dict[str, tk.StringVar] = {}
        cards = ttk.Frame(root)
        cards.grid(row=1, column=0, columnspan=2, sticky="new")
        for i, key in enumerate(("case_count", "valid", "invalid", "duplicate_rate", "validation_failure_rate")):
            cards.columnconfigure(i, weight=1)
            frame = ttk.LabelFrame(cards, text=key.replace("_", " ").title())
            frame.grid(row=0, column=i, sticky="ew", padx=4)
            var = tk.StringVar(value="—")
            self.stat_cards[key] = var
            ttk.Label(frame, textvariable=var, font=("Segoe UI", 15, "bold"), anchor="center").pack(fill="x", padx=8, pady=13)
        bottom = ttk.PanedWindow(root, orient="horizontal")
        bottom.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(14, 0))
        root.rowconfigure(2, weight=1)
        left = ttk.LabelFrame(bottom, text="Distributions", style="Section.TLabelframe")
        right = ttk.LabelFrame(bottom, text="Lengths & Reproducibility", style="Section.TLabelframe")
        bottom.add(left, weight=1)
        bottom.add(right, weight=1)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.stat_tree = ttk.Treeview(left, columns=("group", "value"), show="headings")
        self.stat_tree.heading("group", text="Group")
        self.stat_tree.heading("value", text="Count")
        self.stat_tree.column("group", width=220)
        self.stat_tree.column("value", width=100)
        self.stat_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.stat_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=8)
        self.stat_tree.configure(yscrollcommand=scrollbar.set)
        self.stat_detail = ScrolledText(right, height=12)
        self.stat_detail.pack(fill="both", expand=True, padx=8, pady=8)

    # -- runner ----------------------------------------------------------

    def _build_runner_tab(self) -> None:
        root = ttk.Frame(self.runner_tab, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        controls = ttk.LabelFrame(root, text="Local Model Runner", style="Section.TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.provider_var = tk.StringVar(value="RuleModel")
        ttk.Label(controls, text="Provider:").pack(side="left", padx=(10, 4), pady=9)
        ttk.Combobox(controls, textvariable=self.provider_var, values=("MockModel", "RuleModel", "Both (Mock + Rule)"), state="readonly", width=18).pack(side="left", padx=4)
        self.run_scope_var = tk.StringVar(value="All cases")
        ttk.Label(controls, text="Scope:").pack(side="left", padx=(20, 4))
        ttk.Combobox(controls, textvariable=self.run_scope_var, values=("All cases", "Selected case(s)"), state="readonly", width=18).pack(side="left", padx=4)
        ttk.Button(controls, text="Run", style="Accent.TButton", command=self._run_cases).pack(side="left", padx=(20, 4))
        ttk.Button(controls, text="Clear Outputs", command=self._clear_outputs).pack(side="left", padx=4)
        ttk.Label(controls, text="Paid APIs are disabled in development.", foreground="#9a6712").pack(side="right", padx=10)
        summary = ttk.Frame(root)
        summary.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.runner_summary_var = tk.StringVar(value="No run yet")
        ttk.Label(summary, textvariable=self.runner_summary_var, style="Subtitle.TLabel").pack(side="left")
        self.runner_log = ScrolledText(root, height=20)
        self.runner_log.grid(row=2, column=0, sticky="nsew")

    # -- export ----------------------------------------------------------

    def _build_export_tab(self) -> None:
        root = ttk.Frame(self.export_tab, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        intro = ttk.LabelFrame(root, text="Dataset Export", style="Section.TLabelframe")
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(intro, text="Export the current validated or pending cases. Each output includes Synthetic metadata and reproducibility fields.", wraplength=1000).pack(anchor="w", padx=10, pady=10)
        buttons = ttk.Frame(root)
        buttons.grid(row=1, column=0, sticky="w", pady=(0, 12))
        for label, fmt in (("Export JSONL", "jsonl"), ("Export JSON", "json"), ("Export CSV", "csv"), ("Export Markdown", "markdown")):
            ttk.Button(buttons, text=label, command=lambda f=fmt: self._export_dialog(f)).pack(side="left", padx=(0, 8))
        report_frame = ttk.LabelFrame(root, text="Validation / Evaluation Report", style="Section.TLabelframe")
        report_frame.grid(row=2, column=0, sticky="nsew")
        self.report_text = ScrolledText(report_frame, height=22)
        self.report_text.pack(fill="both", expand=True, padx=8, pady=8)

    # -- state/draft -----------------------------------------------------

    def _load_draft_into_editor(self, draft: DefinitionDraft) -> None:
        for key in ("name", "capability", "difficulty", "task_type"):
            self.editor_vars[key].set(str(getattr(draft, key)))
        self.description_text.set(draft.description)
        self.input_schema_text.set(draft.input_schema)
        self.expected_output_text.set(draft.expected_output)
        self.scoring_text.set(draft.scoring)
        self.constraints_text.set(draft.constraints)

    def _read_draft(self) -> DefinitionDraft:
        return DefinitionDraft(
            name=self.editor_vars["name"].get().strip(),
            capability=self.editor_vars["capability"].get().strip(),
            description=self.description_text.get().strip(),
            difficulty=self.difficulty_var.get() or "Level 2",
            task_type=self.task_type_var.get() or "Memory",
            input_schema=_parse_json_or_text(self.input_schema_text.get(), {}),
            expected_output=_parse_json_or_text(self.expected_output_text.get(), {}),
            scoring=_parse_json_or_text(self.scoring_text.get(), {}),
            constraints=_parse_json_or_text(self.constraints_text.get(), {}),
        )

    def _save_definition(self) -> None:
        self.definition_draft = self._read_draft()
        self.adapter.make_definition(self.definition_draft)
        self.status_var.set(f"Definition saved: {self.definition_draft.name or 'Untitled'}")

    def _clear_definition(self) -> None:
        self.definition_draft = DefinitionDraft()
        self._load_draft_into_editor(self.definition_draft)
        self.status_var.set("Definition reset")

    def _save_definition_json(self) -> None:
        draft = self._read_draft()
        path = filedialog.asksaveasfilename(
            title="Save Benchmark Definition",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(draft.to_dict(), handle, ensure_ascii=False, indent=2)
            self.definition_draft = draft
            self.status_var.set(f"Definition saved → {path}")
        except Exception as exc:
            self._show_error("Save definition failed", str(exc))

    def _load_definition_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Benchmark Definition",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            draft = DefinitionDraft(
                name=str(data.get("name", "")),
                capability=str(data.get("capability", "")),
                description=str(data.get("description", "")),
                difficulty=str(data.get("difficulty", "Level 2")),
                task_type=str(data.get("task_type", data.get("type", "Memory"))),
                input_schema=data.get("input_schema", {}),
                expected_output=data.get("expected_output", {}),
                scoring=data.get("scoring", {}),
                constraints=data.get("constraints", {}),
            )
            self.definition_draft = draft
            self._load_draft_into_editor(draft)
            self.status_var.set(f"Definition loaded ← {Path(path).name}")
        except Exception as exc:
            self._show_error("Load definition failed", str(exc))

    def _split_ratios(self) -> dict[str, float]:
        enabled = [split for split, var in self.split_vars.items() if var.get()]
        if not enabled:
            enabled = ["Test"]
        equal = 1.0 / len(enabled)
        return {split: equal for split in enabled}

    def _selected_distractors(self) -> list[str]:
        return [item for item, var in self.distractor_vars.items() if var.get()]

    def _update_difficulty_preview(self) -> None:
        """Show the concrete generator knobs behind the selected level."""

        level_text = self.difficulty_var.get() if hasattr(self, "difficulty_var") else "Level 2"
        match = re.search(r"[1-5]", level_text or "2")
        level = int(match.group(0)) if match else 2
        profile: Mapping[str, Any] | None = None
        module = self.adapter.generator_module
        if module is not None:
            profiles = getattr(module, "DIFFICULTY_PROFILES", None)
            if isinstance(profiles, Mapping):
                profile = profiles.get(level) or profiles.get(str(level))
        if profile is None:
            profile = {
                1: {"context_length": 120, "distractor_count": 0, "number_of_facts": 2, "conflicting_facts": 0, "time_distance": 0},
                2: {"context_length": 260, "distractor_count": 1, "number_of_facts": 3, "conflicting_facts": 0, "time_distance": 1},
                3: {"context_length": 520, "distractor_count": 3, "number_of_facts": 5, "conflicting_facts": 1, "time_distance": 3},
                4: {"context_length": 900, "distractor_count": 6, "number_of_facts": 7, "conflicting_facts": 2, "time_distance": 7},
                5: {"context_length": 1500, "distractor_count": 10, "number_of_facts": 10, "conflicting_facts": 3, "time_distance": 30},
            }.get(level, {})
        keys = ("context_length", "distractor_count", "number_of_facts", "conflicting_facts", "time_distance")
        shown = ", ".join(f"{key.replace('_', ' ')}={profile.get(key)}" for key in keys if key in profile)
        self.difficulty_preview_var.set(f"Level {level}: {shown or 'engine profile'}")

    # -- actions ---------------------------------------------------------

    def _generate_dataset(self) -> None:
        draft = self._read_draft()
        self.definition_draft = draft
        count = max(1, min(100_000, _as_int(self.count_var.get(), 50)))
        seed = _as_int(self.seed_var.get(), 20261004)
        # Read Tk variables on the main thread; Tcl variables are not safe to
        # access from the worker used for long dataset generation.
        split_ratios = self._split_ratios()
        distractors = self._selected_distractors()
        self._start_job("generate", lambda: self.adapter.generate(draft, count, seed, split_ratios, distractors))

    def _generate_demo(self) -> None:
        self.definition_draft = DefinitionDraft()
        self._load_draft_into_editor(self.definition_draft)
        self.count_var.set("500")
        self.seed_var.set("20261004")
        self._start_job("generate_demo", lambda: self.adapter.generate_demo(500, 20261004))

    def _validate_dataset(self) -> None:
        if not self.cases:
            self._show_info("Validation", "Generate or load a dataset first.")
            return
        self._start_job("validate", lambda: self.adapter.validate(self.cases))

    def _run_cases(self) -> None:
        if not self.cases:
            self._show_info("Runner", "Generate or load a dataset first.")
            return
        selected_rows = self.case_tree.selection() if self.run_scope_var.get() == "Selected case(s)" else ()
        selected_ids = {self._tree_case_map.get(row, row) for row in selected_rows}
        if selected_ids:
            run_cases = [case for case in self.cases if case.id in selected_ids]
        elif self.run_scope_var.get() == "Selected case(s)":
            self._show_info("Runner", "Select one or more cases in Case Browser, then run again.")
            return
        else:
            run_cases = list(self.cases)
        provider = self.provider_var.get()
        self._start_job("run", lambda: self.adapter.run(run_cases, provider))

    def _clear_outputs(self) -> None:
        for case in self.cases:
            case.model_outputs.clear()
        self.last_run = []
        self.runner_log.set("")
        self.runner_summary_var.set("Outputs cleared")
        self._refresh_case_tree()

    def _refresh_all(self) -> None:
        self._refresh_case_tree()
        self._refresh_statistics()
        self._refresh_report()

    def _refresh_case_tree(self) -> None:
        if not hasattr(self, "case_tree"):
            return
        for item in self.case_tree.get_children():
            self.case_tree.delete(item)
        self._tree_case_map.clear()
        needle = self.search_var.get().strip().lower() if hasattr(self, "search_var") else ""
        type_filter = self.filter_type_var.get() if hasattr(self, "filter_type_var") else "All"
        status_filter = self.filter_status_var.get() if hasattr(self, "filter_status_var") else "All"
        row_number = 0
        for case in self.cases:
            haystack = " ".join((case.id, case.prompt, case.context, case.task_type, case.expected_answer if isinstance(case.expected_answer, str) else _safe_text(case.expected_answer))).lower()
            if needle and needle not in haystack:
                continue
            if type_filter != "All" and case.task_type != type_filter:
                continue
            if status_filter != "All" and case.validation_status != status_filter:
                continue
            row_id = f"case-row-{row_number}"
            row_number += 1
            self._tree_case_map[row_id] = case.id
            self.case_tree.insert("", "end", iid=row_id, values=(case.id, case.task_type, case.split, case.difficulty, case.validation_status, case.generator, case.seed))

    def _on_case_select(self, _event: Any = None) -> None:
        selected = self.case_tree.selection()
        if not selected:
            return
        row = selected[0]
        case_id = self._tree_case_map.get(row)
        if case_id is None:
            values = self.case_tree.item(row, "values")
            case_id = str(values[0]) if values else None
        case = next((item for item in self.cases if item.id == case_id), None)
        if case is None:
            return
        self.case_detail_title.configure(text=f"{case.id}  ·  {case.task_type}  ·  {case.split}")
        self.detail_prompt.set(case.prompt)
        self.detail_context.set(case.context)
        self.detail_expected.set(case.expected_answer)
        self.detail_outputs.set(case.model_outputs or "No model output yet")
        errors = "; ".join(case.validation_errors) if case.validation_errors else "none"
        self.detail_meta.configure(text=f"Difficulty: {case.difficulty or '—'}    Generator: {case.generator}    Seed: {case.seed}\nValidation: {case.validation_status}    Errors: {errors}")

    def _refresh_statistics(self) -> None:
        if not hasattr(self, "stat_tree"):
            return
        self.last_stats = self.adapter.stats(self.cases)
        total = self.last_stats.get("case_count", 0)
        self.stat_summary_var.set(f"{total:,} cases · Synthetic dataset")
        valid = sum(1 for case in self.cases if case.validation_status in ("Valid", "PASS", "Pass"))
        invalid = sum(1 for case in self.cases if case.validation_status == "Invalid")
        self.stat_cards["case_count"].set(f"{total:,}")
        self.stat_cards["valid"].set(f"{valid:,}")
        self.stat_cards["invalid"].set(f"{invalid:,}")
        self.stat_cards["duplicate_rate"].set(f"{self.last_stats.get('duplicate_rate', 0) * 100:.2f}%")
        self.stat_cards["validation_failure_rate"].set(f"{self.last_stats.get('validation_failure_rate', 0) * 100:.2f}%")
        for item in self.stat_tree.get_children():
            self.stat_tree.delete(item)
        for group, values in (("Difficulty", self.last_stats.get("difficulty_distribution", {})), ("Task Type", self.last_stats.get("type_distribution", {})), ("Split", self.last_stats.get("split_distribution", {}))):
            for key, value in values.items():
                self.stat_tree.insert("", "end", values=(f"{group}: {key}", value))
        self.last_item_analysis = self.adapter.item_analysis(self.cases, self.last_run)
        detail = {
            "context_length": self.last_stats.get("context_length", {}),
            "answer_length": self.last_stats.get("answer_length", {}),
            "duplicate_rate": self.last_stats.get("duplicate_rate", 0),
            "validation_failure_rate": self.last_stats.get("validation_failure_rate", 0),
            "item_analysis": self.last_item_analysis.get("summary", {
                "all_models_wrong_count": len(self.last_item_analysis.get("all_models_wrong", [])),
                "too_easy_count": len(self.last_item_analysis.get("too_easy", [])),
                "discriminating_count": len(self.last_item_analysis.get("discriminating", [])),
                "problematic_count": len(self.last_item_analysis.get("problematic", [])),
            }),
        }
        self.stat_detail.set(detail)

    def _refresh_report(self) -> None:
        if not hasattr(self, "report_text"):
            return
        report = {
            "benchmark_definition": self.definition_draft.to_dict(),
            "validation": self.last_validation,
            "statistics": self.last_stats,
            "item_analysis": self.last_item_analysis,
            "last_run_summary": self._run_summary(self.last_run),
            "synthetic": True,
        }
        self.report_text.set(report)

    def _run_summary(self, outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not outputs:
            return {"case_count": 0}
        scores = [float(item.get("score", 0)) for item in outputs]
        by_provider: dict[str, dict[str, Any]] = {}
        for provider in sorted({str(item.get("provider", "")) for item in outputs}):
            subset = [item for item in outputs if str(item.get("provider", "")) == provider]
            by_provider[provider] = {
                "case_count": len(subset),
                "pass_count": sum(1 for item in subset if item.get("pass", item.get("passed", False))),
                "mean_score": round(statistics.mean([float(item.get("score", 0)) for item in subset]), 4) if subset else 0,
            }
        return {
            "case_count": len(outputs),
            "provider": outputs[0].get("provider", ""),
            "pass_count": sum(1 for item in outputs if item.get("pass")),
            "pass_rate": round(sum(1 for item in outputs if item.get("pass")) / len(outputs), 4),
            "mean_score": round(statistics.mean(scores), 4),
            "mean_latency_ms": round(statistics.mean([float(item.get("latency_ms", 0)) for item in outputs]), 3),
            "by_provider": by_provider,
        }

    # -- jobs ------------------------------------------------------------

    def _start_job(self, job_name: str, operation: Callable[[], Any]) -> None:
        self.progress.start(10)
        self.status_var.set(f"Working: {job_name}…")

        def worker() -> None:
            try:
                result = operation()
                self.job_queue.put((job_name, result))
            except Exception as exc:  # pragma: no cover - defensive UI path
                self.job_queue.put(("error", (job_name, exc, traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_jobs(self) -> None:
        try:
            while True:
                name, result = self.job_queue.get_nowait()
                self.progress.stop()
                if name == "error":
                    job_name, exc, trace = result
                    self.status_var.set(f"{job_name} failed: {exc}")
                    self._append_runner_log(trace)
                    continue
                if name in ("generate", "generate_demo"):
                    self.cases = list(result)
                    self.last_validation = {}
                    self.last_run = []
                    self.status_var.set(f"Generated {len(self.cases):,} Synthetic cases")
                    self.notebook.select(self.browser_tab)
                elif name == "validate":
                    self.last_validation = dict(result)
                    self.status_var.set(f"Validation {result.get('status', '—')}: {result.get('valid', 0):,} valid / {result.get('invalid', 0):,} invalid")
                    self.notebook.select(self.export_tab)
                elif name == "run":
                    self.last_run = list(result)
                    summary = self._run_summary(self.last_run)
                    provider_bits = []
                    for provider, values in summary.get("by_provider", {}).items():
                        provider_bits.append(f"{provider} {values.get('pass_count', 0):,}/{values.get('case_count', 0):,} ({values.get('mean_score', 0):.3f})")
                    display_summary = " · ".join(provider_bits) if provider_bits else f"{summary.get('provider', '')}: {summary.get('pass_count', 0):,}/{summary.get('case_count', 0):,}"
                    self.runner_summary_var.set(f"{display_summary} · mean latency {summary.get('mean_latency_ms', 0):.2f} ms")
                    self._append_runner_log(summary)
                    self.status_var.set(f"Runner complete: {len(self.last_run):,} case outputs")
                    self.notebook.select(self.runner_tab)
                self._refresh_all()
        except queue.Empty:
            pass
        self.after(100, self._poll_jobs)

    def _append_runner_log(self, value: Any) -> None:
        current = self.runner_log.get()
        text = _safe_text(value)
        self.runner_log.set((current + "\n" if current else "") + text)

    # -- files -----------------------------------------------------------

    def _load_json_dataset(self) -> None:
        path = filedialog.askopenfilename(title="Load JSON/JSONL Dataset", filetypes=[("JSON", "*.json"), ("JSONL", "*.jsonl"), ("All files", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as handle:
                    raw = [json.loads(line) for line in handle if line.strip()]
            else:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                raw = _extract_cases(loaded)
            # A loaded file is independent from the last generated native
            # BenchmarkDataset; do not accidentally validate stale cases.
            self.adapter.last_dataset = None
            self.cases = [_canonical_case(item, i) for i, item in enumerate(raw)]
            self.status_var.set(f"Loaded {len(self.cases):,} cases from {Path(path).name}")
            self._refresh_all()
        except Exception as exc:
            self._show_error("Load failed", str(exc))

    def _export_dialog(self, fmt: str | None = None) -> None:
        if not self.cases:
            self._show_info("Export", "Generate or load a dataset first.")
            return
        if fmt is None:
            fmt = "json"
        extension = "md" if fmt == "markdown" else fmt
        path = filedialog.asksaveasfilename(title=f"Export {fmt.upper()}", defaultextension=f".{extension}", filetypes=[(fmt.upper(), f"*.{extension}"), ("All files", "*.*")])
        if not path:
            return
        try:
            metadata = {
                "name": self.definition_draft.name,
                "capability": self.definition_draft.capability,
                "seed": self.seed_var.get(),
                "generator_version": APP_VERSION,
                "dataset_hash": hashlib.sha256("".join(case.id for case in self.cases).encode()).hexdigest(),
            }
            saved = self.adapter.export(self.cases, path, fmt, metadata)
            self.status_var.set(f"Exported {len(self.cases):,} cases → {saved}")
            self._show_info("Export complete", saved)
        except Exception as exc:
            self._show_error("Export failed", str(exc))

    def _about(self) -> None:
        self._show_info("About", f"{APP_NAME}\nAI 基准测试数据集工厂\nVersion {APP_VERSION}\n\nSynthetic-only development workflow.\nNo paid model API is called by this application.")

    def _show_info(self, title: str, message: str) -> None:
        try:
            messagebox.showinfo(title, message, parent=self)
        except tk.TclError:
            self.status_var.set(message)

    def _show_error(self, title: str, message: str) -> None:
        try:
            messagebox.showerror(title, message, parent=self)
        except tk.TclError:
            self.status_var.set(f"{title}: {message}")


def run_headless_smoke(count: int = 500, seed: int = 20261004) -> dict[str, Any]:
    """Generate and validate a demo dataset without opening a window.

    This helper is useful for CI/build machines without a display and gives
    packaging scripts a single, dependency-free health check for the GUI's
    adapter path.
    """

    adapter = BackendAdapter()
    cases = adapter.generate_demo(count=count, seed=seed)
    validation = adapter.validate(cases)
    stats = adapter.stats(cases)
    return {
        "case_count": len(cases),
        "validation": validation,
        "statistics": stats,
        "synthetic": True,
    }


def main() -> None:
    """Launch the desktop application."""

    # ``--headless-smoke`` deliberately avoids Tk initialization.  It is
    # handy for CI and PyInstaller smoke checks on machines without DISPLAY.
    import sys

    if "--headless-smoke" in sys.argv:
        print(json.dumps(run_headless_smoke(), ensure_ascii=False, indent=2))
        return
    app = BenchmarkFactoryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
