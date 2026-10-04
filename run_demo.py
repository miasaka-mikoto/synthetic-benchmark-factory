#!/usr/bin/env python3
"""Generate and validate the offline Long-Term Memory reference dataset.

This script intentionally has no network or model-provider dependency.  It is
kept as a small, boring entry point so it can be run from a source checkout or
bundled as ``BenchmarkFactory.exe``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def _jsonable(value: Any) -> Any:
    """Convert package dataclasses/enums to deterministic JSON-compatible data."""
    try:
        from synthetic_benchmark_factory.serialization import to_dict

        converted = to_dict(value)
        if converted is not value:
            return converted
    except Exception:
        pass
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


def _canonical(value: Any) -> str:
    """Canonicalize generated data without volatile bookkeeping fields.

    Dataclass instances carry ``created_at`` timestamps for UI provenance.
    Those timestamps must not participate in a dataset hash: two runs with
    the same definition/seed/generator are expected to produce the same
    content hash.  The generator also stores a convenience hash on the
    dataset itself; remove that prior value before calculating the new one.
    """

    def strip_volatile(node: Any, *, root: bool = False) -> Any:
        if isinstance(node, dict):
            result = {}
            for key, item in node.items():
                key = str(key)
                if key in {"created_at", "updated_at"}:
                    continue
                if root and key in {"dataset_hash", "hash", "benchmark_version"}:
                    continue
                result[key] = strip_volatile(item, root=False)
            return result
        if isinstance(node, list):
            return [strip_volatile(item, root=False) for item in node]
        return node

    normalized = strip_volatile(_jsonable(value), root=True)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reproducible_payload(dataset: Any) -> Any:
    """Return content used for a dataset hash, excluding wall-clock fields.

    Case creation timestamps are useful audit metadata but must not make a
    seeded dataset change on every invocation.  The version's own hash is also
    excluded to avoid hashing a value that contains itself.
    """
    raw = _jsonable(dataset)

    def clean(value: Any, *, in_version: bool = False) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if key in {"created_at", "dataset_hash", "statistics"}:
                    continue
                # benchmark_version is derived from the hash in the core model;
                # the config/seed/generator fields below are the source of truth.
                if in_version and key in {"benchmark_version"}:
                    continue
                cleaned[key] = clean(item, in_version=(in_version or key == "version"))
            return cleaned
        if isinstance(value, list):
            return [clean(item, in_version=in_version) for item in value]
        return value

    return clean(raw)


def _dataset_cases(dataset: Any) -> list[Any]:
    if isinstance(dataset, (list, tuple)):
        return list(dataset)
    for attr in ("cases", "items", "records"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return list(value)
    if isinstance(dataset, dict):
        for key in ("cases", "items", "records"):
            if key in dataset:
                return list(dataset[key])
    return []


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _validate(dataset: Any) -> dict[str, Any]:
    """Use the package validator when present, with a conservative fallback."""
    issues: list[Any] = []
    try:
        # Prefer the semantic validator: it includes schema checks plus
        # duplicate, cross-split leakage, ambiguity, impossibility and
        # task-specific checks.  The schema validator remains a safe fallback
        # for a minimal source bundle.
        try:
            from synthetic_benchmark_factory.validator import Validator

            result = Validator().validate_dataset(dataset)
        except ImportError:
            from synthetic_benchmark_factory.schema import validate_dataset

            result = validate_dataset(dataset)
        if isinstance(result, bool):
            return {"valid": result, "issues": [] if result else ["schema validator returned false"]}
        if isinstance(result, dict):
            return result
        if hasattr(result, "to_dict"):
            try:
                report_payload = result.to_dict()
                if isinstance(report_payload, dict):
                    # Keep the compact top-level shape expected by the CLI
                    # while retaining semantic counters for the report.
                    report_payload.setdefault("valid", bool(getattr(result, "valid", False)))
                    return report_payload
            except Exception:
                pass
        issues = list(getattr(result, "issues", []) or [])
        valid = bool(getattr(result, "valid", not issues))
        return {"valid": valid, "issues": [_jsonable(x) for x in issues]}
    except Exception as exc:
        # The fallback catches the most dangerous errors even when this script
        # is copied alongside an older package build.
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []
        for case in _dataset_cases(dataset):
            cid = str(_field(case, "case_id", ""))
            if cid in seen_ids:
                duplicate_ids.append(cid)
            seen_ids.add(cid)
            if not _field(case, "expected_output", _field(case, "ground_truth", None)):
                issues.append({"case_id": cid, "code": "missing_ground_truth"})
        if duplicate_ids:
            issues.append({"code": "duplicate_case_id", "ids": duplicate_ids})
        # Import errors are recorded, not hidden; generated data remains useful
        # for debugging while the process exits non-zero below.
        issues.append({"code": "validator_unavailable", "message": str(exc)})
        return {"valid": not issues, "issues": [_jsonable(x) for x in issues]}


def _mark_case_validation(cases: Iterable[Any]) -> None:
    """Stamp each generated case with its own validation result.

    Dataset-level validation is useful for a pass/fail gate, but leaving every
    item as ``unvalidated`` makes the exported validation-failure statistic
    misleading.  This per-case pass keeps the Case Browser and reports honest.
    """

    try:
        from synthetic_benchmark_factory.schema import validate_benchmark_case
    except Exception:
        return
    for case in cases:
        try:
            result = validate_benchmark_case(case)
            errors = list(getattr(result, "errors", []) or [])
            # Older validators return a plain list of issues.
            if not errors and isinstance(result, (list, tuple)):
                errors = [issue for issue in result if getattr(issue, "is_error", True)]
            setattr(case, "validation_status", "invalid" if errors else "valid")
            setattr(case, "validation_errors", [_jsonable(issue) for issue in errors])
        except Exception as exc:
            setattr(case, "validation_status", "invalid")
            setattr(case, "validation_errors", [{"code": "validator_exception", "message": str(exc)}])


def _statistics(cases: Iterable[Any]) -> dict[str, Any]:
    rows = list(cases)
    split = Counter(_enum_text(_field(c, "split", "")) for c in rows)
    kind = Counter(_enum_text(_field(c, "benchmark_type", _field(c, "task_type", ""))) for c in rows)
    difficulty = Counter(_enum_text(_field(c, "difficulty", "")) for c in rows)
    context_lengths = [len(str(_field(c, "context", ""))) for c in rows]
    answer_lengths = [len(str(_field(c, "expected_output", _field(c, "ground_truth", "")))) for c in rows]
    ids = [str(_field(c, "case_id", "")) for c in rows]
    duplicate_rate = 1.0 - (len(set(ids)) / len(ids)) if ids else 0.0
    failures = sum(1 for c in rows if _enum_text(_field(c, "validation_status", "")) not in ("", "valid", "passed", "pass"))

    def summary(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "p50": 0}
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "max": ordered[-1],
            "mean": round(sum(values) / len(values), 3),
            "p50": ordered[(len(ordered) - 1) // 2],
        }

    return {
        "case_count": len(rows),
        "split_distribution": dict(sorted(split.items())),
        "type_distribution": dict(sorted(kind.items())),
        "difficulty_distribution": dict(sorted(difficulty.items())),
        "context_length": summary(context_lengths),
        "answer_length": summary(answer_lengths),
        "duplicate_rate": round(duplicate_rate, 6),
        "validation_failure_rate": round(failures / len(rows), 6) if rows else 0.0,
    }


def _write_exports(out_dir: Path, dataset: Any, cases: list[Any], manifest: dict[str, Any], validation: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _jsonable(dataset)
    if isinstance(data, dict):
        full = dict(data)
        # Keep dataset.json directly readable by BenchmarkDataset.from_dict.
        # The human-readable manifest is emitted separately below.
        version = dict(full.get("version") or {})
        version.update(
            {
                "benchmark_version": manifest["benchmark_version"],
                "seed": manifest["seed"],
                "generator_version": manifest["generator_version"],
                "config": manifest["config"],
                "dataset_hash": manifest["dataset_hash"],
                "case_count": len(cases),
            }
        )
        full["version"] = version
    else:
        full = {"cases": data, "version": manifest}
    (out_dir / "dataset.json").write_text(json.dumps(full, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Keep a compact manifest record on the first line so ``read_dataset`` can
    # round-trip split/version metadata without duplicating all cases there.
    manifest_dataset = {key: full[key] for key in ("name", "definitions", "version", "metadata") if key in full}
    with (out_dir / "dataset.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"record_type": "manifest", "dataset": manifest_dataset}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        for case in cases:
            handle.write(json.dumps(_jsonable(case), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    # CSV deliberately keeps a stable, human-readable subset.  The complete
    # nested structures remain available in JSON/JSONL.
    columns = ["case_id", "benchmark_type", "split", "difficulty", "prompt", "context", "expected_output", "validation_status"]
    with (out_dir / "dataset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow({column: _enum_text(_field(case, column, "")) for column in columns})

    lines = ["# Synthetic Benchmark Factory dataset", "", f"Cases: {len(cases)}", f"Dataset hash: `{manifest['dataset_hash']}`", "", "| Case | Type | Split | Difficulty | Validation |", "|---|---|---|---|---|"]
    for case in cases:
        lines.append("| {case_id} | {benchmark_type} | {split} | {difficulty} | {validation_status} |".format(**{column: _enum_text(_field(case, column, "")) for column in columns}))
    (out_dir / "dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({"manifest": manifest, "validation": validation, "statistics": _statistics(cases)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_lines = ["# Validation report", "", f"- Valid: **{validation.get('valid', False)}**", f"- Cases: **{len(cases)}**", f"- Dataset hash: `{manifest['dataset_hash']}`", "", "## Statistics", "", "```json", json.dumps(_statistics(cases), ensure_ascii=False, indent=2, sort_keys=True), "```", "", "## Issues", ""]
    issues = validation.get("issues", [])
    report_lines.extend([f"- `{json.dumps(_jsonable(issue), ensure_ascii=False, sort_keys=True)}`" for issue in issues] or ["- None"])
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _write_local_run_artifacts(out_dir: Path, dataset: Any) -> None:
    """Run the two bundled offline providers and save item analysis.

    This is deliberately separate from the formal dataset export: provider
    outputs are experimental observations, not ground truth.  Keeping
    ``attach=False`` also means rerunning a provider cannot mutate the dataset
    hash or accidentally publish a MockModel answer as benchmark truth.
    """
    try:
        from synthetic_benchmark_factory.analysis import analyze_items
        from synthetic_benchmark_factory.reports import build_report, export_report
        from synthetic_benchmark_factory.runner import BenchmarkRunner

        run_report = BenchmarkRunner().run_dataset(dataset, attach=False)
        records = [record.to_dict() if hasattr(record, "to_dict") else _jsonable(record) for record in run_report.records]
        (out_dir / "runs.json").write_text(
            json.dumps({"providers": run_report.providers, "summary": run_report.summary, "records": records, "synthetic": True}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        analysis = analyze_items(records, cases=_dataset_cases(dataset))
        analysis_dict = analysis.to_dict() if hasattr(analysis, "to_dict") else _jsonable(analysis)
        (out_dir / "item_analysis.json").write_text(json.dumps(analysis_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        markdown = analysis.to_markdown() if hasattr(analysis, "to_markdown") else "# Item Analysis\n"
        (out_dir / "item_analysis.md").write_text(markdown, encoding="utf-8")
        evaluation_report = build_report(dataset, run_report, item_analysis=analysis)
        export_report(evaluation_report, out_dir / "evaluation_report.json", format="json")
        export_report(evaluation_report, out_dir / "evaluation_report.md", format="markdown", include_records=False)
    except Exception as exc:
        # The core dataset remains usable if an optional runner integration is
        # unavailable in a minimal build; leave an explicit diagnostic file.
        (out_dir / "run_error.txt").write_text(f"Offline runner unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")


def generate(
    count: int,
    seed: int,
    out_dir: Path,
    *,
    split_ratios: list[float] | None = None,
    difficulty_range: tuple[int, int] | None = None,
) -> int:
    try:
        from synthetic_benchmark_factory.generator import BenchmarkGenerator, generate_demo_dataset

        if split_ratios is None and difficulty_range is None:
            dataset = generate_demo_dataset(count=count, seed=seed)
        else:
            # Keep the demo definition identical to generate_demo_dataset while
            # exposing reproducible CLI knobs for split/difficulty experiments.
            definition = {
                "name": "Long-Term Memory Benchmark",
                "capability": "long-term memory",
                "description": "Recall stable and updated facts after temporal and irrelevant context.",
                "difficulty": 3,
                "task_type": "memory",
                "input_schema": {"context": "string", "question": "string"},
                "expected_output": {"answer": "string", "evidence_fact_id": "string"},
                "scoring": {"method": "exact_match", "fields": ["answer", "evidence_fact_id"]},
                "constraints": ["Use the most recent non-conflicting fact.", "Return JSON only."],
            }
            schedule = None
            if difficulty_range is not None:
                schedule = list(range(difficulty_range[0], difficulty_range[1] + 1))
            dataset = BenchmarkGenerator(definition=definition, seed=seed).generate(
                count=count,
                seed=seed,
                split_ratios=(
                    {"train": split_ratios[0], "validation": split_ratios[1], "test": split_ratios[2], "challenge": split_ratios[3]}
                    if split_ratios is not None
                    else None
                ),
                difficulty_schedule=schedule,
            )
    except ImportError:
        from synthetic_benchmark_factory.generator import BenchmarkGenerator

        dataset = BenchmarkGenerator(seed=seed).generate(count=count)

    cases = _dataset_cases(dataset)
    _mark_case_validation(cases)
    validation = _validate(dataset)
    # ``BenchmarkDataset.compute_statistics`` is called during generation, but
    # at that point cases are still ``unvalidated``.  Recompute after the
    # semantic validator stamps each case so reports and the Case Browser show
    # the truthful valid/invalid counts (without affecting the content hash;
    # statistics are excluded from canonical hashing).
    try:
        recompute_statistics = getattr(dataset, "compute_statistics", None)
        if callable(recompute_statistics):
            recompute_statistics()
    except Exception:
        # Statistics are a convenience projection; a validation result should
        # never be hidden because an older/minimal dataset object lacks this
        # optional method.
        pass
    # Normalize the manifest fields before hashing so the package CLI's
    # ``hash`` command and the generated manifest report the same value.
    generator_version = "1.0.0"
    try:
        version = getattr(dataset, "version")
        version.benchmark_version = "long-term-memory-demo-v1"
        version.seed = seed
        version.generator_version = generator_version
        version.config = {
            "count": count,
            "benchmark_type": "Memory",
            "data_origin": "Synthetic",
            **({"split_ratios": split_ratios} if split_ratios is not None else {}),
            **({"difficulty_range": list(difficulty_range)} if difficulty_range is not None else {}),
        }
        version.case_count = len(cases)
        version.dataset_hash = ""
    except Exception:
        pass
    try:
        from synthetic_benchmark_factory.serialization import dataset_hash as package_dataset_hash

        digest = package_dataset_hash(dataset)
    except Exception:
        canonical = json.dumps(_reproducible_payload(dataset), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest = {
        "benchmark_version": "long-term-memory-demo-v1",
        "seed": seed,
        "generator_version": generator_version,
        "config": {
            "count": count,
            "benchmark_type": "Memory",
            "data_origin": "Synthetic",
            **({"split_ratios": split_ratios} if split_ratios is not None else {}),
            **({"difficulty_range": list(difficulty_range)} if difficulty_range is not None else {}),
        },
        "dataset_hash": f"sha256:{digest}",
    }
    try:
        dataset.version.dataset_hash = manifest["dataset_hash"].removeprefix("sha256:")
    except Exception:
        pass
    _write_exports(out_dir, dataset, cases, manifest, validation)
    _write_local_run_artifacts(out_dir, dataset)
    print(json.dumps({"out": str(out_dir), "cases": len(cases), "valid": validation.get("valid", False), "dataset_hash": manifest["dataset_hash"]}, ensure_ascii=False))
    return 0 if validation.get("valid", False) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the offline Long-Term Memory benchmark demo")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "demo")
    parser.add_argument("--split-ratios", type=float, nargs=4, metavar=("TRAIN", "VALIDATION", "TEST", "CHALLENGE"), help="four split ratios")
    parser.add_argument("--difficulty", type=int, nargs=2, metavar=("MIN", "MAX"), help="inclusive difficulty range (1-5)")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.split_ratios is not None and any(value < 0 for value in args.split_ratios):
        parser.error("--split-ratios cannot contain negative values")
    if args.split_ratios is not None and sum(args.split_ratios) <= 0:
        parser.error("--split-ratios must contain a positive value")
    difficulty_range = tuple(args.difficulty) if args.difficulty is not None else None
    if difficulty_range is not None and not (1 <= difficulty_range[0] <= difficulty_range[1] <= 5):
        parser.error("--difficulty must be an inclusive range from 1 to 5")
    return generate(args.count, args.seed, args.out, split_ratios=args.split_ratios, difficulty_range=difficulty_range)


if __name__ == "__main__":
    raise SystemExit(main())
