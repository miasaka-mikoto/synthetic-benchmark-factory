"""Small offline CLI for inspecting and exporting benchmark datasets.

Generators and desktop UI can import the same functions directly.  This CLI
is intentionally conservative: it never invokes a model or network service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from . import __version__
from .models import BenchmarkDefinition
from .schema import validate_dataset as validate_schema_dataset
from .validator import validate_dataset as validate_semantic_dataset
from .serialization import (
    dataset_hash,
    read_dataset,
    to_json,
    write_dataset,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthetic-benchmark-factory", description="Offline Synthetic Benchmark Factory tools")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    new_def = sub.add_parser("new-definition", help="create a benchmark definition JSON")
    new_def.add_argument("--name", required=True)
    new_def.add_argument("--capability", required=True)
    new_def.add_argument("--description", default="")
    new_def.add_argument("--task-type", default="custom")
    new_def.add_argument("--difficulty", default="1")
    new_def.add_argument("--output", "-o", type=Path, required=True)

    validate = sub.add_parser("validate", help="validate a JSON/JSONL dataset")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--strict", action="store_true", help="return nonzero for warnings too")

    stats = sub.add_parser("stats", help="print dataset statistics")
    stats.add_argument("dataset", type=Path)

    hash_cmd = sub.add_parser("hash", help="print the deterministic dataset hash")
    hash_cmd.add_argument("dataset", type=Path)

    export = sub.add_parser("export", help="convert a dataset to JSON/JSONL/CSV/Markdown")
    export.add_argument("dataset", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--format", choices=("json", "jsonl", "csv", "markdown"), default=None)
    export.add_argument("--case-only-jsonl", action="store_true", help="omit the JSONL manifest record")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "new-definition":
        definition = BenchmarkDefinition(
            name=args.name,
            capability=args.capability,
            description=args.description,
            task_type=args.task_type,
            difficulty=args.difficulty,
            scoring={"exact_match": True},
        )
        write_json(definition, args.output)
        print(args.output)
        return 0
    dataset = read_dataset(args.dataset)
    if args.command == "validate":
        # The semantic validator wraps schema checks and additionally catches
        # duplicate content, cross-split leakage, ambiguous/impossible cases,
        # and task-specific ground-truth problems.  Keep a schema fallback so
        # the CLI remains useful with a minimal copied package.
        try:
            issues = validate_semantic_dataset(dataset)
            payload = issues.to_dict() if hasattr(issues, "to_dict") else {"valid": bool(getattr(issues, "valid", False)), "issues": []}
            valid = bool(getattr(issues, "valid", payload.get("valid", False)))
            warning_count = len(getattr(issues, "warnings", []) or [])
        except Exception:
            issues = validate_schema_dataset(dataset)
            payload = {"valid": issues.valid, "issues": issues.to_dict()}
            valid = bool(issues.valid)
            warning_count = len(getattr(issues, "warnings", []) or [])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if (not valid or (args.strict and warning_count)) else 0
    if args.command == "stats":
        print(to_json(dataset.compute_statistics(), indent=2))
        return 0
    if args.command == "hash":
        print(dataset_hash(dataset))
        return 0
    if args.command == "export":
        write_dataset(dataset, args.output, format=args.format, include_manifest=not args.case_only_jsonl)
        print(args.output)
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
