#!/usr/bin/env python3
"""Convert a generated dataset to one of the supported export formats."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow ``python scripts/export_dataset.py`` from a source checkout without
# installing the package first (the Windows .bat/PowerShell wrappers use the
# same convenience).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_benchmark_factory.serialization import (
    read_dataset,
    write_csv,
    write_json,
    write_jsonl,
    write_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Synthetic Benchmark Factory dataset")
    parser.add_argument("input", type=Path, help="dataset.json, dataset.jsonl, or another supported source")
    parser.add_argument("output", type=Path, help="destination file")
    parser.add_argument("--format", choices=("json", "jsonl", "csv", "markdown"), help="format (defaults to output suffix)")
    args = parser.parse_args()
    fmt = args.format or {".json": "json", ".jsonl": "jsonl", ".csv": "csv", ".md": "markdown", ".markdown": "markdown"}.get(args.output.suffix.lower())
    if not fmt:
        parser.error("cannot infer format; pass --format")
    dataset = read_dataset(args.input)
    writers = {
        "json": write_json,
        "jsonl": write_jsonl,
        "csv": write_csv,
        "markdown": write_markdown,
    }
    writers[fmt](dataset, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
