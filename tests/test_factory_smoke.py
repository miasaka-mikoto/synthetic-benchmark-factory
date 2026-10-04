"""Offline acceptance tests for Synthetic Benchmark Factory.

These tests intentionally exercise the data path rather than the desktop UI:
generate -> validate -> serialize -> reload -> evaluate.  They use only the
local generator and providers and therefore never consume API credits.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from synthetic_benchmark_factory.generator import BenchmarkGenerator, generate_demo_dataset
from synthetic_benchmark_factory.models import BenchmarkCase, BenchmarkDataset, DatasetSplit, ValidationStatus
from synthetic_benchmark_factory.schema import is_valid, validate_dataset
from synthetic_benchmark_factory.serialization import from_json, to_json


def _canonical(dataset) -> str:
    return json.dumps(dataset.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_demo_generation_is_reproducible() -> None:
    left = generate_demo_dataset(count=40, seed=1729)
    right = generate_demo_dataset(count=40, seed=1729)
    assert _canonical(left) == _canonical(right)
    assert hashlib.sha256(_canonical(left).encode()).hexdigest() == hashlib.sha256(_canonical(right).encode()).hexdigest()


def test_demo_has_ground_truth_and_synthetic_marker() -> None:
    dataset = generate_demo_dataset(count=80, seed=11)
    assert dataset.case_count == 80
    assert dataset.cases
    assert all(case.synthetic is True for case in dataset.cases)
    assert all(case.ground_truth is not None for case in dataset.cases)
    assert all(isinstance(case.generator.seed, int) for case in dataset.cases)
    assert dataset.metadata["seed"] == 11
    assert all(case.validation_status in (ValidationStatus.PENDING, ValidationStatus.VALID) for case in dataset.cases)


def test_demo_covers_splits_and_real_difficulty_parameters() -> None:
    dataset = generate_demo_dataset(count=500, seed=20261004)
    splits = {str(case.split.value if hasattr(case.split, "value") else case.split) for case in dataset.cases}
    assert {"train", "validation", "test", "challenge"}.issubset(splits)
    # The demo defaults to a middle difficulty, while the generator contract
    # must still make all five levels materially different.
    level_params = []
    for level in range(1, 6):
        generated = BenchmarkGenerator(
            definition={"name": f"memory-l{level}", "capability": "memory", "task_type": "memory", "difficulty": level},
            seed=20261004,
        ).generate(count=1)
        p = generated.cases[0].difficulty_params
        level_params.append((p.context_length, p.distractor_count, p.number_of_facts, p.conflicting_facts, p.time_distance))
    assert len(set(level_params)) == 5


def test_validator_accepts_reference_dataset() -> None:
    dataset = generate_demo_dataset(count=120, seed=23)
    result = validate_dataset(dataset)
    assert is_valid(result), result


def test_validator_rejects_duplicate_case() -> None:
    dataset = generate_demo_dataset(count=8, seed=5)
    duplicate = dataset.cases[0]
    bad = BenchmarkDataset(
        name=dataset.name,
        definitions=dataset.definitions,
        version=dataset.version,
        cases=[*dataset.cases, duplicate],
    )
    result = validate_dataset(bad)
    assert not is_valid(result)


def test_json_round_trip_preserves_cases() -> None:
    dataset = generate_demo_dataset(count=12, seed=91)
    restored = from_json(to_json(dataset), BenchmarkDataset)
    assert restored.case_count == dataset.case_count
    assert [case.case_id for case in restored.cases] == [case.case_id for case in dataset.cases]
    assert [case.ground_truth for case in restored.cases] == [case.ground_truth for case in dataset.cases]


def test_cli_smoke_writes_exports(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    command = [sys.executable, "run_demo.py", "--count", "20", "--seed", "7", "--out", str(output)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for filename in ("dataset.json", "dataset.jsonl", "dataset.csv", "dataset.md", "manifest.json", "report.json", "report.md", "evaluation_report.json"):
        assert (output / filename).is_file(), filename
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["dataset_hash"].startswith("sha256:")
    evaluation = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
    assert evaluation["statistics"]["valid_case_count"] == 20
    assert evaluation["statistics"]["invalid_case_count"] == 0
