"""Additional offline acceptance checks for the actual benchmark data path."""

from __future__ import annotations

from synthetic_benchmark_factory.analysis import analyze_items
from synthetic_benchmark_factory.generator import BenchmarkGenerator, generate_demo_dataset
from synthetic_benchmark_factory.serialization import dataset_hash
from synthetic_benchmark_factory.validator import validate_dataset
from synthetic_benchmark_factory.versioning import verify_dataset_hash
from synthetic_benchmark_factory.runner import BenchmarkRunner


TASK_TYPES = (
    "extraction",
    "classification",
    "memory",
    "planning",
    "tool_selection",
    "arithmetic",
    "structured_output",
    "instruction_following",
)


def _definition(task_type: str) -> dict:
    return {
        "name": f"{task_type} acceptance",
        "capability": task_type,
        "task_type": task_type,
        "difficulty": 3,
        "input_schema": {},
        "expected_output": {},
        "scoring": {"method": "exact_match"},
        "constraints": {},
    }


def test_all_first_release_types_generate_valid_unique_cases() -> None:
    for task_type in TASK_TYPES:
        dataset = BenchmarkGenerator(_definition(task_type), seed=101).generate(count=40)
        report = validate_dataset(dataset)
        assert report.valid, (task_type, report.to_dict())
        assert report.duplicate_count == 0
        assert report.leakage_count == 0
        assert all(case.ground_truth is not None for case in dataset.cases)


def test_semantic_validator_rejects_duplicate_and_cross_split_leakage() -> None:
    dataset = generate_demo_dataset(count=8, seed=77)
    duplicate = dataset.cases[0]
    dataset.cases.append(duplicate)
    report = validate_dataset(dataset)
    assert not report.valid
    assert report.duplicate_count >= 1

    leaked = generate_demo_dataset(count=4, seed=78)
    leaked.cases[1].prompt = leaked.cases[0].prompt
    leaked.cases[1].context = leaked.cases[0].context
    leaked.cases[1].ground_truth = leaked.cases[0].ground_truth
    leaked.cases[1].expected_output = leaked.cases[0].expected_output
    leaked.cases[0].split = "train"
    leaked.cases[1].split = "test"
    leaked_report = validate_dataset(leaked)
    assert not leaked_report.valid
    assert leaked_report.leakage_count >= 1


def test_runner_and_item_analysis_keep_provider_provenance() -> None:
    dataset = generate_demo_dataset(count=12, seed=91)
    report = BenchmarkRunner().run_dataset(dataset)
    assert {"MockModel", "RuleModel"}.issubset(set(report.providers))
    assert all(record.synthetic for record in report.records)
    assert all(record.provider in {"MockModel", "RuleModel"} for record in report.records)
    analysis = analyze_items(report.records, cases=dataset.cases)
    assert analysis.item_count == dataset.case_count
    assert analysis.summary["provider_count"] == 2


def test_hash_survives_validation_bookkeeping() -> None:
    dataset = generate_demo_dataset(count=16, seed=333)
    before = dataset_hash(dataset)
    assert verify_dataset_hash(dataset)
    validate_dataset(dataset)
    assert dataset_hash(dataset) == before
    assert verify_dataset_hash(dataset)

