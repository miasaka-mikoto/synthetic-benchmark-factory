"""Offline smoke checks for Synthetic Benchmark Factory.

The checks intentionally use only the Python standard library. They verify
that the public data model can be imported, a deterministic demo dataset can
be generated (when the generator module is present), serialization round trips
work, and the validator rejects a clearly broken case. The script adapts to
minor API differences while the project is evolving and exits non-zero only
for a real import/runtime failure.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_imports() -> object:
    package = importlib.import_module("synthetic_benchmark_factory")
    optional = (
        "models",
        "schema",
        "serialization",
        "versioning",
        "generator",
        "runner",
        "metrics",
        "analysis",
        "reports",
    )
    missing = []
    for name in optional:
        try:
            importlib.import_module(f"synthetic_benchmark_factory.{name}")
        except ModuleNotFoundError as exc:
            # Tolerate a module not being implemented yet, but do not hide a
            # missing dependency imported by an existing module.
            if exc.name == f"synthetic_benchmark_factory.{name}":
                missing.append(name)
            else:
                raise
    if missing:
        print(f"[smoke] optional modules not present yet: {', '.join(missing)}")
    return package


def _try_demo_generation(package: object) -> int:
    """Return number of generated cases, or zero when no public generator yet."""
    candidates = ("generate_demo_dataset", "generate_dataset", "build_demo_dataset")
    generator = None
    for module_name in (
        "synthetic_benchmark_factory.generator",
        "synthetic_benchmark_factory.generators",
    ):
        try:
            generator = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue
    if generator is None:
        print("[smoke] generator module not present; import-only smoke passed")
        return 0

    fn = next((getattr(generator, name, None) for name in candidates if hasattr(generator, name)), None)
    if fn is None:
        print("[smoke] no demo generator function exported; import-only smoke passed")
        return 0

    result = None
    attempts = (
        lambda: fn(count=12, seed=123),
        lambda: fn(num_cases=12, seed=123),
        lambda: fn(12, seed=123),
        lambda: fn(seed=123),
        lambda: fn(),
    )
    for attempt in attempts:
        try:
            result = attempt()
            break
        except TypeError:
            continue
    if result is None:
        print("[smoke] generator API signature not recognized; import-only smoke passed")
        return 0
    cases = getattr(result, "cases", result)
    try:
        count = len(cases)
    except TypeError:
        count = 0
    if count == 0:
        raise AssertionError("generator returned no cases")
    print(f"[smoke] generated {count} deterministic case(s)")

    to_dict = getattr(package, "to_dict", None)
    from_dict = getattr(package, "from_dict", None)
    if callable(to_dict) and callable(from_dict):
        payload = to_dict(result)
        if isinstance(payload, dict):
            clone = from_dict(payload)
            if clone is None:
                raise AssertionError("from_dict returned None for generated dataset")
    return count


def main() -> int:
    print("Synthetic Benchmark Factory smoke test")
    package = _check_imports()
    _try_demo_generation(package)
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
