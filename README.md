# Synthetic Benchmark Factory

**AI 基准测试数据集工厂** is an offline-first desktop and Python toolkit for
creating, validating, running, and inspecting reproducible synthetic benchmark
datasets. It is intended for experiments with capabilities such as long-term
memory, instruction following, tool selection, extraction, planning, arithmetic,
classification, and structured output.

The first demo is a 500-case **Long-Term Memory Benchmark**. Every case has a
deterministic ground truth, a split, difficulty parameters, generator metadata,
and validation status. Synthetic cases and model outputs are explicitly labelled
so a MockModel result is never presented as a real-LLM benchmark result.

## Design promises

* No paid model API is called by the generator, validator, demo, or tests.
* A seed, generator version, and canonical configuration reproduce the same
  dataset and dataset hash.
* Cases with duplicate IDs, ambiguous/impossible answers, invalid schemas, or
  leakage are rejected before export.
* Difficulty is controlled by task parameters (context length, facts,
  distractors, conflicts, and time distance), not by a label alone.
* Export formats are JSONL, JSON, CSV, and Markdown.

## Quick start (Windows)

The project runs with the Python 3.11+ standard library. From a checkout:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python run_demo.py --count 500 --seed 20261004 --out artifacts\demo
python -m pytest -q
```

The demo command creates a versioned output directory containing:

* `dataset.jsonl` — a compact manifest record followed by one fully structured case per line;
* `dataset.json` — the complete dataset and manifest;
* `dataset.csv` and `dataset.md` — inspection-friendly exports;
* `report.json` and `report.md` — validation and statistics reports;
* `manifest.json` — seed, generator version, config, and dataset hash;
* `runs.json`, `item_analysis.json/.md` — local MockModel/RuleModel runs and
  per-item analysis (clearly marked synthetic/provider-specific);
* `evaluation_report.json/.md` — combined evaluation report.

No network access or API key is required. If a future provider is configured,
the provider name is recorded on every model output and in the report.

## CLI

The canonical offline demo is:

```text
python run_demo.py [--count N] [--seed INTEGER] [--out PATH]
                   [--split-ratios TRAIN VALIDATION TEST CHALLENGE]
                   [--difficulty MIN MAX]
```

The package CLI is also available when installed:

```text
python -m synthetic_benchmark_factory.cli --help

# desktop editor
python app.py

# CI/build-machine check without a display
python app.py --headless-smoke
```

Use `--count 500 --seed 20261004` for the reference demo. Running it twice with
the same arguments must produce the same manifest hash and case content.

## Project layout

```text
synthetic_benchmark_factory/
  models.py          # benchmark definitions, cases, versions, outputs
  generator.py       # procedural generators and seed handling
  validator.py       # semantic duplicate/leakage/schema/answer checks
  runner.py          # ModelProvider, MockModel, RuleModel, run records
  metrics.py         # exact match, F1, schema, task-specific metrics
  providers.py       # offline MockModel/RuleModel provider adapters
  serialization.py   # canonical JSON/JSONL/CSV/Markdown and hashes
  versioning.py      # reproducible dataset manifests
  analysis.py        # item analysis (wrong/easy/discriminating/problematic)
  reports.py         # statistics and item analysis
  cli.py             # command-line entry points
run_demo.py          # one-command reference dataset generation
tests/               # deterministic unit/integration tests
examples/            # benchmark definitions and config files
docs/                # test reports and data dictionary
artifacts/           # generated output (ignored by source control)
```

The desktop editor uses the same package APIs; it is a view over definitions,
cases, validators, runners, and reports rather than a second data model.

## UI preview

The checked-in preview shows the implemented editor layout, Case Browser,
validation/quality panel, split filters, and explicit Synthetic provider labels:
`artifacts/ui_preview.png` (the SVG source is beside it).

## Data model

### Benchmark definition

`Name`, `Capability`, `Description`, `Difficulty`, `Task Type`, `Input Schema`,
`Expected Output`, `Scoring`, and `Constraints` define what a benchmark tests.
Definitions also carry a generator name/version and a seed policy.

### Case

Each case contains a stable `case_id`, `split` (`train`, `validation`, `test`,
or `challenge`), prompt/context/input, expected answer (ground truth),
difficulty level and parameters, distractors, generator metadata, and
validation status. A case cannot be exported as a formal benchmark item unless
its ground truth is deterministic and validation passes.

### Model output

Outputs include `provider` (`MockModel`, `RuleModel`, or a future provider),
provider version, raw output, latency, token counts, and evaluation metrics.
Synthetic data is marked `data_origin: Synthetic`; provider results are never
silently merged into the ground-truth fields.

## Reproducibility and versioning

The manifest records:

```json
{
  "benchmark_version": "...",
  "seed": 20261004,
  "generator_version": "...",
  "config": {...},
  "dataset_hash": "sha256:..."
}
```

Hashes are computed from canonical JSON (sorted keys, stable separators, no
timestamps). Do not edit generated case files by hand; create a new version
instead. A changed generator or config must produce a changed version/hash.

## Validation and reports

Validation checks include duplicate IDs/content, broken or impossible cases,
ambiguous answers, invalid input/output schemas, and train/test data leakage.
Reports include case counts, split/type/difficulty distributions, context and
answer lengths, duplicate and validation-failure rates, and item analysis:

* all models wrong;
* too-easy items;
* high-discrimination items;
* suspicious/problematic items.

The report identifies the generator and provider for every finding, making it
possible to distinguish a generator defect from a model failure.

## Formal benchmark hygiene

The included `MockModel` and `RuleModel` are development fixtures. They are
useful for exercising the complete scientific loop, but their scores must not
be reported as scores from a real LLM. Keep generated data and model-run
outputs in separate directories when publishing results.

## Building a Windows executable

The supported Windows build uses the desktop editor entry point and includes
all dynamic engine modules:

```powershell
.\build_windows.ps1 -Clean -RunTests
```

With PyInstaller installed, this produces `dist\BenchmarkFactory.exe`; when
PyInstaller is unavailable the script creates a portable source bundle and
explains how to install the optional build dependency. The executable opens
the Benchmark Editor, Generator, Case Browser, Statistics, Runner and Export
views. `BenchmarkFactory.spec` can also be passed directly to PyInstaller.
The provided workspace is Linux-based, so the native Windows `.exe` is not
claimed as a locally runnable artifact here; run the same script on Windows to
produce and smoke-test `dist\BenchmarkFactory.exe`.

## Development checks

```powershell
python -m compileall synthetic_benchmark_factory run_demo.py
python -m pytest -q
python run_demo.py --count 20 --seed 7 --out artifacts\smoke
```

The smoke run is intentionally small; the reference acceptance run uses 500
cases. A failed validation check should fail the command and leave a report
describing the offending case IDs. Fix the generator, then regenerate with a
new generator version and rerun the acceptance suite.

## License and scope

This is a local research and engineering tool. It does not claim that
synthetic benchmarks predict real-world model behavior, and it does not call
paid services during development.
