# Offline acceptance test report

This report is produced by the local acceptance suite. It is intentionally
provider-free: no paid model API, network call, or API key is required.

## Commands

```text
python -m compileall synthetic_benchmark_factory run_demo.py
python -m pytest -q
python run_demo.py --count 500 --seed 20261004 --out artifacts/demo
```

## Acceptance gates

| Gate | Expected result |
|---|---|
| Reproducibility | Same seed/config gives the same case content and dataset hash. |
| Ground truth | Every formal case has a deterministic `ground_truth`. |
| Splits | Train/validation/test/challenge are present and recorded. |
| Difficulty | Levels 1–5 alter concrete generation parameters. |
| Validation | Duplicate IDs/content, broken references, invalid schemas, leakage, ambiguous/impossible cases are detected. |
| Providers | Mock/Rule outputs are labelled; no paid provider is called. |
| Export | JSONL, JSON, CSV, Markdown, manifest and report files are written. |
| Scale | The 500-case demo completes successfully offline. |

The final acceptance run recorded the command output, case count, validation
issue count, split/type/difficulty distributions, and SHA-256 dataset hash
below.

## Latest run

Command:

```text
python run_demo.py --count 500 --seed 20261004 --out artifacts/demo
```

Result: **PASS** — 500 cases, all four splits, five difficulty levels (100 per
level), zero duplicate rate, zero validation failures, and no reported issues.

Reference SHA-256 dataset hash:

```text
sha256:939b13bc0f18623c30da5016969fd2e4eaedaddd3aedb1079610150dbf7e24fb
```

Split counts: train 350, validation 75, test 50, challenge 25. The same command
was run a second time with an identical hash, confirming deterministic output.
