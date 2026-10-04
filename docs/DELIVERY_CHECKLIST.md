# Release checklist

The source checkout is ready for the offline acceptance handoff when these
commands are green:

```text
python -m compileall -q synthetic_benchmark_factory run_demo.py scripts
python -m pytest -q
python qa_smoke.py
python app.py --headless-smoke
python run_demo.py --count 500 --seed 20261004 --out artifacts/demo
python -m synthetic_benchmark_factory validate artifacts/demo/dataset.json
python -m synthetic_benchmark_factory hash artifacts/demo/dataset.json
```

Expected reference facts:

* 500 synthetic Memory cases;
* train/validation/test/challenge: 350/75/50/25;
* difficulty levels 1–5: 100 each;
* zero duplicate, leakage, ambiguous, impossible, and validation-failure cases;
* deterministic dataset hash recorded in `artifacts/demo/manifest.json` and
  `docs/TEST_REPORT.md`;
* `MockModel` and `RuleModel` outputs are in `runs.json` and are labelled
  synthetic/provider-specific;
* JSON, JSONL, CSV, Markdown, validation, statistics, and item-analysis files
  are present under `artifacts/demo`.

The Windows executable build is performed by `build_windows.ps1` or
`build_windows.bat`. On a machine without PyInstaller, the scripts produce a
portable source bundle and leave the source workflow unchanged.

