# Examples

`long_term_memory_config.json` is a human-readable benchmark definition for
the reference demo. It documents the real difficulty knobs and split ratios;
the procedural generator also embeds the same provenance into each case.

Generate a small preview:

```text
python run_demo.py --count 20 --seed 7 --out artifacts/preview
```

Generate the acceptance dataset:

```text
python run_demo.py --count 500 --seed 20261004 --out artifacts/demo
```

