# PyInstaller specification for Synthetic Benchmark Factory.
# Build with: pyinstaller --noconfirm --clean BenchmarkFactory.spec
from pathlib import Path

ROOT = Path(SPECPATH).resolve()
APP = ROOT / "app.py"

a = Analysis(
    [str(APP)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "synthetic_benchmark_factory",
        "synthetic_benchmark_factory.models",
        "synthetic_benchmark_factory.schema",
        "synthetic_benchmark_factory.serialization",
        "synthetic_benchmark_factory.versioning",
        "synthetic_benchmark_factory.generator",
        "synthetic_benchmark_factory.validator",
        "synthetic_benchmark_factory.providers",
        "synthetic_benchmark_factory.runner",
        "synthetic_benchmark_factory.metrics",
        "synthetic_benchmark_factory.evaluation",
        "synthetic_benchmark_factory.evaluator",
        "synthetic_benchmark_factory.provider",
        "synthetic_benchmark_factory.validators",
        "synthetic_benchmark_factory.analysis",
        "synthetic_benchmark_factory.reports",
        "synthetic_benchmark_factory.cli",
        "synthetic_benchmark_factory.__main__",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BenchmarkFactory",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
