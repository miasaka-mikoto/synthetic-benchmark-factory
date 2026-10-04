"""Report structures and offline exporters for benchmark runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analysis import ItemAnalysisReport, analyze_items
from .runner import RunReport, RunRecord, summarize_records


@dataclass
class BenchmarkReport:
    """A portable report combining dataset, run and item-analysis information."""

    title: str = "Synthetic Benchmark Report"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    run_summary: dict[str, Any] = field(default_factory=dict)
    item_analysis: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "dataset": self.dataset,
            "statistics": self.statistics,
            "run_summary": self.run_summary,
            "item_analysis": self.item_analysis,
            "records": self.records,
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_markdown(self) -> str:
        return report_to_markdown(self)


def build_report(
    dataset: Any = None,
    run_report: Any = None,
    *,
    records: Sequence[Any] | None = None,
    item_analysis: ItemAnalysisReport | Mapping[str, Any] | None = None,
    title: str | None = None,
    include_records: bool = True,
) -> BenchmarkReport:
    """Build a report from any combination of dataset and run results."""

    if run_report is not None and records is None:
        records = getattr(run_report, "records", None)
        if records is None and isinstance(run_report, Mapping):
            records = run_report.get("records", [])
    records = list(records or [])
    run_dict = _to_dict(run_report) if run_report is not None else {}
    dataset_dict = _to_dict(dataset) if dataset is not None else {}
    stats = _to_dict(getattr(dataset, "statistics", None)) if dataset is not None else {}
    if dataset is not None and not stats and hasattr(dataset, "compute_statistics"):
        try:
            stats = _to_dict(dataset.compute_statistics())
        except Exception:
            stats = {}
    if item_analysis is None:
        item_analysis_obj = analyze_items(records, cases=getattr(dataset, "cases", None) if dataset is not None else None)
        analysis_dict = item_analysis_obj.to_dict()
    else:
        analysis_dict = _to_dict(item_analysis)
    summary = run_dict.get("summary") if isinstance(run_dict, Mapping) else None
    if not summary:
        summary = summarize_records(records) if records else {}
    record_dicts = [_to_dict(record) for record in records] if include_records else []
    name = title or str(dataset_dict.get("name") or "Synthetic Benchmark Report")
    return BenchmarkReport(
        title=name,
        dataset=dataset_dict,
        statistics=stats,
        run_summary=summary,
        item_analysis=analysis_dict,
        records=record_dicts,
        metadata={
            "synthetic": _is_synthetic(records),
            "record_count": len(records),
            "case_count": len({str(item.get("case_id")) for item in record_dicts if item.get("case_id")}),
        },
    )


def build_evaluation_report(*args: Any, **kwargs: Any) -> BenchmarkReport:
    return build_report(*args, **kwargs)


def report_to_markdown(report: BenchmarkReport | Mapping[str, Any]) -> str:
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    summary = data.get("run_summary") or {}
    stats = data.get("statistics") or {}
    analysis = data.get("item_analysis") or {}
    lines = [
        f"# {data.get('title', 'Synthetic Benchmark Report')}",
        "",
        f"Generated: `{data.get('generated_at', '')}`",
        "",
        "## Run Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("case_count", "record_count", "evaluated_count", "mean_score", "pass_rate", "mean_latency_ms", "p95_latency_ms", "total_tokens", "errors"):
        if key in summary:
            value = summary[key]
            if key == "pass_rate" and isinstance(value, (int, float)):
                value = f"{float(value):.1%}"
            elif isinstance(value, float):
                value = f"{value:.4f}"
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
    if stats:
        lines.extend(["", "## Dataset Statistics", "", "| Metric | Value |", "|---|---:|"])
        for key in ("case_count", "duplicate_rate", "validation_failure_rate", "synthetic_case_count", "valid_case_count", "invalid_case_count"):
            if key in stats:
                value = stats[key]
                if key.endswith("rate") and isinstance(value, (int, float)):
                    value = f"{float(value):.2%}"
                lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        for key in ("difficulty_distribution", "type_distribution", "split_distribution"):
            if key in stats:
                lines.append(f"| {key.replace('_', ' ').title()} | `{json.dumps(stats[key], ensure_ascii=False, sort_keys=True)}` |")
    if analysis:
        lines.extend(["", "## Item Analysis", ""])
        for label in ("all_models_wrong", "too_easy", "discriminating", "problematic"):
            values = analysis.get(label, [])
            lines.append(f"- **{label.replace('_', ' ').title()}**: {len(values)}")
        items = analysis.get("items") or []
        if items:
            lines.extend(["", "| Case | Mean | Spread | Pass rate | Flags |", "|---|---:|---:|---:|---|"])
            for item in items:
                mean = item.get("mean_score")
                mean_text = "" if mean is None else f"{float(mean):.3f}"
                rate = item.get("pass_rate")
                rate_text = "" if rate is None else f"{float(rate):.1%}"
                lines.append(f"| {item.get('case_id', '')} | {mean_text} | {float(item.get('score_spread', 0)):.3f} | {rate_text} | {', '.join(item.get('flags', []))} |")
    return "\n".join(lines) + "\n"


def export_report(report: BenchmarkReport | Mapping[str, Any], path: str | Path, *, format: str | None = None, include_records: bool = True) -> Path:
    """Write JSON, Markdown, CSV or JSONL report files without network access."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    fmt = (format or destination.suffix.lstrip(".") or "json").lower()
    if fmt in {"md", "markdown"}:
        destination.write_text(report_to_markdown(data), encoding="utf-8")
    elif fmt == "csv":
        rows = data.get("records", []) if include_records else []
        _write_csv(destination, rows)
    elif fmt in {"jsonl", "ndjson"}:
        rows = data.get("records", []) if include_records else []
        destination.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")
    else:
        if not include_records:
            data = dict(data)
            data["records"] = []
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return destination


def export_run_report(report: BenchmarkReport | Mapping[str, Any], path: str | Path, **kwargs: Any) -> Path:
    return export_report(report, path, **kwargs)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            output = value.to_dict()
            return dict(output) if isinstance(output, Mapping) else {}
        except Exception:
            return {}
    return {}


def _is_synthetic(records: Sequence[Any]) -> bool:
    values = []
    for record in records:
        item = _to_dict(record)
        if "synthetic" in item:
            values.append(bool(item["synthetic"]))
        elif item.get("metadata", {}).get("synthetic") is not None:
            values.append(bool(item["metadata"]["synthetic"]))
    return all(values) if values else True


__all__ = [
    "BenchmarkReport",
    "Report",
    "build_evaluation_report",
    "build_report",
    "export_report",
    "export_run_report",
    "report_to_markdown",
]

Report = BenchmarkReport
