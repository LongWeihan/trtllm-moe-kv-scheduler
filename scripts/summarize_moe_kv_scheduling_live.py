#!/usr/bin/env python3
"""Summarize MoE-aware KV scheduling live TensorRT-LLM benchmark CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "moe_kv_scheduling_live"
SUMMARY_DIR = RESULT_ROOT / "compare_tables"

WORKLOADS = [
    "balanced_control",
    "repeated_prefix_hot_pressure",
    "mixed_burst",
    "low_reuse_pollution",
]
MODES = [
    "baseline_disabled",
    "admission_only",
    "retention_only",
    "combined",
]


def read_csv(path: Path) -> dict[str, float | str]:
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    out: dict[str, float | str] = {}
    for key, value in row.items():
        if not key or value in (None, ""):
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def get(row: dict[str, float | str], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def pct_change(old: float, new: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for workload in WORKLOADS:
        for mode in MODES:
            path = RESULT_ROOT / mode / f"{workload}.csv"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            data = read_csv(path)
            rows.append(
                {
                    "workload": workload,
                    "mode": mode,
                    "num_samples": get(data, "num_samples"),
                    "num_error_samples": get(data, "num_error_samples"),
                    "total_latency_ms": get(data, "total_latency(ms)"),
                    "ttft_p90_ms": get(data, "p90_time_to_first_token(ms)"),
                    "tpot_p90_ms": get(data, "p90_inter_token_latency(ms)"),
                    "e2e_p90_ms": get(data, "p90_sequence_latency(ms)"),
                    "throughput_tok_s": get(data, "token_throughput(token/sec)"),
                    "seq_throughput_s": get(data, "seq_throughput(seq/sec)"),
                }
            )

    by_workload: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_workload.setdefault(row["workload"], {})[row["mode"]] = row

    deltas = []
    for workload, modes in by_workload.items():
        base = modes.get("baseline_disabled")
        if not base:
            continue
        for mode in MODES:
            if mode == "baseline_disabled" or mode not in modes:
                continue
            current = modes[mode]
            deltas.append(
                {
                    "workload": workload,
                    "mode": mode,
                    "ttft_p90_change_pct": pct_change(base["ttft_p90_ms"], current["ttft_p90_ms"]),
                    "tpot_p90_change_pct": pct_change(base["tpot_p90_ms"], current["tpot_p90_ms"]),
                    "e2e_p90_change_pct": pct_change(base["e2e_p90_ms"], current["e2e_p90_ms"]),
                    "throughput_change_pct": pct_change(base["throughput_tok_s"], current["throughput_tok_s"]),
                }
            )

    summary = {
        "status": "complete" if not missing else "partial",
        "source": "live TensorRT-LLM patched binary with policy toggled by environment variables",
        "baseline_definition": "same patched binary with MoE KV scheduling env vars disabled",
        "rows": rows,
        "deltas": deltas,
        "missing": missing,
    }
    (SUMMARY_DIR / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# MoE KV Scheduling Live Summary",
        "",
        "Latency deltas are lower-is-better. Throughput deltas are higher-is-better.",
        "",
        "## Raw Values",
        "",
        "| Workload | Mode | Samples | Errors | TTFT p90 ms | TPOT p90 ms | E2E p90 ms | Throughput tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['mode']} | {row['num_samples']:.0f} | "
            f"{row['num_error_samples']:.0f} | {row['ttft_p90_ms']:.2f} | "
            f"{row['tpot_p90_ms']:.2f} | {row['e2e_p90_ms']:.2f} | "
            f"{row['throughput_tok_s']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Delta vs Baseline Disabled",
            "",
            "| Workload | Mode | TTFT p90 | TPOT p90 | E2E p90 | Throughput |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['workload']} | {row['mode']} | {fmt_pct(row['ttft_p90_change_pct'])} | "
            f"{fmt_pct(row['tpot_p90_change_pct'])} | {fmt_pct(row['e2e_p90_change_pct'])} | "
            f"{fmt_pct(row['throughput_change_pct'])} |"
        )

    if missing:
        lines.extend(["", "## Missing Files", ""])
        lines.extend(f"- `{item}`" for item in missing)

    (SUMMARY_DIR / "live_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
