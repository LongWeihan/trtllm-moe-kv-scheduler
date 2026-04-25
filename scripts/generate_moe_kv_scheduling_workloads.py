#!/usr/bin/env python3
"""Generate pressure-proof workloads for MoE-aware KV scheduling experiments."""

from __future__ import annotations

import json
from pathlib import Path


VOCAB_BASE = 1000
PROMPT_LEN = 128
OUTPUT_LEN = 96
NUM_SAMPLES = 64
SHARED_PREFIX_LEN = 112


def make_prompt(seed: int, length: int) -> list[int]:
    return [VOCAB_BASE + ((seed * 131 + i * 17) % 10000) for i in range(length)]


def shared_prefix(group: int, length: int = SHARED_PREFIX_LEN) -> list[int]:
    header = [151644, 8948, 198, VOCAB_BASE + group]
    body = [VOCAB_BASE + 100 + group * 31 + (i % 41) for i in range(length - len(header))]
    return header + body


def add_metadata(
    sample: dict,
    *,
    pressure: float,
    reuse_group: int,
    shared_tokens: int,
    expected_reuse: int,
    signal_source: str = "synthetic_hint",
) -> dict:
    sample["moe_pressure_score"] = round(pressure, 4)
    sample["reuse_group"] = reuse_group
    sample["shared_prefix_tokens"] = shared_tokens
    sample["estimated_kv_blocks"] = max(1, (len(sample["input_ids"]) + 31) // 32)
    sample["expected_reuse_count"] = expected_reuse
    sample["signal_source"] = signal_source
    return sample


def balanced_control() -> list[dict]:
    samples: list[dict] = []
    for i in range(NUM_SAMPLES):
        samples.append(
            add_metadata(
                {
                    "task_id": i,
                    "input_ids": make_prompt(10 + i, PROMPT_LEN),
                    "output_len": OUTPUT_LEN,
                },
                pressure=0.35 + 0.05 * (i % 4),
                reuse_group=i,
                shared_tokens=0,
                expected_reuse=0,
            )
        )
    return samples


def repeated_prefix_hot_pressure() -> list[dict]:
    samples: list[dict] = []
    prefixes = [shared_prefix(0), shared_prefix(1)]
    for i in range(NUM_SAMPLES):
        group = i % len(prefixes)
        suffix = make_prompt(1000 + i, PROMPT_LEN - SHARED_PREFIX_LEN)
        samples.append(
            add_metadata(
                {
                    "task_id": i,
                    "input_ids": prefixes[group] + suffix,
                    "output_len": OUTPUT_LEN,
                },
                pressure=0.88 + 0.04 * (i % 3),
                reuse_group=group,
                shared_tokens=SHARED_PREFIX_LEN,
                expected_reuse=NUM_SAMPLES // len(prefixes),
            )
        )
    return samples


def mixed_burst() -> list[dict]:
    samples: list[dict] = []

    # Front-load low-reuse long prompts. A plain FIFO scheduler tends to admit
    # these first, which is exactly the cache pollution case this project tests.
    for i in range(NUM_SAMPLES // 4):
        samples.append(
            add_metadata(
                {
                    "task_id": i,
                    "input_ids": make_prompt(3000 + i * 17, PROMPT_LEN),
                    "output_len": OUTPUT_LEN,
                },
                pressure=0.72 + 0.04 * (i % 3),
                reuse_group=200 + i,
                shared_tokens=0,
                expected_reuse=0,
            )
        )

    hot_prefixes = [shared_prefix(10), shared_prefix(11), shared_prefix(12)]
    for i in range(NUM_SAMPLES - len(samples)):
        group = i % len(hot_prefixes)
        suffix = make_prompt(5000 + i, PROMPT_LEN - SHARED_PREFIX_LEN)
        samples.append(
            add_metadata(
                {
                    "task_id": len(samples),
                    "input_ids": hot_prefixes[group] + suffix,
                    "output_len": OUTPUT_LEN,
                },
                pressure=0.86 + 0.03 * (i % 4),
                reuse_group=10 + group,
                shared_tokens=SHARED_PREFIX_LEN,
                expected_reuse=(NUM_SAMPLES - NUM_SAMPLES // 4) // len(hot_prefixes),
            )
        )
    return samples


def low_reuse_pollution() -> list[dict]:
    samples: list[dict] = []
    for i in range(NUM_SAMPLES):
        samples.append(
            add_metadata(
                {
                    "task_id": i,
                    "input_ids": make_prompt(7000 + i * 23, PROMPT_LEN),
                    "output_len": OUTPUT_LEN,
                },
                pressure=0.86 + 0.03 * (i % 5),
                reuse_group=500 + i,
                shared_tokens=0,
                expected_reuse=0,
            )
        )
    return samples


def write_workload(path: Path, samples: list[dict]) -> None:
    metadata = {
        "workload_name": path.stem,
        "num_requests": len(samples),
        "prompt_len": PROMPT_LEN,
        "output_len": OUTPUT_LEN,
        "signal_source": "synthetic_hint",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "samples": samples}, indent=2), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "workloads" / "moe_kv_scheduling"
    write_workload(out_dir / "balanced_control.json", balanced_control())
    write_workload(out_dir / "repeated_prefix_hot_pressure.json", repeated_prefix_hot_pressure())
    write_workload(out_dir / "mixed_burst.json", mixed_burst())
    write_workload(out_dir / "low_reuse_pollution.json", low_reuse_pollution())


if __name__ == "__main__":
    main()
