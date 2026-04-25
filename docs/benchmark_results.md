# Benchmark Results

## Setup

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Model | `Qwen/Qwen1.5-MoE-A2.7B-Chat` |
| Engine | TensorRT-LLM INT4 weight-only |
| Runtime | TensorRT-LLM C++ `gptManagerBenchmark` |
| Samples | 64 per workload and mode |
| Concurrency | 4 |
| Baseline | Same patched binary with MoE KV scheduling disabled |

## Delta Summary

Latency deltas are lower-is-better. Throughput deltas are higher-is-better.

| Workload | Mode | TTFT p90 | TPOT p90 | E2E p90 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced_control | admission_only | +1.16% | +1.22% | +0.86% | -0.41% |
| balanced_control | retention_only | +1.08% | +0.94% | +0.71% | -0.18% |
| balanced_control | combined | +7.47% | +7.87% | +4.73% | -4.27% |
| repeated_prefix_hot_pressure | admission_only | +0.92% | +2.03% | +2.43% | -0.94% |
| repeated_prefix_hot_pressure | retention_only | +0.02% | +0.65% | +1.24% | +0.11% |
| repeated_prefix_hot_pressure | combined | -0.02% | +1.32% | +1.01% | -0.75% |
| mixed_burst | admission_only | **-3.42%** | **-4.48%** | **-2.39%** | **+3.40%** |
| mixed_burst | retention_only | +0.12% | -1.40% | +2.29% | +0.49% |
| mixed_burst | combined | -3.00% | -3.54% | -0.38% | +1.93% |
| low_reuse_pollution | admission_only | +0.46% | +0.79% | -0.31% | -0.34% |
| low_reuse_pollution | retention_only | -1.04% | -1.03% | -2.30% | -0.88% |
| low_reuse_pollution | combined | +2.15% | +0.39% | +3.91% | -1.84% |

## Main Finding

The clearest improvement appears on `mixed_burst` with admission-only scheduling:

```text
TTFT p90:       -3.42%
TPOT p90:       -4.48%
E2E p90:        -2.39%
Throughput:     +3.40%
```

This matches the intended pressure scenario: low-reuse prompts arrive before high-reuse shared-prefix prompts, and the policy can change admission order once KV pressure crosses the watermark.

## Negative and Neutral Findings

Balanced control regresses when the combined policy is enabled. That means the policy should remain pressure-gated and default-off.

Repeated-prefix hot pressure is largely neutral. When most requests are already high reuse, the admission policy has little useful contrast between candidates.

Retention-only is weaker than admission-only because it acts after request admission. It can protect prefix blocks, but it does not prevent a low-value request from consuming KV capacity first.

## Excluded Run

An additional pressure sweep with a smaller KV budget was excluded from headline results. TensorRT-LLM reduced max sequence length and output length for that run, so it was no longer comparable to the main workload.
