# MoE KV Scheduling Live Summary

Latency deltas are lower-is-better. Throughput deltas are higher-is-better.

## Raw Values

| Workload | Mode | Samples | Errors | TTFT p90 ms | TPOT p90 ms | E2E p90 ms | Throughput tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_control | baseline_disabled | 64 | 0 | 1509.21 | 14.99 | 2922.15 | 132.03 |
| balanced_control | admission_only | 64 | 0 | 1526.76 | 15.17 | 2947.27 | 131.48 |
| balanced_control | retention_only | 64 | 0 | 1525.49 | 15.13 | 2942.91 | 131.78 |
| balanced_control | combined | 64 | 0 | 1622.01 | 16.17 | 3060.30 | 126.39 |
| repeated_prefix_hot_pressure | baseline_disabled | 64 | 0 | 3176.58 | 21.60 | 4153.17 | 92.62 |
| repeated_prefix_hot_pressure | admission_only | 64 | 0 | 3205.66 | 22.04 | 4254.15 | 91.75 |
| repeated_prefix_hot_pressure | retention_only | 64 | 0 | 3177.16 | 21.74 | 4204.76 | 92.72 |
| repeated_prefix_hot_pressure | combined | 64 | 0 | 3175.89 | 21.89 | 4195.23 | 91.93 |
| mixed_burst | baseline_disabled | 64 | 0 | 3182.01 | 21.63 | 4223.54 | 92.42 |
| mixed_burst | admission_only | 64 | 0 | 3073.34 | 20.66 | 4122.78 | 95.56 |
| mixed_burst | retention_only | 64 | 0 | 3185.70 | 21.33 | 4320.21 | 92.87 |
| mixed_burst | combined | 64 | 0 | 3086.67 | 20.87 | 4207.62 | 94.20 |
| low_reuse_pollution | baseline_disabled | 64 | 0 | 3106.89 | 20.99 | 4169.84 | 94.04 |
| low_reuse_pollution | admission_only | 64 | 0 | 3121.04 | 21.16 | 4156.85 | 93.72 |
| low_reuse_pollution | retention_only | 64 | 0 | 3074.44 | 20.78 | 4073.95 | 93.21 |
| low_reuse_pollution | combined | 64 | 0 | 3173.73 | 21.08 | 4332.71 | 92.30 |

## Delta vs Baseline Disabled

| Workload | Mode | TTFT p90 | TPOT p90 | E2E p90 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced_control | admission_only | +1.16% | +1.22% | +0.86% | -0.41% |
| balanced_control | retention_only | +1.08% | +0.94% | +0.71% | -0.18% |
| balanced_control | combined | +7.47% | +7.87% | +4.73% | -4.27% |
| repeated_prefix_hot_pressure | admission_only | +0.92% | +2.03% | +2.43% | -0.94% |
| repeated_prefix_hot_pressure | retention_only | +0.02% | +0.65% | +1.24% | +0.11% |
| repeated_prefix_hot_pressure | combined | -0.02% | +1.32% | +1.01% | -0.75% |
| mixed_burst | admission_only | -3.42% | -4.48% | -2.39% | +3.40% |
| mixed_burst | retention_only | +0.12% | -1.40% | +2.29% | +0.49% |
| mixed_burst | combined | -3.00% | -3.54% | -0.38% | +1.93% |
| low_reuse_pollution | admission_only | +0.46% | +0.79% | -0.31% | -0.34% |
| low_reuse_pollution | retention_only | -1.04% | -1.03% | -2.30% | -0.88% |
| low_reuse_pollution | combined | +2.15% | +0.39% | +3.91% | -1.84% |
