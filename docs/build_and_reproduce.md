# Build and Reproduce

## Patch Layout

```text
trtllm_patch/0001-moe-aware-kv-scheduling.patch
trtllm_patch/0000-local-build-fixes.patch
```

`0001-moe-aware-kv-scheduling.patch` contains the actual optimization. `0000-local-build-fixes.patch` contains workspace-specific build fixes for Conan and Torch runtime linkage.

## Apply

```bash
cd /workspace/TensorRT-LLM
git apply /workspace/project/trtllm_patch/0001-moe-aware-kv-scheduling.patch
```

If needed in the same local container environment:

```bash
git apply /workspace/project/trtllm_patch/0000-local-build-fixes.patch
```

## Build Output Validated

The following artifacts were successfully built in this workspace:

```text
cpp/build/tensorrt_llm/libtensorrt_llm.so
cpp/build/benchmarks/gptManagerBenchmark
```

Build log:

```text
logs/build_moe_kv_scheduler.log
```

## Workloads

```bash
python3 scripts/generate_moe_kv_scheduling_workloads.py
```

Generated files:

```text
workloads/moe_kv_scheduling/balanced_control.json
workloads/moe_kv_scheduling/repeated_prefix_hot_pressure.json
workloads/moe_kv_scheduling/mixed_burst.json
workloads/moe_kv_scheduling/low_reuse_pollution.json
```

## Live Benchmark

```bash
PROJECT_DIR=/workspace/project \
TRTLLM_DIR=/workspace/TensorRT-LLM \
ENGINE_DIR=/workspace/engine \
bash scripts/run_moe_kv_scheduling_live_matrix.sh
```

## Summarize

```bash
python3 scripts/summarize_moe_kv_scheduling_live.py
```

Outputs:

```text
results/moe_kv_scheduling_live/compare_tables/live_summary.md
results/moe_kv_scheduling_live/compare_tables/live_summary.json
```
