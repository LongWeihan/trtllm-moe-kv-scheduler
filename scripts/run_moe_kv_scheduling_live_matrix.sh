#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/project}"
ENGINE_DIR="${ENGINE_DIR:-/workspace/engine}"
TRTLLM_DIR="${TRTLLM_DIR:-/workspace/TensorRT-LLM}"

export LD_LIBRARY_PATH="${TRTLLM_DIR}/cpp/build/tensorrt_llm:${TRTLLM_DIR}/cpp/build/tensorrt_llm/plugins:/usr/local/tensorrt/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/hpcx/ompi/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RESULT_ROOT="${PROJECT_DIR}/results/moe_kv_scheduling_live"
LOG_ROOT="${PROJECT_DIR}/logs/moe_kv_scheduling_live"
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

WORKLOADS=(
  "balanced_control:balanced_control.json:384"
  "repeated_prefix_hot_pressure:repeated_prefix_hot_pressure.json:256"
  "mixed_burst:mixed_burst.json:256"
  "low_reuse_pollution:low_reuse_pollution.json:256"
)

MODES=(
  "baseline_disabled"
  "admission_only"
  "retention_only"
  "combined"
)

clear_policy_env() {
  unset TRTLLM_MOE_KV_SCHED_TELEMETRY_ENABLE
  unset TRTLLM_MOE_KV_ADMISSION_ENABLE
  unset TRTLLM_MOE_KV_RETENTION_ENABLE
  unset TRTLLM_MOE_KV_HIGH_WATERMARK_PCT
  unset TRTLLM_MOE_KV_WEIGHT_REUSE
  unset TRTLLM_MOE_KV_WEIGHT_RECOMPUTE
  unset TRTLLM_MOE_KV_WEIGHT_PRESSURE
  unset TRTLLM_MOE_KV_WEIGHT_POLLUTION
  unset TRTLLM_MOE_KV_MIN_ADMIT_UTILITY
  unset TRTLLM_MOE_KV_DEFER_MARGIN
  unset TRTLLM_MOE_KV_PREFIX_PRIORITY_BASE
  unset TRTLLM_MOE_KV_PREFIX_PRIORITY_MAX
  unset TRTLLM_MOE_KV_DECODE_PRIORITY
  unset TRTLLM_MOE_KV_DURATION_MS
}

set_common_policy_env() {
  export TRTLLM_MOE_KV_HIGH_WATERMARK_PCT=50
  export TRTLLM_MOE_KV_WEIGHT_REUSE=1.35
  export TRTLLM_MOE_KV_WEIGHT_RECOMPUTE=1.00
  export TRTLLM_MOE_KV_WEIGHT_PRESSURE=0.55
  export TRTLLM_MOE_KV_WEIGHT_POLLUTION=1.10
  export TRTLLM_MOE_KV_MIN_ADMIT_UTILITY=35
  export TRTLLM_MOE_KV_DEFER_MARGIN=8
  export TRTLLM_MOE_KV_PREFIX_PRIORITY_BASE=45
  export TRTLLM_MOE_KV_PREFIX_PRIORITY_MAX=95
  export TRTLLM_MOE_KV_DECODE_PRIORITY=30
  export TRTLLM_MOE_KV_DURATION_MS=9000
}

set_mode() {
  local mode="$1"
  clear_policy_env
  case "${mode}" in
    baseline_disabled)
      ;;
    admission_only)
      set_common_policy_env
      export TRTLLM_MOE_KV_ADMISSION_ENABLE=1
      ;;
    retention_only)
      set_common_policy_env
      export TRTLLM_MOE_KV_RETENTION_ENABLE=1
      ;;
    combined)
      set_common_policy_env
      export TRTLLM_MOE_KV_ADMISSION_ENABLE=1
      export TRTLLM_MOE_KV_RETENTION_ENABLE=1
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local workload_name="$1"
  local dataset="$2"
  local kv_tokens="$3"
  local mode="$4"

  local out_dir="${RESULT_ROOT}/${mode}"
  mkdir -p "${out_dir}"
  local csv="${out_dir}/${workload_name}.csv"
  local log="${LOG_ROOT}/${mode}_${workload_name}.log"

  set_mode "${mode}"
  echo "Running workload=${workload_name} mode=${mode} kv_tokens=${kv_tokens}" | tee "${log}"

  "${TRTLLM_DIR}/cpp/build/benchmarks/gptManagerBenchmark" \
    --engine_dir "${ENGINE_DIR}" \
    --api executor \
    --type inflight \
    --streaming true \
    --scheduler_policy max_utilization \
    --dataset "${PROJECT_DIR}/workloads/moe_kv_scheduling/${dataset}" \
    --max_num_samples 64 \
    --warm_up 0 \
    --eos_id -1 \
    --pad_id 151643 \
    --max_batch_size 4 \
    --max_num_tokens 512 \
    --concurrency 4 \
    --max_tokens_in_paged_kvcache "${kv_tokens}" \
    --output_csv "${csv}" 2>&1 | tee -a "${log}"
}

for workload in "${WORKLOADS[@]}"; do
  IFS=":" read -r workload_name dataset kv_tokens <<< "${workload}"
  for mode in "${MODES[@]}"; do
    run_one "${workload_name}" "${dataset}" "${kv_tokens}" "${mode}"
  done
done

python3 "${PROJECT_DIR}/scripts/summarize_moe_kv_scheduling_live.py"
