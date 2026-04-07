#!/bin/bash

set -ex
exec &> >(tee "nccl.txt")

export VLLM_RPC_TIMEOUT=600000
#export VLLM_NVTX_SCOPES_FOR_PROFILING=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

PORT=9001

# model config
#MODEL="/huggingface2/Llama-3.1-8B-Instruct"
MODEL="/huggingface/Llama-3.1-70B-Instruct"
#MODEL="/huggingface/Llama-3.1-70B"
#MODEL="/purdue/yechen3/llama-models/models--meta-llama--Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac"
#MODEL="/huggingface/qwen3-30b-a3b"
GPU_UTIL=0.90
MAX_MODEL_LEN=15360 # max token len for 1 request (ISL+OSL)
MAX_BATCHED_TOKEN=65536 #128000 # max token len for all reqeust (ISL+OSL)
MAX_SEQ=64 # number of batch

# benchmark config
QPS=(INF)
NUM_PROMPTS=1 # total number of requests sent to vllm server
INPUT=2048  # (896, 768, 512, 256, 128)
OUTPUT=2 #128 # (128, 256, 512, 768, 896)

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep -u $USER pt_main_thread | xargs -r kill -15
  pgrep -u $USER python3 | xargs -r kill -15
  # vLLM now names the process with VLLM prefix after https://github.com/vllm-project/vllm/pull/21445
  pgrep -u $USER VLLM | xargs -r kill -15
  for port in $PORT; do lsof -t -i:$port | xargs -r kill -15; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

TP_SIZE=4
#CUDA_VISIBLE_DEVICES=7 \
#CUDA_VISIBLE_DEVICES=4,5,6,7 \
launch_chunked_prefill() {
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nsys profile \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --gpu-metrics-devices=0 \
    --cuda-graph-trace=node \
    --delay=70 \
  vllm serve $MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_BATCHED_TOKEN \
    --enable-chunked-prefill \
    --max-num-seqs $MAX_SEQ \
    --no-enable-prefix-caching \
    --disable-custom-all-reduce \
    --gpu-memory-utilization $GPU_UTIL &
  wait_for_server $PORT
  sleep 1
}
#--delay=50 \
launch_chunked_prefill_sp() {
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_DISABLE_SYMMETRIC_MEMORY=1 \
  nsys profile \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --gpu-metrics-devices=0 \
    --cuda-graph-trace=node \
  vllm serve $MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_BATCHED_TOKEN \
    --enable-chunked-prefill \
    --max-num-seqs $MAX_SEQ \
    --no-enable-prefix-caching \
    --disable-custom-all-reduce \
    --gpu-memory-utilization $GPU_UTIL \
    --compilation-config '{
      "pass_config": {
        "enable_sp": true,
        "fuse_gemm_comms": true,
        "sp_min_token_num": 512
      },
      "use_inductor_graph_partition": true
    }' &
  wait_for_server $PORT
  sleep 1
}
#--disable-custom-all-reduce \
#"compile_sizes": [512, 1024, 2048],
# disable cuda graph
#--enforce-eager \
#--delay=75 \
launch_chunked_prefill_sp_ncu() {
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ncu \
    --target-processes all \
    --replay-mode application \
    --app-replay-mode relaxed \
    --app-replay-match name \
    --set basic \
    --kernel-name 'regex:.*gemm.*|.*mma.*|.*all_reduce.*|.*reduce_scatter.*|.*all_gather.*' \
    --launch-count 400 \
    --kill yes \
    -f \
    -o "ncu_vllm_prefill_$(date +%Y%m%d_%H%M%S)" \
  vllm serve $MODEL \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_BATCHED_TOKEN \
    --enable-chunked-prefill \
    --max-num-seqs $MAX_SEQ \
    --no-enable-prefix-caching \
    --disable-custom-all-reduce \
    --gpu-memory-utilization $GPU_UTIL \
    --compilation-config '{
      "pass_config": {
        "enable_sp": true,
        "fuse_gemm_comms": true,
        "sp_min_token_num": 512
      },
      "use_inductor_graph_partition": true
    }' &
  wait_for_server $PORT
  sleep 1
}

benchmark() {
  results_folder="./results"
  dataset_name="sonnet"
  #dataset_path="../sonnet_4x.txt"
  dataset_path="./sonnet_4x.txt"
  num_prompts=$NUM_PROMPTS
  qps=$1
  input_len=$2
  output_len=$3
  tag=$4

  vllm bench serve \
    --backend vllm \
    --model $MODEL \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --num-prompts $NUM_PROMPTS \
	  --sonnet-input-len $input_len \
	  --sonnet-output-len $output_len \
    --port $PORT \
    --save-result \
    --result-dir $results_folder \
    --result-filename "$tag"-qps-"$qps"-i-"$input_len"-o-"$output_len"-b-"$NUM_PROMPTS".json \
    --request-rate "$qps" \
	  --percentile-metrics "ttft,tpot,itl,e2el" \
	  --metric-percentiles "50, 95" \
    --ignore-eos

  sleep 2
}

LOG_TIMESTAMP_PREV_EPOCH=""

log_timestamp() {
  local func_name=$1
  local now_human now_epoch delta

  now_human=$(date '+%Y-%m-%d %H:%M:%S %Z')
  now_epoch=$(date '+%s.%N')

  if [[ -n "$LOG_TIMESTAMP_PREV_EPOCH" ]]; then
    delta=$(awk -v now="$now_epoch" -v prev="$LOG_TIMESTAMP_PREV_EPOCH" 'BEGIN { printf "+%.3fs", now - prev }')
  else
    delta="+0.000s"
  fi

  LOG_TIMESTAMP_PREV_EPOCH="$now_epoch"
  echo "[${now_human}] [${delta}] func starting ${func_name}"
}

main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  uv pip install quart httpx matplotlib aiohttp datasets

  rm -rf results
  mkdir results

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  log_timestamp "launch_chunked_prefill"
  #launch_chunked_prefill
  launch_chunked_prefill_sp
  #launch_chunked_prefill_sp_ncu
  for qps in "${QPS[@]}"; do
    log_timestamp "benchmark"
	  benchmark $qps $INPUT $OUTPUT chunked_prefill
  done
  log_timestamp "kill_gpu_processes"
  kill_gpu_processes

}


main "$@"
