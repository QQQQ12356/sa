#!/usr/bin/env bash
# HotpotQA 本地 Beacon 生成与评分。参数见 docs/37_hotpotqa_vllm.md。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此入口仅支持 Linux。' >&2; exit 2; }
(( $# == 0 )) || { echo '请在脚本顶部设置参数。' >&2; exit 2; }
cd -- "$PROJECT_ROOT"

export HF_ENDPOINT=https://huggingface.co
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SA_BENCHMARK_DIR=artifacts/benchmarks/hotpotqa/v1

# 训练实验自动解析 Beacon 权重和压缩配置。
export SA_TRAINING_RUN=artifacts/experiments/beacon/openseeker/REPLACE_WITH_RUN
export SA_CHECKPOINT=''
export SA_PROMPT_DATA=''  # benchmark 提示词不接受覆盖。
export SA_RUN_NAME=beacon-hotpotqa
export SA_RESUME_FROM=''  # 填已完成的同方法测试目录，LIMIT 填累计目标题数。
export SA_EVALUATION_DIR=''  # 填已完成的同方法测试目录时，仅评分。
export SA_SCORE_AFTER_GENERATION=false

# 与 Baseline/Plain 共用 HotpotQA 提示词；每张可见 GPU 加载一个完整模型。
export SA_GENERATION__PROMPT_PROFILE=benchmark
export SA_GENERATION__NUM_WORKERS=1

export SA_GENERATION__LIMIT=-1  # -1 全量，首次可设 1。
export SA_GENERATION__MAX_TURNS=200
export SA_GENERATION__MAX_CONTEXT=32768
export SA_GENERATION__MAX_NEW_TOKENS=8192
export SA_GENERATION__MAX_TOOL_TOKENS=32768
export SA_GENERATION__TOOL_COUNT=200
export SA_GENERATION__TOOL_CALL_RETRIES=1
export SA_GENERATION__DECODING=sample
export SA_GENERATION__REPETITION_PENALTY=1.0
export SA_GENERATION__TEMPERATURE=1.0
export SA_GENERATION__TOP_P=1.0
export SA_GENERATION__GENERATION_SEED=42

# 使用 32 号脚本构建的 SQLite BM25 索引。
export SA_GENERATION__TITLE_WEIGHT=1.0
export SA_GENERATION__VERIFY_INDEX=true
export SA_GENERATION__TOPK=5
export SA_GENERATION__SNIPPET_TOKENS=512
export SA_GENERATION__READ_TOKENS=4096
export SA_GENERATION__ALLOW_VISIT=true

# 默认只跑官方指标。开启语义裁判时填本地权重目录。
export SA_JUDGE__LLM_JUDGE=false
export SA_JUDGE__JUDGE_MODEL=''
export SA_JUDGE__JUDGE_ATTEMPTS=3
export SA_JUDGE__JUDGE_MAX_TOKENS=4096
export SA_JUDGE__JUDGE_MAX_CONTEXT=32768
export SA_JUDGE__JUDGE_ENABLE_THINKING=false
export SA_JUDGE__JUDGE_TEMPERATURE=0.0
export SA_JUDGE__JUDGE_TOP_P=0.8
export SA_JUDGE__JUDGE_TOP_K=20
export SA_JUDGE__SEED=42

export SA_ENTRY=beacon/47_run_hotpotqa
export SA_LAUNCHER="$SCRIPT_DIR/47_run_hotpotqa.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/beacon/py/47_run_hotpotqa.py"
