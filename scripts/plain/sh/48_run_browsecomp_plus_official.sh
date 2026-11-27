#!/usr/bin/env bash
# BrowseComp-Plus official 独立入口。说明见 docs/21_browsecomp_sft.md。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
[[ $(uname -s) == Linux ]] || { echo '此入口仅支持 Linux。' >&2; exit 2; }
unset SA_SERVE_PORT
if (( $# )); then
  [[ $# == 2 && $1 == serve && $2 =~ ^[0-9]+$ ]] || { echo '用法：bash 本脚本 [serve PORT]' >&2; exit 2; }
  export SA_SERVE_PORT="$2"
fi
cd -- "$PROJECT_ROOT"

# 由 cu126 Python 启动。通过 CUDA_VISIBLE_DEVICES 为服务或向量检索选择 GPU。
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SA_BENCHMARK_DIR=artifacts/benchmarks/browsecomp-plus/v2

# 填写本方法的 Qwen3.5-4B 训练实验根目录。
export SA_TRAINING_RUN=artifacts/experiments/plain/browsecomp-sft/REPLACE_WITH_RUN
export SA_CHECKPOINT=''  # 空值使用最终模型；也可填 checkpoint-200。
export SA_RUN_NAME=plain-bcp4b-official
export SA_OVERLAP=filter  # off 保留完整题集；filter 对训练实验排除重合题。
export SA_OVERLAP_THRESHOLD=0.30
export SA_RESUME_FROM=''  # 从已完成评测扩展题数时填写旧评测根目录。
export SA_EVALUATION_DIR=''  # 填已有评测根目录时，仅重新评分。
export SA_SCORE_AFTER_GENERATION=false  # EVALUATION_DIR 为空时生效。false 仅生成。

# 独立 official 实验入口，不使用浏览器协议。
export SA_GENERATION__PROMPT_PROFILE=official
export SA_GENERATION__SNIPPET_TOKENIZER_PATH=/models/Qwen3-0.6B  # 本地官方片段 tokenizer，必填。
export SA_PROMPT_DATA=''  # 固定官方任务模板，不接受训练 system 覆盖。
export SA_GENERATION__NUM_WORKERS=1
# 本入口工具返回长度与调用预算。
export SA_GENERATION__SNIPPET_TOKENS=512
export SA_GENERATION__READ_TOKENS=4096
export SA_GENERATION__MAX_TOOL_TOKENS=32768
export SA_GENERATION__TOOL_CALL_RETRIES=1
export SA_GENERATION__ALLOW_VISIT=false

# 推理端口支持逗号分隔的本机副本。每个副本最多同时处理 32 题。
export SA_GENERATION__VLLM_URLS=http://127.0.0.1:8020/v1
export SA_GENERATION__VLLM_CONCURRENCY=32
export SA_GENERATION__VLLM_TIMEOUT=300
export SA_GENERATION__PROMPT_DATE=''  # 本入口不向 system 追加日期。

# LIMIT=-1 全量，首次检查建议 1 至 5 题。
export SA_GENERATION__LIMIT=-1
# 0 为 No CM；5 保留最近五个成功工具轮次。对照实验使用独立 RUN_NAME。
export SA_GENERATION__OBSERVATION_WINDOW=0
export SA_GENERATION__MAX_TURNS=200
export SA_GENERATION__MAX_CONTEXT=131072
export SA_GENERATION__MAX_NEW_TOKENS=10000
export SA_GENERATION__TOOL_COUNT=4096
export SA_GENERATION__DECODING=sample
export SA_GENERATION__REPETITION_PENALTY=1.0
export SA_GENERATION__TEMPERATURE=0.7
export SA_GENERATION__TOP_P=0.8
export SA_GENERATION__GENERATION_SEED=42

# bm25=官方 Lucene（31）；qwen3-embedding-8b=官方向量 + FAISS（31、33）。
export SA_GENERATION__RETRIEVER=bm25
export SA_GENERATION__VECTOR_INDEX_DIR=''  # 本地向量模式填 33 的输出目录。
export SA_GENERATION__EMBEDDING_MODEL_PATH=''  # 本地向量模式填匹配的 Qwen3-Embedding-8B。
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
export SA_GENERATION__VERIFY_INDEX=true
export SA_GENERATION__TITLE_WEIGHT=1.0  # 仅旧版或自定义语料的 SQLite BM25 使用。
export SA_GENERATION__TOPK=5

# openai 使用 OPENAI_API_KEY（从 Linux 环境传入）；local 使用本地裁判路径。
export SA_JUDGE__JUDGE_BACKEND=openai
export SA_JUDGE__JUDGE_MODEL=gpt-5-mini
export SA_JUDGE__JUDGE_URL=https://api.openai.com/v1
export SA_JUDGE__JUDGE_MAX_TOKENS=4096
export SA_JUDGE__JUDGE_ATTEMPTS=3
export SA_JUDGE__JUDGE_TIMEOUT=120

export SA_ENTRY=plain/48_run_browsecomp_plus_official
export SA_LAUNCHER="$SCRIPT_DIR/48_run_browsecomp_plus_official.sh"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/plain/py/48_run_browsecomp_plus_official.py"
