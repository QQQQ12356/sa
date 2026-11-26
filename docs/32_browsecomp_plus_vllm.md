# 32 BrowseComp-Plus：Baseline、Plain 与 Beacon

HotpotQA 的独立数据准备、生成和评分见 [37 HotpotQA vLLM](37_hotpotqa_vllm.md)。

三种方法共享 Qwen3.5 的 chat template、search/open/find 工具协议、观测遮蔽和评测格式。Baseline/Plain 客户端在 cu126 环境中加载 tokenizer 并渲染 token ID，请求本机 vLLM 的 `/v1/completions`。Beacon 使用相同 token ID 与浏览器，本地执行 `beacon_generate` 压缩工具正文。每题的浏览器游标、工具计数和轨迹独立保存。检索复用项目的本地 Lucene 或 FAISS 和 SQLite，生成结束后使用 BrowseComp-Plus 官方判分模板分别计算答案和检索指标。[上游运行参数](https://github.com/i-DeepSearch/observation-masking/blob/main/run_agent.sh) · [上游模型客户端](https://github.com/i-DeepSearch/observation-masking/blob/main/utils/openai_generator.py) · [上游遮蔽算法](https://github.com/i-DeepSearch/observation-masking/blob/main/tools/context_management.py)

上游 README 将 Qwen3.5-4B + BM25 列为实验设置，部署示例使用 9B。本项目将其 Qwen3.5 部署和原生工具流程应用于 4B，复用本项目的数据版本、检索索引、答题清单和裁判配置。与论文成绩对齐时需另核对题序、提示词、模型/索引 revision、窗口及裁判。[上游实验说明](https://github.com/i-DeepSearch/observation-masking#analysis--findings) · [参考提交 681de57](https://github.com/i-DeepSearch/observation-masking/tree/681de57410072a33d1f62fc5f39b1c5208900675)

## 1. 准备权重和语料

- 按 [10 环境](10_environment.md) 准备 cu126 客户端和独立的 `.venv-vllm`。
- 按 [20 数据](20_data.md#8-准备-browsecomp-plus) 完成 31 的 `download → prepare → index`。三种方法使用同一个 `SA_BENCHMARK_DIR`。
- Baseline 在自己的 46 脚本填写 `SA_MODEL_PATH`，指向下载的原始 Qwen3.5-4B commit 目录。
- Plain 在自己的 46 脚本填写 `SA_TRAINING_RUN`，指向同方法的正式训练目录。默认从 `train_summary.json` 读取最终模型。周期检查点设置 `SA_CHECKPOINT`，参见 [33 检查点](33_checkpoint_evaluation.md)。

Beacon 在 `scripts/beacon/sh/46_run_browsecomp_plus.sh` 填写 `SA_TRAINING_RUN` 和可选 `SA_CHECKPOINT`，默认 `PROMPT_PROFILE=qwen-native`。直接执行该脚本，无需服务终端。`NUM_WORKERS=1` 表示一个本地模型副本；多副本需要足够的可见 GPU。fast tokenizer 的 offset mapping 用于定位工具正文，模板对工具正文做非原样转换时会报错，避免压缩错误区间。

```bash
bash scripts/beacon/sh/46_run_browsecomp_plus.sh
```

旧 Beacon 的 42、以及 46 的 benchmark/training profile 仅供复现旧实验。新对照使用 qwen-native，`PROMPT_DATA` 留空。表中的 vLLM 配置仅适用于 Baseline/Plain，其余检索与预算默认值适用于三个方法。

## 2. Baseline/Plain 启动服务并生成

在 cu126 环境中开启服务终端。第一次验证可将选定脚本顶部的 `SA_GENERATION__LIMIT` 改为 1、`SA_GENERATION__VLLM_CONCURRENCY` 改为 1。显存不足时先减小 `SA_GENERATION__MAX_CONTEXT`，服务和客户端共用此值。

```bash
# 按待测方法执行一条。服务前台运行，Ctrl+C 停止。
bash scripts/baseline/sh/46_run_browsecomp_plus.sh serve 8010
bash scripts/plain/sh/46_run_browsecomp_plus.sh serve 8020
```

等待服务就绪，在另一个已激活 cu126 的终端执行同一个方法的入口。

```bash
bash scripts/baseline/sh/46_run_browsecomp_plus.sh
bash scripts/plain/sh/46_run_browsecomp_plus.sh
```

多副本时，在脚本的 `SA_GENERATION__VLLM_URLS` 中列出本机端口，例如 `http://127.0.0.1:8010/v1,http://127.0.0.1:8011/v1`，为每个端口分别运行 `serve PORT`，通过命令前的 `CUDA_VISIBLE_DEVICES=...` 指定各自 GPU。服务模型名和权重路径会在生成前核验。一个副本使用多张卡时传入 `SA_VLLM_TP` 及足够的可见卡。服务和 dense 查询编码可放在不同 GPU。

## 3. 参数

以下生成参数在脚本中加 `SA_GENERATION__` 前缀。

| 参数 | 默认 | 作用 |
|---|---|---|
| `VLLM_URLS` | Baseline 8010，Plain 8020 | 逗号分隔的本机 `http://127.0.0.1:PORT/v1` 地址。题目按顺序轮转分配 |
| `VLLM_CONCURRENCY`、`VLLM_TIMEOUT` | 32、300 | 每副本并发题数，以及单次 HTTP 请求超时秒数 |
| `RETRIEVER` | `bm25` | 官方 Lucene BM25。向量检索改 `qwen3-embedding-8b` |
| `VECTOR_INDEX_DIR`、`EMBEDDING_MODEL_PATH`、`EMBEDDING_DEVICE` | 空、空、`cuda:0` | 向量索引目录、与清单匹配的 embedding 模型 commit、查询编码设备。前两项取自 33 输出。[向量准备](35_browsecomp_plus_dense.md) |
| `TOPK`、`TITLE_WEIGHT`、`VERIFY_INDEX` | 15、1.0、true | 单次查询默认返回数、旧版 SQLite BM25 的标题权重、索引 SHA256 核验 |
| `LIMIT`、`PROMPT_DATE` | -1、空 | -1 取全部题，正数取筛选后的前 N 题。日期可填 `YYYY-MM-DD`，留空用运行日期，续测沿用原日期 |
| `OBSERVATION_WINDOW` | 0 | 0 保留全部轮次。正整数保留最近 N 个含成功工具结果的轮次，较旧观测替换成占位文本，原始轨迹保留 |
| `MAX_TURNS`、`TOOL_COUNT` | 200、4096 | 单题生成轮数、工具调用次数上限 |
| `MAX_CONTEXT`、`MAX_NEW_TOKENS` | 131072、8192 | 单题每轮输入加输出总预算、单轮输出上限，输出也受剩余上下文限制 |
| `DECODING`、`TEMPERATURE`、`TOP_P` | sample、1.0、1.0 | 采样策略、采样温度和累积概率阈值。greedy 使用温度 0 |
| `REPETITION_PENALTY`、`GENERATION_SEED` | 1.0、42 | 重复惩罚和采样随机种子 |

`SA_VLLM_BIN` 默认 `.venv-vllm/bin/vllm`，`SA_VLLM_GPU_MEMORY` 默认 0.90，`SA_VLLM_TP` 默认 1。以上从服务终端传入。可在两个终端通过环境变量 `SA_VLLM_API_KEY` 设置相同密钥。比较 BM25 和向量检索时保持模型、题集、预算和遮蔽设置一致，分别设置 `SA_RUN_NAME`。Plain 的 `SA_OVERLAP=filter` 根据训练数据移除重合题，`SA_OVERLAP_THRESHOLD` 是匹配阈值，默认 0.30。跨方法比较全题集时保留 `off`。

## 4. 评分与结果

默认 `SA_SCORE_AFTER_GENERATION=false`，仅生成。将其改为 true 可连续生成与评分。对已有结果评分时将完整评测根目录填入同方法脚本的 `SA_EVALUATION_DIR`，直接运行 46。远程裁判设置 `SA_JUDGE__JUDGE_BACKEND=openai`、`SA_JUDGE__JUDGE_MODEL` 与 `SA_JUDGE__JUDGE_URL`，并从终端传入 `OPENAI_API_KEY`。本地裁判设置 `local` 并将 `SA_JUDGE__JUDGE_MODEL` 改为本地权重路径，可先停止 vLLM 以释放显存。`JUDGE_MAX_TOKENS`、`JUDGE_ATTEMPTS`、`JUDGE_TIMEOUT` 分别限制裁判输出长度、尝试次数和单次请求时间。

Baseline 的结果位于 `artifacts/experiments/baseline/untrained/evaluation/browsecomp-plus/`。Plain/Beacon 的结果位于所属训练目录下的 `evaluation/browsecomp-plus/`。每个任务都有独立目录，`results/predictions.jsonl` 存逐题结果和完整轨迹，`results/manifest.json` 记录实际模型、题目和索引，`scores/*/results/summary.metrics.json` 存评分。扩充题目数和中断恢复见 [36 续测](36_browsecomp_plus_resume.md)。

## 5. BM25 与向量对照

在三个方法各自的 46 脚本中采用同一组设置，分别创建新运行：

```bash
# BM25：先完成 31 的 download → prepare → index。
export SA_GENERATION__RETRIEVER=bm25
export SA_GENERATION__VECTOR_INDEX_DIR=''
export SA_GENERATION__EMBEDDING_MODEL_PATH=''

# Qwen embedding：先完成 33 的模型与向量准备。
export SA_GENERATION__RETRIEVER=qwen3-embedding-8b
export SA_GENERATION__VECTOR_INDEX_DIR='/path/to/matching/vector/snapshot'
export SA_GENERATION__EMBEDDING_MODEL_PATH='/path/to/Qwen3-Embedding-8B/commit'
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
```

实际应将所选配置写入 SH 后执行入口。向量路径与模型必须来自同一索引清单，具体准备方式见 [35 检索索引](35_browsecomp_plus_dense.md)。官方分片直读可省略 SQLite index 阶段，所需 parquet 必须保留。BM25 与转换/自编码向量需要 SQLite 文档库。

对照时固定 `BENCHMARK_DIR`、`LIMIT`、`PROMPT_DATE`、`TOPK`、全部生成预算、解码参数、`OBSERVATION_WINDOW` 与裁判。`RUN_NAME` 分别标记方法和检索器，如 `beacon_bm25`、`beacon_dense`。`OVERLAP=off` 保留完整题集；若筛选训练重合，必须核对各方法实际 selected IDs，题集不同的结果不能直接比较。相同 seed 在不同推理后端不保证逐 token 相同。

Beacon 的原生浏览器与 Baseline/Plain 一样，每条工具结果最多保留 12000 字符的推理可见文本，完整结果保存在轨迹。`MAX_TOOL_TOKENS`、`TOOL_CALL_RETRIES`、`SNIPPET_TOKENS`、`READ_TOKENS`、`ALLOW_VISIT` 仅在 Beacon 旧 benchmark/training 协议生效；原生协议按 browser 的 search/open/find 参数控制读取。`MAX_CONTEXT` 按压缩前的输入 token 数和输出预算检查，三种方法口径相同。
