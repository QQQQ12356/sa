# 37 HotpotQA：Baseline、Plain 与 Beacon

Baseline、Plain 和 Beacon 共用 HotpotQA 提示词、fullwiki 题目和独立 Wikipedia abstracts 文档库。Baseline/Plain 生成由本机 vLLM 执行，Beacon 本地加载压缩模型，检索使用 SQLite BM25，`search` 和 `visit` 返回带零基句子编号的正文。模型输出 `{"answer": "...", "sp": [["文档标题", 0]]}`，供原有 HotpotQA 评分器计算答案、支持事实和联合指标。

## 1. 准备数据和模型

1. 激活 cu126 环境，并按 [10 环境安装](10_environment.md) 创建独立的 vLLM 服务环境。
2. 按 [20 数据准备](20_data.md#9-准备-hotpotqa) 运行 `scripts/data/sh/32_prepare_hotpotqa.sh` 的 `download → prepare → index`。默认数据目录为 `artifacts/benchmarks/hotpotqa/v1`，同时用于生成与评分。
3. 在对应的 47 号脚本修改下表中的模型设置。Plain 的周期 LoRA 权重先按 [33 检查点](33_checkpoint_evaluation.md) 合并。

| 方法 | 脚本 | 模型设置 | 默认端口 |
|---|---|---|---|
| Baseline | `scripts/baseline/sh/47_run_hotpotqa.sh` | `SA_MODEL_PATH` 指向原始 Qwen3.5-4B 本地目录 | 8030 |
| Plain | `scripts/plain/sh/47_run_hotpotqa.sh` | `SA_TRAINING_RUN` 指向 Plain 训练实验。空的 `SA_CHECKPOINT` 使用最终权重 | 8040 |

Beacon 使用 `scripts/beacon/sh/47_run_hotpotqa.sh`，填写 `SA_TRAINING_RUN` 和可选 `SA_CHECKPOINT`。默认 `PROMPT_PROFILE=benchmark`，统一提示词不接受 `PROMPT_DATA` 覆盖。`NUM_WORKERS=1` 是本地模型副本数，多副本按 GPU 分配；其余预算、检索参数及裁判默认值与下表一致。直接执行，无需 `serve`：

```bash
bash scripts/beacon/sh/47_run_hotpotqa.sh
```

旧 44/45 入口保留原训练提示词及裁判默认值，用于旧实验复现。推荐对照使用 47。HotpotQA 当前统一入口仅使用 BM25，Qwen embedding 对照用于 BrowseComp-Plus。

## 2. 启动服务、生成和评分

两个终端都激活 cu126 环境。首次检查时在脚本内将 `SA_GENERATION__LIMIT` 和 `SA_GENERATION__VLLM_CONCURRENCY` 改为 1。

```bash
# 终端一，按待测方法选择一条
bash scripts/baseline/sh/47_run_hotpotqa.sh serve 8030
bash scripts/plain/sh/47_run_hotpotqa.sh serve 8040

# 终端二，服务就绪后运行同方法的脚本
bash scripts/baseline/sh/47_run_hotpotqa.sh
bash scripts/plain/sh/47_run_hotpotqa.sh
```

- **只生成**：保持 `SA_EVALUATION_DIR` 为空、`SA_SCORE_AFTER_GENERATION=false`。
- **生成后评分**：保持 `SA_EVALUATION_DIR` 为空，把 `SA_SCORE_AFTER_GENERATION` 改为 `true`。
- **只评分**：把同方法已完成的 HotpotQA 测试根目录填入 `SA_EVALUATION_DIR`，执行 47 号脚本。只算官方指标时可关闭 vLLM，也无需 GPU。题目版本和题数从原测试记录读取。
- **续测**：把已完成的测试根目录填入 `SA_RESUME_FROM`，清空 `SA_EVALUATION_DIR`，把 `LIMIT` 改为更大的累计题数。模型、题集和生成预算沿用原测试。[36 续测](36_browsecomp_plus_resume.md)

## 3. 生成参数

表中的生成字段在 Bash 中使用 `SA_GENERATION__` 前缀。其余顶层参数包括 `SA_BENCHMARK_DIR`（已准备的数据版本）、`SA_RUN_NAME`（新评测的标签）和第 2 节的运行模式。服务侧使用 `SA_VLLM_BIN`（可执行文件）、`SA_VLLM_TP`（单副本 GPU 数，默认 1）、`SA_VLLM_GPU_MEMORY`（显存使用比例，默认 0.90）。如启用 `SA_VLLM_API_KEY`，两终端使用相同值。多副本时一端口一服务，用命令前的 `CUDA_VISIBLE_DEVICES` 分配显卡。

| 字段 | 默认 | 含义 |
|---|---|---|
| `VLLM_URLS` | Baseline 8030，Plain 8040 | 一个或多个本机 `http://127.0.0.1:PORT/v1` 地址，以逗号分隔 |
| `VLLM_CONCURRENCY`、`VLLM_TIMEOUT` | 32、300 | 每副本最大并发题数，单次 HTTP 请求超时秒数 |
| `LIMIT` | -1 | -1 使用全量，正数取题集前 N 题 |
| `MAX_TURNS`、`TOOL_COUNT` | 200、200 | 每题生成轮数和工具调用数量上限 |
| `MAX_CONTEXT`、`MAX_NEW_TOKENS` | 32768、8192 | 每轮输入与输出的总预算，单轮最大输出。服务与客户端共用上下文上限 |
| `MAX_TOOL_TOKENS`、`TOOL_CALL_RETRIES` | 32768、1 | 单条工具返回的 token 上限，工具调用格式错误时允许修正的次数 |
| `DECODING`、`TEMPERATURE`、`TOP_P` | sample、1.0、1.0 | 解码方式、采样温度和累积概率阈值。greedy 使用温度 0 |
| `REPETITION_PENALTY`、`GENERATION_SEED` | 1.0、42 | 重复惩罚系数和采样随机种子 |
| `TOPK`、`TITLE_WEIGHT`、`VERIFY_INDEX` | 5、1.0、true | 每条查询的返回上限、BM25 标题权重、启动时扫描索引 SHA256。标题权重须与建索引时一致 |
| `SNIPPET_TOKENS`、`READ_TOKENS`、`ALLOW_VISIT` | 512、4096、true | 搜索摘录与正文读取各自的单篇 token 上限，是否开启 `visit` |

## 4. 评分与产物

官方答案 EM/F1、支持事实指标和联合指标始终计算。支持事实取模型输出的 `sp`，分数以本次选中的题集为分母。可通过 `SA_JUDGE__LLM_JUDGE=true` 另算语义准确率，填写本地 `SA_JUDGE__JUDGE_MODEL`。该评分直接加载裁判模型，需要相应显存。

| 裁判字段（前缀 `SA_JUDGE__`） | 默认 | 含义 |
|---|---|---|
| `LLM_JUDGE`、`JUDGE_MODEL` | false、空 | 是否执行语义评分、本地裁判权重目录 |
| `JUDGE_ATTEMPTS` | 3 | 每题裁判的最大尝试次数 |
| `JUDGE_MAX_TOKENS`、`JUDGE_MAX_CONTEXT` | 4096、32768 | 单次裁判输出和总上下文上限 |
| `JUDGE_ENABLE_THINKING` | false | 裁判模板是否启用思考 |
| `JUDGE_TEMPERATURE`、`JUDGE_TOP_P`、`JUDGE_TOP_K`、`SEED` | 0.0、0.8、20、42 | 裁判采样参数，温度为 0 时使用确定性解码 |

Baseline 的结果位于 `artifacts/experiments/baseline/untrained/evaluation/hotpotqa/`，Plain/Beacon 位于所属训练实验的 `evaluation/hotpotqa/`。推理输出为 `results/predictions.jsonl`，评分摘要为 `scores/*/results/summary.metrics.json`。评分目录还包含 `per_question.json` 和官方格式的 `hotpot_predictions.json`。
