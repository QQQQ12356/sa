# 35 BrowseComp-Plus 使用官方 BM25 和 Qwen3-Embedding-8B 检索

使用默认 `v2` 数据时，`bm25` 用 Pyserini 读取 Tevatron 官方预建 Lucene 索引，`qwen3-embedding-8b` 用本地模型编码查询并使用 CPU FAISS `IndexFlatIP` 排序。BM25 和转换后的向量索引从 SQLite 读取文档正文。官方向量直读模式从语料 parquet 读取正文。

官方向量直读模式在评测启动时将四个预编码分片（约 1.64 GB）逐个加入内存 FAISS。自定义语料可使用 `encode`。 [官方 Lucene 索引](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/main/bm25) · [官方向量分片](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/main/qwen3-embedding-8b) · [参考检索实现](https://github.com/i-DeepSearch/observation-masking/blob/main/tools/backend.py)

## 1. 准备题目、语料和文档库

在已激活的 Linux 评测环境中安装检索依赖。官方 BM25 排序和 三种方法原生浏览器的摘要高亮均需要 Java 21，确保 `java -version` 显示 21 且 `JAVA_HOME` 指向同一安装。

```bash
python -m pip install -r requirements.browsecomp-plus.txt
java -version
```

检索依赖固定为 Pyserini 1.4.0 和 FAISS 1.13.2，并使用训练版本清单作为 pip 约束。Pyserini 1.4.0 使用 Lucene 9.9.1，支持项目的 PyTorch 2.7.0+cu126。旧的 Pyserini 1.5.0 要求 PyTorch 至少为 2.9，安装时可能替换原 CUDA wheel。 [Pyserini 1.4.0 发布说明](https://github.com/castorini/pyserini/blob/master/docs/release-notes/release-notes-v1.4.0.md) · [Pyserini 1.5.0 依赖](https://pypi.org/pypi/pyserini/1.5.0/json)

官方向量直读模式只需完成 [31 数据准备](20_data.md#8-准备-browsecomp-plus)的 `download → prepare`，保留原始 `raw/corpus` parquet。BM25、已有的转换索引和自定义向量索引还需要完成 `index` 阶段。评测的 `SA_BENCHMARK_DIR` 和 33 脚本的 `SA_VERSION` 应指向同一数据版本。

官方向量要求完整的官方语料及完全匹配的文档 ID。转换索引和自定义索引的 `prepare` 阶段从 `prepared/` 读取 `manifest.json` 和 `corpus.jsonl`，也可通过 `SA_PREPARED_DIR` 指定其他位置。

## 2. 准备向量索引

在 Linux 上编辑并执行：

```bash
bash scripts/data/sh/33_prepare_browsecomp_plus_dense.sh
```

1. 设置 `SA_PHASE=download`、`SA_SOURCE=official` 和 `HF_HUB_OFFLINE=0` 后运行。记录打印的模型 commit 目录和 `SA_OFFICIAL_INDEX_DIR`。模型保存在 `artifacts/models/qwen3-embedding-8b/<commit>/`，官方文档向量快照保存在目标 benchmark 版本的 `raw/qwen3-embedding-8b/<commit>/`。
2. 将下载输出的模型 commit 目录填入评测的 `SA_GENERATION__EMBEDDING_MODEL_PATH`，将官方分片快照根目录填入 `SA_GENERATION__VECTOR_INDEX_DIR`。此前已下载的快照可直接填原来的 `SA_OFFICIAL_INDEX_DIR`。

如需生成旧格式的转换索引，可继续运行 `SA_PHASE=prepare`，填写 `SA_EMBEDDING_MODEL_PATH` 和 `SA_OFFICIAL_INDEX_DIR`。转换产物位于 `vector_indexes/<INDEX_PROFILE>/`，包含 `vectors.npy`、`docids.json` 和 `manifest.json`。

自行编码时，从第一步开始设 `SA_SOURCE=encode`。下载阶段只下载 embedding 模型。准备阶段从本地 prepared 语料逐批编码每篇文档的正文，并按 `SA_DOC_MAX_LENGTH` 截取编码输入。每篇文档对应一个向量，原始正文和访问接口继续由 SQLite 提供。换用新 `SA_INDEX_PROFILE` 以便保留已有产物。失败后检查该次 `jobs/` 日志。若输出目录已留下部分文件，请换新 profile 重跑。

### 准备脚本参数

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `PYTHON_BIN` | 当前环境的 `python` | 使用装有仓库训练依赖的 Linux 环境。也可在启动前设置解释器绝对路径。 |
| `HF_ENDPOINT` | `https://huggingface.co` | 下载使用的 Hub 地址，可改成 Linux 可访问的镜像。 |
| `HF_HUB_OFFLINE` | `0` | 下载时使用 0。准备阶段仅访问本地文件，可设 1。 |
| `CUDA_VISIBLE_DEVICES` | `0` | 自行编码使用的物理 GPU 编号。导入官方向量可在 CPU 任务上运行。 |
| `SA_VERSION` | `v2` | 题目和语料版本，与数据准备 31 及评测的 benchmark 目录一致。 |
| `SA_INDEX_PROFILE` | `qwen3-embedding-8b-official` | 向量产物版本名。只允许字母、数字、下划线和连字符；改变输入或配置时换名。 |
| `SA_PHASE` | `download` | `download` 下载权重和所需官方向量；`prepare` 转换官方向量或自行编码。 |
| `SA_SOURCE` | `official` | `official` 导入四个官方分片；`encode` 对本地 prepared 语料自行编码。 |
| `SA_MODEL_REVISION` | `main` | 下载阶段的 Qwen3-Embedding-8B revision。正式实验可填固定 commit；程序对 `main` 解析并记录实际 commit。 |
| `SA_INDEX_REVISION` | 空 | 官方向量下载沿用同版本 31 记录的索引 commit。显式填写时须与 31 一致。 |
| `SA_EMBEDDING_MODEL_PATH` | 空 | 准备阶段必填。填模型下载输出的 commit 目录，包含模型文件和 `download_manifest.json`。同目录用于评测的查询编码。 |
| `SA_OFFICIAL_INDEX_DIR` | 空 | 官方导入必填。填向量下载输出的快照根目录，目录内有 `download_manifest.json`。自行编码可留空。 |
| `SA_PREPARED_DIR` | 空 | 仅 `prepare` 使用，适用于 `official` 和 `encode`。填含 `manifest.json` 和 `corpus.jsonl` 的目录。相对路径以项目根目录为准。留空使用 `artifacts/benchmarks/browsecomp-plus/<VERSION>/prepared/`。向量索引的输出位置仍由 `SA_VERSION` 和 `SA_INDEX_PROFILE` 决定。 |
| `SA_DEVICE` | `cuda:0` | 仅用于自行编码。`cuda:0` 是第一张可见 GPU，使用 FP16；`cpu` 用 FP32，编码通常更慢。 |
| `SA_BATCH_SIZE` | `4` | 自行编码的每批文档数。遇到显存不足先减小该值。 |
| `SA_DOC_MAX_LENGTH` | `4096` | 自行编码的单篇输入 token 上限，范围 1–32768。影响向量覆盖的文本范围。 |

## 3. 运行评测和评分

把下述参数填入 Baseline、Plain 或 Beacon 的 46 脚本。Beacon 也可分别运行 42 和 43。

按 [32 vLLM 评测](32_browsecomp_plus_vllm.md) 或 [31 Beacon](31_beacon.md) 配置权重。向量模式填写以下四项。

```bash
export SA_GENERATION__RETRIEVER=qwen3-embedding-8b
export SA_GENERATION__VECTOR_INDEX_DIR=/path/to/official-vector-snapshot
export SA_GENERATION__EMBEDDING_MODEL_PATH=/path/to/embedding-model-commit
export SA_GENERATION__EMBEDDING_DEVICE=cuda:0
```

| 变量 | 默认值 | 作用和修改提示 |
|---|---|---|
| `SA_GENERATION__RETRIEVER` | `bm25` | 可改为 `qwen3-embedding-8b` 启用向量检索。 |
| `SA_GENERATION__VECTOR_INDEX_DIR` | 空 | 向量模式必填，填写 33 下载的官方分片快照根目录，或已有的转换索引目录。 |
| `SA_GENERATION__EMBEDDING_MODEL_PATH` | 空 | 向量模式必填，填写本地模型 commit 目录。已有转换索引使用其清单绑定的模型。 |
| `SA_GENERATION__EMBEDDING_DEVICE` | `cuda:0` | 查询编码设备。每个生成进程各加载一份 embedding 模型，且每个 worker 只见到其分配 GPU。多 worker 用 `cuda:0` 表示各自的卡。单 worker 显存紧张可改为 `cpu`，需要为 FP32 权重及推理留足内存。 |
| `SA_GENERATION__VERIFY_INDEX` | `true` | 校验所用语料、索引文件的 SHA256。关闭时仍核对来源和文件清单。 |
| `SA_GENERATION__TITLE_WEIGHT` | `1.0` | 用于旧版或自定义语料的 SQLite FTS5 排序。 |
| `SA_GENERATION__TOPK` | `15`（46）；`5`（42） | 每条查询取回的不同文档数。 |
| `SA_GENERATION__SNIPPET_TOKENS` / `SA_GENERATION__READ_TOKENS` | `512` / `4096` | 仅 HotpotQA/旧 Beacon 协议控制每篇摘录和 visit 的 token 预算；推荐原生 BCP 由 browser 参数控制。 |

首次以 `SA_GENERATION__LIMIT=5` 跑小样本，检查 `results/manifest.json` 的 `index.retriever`、`index.ranking` 和向量模式下的 `index.vector_index`。官方向量直读模式的 `index.version` 为 `parquet-corpus-v1`。每个评测进程都加载一份 CPU FAISS 索引，全量向量约 1.53 GiB，直读还会将语料载入内存。

评分继续使用同方法的 46，Beacon 也可用 43，将 `SA_EVALUATION_DIR` 指向刚生成的测试记录目录。它复用已生成的逐题预测和题目范围。正式对照时，固定 checkpoint、题目范围、重合筛选、提示词、TopK、工具观测预算与解码参数，仅切换检索器。比较答案准确率、`evidence_recall`、`gold_recall` 和推理耗时。

## 4. 索引核验与排错

- 官方直读在评测时核对语料和向量的官方快照、分片维度及完整 docid 对应关系，默认校验文件 SHA256。转换索引继续使用现有的语料摘要、模型快照与归一化校验。官方文档向量的分词和截断以预编码分片为准。
- 遇到语料摘要或文档 ID 不匹配，检查 31 和 33 的 `SA_VERSION`、评测 `SA_BENCHMARK_DIR` 以及向量目录。自定义语料使用 `encode` 生成自身的向量索引。
- 模型快照不匹配时，恢复构建向量用的模型目录或使用新 profile 重新准备。评测中的 `SA_MODEL_PATH` 或 `SA_TRAINING_RUN` 仍负责选择作答模型。
- GPU 显存不足时，先降准备阶段的 `SA_BATCH_SIZE`。评测阶段可让查询编码使用 CPU，或使用能同时容纳 Agent 与 embedding 模型的 GPU。
