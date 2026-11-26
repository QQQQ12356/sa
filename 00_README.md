# SearchAgent 实验手册

在 Linux/cu126 环境中进行训练和正式评测；Windows 可编辑代码并运行不依赖 CUDA 的测试。先完成环境与共享数据准备，再修改对应 Bash 入口中的参数。

| 方法 | 训练/模型来源 | BrowseComp-Plus | HotpotQA |
|---|---|---|---|
| Baseline | 原始本地 Qwen3.5-4B | `scripts/baseline/sh/46_run_browsecomp_plus.sh` | `scripts/baseline/sh/47_run_hotpotqa.sh` |
| Plain | [Plain 训练](docs/30_plain.md)实验及可选检查点 | `scripts/plain/sh/46_run_browsecomp_plus.sh` | `scripts/plain/sh/47_run_hotpotqa.sh` |
| Beacon | [Beacon 训练](docs/31_beacon.md)实验及可选检查点 | `scripts/beacon/sh/46_run_browsecomp_plus.sh` | `scripts/beacon/sh/47_run_hotpotqa.sh` |

三种方法的推荐评测入口共享题集、提示词、工具和默认预算。BrowseComp-Plus 支持官方 Lucene BM25 与 Qwen3-Embedding-8B；HotpotQA 使用 SQLite BM25。Baseline/Plain 由本机 vLLM 推理，Beacon 在 cu126 环境中本地加载压缩模型。Beacon 的压缩配置从训练检查点读取。完整步骤分别见 [BrowseComp-Plus](docs/32_browsecomp_plus_vllm.md) 和 [HotpotQA](docs/37_hotpotqa_vllm.md)。

## 文档导航

按用途保留以下核心文档，参数含义以参数手册与对应专题为准；每份 Bash 脚本列出该入口的实际默认值。

| 分类 | 核心文档 | 阅读目的 |
|---|---|---|
| 安装 | [10 环境](docs/10_environment.md) | cu126、Java、独立 vLLM 环境及依赖 |
| 数据 | [20 数据准备](docs/20_data.md) · [21 BrowseComp SFT](docs/21_browsecomp_sft.md) · [35 检索索引](docs/35_browsecomp_plus_dense.md) | 模型、训练集、题集、BM25 和向量索引 |
| 训练 | [30 Plain](docs/30_plain.md) · [31 Beacon](docs/31_beacon.md) | 训练计划、配置和压缩参数 |
| 评测 | [32 BrowseComp-Plus](docs/32_browsecomp_plus_vllm.md) · [37 HotpotQA](docs/37_hotpotqa_vllm.md) | 三方法的生成、评分与公平对照 |
| 评测维护 | [33 检查点](docs/33_checkpoint_evaluation.md) · [34 训练轨迹评测](docs/34_training_data_evaluation.md) · [36 续测与恢复](docs/36_browsecomp_plus_resume.md) | 周期权重、训练诊断和扩充题数 |
| 参数与结果 | [60 参数](docs/60_parameters.md) · [70 产物](docs/70_artifacts.md) · [82 Excel 导出](docs/82_benchmark_excel.md) | 参数类型/含义、实验归档和结果汇总 |
| 在线 BrowseComp | [31 在线入口](docs/31_beacon.md#5-配置测试) · [11 摘要服务](docs/11_sglang_summary.md) | 仅在线搜索与页面摘要实验需要 |

## 代码与验证

- `scripts/<方法>/{sh,py}`：方法入口。编辑 SH，PY 负责类型解析与调用。
- `search_comp/experiments/`：共享配置、模型来源、评测调度与归档。
- `search_comp/evaluation/`：模型生成、工具协议、答案解析与续测。
- `search_comp/offline/`：语料、检索器、浏览器和评分。
- `search_comp/workflows/`：准备、运行、评分与报告的实现。
- `tests/`：配置契约、工具流程、检索和模型测试。新产物统一位于 `artifacts/`。

正式依赖环境中执行 `python -m pytest -q`。只检查本次评测对齐可执行：

```bash
python -m pytest -q tests/test_beacon_evaluation.py tests/test_experiment_contracts.py tests/test_shell_config.py
```

这些测试使用模拟模型和检索器核对路由、预算与结果；需要 `json5`，不需要下载权重或索引。真实模型、Java/Lucene、FAISS 和多 GPU 验证须在 Linux 实验环境执行。
