# 00 SearchAgent 实验手册

| 方法 | 训练 | 评测 |
|---|---|---|
| Baseline | 使用原始 Qwen3.5-4B 权重 | [BrowseComp-Plus · vLLM](docs/32_browsecomp_plus_vllm.md) |
| Plain | [OpenSeeker、Research、BrowseComp SFT 或混合数据](docs/30_plain.md) | [BrowseComp-Plus · vLLM](docs/32_browsecomp_plus_vllm.md) |
| Beacon | [压缩算法配置与训练](docs/31_beacon.md) | [BrowseComp-Plus、BrowseComp、HotpotQA](docs/31_beacon.md#5-配置测试) |

在 Linux 项目目录下运行 Bash 入口。先激活现有 cu126 环境，完成共享数据准备，再编辑对应方法脚本顶部的变量。Baseline 和 Plain 测 BrowseComp-Plus 时用同一套 Qwen 原生工具调用、固定语料和 BM25 / Qwen3-Embedding-8B 两种检索器。Beacon 的训练、原生推理和评测入口位于自己的目录。

- 环境：[10 cu126、Java 和独立 uv/vLLM](docs/10_environment.md) · [11 Beacon 在线摘要服务](docs/11_sglang_summary.md)
- 数据：[20 模型、训练集和 benchmark](docs/20_data.md) · [21 BrowseComp SFT](docs/21_browsecomp_sft.md) · [35 向量索引](docs/35_browsecomp_plus_dense.md)
- 评测：[32 Baseline/Plain 推理与评分](docs/32_browsecomp_plus_vllm.md) · [33 检查点](docs/33_checkpoint_evaluation.md) · [34 训练轨迹评测](docs/34_training_data_evaluation.md) · [36 续测与中断恢复](docs/36_browsecomp_plus_resume.md)
- 参数与产物：[60 参数](docs/60_parameters.md) · [70 实验目录](docs/70_artifacts.md) · [82 Excel 导出](docs/82_benchmark_excel.md)

`scripts/<方法>/{sh,py}` 保存各方法入口，`scripts/data` 准备共享资产，`search_comp/` 包含实现，`tests/` 包含单元与集成测试。模型、数据、训练和评分产物写入 `artifacts/`。在有依赖的环境中运行 `python -m pytest -q` 检查代码。
