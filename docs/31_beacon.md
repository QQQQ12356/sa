# 31 Beacon 完整实验指南

本文字段名称表示逻辑参数：SH 中普通变量加 `SA_` 前缀，例如 `MODEL_PATH` 对应 `SA_MODEL_PATH`；嵌套键用双下划线，例如 `TRAINING.max_length` 对应 `SA_TRAINING__MAX_LENGTH`。环境变量如 `HF_ENDPOINT` 不加前缀。实际可编辑名称、类型与默认值见 [60 参数](60_parameters.md)。

## 1. 实验顺序

安装环境 → 准备本地模型 → 准备一种训练数据 → 准备三个 benchmark → 校验训练计划 → 训练 → 三项测试 → 三项独立评分 → 检查产物。

公共前置步骤按 [10 环境准备](10_environment.md) 和 [20 数据准备](20_data.md) 执行。每个 SH 文件顶部包含完整待改参数；执行时只传脚本文件名。

## 2. 选择数据并修改训练脚本

| 训练数据 | 入口 | `SA_DATASET_DIR` 指向 |
|---|---|---|
| OpenSeeker | `scripts/beacon/sh/10_train_openseeker.sh` | `artifacts/datasets/openseeker/prepared/<PROFILE>` |
| Research | `scripts/beacon/sh/11_train_research.sh` | `artifacts/datasets/research/prepared/<PROFILE>` |
| [BrowseComp SFT](21_browsecomp_sft.md) | `scripts/beacon/sh/11_train_research.sh`，设 `SA_DATASET_NAME=browsecomp-sft` | `artifacts/datasets/browsecomp-sft/prepared/<PROFILE>` |
| [混合数据](20_data.md#5-可选混合-openseeker-与-research) | `scripts/beacon/sh/11_train_research.sh`，设 `SA_DATASET_NAME=mixed` | `artifacts/datasets/mixed/prepared/<PROFILE>` |

在选定文件中修改：

1. `SA_MODEL_PATH`：本地基座路径，与该数据 manifest 的 tokenizer 身份完全相同。
2. `SA_DATASET_DIR`：对应 prepared 版本，不能交叉使用数据身份。
3. `SA_MODEL_FAMILY`：默认 qwen35；使用 Qwen3 时改 qwen3 并重新按其 tokenizer 预处理。
4. `SA_RUN_NAME`：如 `qwen35_L16384_s42`；创建目录时自动加 UTC 时间和随机后缀。
5. `TRAINING`：学习率、长度、梯度累积、训练轮数/步数、随机种子、保存与评估间隔。
6. `ENVIRONMENT`：GPU 编号；多卡同步 `NPROC_PER_NODE`、`DISTRIBUTED_BACKEND`，ZeRO 配置位于本文件 `DISTRIBUTED`。

## 3. Beacon 专用参数

默认 `beacon_layout_policy='configured'`，`beacon_information_mode='whole'`，`beacon_window_enabled=false`，按完整工具观测区段压缩；默认固定比例 `beacon_ratio=16`。只压缩工具观测体，保留原始 system、问题和 assistant 监督协议。

| 目标 | 修改项 | 联动要求 |
|---|---|---|
| 固定压缩比例 | `SA_BEACON__BEACON_RATIO` | 训练与测试读取同一 checkpoint 配置 |
| 随机压缩训练 | `beacon_compression_mode='random'`、`beacon_compression_ratios` | 推理仍使用固定 `beacon_ratio` |
| 窗口布局 | `beacon_information_mode='window'`、`beacon_window_enabled=true`、`beacon_window`、`beacon_stride` | 保持 `beacon_layout_policy='configured'`；stride 不大于 window |
| 兼容整段布局 | `SA_TRAINING__BEACON_LAYOUT_POLICY='whole'` | 会规范化部分布局字段；先检查归档 train.yaml |
| 输出蒸馏 | `beacon_output_distill_weight`、teacher 参数 | 大于 0 会额外加载 teacher；预留内存与显存 |
| 显存优化 | `beacon_checkpoint_blocks`、`beacon_checkpoint_loss`、CPU offload | 训练默认开启；不用普通 gradient_checkpointing 代替 |

逐项默认值和字段说明见 [60 参数手册](60_parameters.md)。Beacon 多卡训练支持 DDP 和 ZeRO-2。推荐 46/47 评测入口默认使用 `sample`；旧入口默认 `greedy`。设置 `SA_GENERATION__DECODING=sample` 可启用按题设种子的采样，并用 `SA_GENERATION__TEMPERATURE` 和 `SA_GENERATION__TOP_P` 调整采样。

## 4. 校验计划与训练

先将选定文件 `SA_PLAN_ONLY=true` 后运行；要求 prepared 数据已存在，程序校验来源、协议、tokenizer、长度与划分，归档配置但不加载模型。计划目录的 `operation=plan`，不能直接用于测试。

将同一文件 `SA_PLAN_ONLY=false` 后运行正式训练。例如选择 OpenSeeker：

```bash
bash scripts/beacon/sh/10_train_openseeker.sh
```

选择 Research 则运行：

```bash
bash scripts/beacon/sh/11_train_research.sh
```

每次正式运行建立新实验，保存 `train.yaml`、`experiment.json`、脚本副本、日志和模型。默认 LoRA 结束后自动合并，成功的 `models/model/train_summary.json` 中 `final_model_path` 是测试使用的模型。最终模型模式要求训练成功。若最终合并失败，可另选该实验中已完整保存的周期检查点。

## 5. 配置测试

正式对照实验使用下面两个统一入口，完整生成、评分与参数说明集中在对应评测文档。

| Benchmark | 推荐入口 | 共享行为 |
|---|---|---|
| BrowseComp-Plus | `scripts/beacon/sh/46_run_browsecomp_plus.sh` | [32 评测说明](32_browsecomp_plus_vllm.md)：与 Baseline/Plain 使用相同 Qwen 原生浏览器、BM25/向量检索和预算 |
| HotpotQA | `scripts/beacon/sh/47_run_hotpotqa.sh` | [37 评测说明](37_hotpotqa_vllm.md)：统一 benchmark 提示词、BM25、答案/支持事实指标 |

填写 `SA_TRAINING_RUN`；`SA_CHECKPOINT` 留空使用成功训练的最终模型，或填写同实验的 `checkpoint-200`。周期检查点也可来自运行中或失败的正式训练，文件完整性要求见 [33 检查点](33_checkpoint_evaluation.md)。`SA_PROMPT_DATA` 在推荐提示词下留空。混合训练实验也能直接使用推荐提示词。

```bash
bash scripts/beacon/sh/46_run_browsecomp_plus.sh
bash scripts/beacon/sh/47_run_hotpotqa.sh
```

Beacon 使用本地模型与 fast tokenizer，通过原生模板定位工具正文的 token 区间，仅压缩这些区间。问题、system、assistant 推理、工具调用和协议边界保留。BM25 与向量模式使用相同区域标记。`SA_GENERATION__NUM_WORKERS` 是本地模型副本数，每张可见 GPU 加载一份模型；不使用 vLLM `serve`。

首次将 `LIMIT` 设为 1–5；正式运行改 `-1`。默认 `SCORE_AFTER_GENERATION=false` 只生成，可独立安排裁判资源。跨方法请固定相同题集、日期、预算、检索器和观测窗口；压缩配置取自所选权重。

## 6. 评分与兼容入口

46/47 均支持三种模式：`EVALUATION_DIR` 为空时生成；同时启用 `SCORE_AFTER_GENERATION=true` 时接着评分；填写同方法同 benchmark 的成功测试根目录时只评分。题集和 limit 自动沿用原测试，结果写入新的 `scores/` 子目录。BrowseComp-Plus 支持 local/openai 裁判；HotpotQA 默认只算官方指标，可额外开启本地语义裁判。详细参数见对应评测文档与 [60 参数](60_parameters.md)。

保留以下入口用于在线任务或复现旧实验。旧协议与推荐原生浏览器协议不同，请分别记录结果，续测时沿用原入口及原配置。

| 用途 | 生成 | 独立评分 |
|---|---|---|
| 在线 BrowseComp | `40_test_browsecomp.sh` | `41_score_browsecomp.sh` |
| 旧 BrowseComp-Plus 工具协议 | `42_test_browsecomp_plus.sh` | `43_score_browsecomp_plus.sh` |
| 旧 HotpotQA 训练提示词 | `44_test_hotpotqa.sh` | `45_score_hotpotqa.sh` |

这些文件均位于 `scripts/beacon/sh/`。在线环境见 [11 摘要服务](11_sglang_summary.md)。旧 43 使用本地裁判目录，旧 45 默认开启本地语义裁判；统一 47 默认关闭语义裁判。评分失败/缺失生成保持在分母内；不完整裁判不能作为完整成绩。续测与中断恢复见 [36](36_browsecomp_plus_resume.md)。

## 7. 归档与排查

实验归档在 `artifacts/experiments/beacon/<dataset>/<run>/`。依次检查根 `job.json` 的状态、`console.log`、模型训练摘要、各测试 `evaluation.json` 和评分指标。

使用 `scripts/tools/sh/81_prediction_report.sh` 检查预测轨迹；将 `METHOD='beacon'` 并填写对应 benchmark 和测试目录。目录规范见 [70 产物规范](70_artifacts.md)。遇到失败保留原记录，修正参数后重新运行，程序不自动覆盖或继续旧训练。
