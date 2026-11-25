# 30 Plain 完整实验指南

本文字段名称表示逻辑参数：SH 中普通变量加 `SA_` 前缀，例如 `MODEL_PATH` 对应 `SA_MODEL_PATH`；嵌套键用双下划线，例如 `TRAINING.max_length` 对应 `SA_TRAINING__MAX_LENGTH`。环境变量如 `HF_ENDPOINT` 不加前缀。实际可编辑名称、类型与默认值见 [60 参数](60_parameters.md)。

## 1. 实验顺序

安装环境 → 准备本地模型 → 准备一种训练数据 → 准备 BrowseComp-Plus → 校验训练计划 → 训练 → 测试 → 评分 → 检查产物。

公共前置步骤按 [10 环境准备](10_environment.md) 和 [20 数据准备](20_data.md) 执行。每个 SH 文件顶部包含完整待改参数；执行时只传脚本文件名。

## 2. 选择数据并修改训练脚本

| 训练数据 | 入口 | `SA_DATASET_DIR` 指向 |
|---|---|---|
| OpenSeeker | `scripts/plain/sh/10_train_openseeker.sh` | `artifacts/datasets/openseeker/prepared/<PROFILE>` |
| Research | `scripts/plain/sh/11_train_research.sh` | `artifacts/datasets/research/prepared/<PROFILE>` |
| BrowseComp SFT | `scripts/plain/sh/11_train_research.sh`，设 `SA_DATASET_NAME=browsecomp-sft` | `artifacts/datasets/browsecomp-sft/prepared/<PROFILE>` |

在选定文件中修改：

1. `SA_MODEL_PATH`：本地基座路径，与该数据 manifest 的 tokenizer 身份完全相同。
2. `SA_DATASET_DIR`：对应 prepared 版本，不能交叉使用数据身份。
3. `SA_MODEL_FAMILY`：默认 qwen35；使用 Qwen3 时改 qwen3 并重新按其 tokenizer 预处理。
4. `SA_RUN_NAME`：如 `qwen35_L16384_s42`；创建目录时自动加 UTC 时间和随机后缀。
5. `TRAINING`：学习率、长度、梯度累积、训练轮数/步数、随机种子、保存与评估间隔。
6. `ENVIRONMENT`：GPU 编号；多卡同步 `NPROC_PER_NODE`、`DISTRIBUTED_BACKEND`，ZeRO 配置位于本文件 `DISTRIBUTED`。

## 3. Plain 专用参数

`SA_TRAINING__USE_LORA=true` 默认训练 LoRA；`lora_r`、`lora_alpha`、`lora_dropout`、`lora_target_modules` 控制适配器。设为 false 时转为全参数训练，需重新评估显存和优化器配置。

默认启用普通 gradient checkpointing、输出 loss 分块/重算及 CPU activation offload。修改 `plain_loss_chunk_size`、`plain_cpu_offload_threshold` 时保持完整轨迹上限一致。Plain 支持单卡、DDP、ZeRO-2、ZeRO-3；多卡参数在同一训练脚本中修改。

## 4. 校验计划与训练

先将选定文件 `SA_PLAN_ONLY=true` 后运行；要求 prepared 数据已存在，程序校验来源、协议、tokenizer、长度与划分，归档配置但不加载模型。计划目录的 `operation=plan`，不能直接用于测试。

将同一文件 `SA_PLAN_ONLY=false` 后运行正式训练。例如选择 OpenSeeker：

```bash
bash scripts/plain/sh/10_train_openseeker.sh
```

选择 Research 则运行：

```bash
bash scripts/plain/sh/11_train_research.sh
```

每次正式运行建立新实验，保存 `train.yaml`、`experiment.json`、脚本副本、日志和模型。默认 LoRA 结束后自动合并，成功的 `models/model/train_summary.json` 中 `final_model_path` 是测试使用的模型。最终模型模式要求训练成功。若最终合并失败，可另选该实验中已完整保存的周期检查点。


## 5. 测试与评分

在 `scripts/plain/sh/46_run_browsecomp_plus.sh` 填写 `SA_TRAINING_RUN` 和 `SA_BENCHMARK_DIR`。默认使用最终权重，周期权重按 [33 检查点](33_checkpoint_evaluation.md) 准备。编辑 `SA_GENERATION__RETRIEVER` 选择 BM25 或向量检索，首次运行可将 `SA_GENERATION__LIMIT` 设为 1。

```bash
# 终端一启动模型服务
bash scripts/plain/sh/46_run_browsecomp_plus.sh serve 8020
# 终端二生成和评测
bash scripts/plain/sh/46_run_browsecomp_plus.sh
```

两终端均激活 cu126 环境。需要评分时设置 `SA_SCORE_AFTER_GENERATION=true`，也可以填 `SA_EVALUATION_DIR` 单独对已有结果评分。[32 vLLM 部署、全部生成参数和评分配置](32_browsecomp_plus_vllm.md)
