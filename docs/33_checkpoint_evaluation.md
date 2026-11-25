# 33 选择测试检查点

Baseline 通过 `SA_MODEL_PATH` 指向下载的原始 Qwen3.5-4B。Plain 和 Beacon 在各自脚本的 `SA_TRAINING_RUN` 填写同方法的正式训练实验根目录，目录中有 `job.json` 与 `experiment.json`。计划任务只有配置。默认留空 `SA_CHECKPOINT`，使用成功训练的 `models/model/train_summary.json` 中 `final_model_path`。

要评测已保存的周期权重，将 `SA_CHECKPOINT` 改为 `checkpoint-200`，也可填该训练实验 `models/model/checkpoint-200` 的完整路径。训练仍在运行或任务失败时可用已完整保存的检查点。确认该目录在评测期间会被保留。

| 权重类型 | Plain | Beacon |
|---|---|---|
| 全量权重 | vLLM 读取原目录，包含 config 与完整权重或分片 | 原生加载器读取模型参数和 Beacon 配置 |
| 最终 LoRA | 训练自动合并，测试读取 `final_model_path` | 训练流程保存 Beacon 权重和配套配置 |
| 周期 LoRA | 先生成同级的 `checkpoint-200_merged`，再设置原步数 `checkpoint-200` | 直接加载 adapter 和对应的 `beacon_state.pt` |

用 cu126 环境合并 Plain 的周期 adapter。将环境变量 `TRAINING_RUN` 和 `MODEL_PATH` 改为实际实验目录和训练基座路径。

```bash
python -m search_comp.trainer.merge_lora \
  --base_model_path "$MODEL_PATH" \
  --adapter_path "$TRAINING_RUN/models/model/checkpoint-200"
```

Plain 的 46 会从原检查点解析到同级的 `checkpoint-200_merged`。同一次实验的不同权重应使用独立的 `SA_RUN_NAME`。启动 vLLM 后检查 `/v1/models` 的名称与路径，测试完成后检查 `results/manifest.json` 中的 `model` 字段。Beacon 的周期检查点与训练提示词按 [31 Beacon](31_beacon.md) 配置。

已有结果在 Baseline/Plain 的 46 或 Beacon 的对应评分入口填入 `SA_EVALUATION_DIR`。读取 `results/predictions.jsonl` 核对题目、轨迹和答案。使用 `scripts/tools/sh/81_prediction_report.sh` 可将轨迹和匹配的评分记录渲染为同一份 HTML 报告。
