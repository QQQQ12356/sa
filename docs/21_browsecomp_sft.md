# BrowseComp-Plus SFT 数据接入

下载和预处理在 Linux 上运行，使用仓库的训练环境和目标模型的本地 tokenizer。预处理使用 CPU。来源固定为 [BrowseComp-Plus SFT 数据集](https://huggingface.co/datasets/AmanPriyanshu/tool-reasoning-sft-RESEARCH-grill-lab-browsecomp-plus-runs-data-cleaned-rectified/tree/1a67da5e15590e15fc962035a75cd427a47070b2)。

| 原始分片 | 内容 | 训练使用方式 |
|---|---|---|
| `syn_planner.parquet` | 19,505 条网页搜索与抓取轨迹，包含 HotpotQA 等六个来源。 | 默认选用。保留实际评测题文件用于去重。 |
| `data.parquet` | 30,355 条 BrowseComp 运行记录，`user` 消息可能含题目的长改写或原题。已确认与 BrowseComp、BrowseComp-Plus 评测同题。 | 显式选择，并提供原题文件用于归并。只要实验仍用这两套题评测，就把它们保留在受保护列表中。若实验专门研究这些轨迹，可使用独立评测集，按实验范围调整保护列表。 |

项目统一称该数据集为 **BrowseComp-Plus SFT**，与 Research（`AmanPriyanshu/tool-reasoning-sft-RESEARCH-OpenSeeker-v1-Data`）是两个不同的数据集。为兼容已有配置、数据和实验引用，内部身份继续使用 `browsecomp-sft`，脚本文件名仍含 `browsecomp_sft`，产物路径继续使用 `artifacts/datasets/browsecomp-sft/`；这些标识都指 BrowseComp-Plus SFT。

## 下载与准备评测题

1. 执行 `bash scripts/data/sh/12_download_browsecomp_sft.sh`。下载产物位于 `artifacts/datasets/browsecomp-sft/raw/<commit>/`。旧快照可继续填写原来的 `SA_SNAPSHOT_DIR`，已有 Parquet 可从 `SA_INPUTS` 读取。两个 Parquet 合计约 2.13 GB，主分片只有一个大 row group。处理主分片时建议先降低 `SA_WORKERS` 并检查 Linux 主机可用内存。
2. 准备所有计划评测的题目，参照 [模型、训练数据与 benchmark 准备](20_data.md)。默认保护列表包含 BrowseComp、BrowseComp-Plus、HotpotQA。BrowseComp 题目由 30 的 `download` 生成，另两套题目由 31/32 的 `prepare` 生成。已有题目 JSONL 时直接用其真实路径。这个步骤只需题目文件，重合检查无需建立索引。
3. 修改 `scripts/data/sh/25_prepare_browsecomp_sft.sh`，运行 `bash scripts/data/sh/25_prepare_browsecomp_sft.sh`。

### 下载参数

| SH 变量 | 默认值或修改提示 |
|---|---|
| `PYTHON_BIN` | 当前 Linux 环境的 `python`。也可在启动脚本前指定环境的 `bin/python`。 |
| `HF_ENDPOINT` | 默认 `https://huggingface.co`。网络需走镜像时改为可访问的 Hub 地址。 |
| `HF_HUB_OFFLINE` | 下载使用 `0`。 |
| `CUDA_VISIBLE_DEVICES` | 下载和计长无需 GPU。 |
| `SA_SOURCE` | 固定 `browsecomp-sft`。 |
| `SA_REVISION` | 默认固定到已核验的 commit。切换 revision 后检查文件名、列和工具协议。 |

### 预处理参数

| SH 变量 | 作用和修改提示 |
|---|---|
| `PYTHON_BIN`、`HF_ENDPOINT`、`HF_HUB_OFFLINE`、`CUDA_VISIBLE_DEVICES` | 同下载脚本。预处理默认 `HF_HUB_OFFLINE=1`，使用本地模型文件。 |
| `SA_SOURCE` | 固定 `browsecomp-sft`。输出到 `artifacts/datasets/browsecomp-sft/prepared/<PROFILE>/`。 |
| `SA_SYSTEM_PROMPT_MODE` | 默认 `by-shard`，分别使用下两项；也可统一指定 `normalized`、`original`、`official` 或 `official-read`。后两项只适用于主分片。切换模式须更换 `SA_PROFILE` 并重新计长。 |
| `SA_MAIN_PROMPT_MODE` | `by-shard` 时的主分片模式，默认 `official`：对齐官方 search-only 任务。`official-read` 额外允许 get_document；`normalized/original` 保留旧处理选择。 |
| `SA_PLANNER_PROMPT_MODE` | `by-shard` 时的辅助分片模式，默认 `normalized`；也可选 `original`。辅助分片不使用主任务的官方模板。 |
| `SA_SNAPSHOT_DIR` | 下载输出的 commit 目录。用快照时将 `SA_INPUTS` 留为 `[]`。 |
| `SA_INPUTS` | 已有本地文件时填写 JSON 路径数组，如 `["/data/syn_planner.parquet"]`，并清空 `SA_SNAPSHOT_DIR`。按数组顺序处理。 |
| `SA_SHARDS` | 仅在使用快照时生效。默认 `["syn_planner.parquet"]`。选主分片填 `["data.parquet"]`，同时处理时按快照清单顺序读取。 |
| `SA_CANONICAL_QUESTIONS` | 主分片必填，建议用 BrowseComp 的 `questions.jsonl`。原题从最佳匹配读取，用于归一化训练问题和生成分组哈希。辅助分片可留空。 |
| `SA_PROTECTED_QUESTIONS` | 评测题 JSONL 的路径数组，每行提供 `query` 或 `question`、`answer` 和可选 `id`。默认三套评测，路径随版本调整。缺少文件会报错。只有明确允许这些题进入训练时才移除对应项。 |
| `SA_OVERLAP_THRESHOLD` | 默认 `0.30`，范围 `(0, 1]`。影响改写匹配，调低可提高检出率，也会增加误报。精确和嵌入匹配优先。 |
| `SA_PROFILE` | 输出版本名。修改输入、计长、保护列表、匹配或划分策略后换新名称。 |
| `SA_MODELS` | 本地 tokenizer 路径的 JSON 数组。至少一个，与训练 `SA_MODEL_PATH` 的路径字符串完全一致。 |
| `SA_MIN_LENGTH`、`SA_MAX_LENGTH` | 完整轨迹 token 范围，默认 1 至 16,384。按当前模板实际编码，超长整条剔除。训练上限至少为预处理上限。 |
| `SA_CORRECTNESS` | 使用 `any`。当前源没有独立正确性标签，`Correct` 和 `Incorrect` 会筛空这些数据。 |
| `SA_OBSERVATIONS` | 默认 `with`，保留带观测的轨迹。可取 `any`、`with`、`without`。 |
| `SA_WORKERS` | CPU 计长进程数，默认 4。内存紧张时降低。 |
| `SA_SEED` | 默认 42。控制按问题哈希分组后的确定性划分。 |
| `SA_VAL_FRACTION`、`SA_TEST_FRACTION` | 默认各 0.02，允许 0，二者和必须小于 1。实际样本数由哈希和筛选结果决定。 |
| `SA_OFFLINE` | 默认 `true`，只从本地加载 tokenizer。 |

主分片要求原题匹配唯一，最佳与次佳覆盖率差小于 0.10 的轨迹记入 `unresolved_benchmark_question`。辅助分片使用原问题，并保留上游来源及 `qid` 供追溯。两种分片都按问题文本生成 `question_id`，同一题仅保留第一条符合筛选条件的轨迹。保护列表在计长前检查，命中时写入 `rejected.jsonl`。原题映射后也检查保护列表。归并后的同题不会分配到多个本地划分。

`normalized` 模式沿用当前转换：辅助分片的 `web_search(query)` 映射为 `search(query=[...])`，`crawl_webpage(url)` 映射为 `visit(url, goal)`，其中 `goal` 用当前问题补齐。主分片的 `search(parameters.user_query)` 统一为 `search(arguments.query)`，使用只读检索轨迹的 search-only prompt。未支持的工具调用进入拒绝清单。

`original` 模式不替换、裁剪或追加 system，不改写工具名称、`parameters/arguments`、调用标签或最终答案文本；只将导出的 reasoning/action 合并为 assistant 轮次，将工具观测转为可屏蔽的消息。训练继续使用相同的 ChatML 编码、观测掩码和完整轨迹计长。主分片的原始 user 消息保留；原题映射仍用于 `question`/`question_id` 分组和评测题保护。每行记录 `system_prompt_mode=original`、`call_protocol=research-original`，清单记录模式和适配器版本。二次筛选与混合保留这些字段，不会重新转换。

normalized/original 都保留历史观测和示范回答，内部保留划分仅用于训练诊断。原始 system 是该 SFT 导出中的原文，不保证等于 BrowseComp-Plus 或 observation-masking 的提示词。

## 主分片的官方任务对齐

本节只涉及 Plain/Beacon。官方任务与浏览器实验使用不同入口，不能在同一入口切换协议，也不能用另一个入口为已有实验评分：

| 入口（位于 `scripts/plain/sh/` 或 `scripts/beacon/sh/`） | 固定协议 | 用途 |
|---|---|---|
| `46_run_browsecomp_plus.sh` | `qwen-native` | 浏览器 search/open/find，保留现有默认设置。 |
| `48_run_browsecomp_plus_official.sh` | `official` | 官方任务模板，search(query)，固定 top-5 和正文前 512 tokens；默认不开放全文读取。 |
| `49_run_browsecomp_plus_original.sh` | `original` | 复用 original 模式训练数据的原始 system 和调用协议。 |

主分片处理在 25 号脚本设置 `SA_SHARDS='["data.parquet"]'`、`SA_SYSTEM_PROMPT_MODE=by-shard`、`SA_MAIN_PROMPT_MODE=official`，填写 `SA_CANONICAL_QUESTIONS`，使用新 `SA_PROFILE`。程序用原题生成与官方固定版本 `0469490` 一致的 user 任务模板，把 `search(parameters.user_query)` 改为 `search(arguments.query)`，只保留已有 Explanation/Exact Answer/Confidence 的回答并去掉外层 answer 标签。不会编造缺失字段；get_document 轨迹在 search-only 模式下进入拒绝清单。需要保留这些轨迹时显式选 `official-read`，并在 48 中设置 `SA_GENERATION__ALLOW_VISIT=true`。

辅助分片可另用新 PROFILE 运行，选择 `syn_planner.parquet` 与 normalized/original。也可在同一次 by-shard 预处理中显式选择两个分片；它们各自适配，行记录包含 `sft_shard` 与有效模式，清单汇总 `sft_shards/prompt_modes`。问题分组、受保护评测题过滤及长度过滤在所有模式下保持有效。主分片与官方题目高度重合，严格保护后可能没有可训练样本；不能通过移除保护题来宣称独立测试。

训练仍使用 11 号入口和新的 prepared 目录。正式评测使用对应方法的 48，填写 `SA_TRAINING_RUN`、benchmark/检索资产，以及 `SA_GENERATION__SNIPPET_TOKENIZER_PATH`（本地 `Qwen/Qwen3-0.6B` tokenizer）。可用 01_download_model 下载该模型；已有本地 tokenizer 可直接使用。Plain 仍通过 vLLM，Beacon 本地加载；模型后端不随任务模式改变。

48 使用原始 Parquet 正文（包含头部文本）生成片段，返回 docid、可用时的检索分数、snippet；可选 get_document 返回全文，不按 READ_TOKENS 截短，过大的工具返回由 MAX_TOOL_TOKENS 报错。它不使用浏览器高亮、行号、日期或观测遮蔽。默认生成 temperature=0.7、top_p=0.8、max_new_tokens=10000；其余预算见 SH。训练与评测共用 task/system 和调用参数定义。

这是**官方任务与检索返回设置的对齐入口**：user 模板与固定上游版本逐字校验，但工具 system 包装、ChatML 调用和 Agent 循环仍采用本项目适配器，未直接运行 Qwen-Agent。历史 SFT 工具观测保留上游内容，不重新检索生成，也不保证是 top-5/512。不能将这个适配入口表述为完整复现官方全部实验。[上游任务模板](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/search_agent/prompts.py) · [上游工具返回](https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/searcher/tools.py)

## 保留原始 system 的训练与推理

1. 在 `25_prepare_browsecomp_sft.sh` 设置 `SA_SYSTEM_PROMPT_MODE=original`，使用新版本名，如 `browsecomp_sft_original_qwen35_L16384_s42`，重新预处理。评测题保护列表保持有效。
2. Plain/Beacon 的 `11_train_research.sh` 设置 `SA_DATASET_NAME=browsecomp-sft`、`SA_DATASET_DIR` 为新产物，按原训练流程运行。
3. 对应方法的 `49_run_browsecomp_plus_original.sh` 使用 `SA_GENERATION__PROMPT_PROFILE=original`。`SA_PROMPT_DATA` 留空时读取训练实验的 train.jsonl 并校验训练时文件摘要；若其中含多套 system，程序报错，需提供只含一种 system 的 original prepared JSONL。不会任取第一套提示词。

`original` 评测按训练原文加载 system，不追加日期、native 浏览器提示或本地工具说明，并使用训练时的 ChatML/原始调用协议。支持 `search(query/user_query)`、`web_search(query)`、`get_document(docid)`、`crawl_webpage(url)` 和 `visit(url, goal)`，只读取固定本地语料，不访问外部网页。Plain 使用现有 vLLM completion 服务，Beacon 本地加载；工具返回长度分别由 `TOPK`、`SNIPPET_TOKENS` 和 `READ_TOKENS` 控制。84 号训练轨迹诊断也会复用原始 system 和调用协议。

如需对照 BrowseComp-Plus 原始主实验，将 `TOPK=5`、`SNIPPET_TOKENS=512`，并按目标实验决定是否开放全文读取。当前片段按生成模型 tokenizer 计长，官方主实验使用 Qwen3-0.6B tokenizer；本地文档返回结构、正文处理与上游仍可能不同。保留原始 system 和调用协议解决训练/推理的提示词转换问题，不构成完整复现官方实验的声明。`qwen-native` 继续保留为独立的浏览器评测配置，切换到它不会自动继承原始 system。

## 训练与混合

- 二次筛选使用 `26_select_browsecomp_sft.sh`。填写 `SA_PREPARED`、新 `SA_PROFILE`、tokenizer 和筛选条件。程序复用父版本计长，输出仍在 `artifacts/datasets/browsecomp-sft/prepared/`。
- 单独训练复用 Plain 或 Beacon 的 `11_train_research.sh`。设置 `SA_DATASET_NAME=browsecomp-sft`，`SA_DATASET_DIR` 指向所选 prepared 版本。实验归档到 `artifacts/experiments/<方法>/browsecomp-sft/`。
- 旧版写在 `artifacts/datasets/research/prepared/` 的 BrowseComp-Plus SFT 产物可继续使用。新训练使用 `SA_DATASET_NAME=browsecomp-sft`，旧训练实验可继续从原目录评测。保留旧实验引用的文件路径。
- 混合时使用 `24_mix_datasets.sh`，把 SFT 目录填入 `SA_BROWSECOMP_SFT_PREPARED`。任选两个或三个来源，空目录对应的比例项省略。顺序为 OpenSeeker:Research:BrowseComp-Plus SFT；允许 `2:0:1` 或 `0:0:1`，至少一项大于 0。比例为 0 仅关闭该来源的训练采样，验证/测试仍保留，并按 `test > validation > train` 排除低优先级同题副本。完全移除来源时清空其目录。
- 使用混合训练产物时，将训练脚本的 `SA_DATASET_NAME` 设为 `mixed`。使用旧训练提示词评测时，测试脚本中的 `SA_PROMPT_DATA` 选一个与评测工具能力匹配、仅含一种 system/tools 组合的规范化 JSONL。单独训练但同时纳入两个 SFT 分片时，同样可用此变量显式选择一种提示词。
- 预处理结束先检查 `manifest.json` 的输入文件指纹、题目参考文件指纹、计长分布与划分数量。查看 `rejected.jsonl` 的 `protected_question`、`unresolved_benchmark_question` 和工具格式错误，确认训练集非空后再启动训练。

## BrowseComp 与 BrowseComp-Plus 测试前筛选

推荐 46/47 使用统一 benchmark 提示词，`SA_PROMPT_DATA` 留空。Beacon 的旧 40 和 42 默认启用筛选，统一 46 默认关闭。Plain 在 46 中设置 `SA_OVERLAP=filter` 后启用。测试从训练实验读取实际 `train.jsonl`，验证其训练时记录的 SHA256，在生成前排除重合题。训练版本和官方题文件需要保留在实验记录所指路径。

| SH 变量 | 作用和修改提示 |
|---|---|
| `SA_OVERLAP` | `filter` 生成 `overlap.json` 并排除匹配题。需要复跑完整历史题集时显式设为 `off`，并记录实验口径。 |
| `SA_OVERLAP_THRESHOLD` | 改写匹配阈值，默认 0.30。较低值更容易检出改写。 |
| `SA_GENERATION__LIMIT` | `-1` 使用筛选后全部题目。正数取筛选后按官方顺序的前 N 条。 |

匹配先比较大小写和符号归一后的词序列，再检查长问题是否被完整嵌入。改写匹配要求至少 5 个不同的四词片段重合，并达到参考题的覆盖率阈值。大幅改写或翻译可能漏检，共享长片段的不同题可能误匹配。报告记录排除 ID、匹配方式、覆盖率、示例训练行 ID 和输入指纹。评分复用该报告及题数限制。全被排除时测试停止并保留报告。由此产生的分数应与相同题目子集比较。
