# 10 Linux 环境准备

使用 Linux x86_64、NVIDIA GPU、Git 和 Java 21。项目沿用 Python 3.11、PyTorch 2.7.0+cu126 的训练与检索环境。Baseline 和 Plain 的模型服务单独使用 Python 3.12 的 uv 环境。两个环境可在同一 Linux 主机上使用各自的 PyTorch 和 CUDA wheel，GPU 驱动需要覆盖两套运行时。

## 1. 使用已安装的 cu126 环境

```bash
conda activate search-comp-qwen35-cu126
python -m pip install -r requirements.browsecomp-plus.txt
python -m pip check
java -version
python -c 'import torch, httpx, json5; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`requirements.browsecomp-plus.txt` 增加了 vLLM 客户端的 HTTP 和工具参数解析依赖，继续约束原 PyTorch 版本。Java 21 用于 Lucene 读索引以及原生浏览器摘要高亮，`JAVA_HOME` 应指向同一 Java 安装。训练和 Beacon 评测继续使用 cu126 环境。在线 BrowseComp 的网页读取和评分环境见 [11 Beacon 在线摘要服务](11_sglang_summary.md)。

从零安装时可用已有 Conda 描述文件准备 Python 和系统工具，也可直接用 uv 建立 Python 环境：

```bash
# Conda 方案
conda env create -f environment.train-cu126.yml
conda activate search-comp-qwen35-cu126
python -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.0+cu126'
python -m pip install -r requirements.txt -r requirements.browsecomp-plus.txt
```

```bash
# uv 方案。系统另行安装 Git 和 Java 21。
uv venv .venv-cu126 --python 3.11
uv pip install --python .venv-cu126/bin/python --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.0+cu126'
uv pip install --python .venv-cu126/bin/python -r requirements.txt -r requirements.browsecomp-plus.txt
source .venv-cu126/bin/activate
```

还可用 `bash scripts/tools/sh/00_setup_environment.sh` 安装项目原有训练依赖。训练使用 ZeRO 时，按该脚本配置 DeepSpeed，并在当前 cu126 环境编译 CUDA 扩展。单卡或 DDP 可直接使用上述环境。若出现 `causal_conv1d_cuda.so: undefined symbol`，核对当前解释器、PyTorch 和构建扩展时的 CUDA 工具链，再在同环境重建该扩展。

## 2. 安装 vLLM 服务

按上游的独立 uv 环境方式安装。固定版本支持原始多模态 Qwen3.5 和 Plain 训练产出的纯文本 Qwen3.5。[上游部署方式](https://github.com/i-DeepSearch/observation-masking/blob/main/assets/docs/deployment.md) · [vLLM 安装说明](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/) · [vLLM 模型注册表](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/registry.py)

```bash
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python --torch-backend=auto -r requirements.vllm.txt
.venv-vllm/bin/vllm --version
.venv-vllm/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`--torch-backend=auto` 根据 NVIDIA 驱动选择 PyTorch 源，还需确认所装 vLLM wheel 的 CUDA 版本满足驱动与 GPU 要求。若当前 Linux 主机只能支撑 cu126，先为服务准备兼容的驱动和 wheel，再执行 [32 号流程](32_browsecomp_plus_vllm.md)。训练和检索仍可继续使用原环境。也可用 `SA_VLLM_BIN=/path/to/venv/bin/vllm` 指定已有兼容服务环境。

## 3. 运行时环境变量

| 变量 | 用途 |
|---|---|
| `PYTHON_BIN` | Bash 入口用的 cu126 Python，默认当前环境的 `python` |
| `HF_ENDPOINT` | Hugging Face 下载地址，可在下载脚本中改为可访问的镜像 |
| `HF_HUB_OFFLINE` | 下载设 0，使用本地模型和数据时设 1 |
| `CUDA_VISIBLE_DEVICES` | 物理 GPU 编号。使用多卡训练时与 `NPROC_PER_NODE` 配套 |
| `HF_HOME`、`TORCH_HOME`、`XDG_CACHE_HOME` | 已有缓存路径可复用。默认位于 `artifacts/cache/` |

下载需要身份验证时，通过 `hf auth login` 或终端环境变量 `HF_TOKEN` 配置。裁判与 vLLM 密钥也从环境传入，脚本会作为实验产物归档。
