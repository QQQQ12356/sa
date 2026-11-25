"""Serve one local vLLM replica for a baseline or Plain evaluation."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from search_comp.evaluation.checkpoints import resolve_checkpoint
from search_comp.evaluation.vllm_client import endpoint_urls, served_name
from .benchmarks import resolve_target
from .runtime import configure, read_json


def serving_model(run, model, kind, origin):
    model, _ = resolve_checkpoint(run, model, kind, require_weights=True)
    directory = Path(model)
    if (directory / "adapter_config.json").is_file():
        if origin == "base-qwen35":
            raise ValueError("Baseline requires the original full Qwen3.5-4B weights")
        # Plain training already merges its final adapter. Periodic adapters can
        # be merged with the same utility into its default sibling directory.
        merged = directory.with_name(directory.name + "_merged")
        if (not merged.is_dir() or merged.resolve() != merged
                or (merged / "adapter_config.json").exists()):
            raise ValueError(f"Merge the Plain adapter with search_comp.trainer.merge_lora: {directory}")
        model, _ = resolve_checkpoint(run, merged, kind, require_weights=True)
        directory = Path(model)
    config = read_json(directory / "config.json")
    text = config.get("text_config", config)
    dimensions = (text.get("hidden_size"), text.get("intermediate_size"), text.get("num_hidden_layers"))
    supported = {"qwen3_5"} if origin == "base-qwen35" else {"qwen3_5", "qwen3_5_text"}
    if config.get("model_type") not in supported or dimensions != (2560, 9216, 32):
        raise ValueError(f"Expected Qwen3.5-4B weights: {directory}")
    return model


def serve(method, settings):
    configure(settings["ENVIRONMENT"])
    _, run, model, kind, origin = resolve_target(method, settings.get("TRAINING_RUN"),
                                                settings.get("MODEL_PATH"), settings.get("CHECKPOINT"))
    generation = settings["GENERATION"]
    port = int(os.environ["SA_SERVE_PORT"])
    urls = endpoint_urls(generation["vllm_urls"])
    if port not in {urlsplit(url).port for url in urls}:
        raise ValueError("serve port must appear in SA_GENERATION__VLLM_URLS")
    directory = serving_model(run, model, kind, origin)
    binary = Path(os.environ.get("SA_VLLM_BIN", ".venv-vllm/bin/vllm")).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"vLLM executable missing: {binary}; see docs/10_environment.md")
    command = [
        str(binary), "serve", directory, "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", served_name(origin), "--enable-prefix-caching",
        "--generation-config", "vllm", "--max-model-len", str(generation["max_context"]),
        "--tensor-parallel-size", os.environ.get("SA_VLLM_TP", "1"),
        "--gpu-memory-utilization", os.environ.get("SA_VLLM_GPU_MEMORY", "0.90"),
        "--max-num-seqs", str(generation["vllm_concurrency"]),
    ]
    if os.environ.get("SA_VLLM_API_KEY"):
        os.environ["VLLM_API_KEY"] = os.environ["SA_VLLM_API_KEY"]
    os.environ["PATH"] = str(binary.parent) + os.pathsep + os.environ.get("PATH", "")
    if (binary.parent.parent / "pyvenv.cfg").is_file():
        os.environ["VIRTUAL_ENV"] = str(binary.parent.parent)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.execv(binary, command)
