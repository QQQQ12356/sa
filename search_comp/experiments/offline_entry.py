"""Typed settings for method-owned evaluation and scoring launchers."""
import os

from .shell_config import load_settings


def launcher_schema(method, benchmark="browsecomp-plus"):
    """Describe common fields once, then specialize by backend and benchmark."""
    if method not in {"baseline", "plain", "beacon"}:
        raise ValueError("unsupported evaluation method")
    if benchmark not in {"browsecomp-plus", "hotpotqa"}:
        raise ValueError("unsupported offline benchmark")
    schema = {
        f"ENVIRONMENT__{name}": {"env": name, "type": "str"}
        for name in ("HF_ENDPOINT", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")
    }
    for kind, names in {
        "str": ("BENCHMARK_DIR RUN_NAME GENERATION__PROMPT_PROFILE GENERATION__DECODING "
                "GENERATION__RETRIEVER GENERATION__EMBEDDING_DEVICE JUDGE__JUDGE_BACKEND "
                "JUDGE__JUDGE_MODEL JUDGE__JUDGE_URL"),
        "optional": ("PROMPT_DATA RESUME_FROM EVALUATION_DIR GENERATION__VECTOR_INDEX_DIR "
                     "GENERATION__EMBEDDING_MODEL_PATH GENERATION__PROMPT_DATE"),
        "bool": "SCORE_AFTER_GENERATION GENERATION__VERIFY_INDEX GENERATION__ALLOW_VISIT",
        "int": ("GENERATION__LIMIT GENERATION__NUM_WORKERS GENERATION__OBSERVATION_WINDOW "
                "GENERATION__MAX_CONTEXT GENERATION__MAX_NEW_TOKENS GENERATION__MAX_TURNS "
                "GENERATION__MAX_TOOL_TOKENS GENERATION__TOOL_COUNT GENERATION__TOOL_CALL_RETRIES "
                "GENERATION__TOPK GENERATION__SNIPPET_TOKENS GENERATION__READ_TOKENS "
                "GENERATION__GENERATION_SEED JUDGE__JUDGE_MAX_TOKENS JUDGE__JUDGE_ATTEMPTS"),
        "float": ("GENERATION__TITLE_WEIGHT GENERATION__REPETITION_PENALTY "
                  "GENERATION__TEMPERATURE GENERATION__TOP_P JUDGE__JUDGE_TIMEOUT"),
    }.items():
        schema.update({name: {"env": f"SA_{name}", "type": kind} for name in names.split()})
    method_fields = {"MODEL_PATH": "str"} if method == "baseline" else {
        "TRAINING_RUN": "str", "CHECKPOINT": "optional", "OVERLAP": "str", "OVERLAP_THRESHOLD": "float",
    }
    if method in {"baseline", "plain"}:
        method_fields.update(GENERATION__VLLM_URLS="str",
                             GENERATION__VLLM_CONCURRENCY="int", GENERATION__VLLM_TIMEOUT="float")
        if not (method == 'plain' and benchmark == 'browsecomp-plus'):
            for name in ("PROMPT_DATA", "GENERATION__PROMPT_PROFILE", "GENERATION__NUM_WORKERS"):
                del schema[name]
        if benchmark == "browsecomp-plus" and method != 'plain':
            for name in ("MAX_TOOL_TOKENS", "TOOL_CALL_RETRIES", "SNIPPET_TOKENS", "READ_TOKENS", "ALLOW_VISIT"):
                del schema[f"GENERATION__{name}"]
    if benchmark == "hotpotqa":
        for name in ("GENERATION__RETRIEVER", "GENERATION__VECTOR_INDEX_DIR", "GENERATION__EMBEDDING_MODEL_PATH",
                     "GENERATION__EMBEDDING_DEVICE", "GENERATION__OBSERVATION_WINDOW",
                     "JUDGE__JUDGE_BACKEND", "JUDGE__JUDGE_URL", "JUDGE__JUDGE_TIMEOUT",
                     "GENERATION__PROMPT_DATE"):
            del schema[name]
        for name in ("OVERLAP", "OVERLAP_THRESHOLD"):
            method_fields.pop(name, None)
        method_fields.update(JUDGE__LLM_JUDGE="bool", JUDGE__JUDGE_MAX_CONTEXT="int",
                             JUDGE__JUDGE_ENABLE_THINKING="bool", JUDGE__JUDGE_TEMPERATURE="float",
                             JUDGE__JUDGE_TOP_P="float", JUDGE__JUDGE_TOP_K="int", JUDGE__SEED="int")
    schema.update({name: {"env": f"SA_{name}", "type": kind} for name, kind in method_fields.items()})
    if method in {'plain', 'beacon'} and benchmark == 'browsecomp-plus':
        schema['GENERATION__SNIPPET_TOKENIZER_PATH'] = {'env': 'SA_GENERATION__SNIPPET_TOKENIZER_PATH', 'type': 'optional'}
    return schema


def launch(method, filename, schema, *, benchmark="browsecomp-plus"):
    from .benchmarks import run_offline_benchmark

    settings = load_settings(filename, schema)
    if method in {'plain', 'beacon'} and benchmark == 'browsecomp-plus':
        from pathlib import Path
        number = Path(filename).stem.split('_', 1)[0]
        expected = {'46': 'qwen-native', '48': 'official', '49': 'original'}.get(number)
        if expected and settings['GENERATION']['prompt_profile'] != expected:
            raise ValueError(f'{number} 入口固定使用 {expected}；浏览器/官方/原始实验须使用各自入口')
        if expected and settings.get('EVALUATION_DIR'):
            from .runtime import path, read_json
            record = path(settings['EVALUATION_DIR']) / 'evaluation.json'
            if record.is_file() and read_json(record).get('generation', {}).get('prompt_profile') != expected:
                raise ValueError('已有评测的提示词协议与当前入口不同，请使用原实验入口评分')
    if method in {"baseline", "plain"}:
        if method == 'plain' and benchmark == 'browsecomp-plus':
            if settings['GENERATION']['prompt_profile'] not in {'qwen-native', 'original', 'official'}:
                raise ValueError('invalid Plain BrowseComp-Plus prompt profile')
        else:
            settings["PROMPT_DATA"] = None
            settings["GENERATION"]["prompt_profile"] = "qwen-native" if benchmark == "browsecomp-plus" else "benchmark"
        if "SA_SERVE_PORT" in os.environ:
            from .vllm_serve import serve

            serve(method, settings)
            return
        if not settings["EVALUATION_DIR"] and not settings["GENERATION"]["vllm_urls"]:
            raise ValueError("填写 SA_GENERATION__VLLM_URLS 后再生成")
    run_offline_benchmark(method=method, benchmark=benchmark,
                          **{key.lower(): value for key, value in settings.items()})
