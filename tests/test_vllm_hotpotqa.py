"""Exercise the HotpotQA launcher, local BM25 and official metrics with mock vLLM."""

import json

import httpx
import pytest
from transformers import AutoTokenizer

from search_comp.data.openseeker import CALL_END
from search_comp.evaluation.vllm_client import served_name
from search_comp.experiments import benchmarks
from search_comp.experiments.runtime import write_json
from search_comp.workflows import offline_benchmark
from test_experiment_contracts import parameters
from test_offline_benchmarks import CharacterTokenizer, prepared_hotpot


@pytest.mark.parametrize("method,port", [("baseline", 8030), ("plain", 8040)])
def test_hotpotqa_vllm_generation_and_score_only(tmp_path, monkeypatch, method, port):
    assets = prepared_hotpot(tmp_path)
    model = tmp_path / "qwen"
    model.mkdir()
    write_json(model / "config.json", dict(model_type="qwen3_5", text_config=dict(
        hidden_size=2560, intermediate_size=9216, num_hidden_layers=32,
    )))
    (model / "model.safetensors").touch()
    origin = "base-qwen35" if method == "baseline" else "checkpoint"
    target = {"model_path": str(model)}
    owner = tmp_path / "artifacts/experiments/baseline/untrained"
    if method == "plain":
        owner = tmp_path / "plain/run"
        output = owner / "models/model"
        output.mkdir(parents=True)
        write_json(owner / "job.json", dict(operation="train", status="complete"))
        write_json(owner / "experiment.json", dict(method="plain", dataset_name="mixed",
                                                  training_config=dict(model_name_or_path=str(model))))
        write_json(output / "train_summary.json", dict(final_model_path=str(model)))
        target = {"training_run": str(owner)}

    monkeypatch.setattr(benchmarks, "ROOT", tmp_path)
    monkeypatch.setattr(benchmarks, "configure", lambda _: None)
    monkeypatch.setattr(offline_benchmark, "enforce_offline", lambda: None)
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    sent = []

    def route(request):
        assert request.url.port == port
        if request.url.path == "/v1/models":
            card = dict(id=served_name(origin), root=str(model), max_model_len=32768)
            return httpx.Response(200, json={"data": [card]})
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        prompt = tokenizer.decode(body["prompt"])
        assert "REFERENCE_CONTEXT" not in prompt
        assert body["include_stop_str_in_output"]
        assert body["stop"] == [CALL_END, "</answer>"]
        sent.append(prompt)
        if len(sent) == 1:
            response = ('<tool_calls_begin><tool_call>{"name":"search","arguments":{"query":["bridge"]}}'
                        '</tool_call>' + CALL_END)
        elif len(sent) == 2:
            assert "local://1" in prompt
            response = ('<tool_calls_begin><tool_call>{"name":"visit","arguments":{"url":"local://1","goal":"read"}}'
                        '</tool_call>' + CALL_END)
        else:
            assert "[0]" in prompt
            response = '<answer>{"answer":"red blue","sp":[["Bridge",0]]}</answer>'
        return httpx.Response(200, json=dict(choices=[dict(text=response, finish_reason="stop")],
                                             usage=dict(completion_tokens=10)))

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(route), **kw))
    defaults = parameters(offline_benchmark.ROOT / "scripts" / method / "py/47_run_hotpotqa.py")
    generation = dict(defaults["GENERATION"], prompt_profile="benchmark", limit=1, vllm_concurrency=1)
    options = dict(method=method, benchmark="hotpotqa", benchmark_dir=assets, prompt_data=None,
                   run_name="smoke", resume_from=None, score_after_generation=False,
                   generation=generation, judge=defaults["JUDGE"], environment={}, **target)
    evaluation = benchmarks.run_offline_benchmark(evaluation_dir=None, **options)
    assert evaluation.is_relative_to(owner / "evaluation/hotpotqa")
    row = json.loads((evaluation / "results/predictions.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "completed"
    assert row["generation_config"]["tool_protocol"] == "openseeker"
    assert row["visited_docids"] == ["1"]
    assert json.loads(row["final_response"])["sp"] == [["Bridge", 0]]

    def unexpected_client(**kwargs):
        pytest.fail("score-only attempted to start a vLLM client")

    monkeypatch.setattr(httpx, "Client", unexpected_client)
    scores = benchmarks.run_offline_benchmark(evaluation_dir=evaluation, **options)
    metrics = json.loads((scores / "results/summary.metrics.json").read_text(encoding="utf-8"))
    assert metrics["em"] == metrics["sp_em"] == metrics["joint_em"] == 1
    assert "llm_judge_accuracy" not in metrics
