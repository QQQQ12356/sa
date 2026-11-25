"""Exercise HTTP generation, browser state, and method isolation without a GPU."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import httpx
import pytest
from transformers import AutoTokenizer

from search_comp.evaluation.vllm_client import VllmClient, endpoint_urls, served_name
from search_comp.experiments import vllm_serve
from search_comp.offline import native_browser
from search_comp.workflows import offline_benchmark


@pytest.mark.parametrize("origin", ["base-qwen35", "checkpoint"])
@pytest.mark.parametrize("retriever", ["bm25", "qwen3-embedding-8b"])
def test_vllm_uses_native_tools_and_preserves_question_order(tmp_path, monkeypatch, origin, retriever):
    args = offline_benchmark.make_parser().parse_args([
        "run", "--benchmark", "browsecomp-plus", "--root", str(tmp_path),
        "--prompt-profile", "qwen-native", "--prompt-date", "2026-11-25",
        "--kind", "plain", "--model-origin", origin,
        "--vllm-urls", "http://127.0.0.1:8010/v1,http://127.0.0.1:8011/v1",
        "--vllm-concurrency", "1", "--retriever", retriever,
    ])
    model = str(tmp_path / "weights")
    name = served_name(origin)
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "browser.search"
            question = int(messages[1]["content"])
            if len(messages) > 2:
                assert messages[-1]["content"].startswith(f"[0] bridge-{question}")
            return [question, len(messages)]

        def encode(self, prompt, *, add_special_tokens):
            return prompt

    def route(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [dict(id=name, root=model, max_model_len=131072)]})
        assert request.url.path == "/v1/completions"
        payload = json.loads(request.content)
        question, message_count = payload["prompt"]
        assert request.url.port == 8010 + question % 2
        assert payload["model"] == name and payload["temperature"] == 0
        assert payload["seed"] == 42 and "<tool_response>" in payload["stop"]
        seen.append((question, message_count))
        if message_count == 2:
            reply = f'<tool_call>{{"name":"browser.search","arguments":{{"query":"bridge-{question}"}}}}</tool_call>'
        else:
            reply = f"Explanation: see bridge.\nExact Answer: {question}\nConfidence: 90%"
        return httpx.Response(200, json=dict(choices=[dict(text=reply, finish_reason="stop")],
                                             usage=dict(completion_tokens=10)))

    http_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: http_client(transport=httpx.MockTransport(route), **kw))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(vllm_serve, "serving_model", lambda *a: model)
    monkeypatch.setattr(native_browser, "load_highlighter", lambda _: lambda query, text: text)
    doc = dict(docid="1", title="Bridge", source_url="https://example.invalid/bridge", text="Red.")
    rankers = []

    def search(query, topk):
        rankers.append(retriever)
        return [doc]

    index = SimpleNamespace(search=search, close=lambda: None)
    monkeypatch.setattr(offline_benchmark, "LocalIndex", lambda *a, **kw: nullcontext(object()))
    monkeypatch.setattr(offline_benchmark, "LuceneIndex", lambda *a: index)
    monkeypatch.setattr(offline_benchmark, "DenseIndex", lambda *a: index)

    rows = [dict(id=str(i), query=str(i), answer=str(i)) for i in range(4)]
    result = offline_benchmark.run_phase(args, tmp_path, rows, {}, {"vector_index": {}, "ranking": {}})
    predictions = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    assert [row["prediction"] for row in predictions] == ["0", "1", "2", "3"]
    assert [row["sample_index"] for row in predictions] == list(range(4))
    assert all(row["retrieved_docids"] == ["1"] and row["generation_config"]["served_model"] == name
               for row in predictions)
    assert sorted(seen) == [(i, n) for i in range(4) for n in (2, 4)]
    assert rankers == [retriever] * 4


def test_vllm_rejects_remote_endpoint_and_mismatched_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="local"):
        endpoint_urls("https://example.invalid:8020/v1")
    response = dict(data=[dict(id=served_name("checkpoint"), root=str(tmp_path / "other"))])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))) as http:
        with pytest.raises(ValueError, match="must serve"):
            VllmClient(http, "http://127.0.0.1:8020/v1", str(tmp_path), "checkpoint", {}, 100)


def test_vllm_selects_merged_plain_weights_and_rejects_a_baseline_adapter(tmp_path):
    adapter = tmp_path / "checkpoint-200"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").touch()
    with pytest.raises(ValueError, match="original full"):
        vllm_serve.serving_model(None, adapter, "plain", "base-qwen35")
    with pytest.raises(ValueError, match="Merge the Plain adapter"):
        vllm_serve.serving_model(None, adapter, "plain", "checkpoint")
    merged = tmp_path / "checkpoint-200_merged"
    merged.mkdir()
    (merged / "config.json").write_text(json.dumps(dict(
        model_type="qwen3_5_text", hidden_size=2560, intermediate_size=9216, num_hidden_layers=32,
    )), encoding="utf-8")
    (merged / "model.safetensors").touch()
    assert vllm_serve.serving_model(None, adapter, "plain", "checkpoint") == str(merged.resolve())
