"""Preserved export systems/actions use the same serializer during SFT and inference."""
import json
import sys
from types import SimpleNamespace
import pytest

from search_comp.evaluation.tool_agent import run_agent
from search_comp.offline.tools import OriginalLocalTools


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8')


def test_original_agent_executes_source_actions_and_reuses_source_system():
    document = dict(docid='42', title='Building', text='Evidence', url='local://42',
                    source_url='https://example.com', sentences=[])
    index = SimpleNamespace(search=lambda query, topk: [document], get=lambda docid: document,
                            get_url=lambda url: document)
    tools = OriginalLocalTools(index, Tokenizer())
    responses = iter([
        '<tool_call>{"name":"search","parameters":{"user_query":"architect"}}</tool_call>',
        '<tool_call>{"name":"get_document","parameters":{"docid":"42"}}</tool_call>',
        '<answer>Building</answer>',
    ])
    system = 'Original export system\n'
    inputs = []

    def generate(encoded, budget, *, stop_texts):
        inputs.append(bytes(encoded['input_ids']).decode('utf-8'))
        assert '</tool_call>' in stop_texts
        text = next(responses)
        return text, True, len(text)

    result = run_agent([dict(role='system', content=system), dict(role='user', content='Which building?')],
                       Tokenizer(), generate, tools, max_context=10000, call_protocol='research-original')
    assert result['status'] == 'completed'
    assert result['prediction'] == 'Building'
    assert result['messages'][0]['content'] == system
    assert tools.retrieved == {'42'}
    assert tools.visited == {'42'}
    assert '<tool_calls_begin>' not in inputs[-1]
    assert '"parameters"' in inputs[-1]


def test_original_crawl_uses_offline_corpus_only():
    document = dict(docid='42', title='Building', text='Evidence', url='local://42',
                    source_url='https://example.com', sentences=[])
    documents = SimpleNamespace(get_url=lambda url: document)
    tools = OriginalLocalTools(SimpleNamespace(documents=documents), Tokenizer())
    replies = tools.execute([dict(name='crawl_webpage', arguments={'url': 'https://example.com'})])
    assert 'Evidence' in replies[0]['content']
    assert tools.visited == {'42'}


def test_training_diagnostic_reuses_original_system(tmp_path):
    from search_comp.data.browsecomp_sft import normalize_browsecomp_sft
    from search_comp.experiments.training_data_evaluation import read_samples
    from test_browsecomp_sft import planner_row

    row = normalize_browsecomp_sft(planner_row(), 'browsecomp-sft/syn_planner.parquet', 0,
                                  system_prompt_mode='original')
    filename = tmp_path / 'train.jsonl'
    filename.write_text(json.dumps(row) + '\n', encoding='utf-8')
    samples, _ = read_samples(filename, -1, True)
    assert samples[0]['system'] == row['messages'][0]['content']
    assert samples[0]['call_protocol'] == 'research-original'


@pytest.mark.parametrize('profile', ['original', 'official'])
def test_plain_vllm_original_uses_shared_retriever_and_preserves_stop_tags(tmp_path, monkeypatch, profile):
    import httpx
    from search_comp.experiments import vllm_serve
    from search_comp.evaluation.vllm_client import served_name
    from search_comp.workflows import offline_benchmark

    args = offline_benchmark.make_parser().parse_args([
        'run', '--benchmark', 'browsecomp-plus', '--prompt-profile', profile,
        '--kind', 'plain', '--vllm-urls', 'http://127.0.0.1:8020/v1', '--vllm-concurrency', '1',
    ])
    model = str(tmp_path / 'weights')
    seen = []

    def route(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [dict(id=served_name('checkpoint'), root=model)]})
        payload = json.loads(request.content)
        assert payload['include_stop_str_in_output'] is True
        assert '</tool_call>' in payload['stop']
        prompt = bytes(payload['prompt']).decode('utf-8')
        seen.append(prompt)
        if profile == 'original':
            assert prompt.startswith('<|im_start|>system\nSource system\n<|im_end|>')
        else:
            assert 'Your response should be in the following format:' in prompt
            assert 'browser.' not in prompt
        call = ('<tool_call>{"name":"search","parameters":{"user_query":"architect"}}</tool_call>'
                if profile == 'original' else '<tool_call>{"name":"search","arguments":{"query":"architect"}}</tool_call>')
        final = ('<answer>Building</answer>' if profile == 'original' else
                 'Explanation: Evidence [42].\nExact Answer: Building\nConfidence: 90%')
        text = final if '<|im_start|>user\n<tool_response>' in prompt else call
        return httpx.Response(200, json=dict(choices=[dict(text=text, finish_reason='stop')],
                                           usage={'completion_tokens': 10}))

    http_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: http_client(transport=httpx.MockTransport(route), **kw))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setattr(vllm_serve, 'serving_model', lambda *a: model)
    doc = dict(docid='42', title='Building', text='Evidence', sentences=[],
               url='local://42', source_url='https://example.com')
    index = SimpleNamespace(search=lambda query, topk: [doc])
    monkeypatch.setattr(offline_benchmark, 'open_documents', lambda *a, **kw: index)
    monkeypatch.setattr(offline_benchmark, 'open_retrieval', lambda *a: index)
    prompt = ('Source system\n', 'source.jsonl') if profile == 'original' else offline_benchmark.agent_prompt(args)
    target = offline_benchmark.run_phase(args, tmp_path, [dict(id='1', query='Which building?', answer='Building')],
                                        {}, {}, prompt=prompt)
    row = json.loads(target.read_text(encoding='utf-8'))
    assert row['prediction'] == 'Building'
    assert row['retrieved_docids'] == ['42']
    assert row['generation_config']['tool_protocol'] == ('research-original' if profile == 'original' else 'official-task')
    assert len(seen) == 2
