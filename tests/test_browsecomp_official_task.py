"""Official task alignment is shared by training/inference and isolated from browsing."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from search_comp.data.browsecomp_sft import normalize_browsecomp_sft
from search_comp.data.browsecomp_plus_official import task_prompt, system_prompt
from search_comp.data.openseeker import encode, parse_calls
from search_comp.offline.tools import OfficialLocalTools
from search_comp.evaluation.tool_agent import ToolError
from search_comp.workflows.offline_benchmark import agent_prompt, evaluation_messages
from search_comp.experiments.offline_entry import launcher_schema
from search_comp.experiments.shell_config import launcher_defaults
from test_browsecomp_sft import planner_row
from test_original_sft_inference import Tokenizer


def main_row(read=False):
    row = planner_row()
    row.pop('data_source')
    row['canonical_question'] = 'Which building was designed by the architect?'
    row['messages'][3]['content'] = '<tool_call>{"name":"search","parameters":{"user_query":"architect"}}</tool_call>'
    if read:
        row['messages'][5]['content'] = '<tool_call>{"name":"get_document","parameters":{"docid":"42"}}</tool_call>'
    else:
        row['messages'][5]['content'] = '<tool_call>{"name":"search","parameters":{"user_query":"building"}}</tool_call>'
    row['messages'][-1]['content'] = '<answer>Explanation: Evidence [42].\nExact Answer: Building\nConfidence: 90%</answer>'
    return row


@pytest.mark.parametrize('read,expected', [
    (False, '05978d39b1ca758b5c4a8aed2e39e86198777c1e3f91661ea7eff81af8ebe8fc'),
    (True, 'cbfeaad789da61f5b301c7f08577eb2e157706c764a0186de788fe0c1bea61b0'),
])
def test_task_text_matches_the_pinned_upstream_template(read, expected):
    # Fingerprints were measured from upstream 0469490's formatted task templates.
    assert hashlib.sha256(task_prompt('AUDIT_QUESTION', read).encode()).hexdigest() == expected


@pytest.mark.parametrize('read', [False, True])
def test_main_training_and_official_inference_have_identical_initial_messages(read):
    raw = main_row(read)
    mode = 'official-read' if read else 'official'
    row = normalize_browsecomp_sft(raw, 'browsecomp-sft/data.parquet', 0, system_prompt_mode=mode)
    args = SimpleNamespace(prompt_profile='official', allow_visit=read)
    system, source = agent_prompt(args)
    messages = evaluation_messages(args, system, row['question'])
    assert messages == row['messages'][:2]
    assert row['sft_shard'] == 'main'
    assert row['experiment_protocol'] == source
    assert 'browser.' not in system
    assert 'Today' not in system
    assert '<answer>' not in row['messages'][-1]['content']
    assert 'Exact Answer: Building' in row['messages'][-1]['content']
    calls = parse_calls(row['messages'][2]['content'], row['call_protocol'])
    assert calls == [dict(name='search', arguments={'query': 'architect'})]
    encoded = encode(row['messages'], Tokenizer(), call_protocol=row['call_protocol'])
    assert encoded['n_searches'] == 2


def test_by_shard_keeps_planner_independent_and_rejects_wrong_official_tools():
    main = normalize_browsecomp_sft(main_row(), 'browsecomp-sft/data.parquet', 0, system_prompt_mode='by-shard')
    planner = normalize_browsecomp_sft(planner_row(), 'browsecomp-sft/syn_planner.parquet', 0,
                                      system_prompt_mode='by-shard', planner_prompt_mode='original')
    assert main['system_prompt_mode'] == 'official'
    assert planner['system_prompt_mode'] == 'original'
    assert planner['sft_shard'] == 'planner'
    assert planner['messages'][0]['content'] == 'Original system'
    with pytest.raises(ValueError, match='main shard'):
        normalize_browsecomp_sft(planner_row(), 'browsecomp-sft/syn_planner.parquet', 0, system_prompt_mode='official')
    with pytest.raises(ValueError, match='not enabled'):
        normalize_browsecomp_sft(main_row(True), 'browsecomp-sft/data.parquet', 0, system_prompt_mode='official')


def test_incomplete_official_answers_are_not_fabricated():
    raw = main_row()
    raw['messages'][-1]['content'] = '<answer>Building</answer>'
    with pytest.raises(ValueError, match='no fields are fabricated'):
        normalize_browsecomp_sft(raw, 'browsecomp-sft/data.parquet', 0, system_prompt_mode='official')


def test_official_results_use_raw_prefix_and_distinct_snippet_tokenizer():
    seen = []
    doc = dict(docid='42', text='Parsed body', raw_text='Header\n' + 'x' * 600, score=0.75)
    index = SimpleNamespace(search=lambda query, k: seen.append((query, k)) or [doc], get=lambda docid: doc)
    tools = OfficialLocalTools(index, object(), snippet_tokenizer=Tokenizer(), allow_visit=False)
    reply = tools.execute([dict(name='search', arguments={'query': 'building'})])
    text = reply[0]['content'].removeprefix('<tool_response>').removesuffix('</tool_response>')
    result = json.loads(text)
    assert result == [dict(docid='42', score=0.75, snippet=doc['raw_text'][:512])]
    assert seen == [('building', 5)]
    with pytest.raises(ToolError, match='unknown_tool'):
        tools.execute([dict(name='browser.search', arguments={'query': 'building'})])
    with pytest.raises(ToolError, match='unknown_tool'):
        tools.execute([dict(name='get_document', arguments={'docid': '42'})])
    reading = OfficialLocalTools(index, object(), snippet_tokenizer=Tokenizer(), allow_visit=True, read_tokens=1)
    assert reading.get_document('42')['text'] == doc['raw_text']


@pytest.mark.parametrize('method', ['plain', 'beacon'])
def test_separate_launchers_have_explicit_profiles_and_official_defaults(method):
    root = Path(__file__).resolve().parents[1]
    schema = launcher_schema(method)
    for name, profile in [('46_run_browsecomp_plus', 'qwen-native'),
                          ('48_run_browsecomp_plus_official', 'official'),
                          ('49_run_browsecomp_plus_original', 'original')]:
        settings = launcher_defaults(root / 'scripts' / method / 'sh' / f'{name}.sh', schema)
        assert settings['GENERATION']['prompt_profile'] == profile
        if profile == 'official':
            assert settings['GENERATION']['topk'] == 5
            assert settings['GENERATION']['snippet_tokens'] == 512
            assert settings['GENERATION']['allow_visit'] is False
            assert settings['GENERATION']['observation_window'] == 0
            assert settings['GENERATION']['snippet_tokenizer_path']


def test_main_mode_reaches_preparation_worker(monkeypatch):
    from search_comp.workflows import prepare_openseeker as workflow
    monkeypatch.setattr(workflow, 'TOKENIZERS', {'mock': Tokenizer()}, raising=False)
    monkeypatch.setattr(workflow, 'SYSTEM_PROMPT_MODE', dict(system_prompt_mode='by-shard',
        main_prompt_mode='official-read', planner_prompt_mode='normalized'), raising=False)
    row, facts, error = workflow.measure(('browsecomp-sft/data.parquet', 0, main_row(True)))
    assert error is None
    assert row['system_prompt_mode'] == 'official-read'
    assert facts['sft_shard'] == 'main'
    assert facts['tools']['get_document'] == 1


@pytest.mark.parametrize('number,wrong', [('46', 'official'), ('48', 'qwen-native'), ('49', 'official')])
def test_entry_cannot_switch_experiment_protocol(monkeypatch, number, wrong):
    from search_comp.experiments import offline_entry
    monkeypatch.setattr(offline_entry, 'load_settings', lambda *a: {'GENERATION': {'prompt_profile': wrong}})
    with pytest.raises(ValueError, match='入口固定'):
        offline_entry.launch('beacon', f'{number}_run.py', {})


def test_scoring_cannot_cross_experiment_entries(tmp_path, monkeypatch):
    from search_comp.experiments import offline_entry
    (tmp_path / 'evaluation.json').write_text(json.dumps({'generation': {'prompt_profile': 'qwen-native'}}), encoding='utf-8')
    monkeypatch.setattr(offline_entry, 'load_settings', lambda *a: {
        'GENERATION': {'prompt_profile': 'official'}, 'EVALUATION_DIR': str(tmp_path)})
    with pytest.raises(ValueError, match='使用原实验入口'):
        offline_entry.launch('beacon', '48_run.py', {})


def test_official_document_view_preserves_raw_corpus_headers(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    from search_comp.offline.index import ParquetCorpus
    from search_comp.experiments.runtime import digest
    filename = tmp_path / 'raw/corpus/data/corpus.parquet'
    filename.parent.mkdir(parents=True)
    text = '---\ntitle: Building\n---\nEvidence'
    pq.write_table(pa.table({'docid': ['42'], 'url': ['https://example.com'], 'text': [text]}), filename)
    metadata = dict(files={'data/corpus.parquet': digest(filename)}, count=1)
    browser = ParquetCorpus(tmp_path, metadata)
    official = ParquetCorpus(tmp_path, metadata, preserve_raw=True)
    assert browser.get('42')['text'] == 'Evidence'
    assert 'raw_text' not in browser.get('42')
    assert official.get('42')['raw_text'] == text
    assert official.manifest['count'] == 1
