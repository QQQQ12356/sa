import json
import sys
from types import SimpleNamespace

import pytest

from search_comp.data.browsecomp_sft import REPO, SHARDS, normalize_browsecomp_sft
from search_comp.data.openseeker import parse_calls, question_key
from search_comp.data.question_overlap import QuestionIndex
from search_comp.evaluation.training_overlap import audit_training_overlap, select_questions
from search_comp.experiments.runtime import digest
from search_comp.experiments import data as dataset_jobs
from search_comp.workflows.prepare_openseeker import prepare_sft_input


QUESTION = "Which historic building was designed by the same architect as the palace?"


def test_main_rewrites_share_canonical_question_and_respect_protection():
    official = {"id": "42", "query": QUESTION, "answer": "reference"}
    messages = [
        {"role": "system", "content": "Original system"},
        {"role": "user", "content": f"We need to answer this puzzle: {QUESTION}"},
        {"role": "reasoning", "content": "Search for the architect."},
        {"role": "tool_call", "content": '<tool_call>{"name":"search","parameters":{"user_query":"architect"}}</tool_call>'},
        {"role": "tool_output", "content": "<tool_response>historical evidence</tool_response>"},
        {"role": "answer", "content": "<answer>demonstration</answer>"},
    ]
    raw = {"messages": json.dumps(messages), "category": "browsecomp-openai/gpt-oss-20b"}
    index = QuestionIndex([official])
    assert prepare_sft_input(raw, index, index) == "protected_question"
    assert prepare_sft_input(raw, index, QuestionIndex([])) is None

    result = normalize_browsecomp_sft(raw, "browsecomp-sft/data.parquet", 0)
    assert result["question_id"] == question_key(QUESTION)
    assert result["original_question"] == messages[1]["content"]
    assert result["benchmark_question_id"] == "42"
    assert result["answer"] == "demonstration"
    assert parse_calls(result["messages"][2]["content"]) == [
        {"name": "search", "arguments": {"query": ["architect"]}},
    ]


def test_overlap_filters_before_limit_and_audits_the_training_fingerprint(tmp_path):
    questions = [
        {"id": 10, "query": QUESTION, "answer": "reference"},
        {"id": 20, "query": "What is the capital of Japan?", "answer": "Tokyo"},
    ]
    question_file = tmp_path / "questions.jsonl"
    question_file.write_text("".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8")
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(dict(id="sample", question="Rewritten question",
                                         original_question=f"Find the answer: {QUESTION}")) + "\n", encoding="utf-8")
    report = audit_training_overlap(question_file, train_file, digest(train_file), 0.30)
    assert report["excluded_ids"] == ["10"]
    assert report["matches"]["10"]["method"] == "embedded"
    report_file = tmp_path / "overlap.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    assert select_questions(questions, 1, report_file) == questions[1:]


def planner_row(question=QUESTION):
    return dict(data_source="hotpotqa", qid="source-42", messages=[
        dict(role="system", content="Original system"),
        dict(role="user", content=question),
        dict(role="reasoning", content="Search, then read the page."),
        dict(role="tool_call", content='<tool_call>{"name":"web_search","arguments":{"query":"architect"}}</tool_call>'),
        dict(role="tool_output", content="Found a page."),
        dict(role="tool_call", content='<tool_call>{"name":"crawl_webpage","arguments":{"url":"https://example.com"}}</tool_call>'),
        dict(role="tool_output", content="Historical evidence."),
        dict(role="answer", content="The historic building."),
    ])


def test_planner_conversion_preserves_provenance_and_checks_protection():
    raw = planner_row()
    original = json.dumps(raw)
    protected = QuestionIndex([dict(id="42", query=QUESTION)])
    assert prepare_sft_input(raw, None, protected) == "protected_question"
    assert prepare_sft_input(raw, None, QuestionIndex([])) is None
    result = normalize_browsecomp_sft(raw, "browsecomp-sft/syn_planner.parquet", 0)
    assert result["upstream_source"] == "hotpotqa"
    assert result["upstream_qid"] == "source-42"
    assert result["training_only"] is True
    assert result["question_id"] == question_key(QUESTION)
    calls = [call for message in result["messages"] if message["role"] == "assistant"
             for call in parse_calls(message["content"])]
    assert calls == [dict(name="search", arguments={"query": ["architect"]}),
                     dict(name="visit", arguments={"url": "https://example.com", "goal": QUESTION})]
    assert json.dumps(raw) == original


def test_canonical_mapping_checks_the_mapped_question_against_protection():
    # A paraphrase can match the canonical index while falling below the protection threshold.
    raw = planner_row("Rewritten question")
    raw.pop("data_source")
    canonical = SimpleNamespace(matches=lambda _: [(dict(query=QUESTION, id="42"), 0.8, "four_gram")])
    protected = QuestionIndex([dict(query=QUESTION, id="42")])
    assert prepare_sft_input(raw, canonical, protected) == "protected_question"
    assert "canonical_question" not in raw


def test_download_retains_both_original_shards_and_pinned_fingerprints(tmp_path, monkeypatch):
    from search_comp.workflows.download_openseeker import run

    revision = "test-commit"
    calls = []

    class Hub:
        def __init__(self, **kwargs):
            pass

        def dataset_info(self, repo, **kwargs):
            assert repo == REPO
            assert kwargs["revision"] == revision
            return SimpleNamespace(sha=revision)

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        directory = tmp_path / revision
        directory.mkdir()
        for shard in SHARDS:
            (directory / shard).write_bytes(b"original parquet placeholder")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Hub, snapshot_download=download))
    run(SimpleNamespace(source="browsecomp-sft", revision=revision,
                        endpoint="https://huggingface.co", output=tmp_path))
    manifest = json.loads((tmp_path / revision / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repo"] == REPO
    assert manifest["revision"] == revision
    assert manifest["jsonl_inputs"] == []
    assert len(manifest["parquet_inputs"]) == len(SHARDS)
    assert calls[0][1]["allow_patterns"] == [*SHARDS, "README.md"]
    for item in manifest["files"]:
        assert item["sha256"] == digest(item["path"])


def test_snapshot_cannot_silently_omit_a_selected_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_jobs, "configure", lambda _: None)
    (tmp_path / "download_manifest.json").write_text(json.dumps(dict(
        repo=REPO, parquet_inputs=[str(tmp_path / "syn_planner.parquet")], revision="test-commit")),
        encoding="utf-8")
    with pytest.raises(ValueError, match="缺少所选 SA_SHARDS"):
        dataset_jobs.prepare_dataset(
            dataset_name="browsecomp-sft", source="browsecomp-sft", snapshot_dir=tmp_path,
            inputs=[], shards=list(SHARDS), profile="test", models=["mock"], max_length=100,
            min_length=1, correctness="any", observations="any", workers=1, seed=42,
            val_fraction=0.02, test_fraction=0.02, offline=True, environment={})


@pytest.mark.parametrize('serialized', [True, False])
def test_original_preserves_system_question_actions_and_answer(serialized):
    from search_comp.data.openseeker import encode
    raw = planner_row()
    system = '  Original system\r\nUse web_search and crawl_webpage. 中文\n'
    raw['messages'][0]['content'] = system
    raw['messages'][-1]['content'] = '<answer>Building</answer>'
    original = json.loads(json.dumps(raw))
    raw['canonical_question'] = 'Canonical grouping question'
    if serialized:
        raw['messages'] = json.dumps(raw['messages'])
    result = normalize_browsecomp_sft(raw, 'browsecomp-sft/syn_planner.parquet', 0,
                                      system_prompt_mode='original')
    assert result['messages'][0]['content'] == system
    assert result['messages'][1]['content'] == QUESTION
    assert result['question_id'] == question_key('Canonical grouping question')
    assert result['messages'][-1]['content'] == '<answer>Building</answer>'
    assert result['answer'] == 'Building'
    assert result['call_protocol'] == 'research-original'
    calls = [call for msg in result['messages'] if msg['role'] == 'assistant'
             for call in parse_calls(msg['content'], result['call_protocol'])]
    assert [call['name'] for call in calls] == ['web_search', 'crawl_webpage']
    assert original['messages'][3]['content'] in result['messages'][2]['content']

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode('utf-8'))

    encoded = encode(result['messages'], Tokenizer(), call_protocol=result['call_protocol'])
    assert encoded['n_searches'] == 2
    assert len(encoded['regions']) == 2
    for start, end in encoded['regions']:
        assert encoded['labels'][start:end] == [-100] * (end-start)


def test_original_supports_parameters_and_get_document_without_conversion():
    raw = planner_row()
    raw.pop('data_source')
    raw['messages'][3]['content'] = '<tool_call>{"name":"search","parameters":{"user_query":"architect"}}</tool_call>'
    raw['messages'][5]['content'] = '<tool_call>{"name":"get_document","parameters":{"docid":"42"}}</tool_call>'
    result = normalize_browsecomp_sft(raw, 'browsecomp-sft/data.parquet', 0, system_prompt_mode='original')
    assert result['messages'][0]['content'] == raw['messages'][0]['content']
    assert parse_calls(result['messages'][4]['content'], 'research-original') == [
        dict(name='get_document', arguments={'docid': '42'})]
    with pytest.raises(ValueError, match='envelope'):
        parse_calls(result['messages'][4]['content'])
    with pytest.raises(ValueError, match='unsupported BrowseComp-Plus SFT tool'):
        normalize_browsecomp_sft(raw, 'browsecomp-sft/data.parquet', 0)


def test_original_system_is_reused_without_native_or_local_additions(tmp_path):
    from search_comp.data.agent_prompt import resolve_original_system_prompt
    from search_comp.workflows.offline_benchmark import agent_prompt
    result = normalize_browsecomp_sft(planner_row(), 'browsecomp-sft/syn_planner.parquet', 0,
                                      system_prompt_mode='original')
    filename = tmp_path / 'train.jsonl'
    filename.write_text(json.dumps(result) + '\n', encoding='utf-8')
    system, _ = agent_prompt(SimpleNamespace(prompt_profile='original', prompt_data=filename, run_dir=None))
    assert system == result['messages'][0]['content']
    result['messages'][0]['content'] = 'Another system'
    with filename.open('a', encoding='utf-8') as out:
        out.write(json.dumps(result) + '\n')
    with pytest.raises(ValueError, match='mixed training systems'):
        resolve_original_system_prompt(filename)
    with pytest.raises(ValueError, match='requires PROMPT_DATA'):
        resolve_original_system_prompt()


def test_original_profile_rejects_normalized_data(tmp_path):
    from search_comp.data.agent_prompt import resolve_original_system_prompt
    result = normalize_browsecomp_sft(planner_row(), 'browsecomp-sft/syn_planner.parquet', 0)
    filename = tmp_path / 'train.jsonl'
    filename.write_text(json.dumps(result) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SYSTEM_PROMPT_MODE=original'):
        resolve_original_system_prompt(filename)


def test_invalid_system_prompt_mode_fails_before_loading_data():
    with pytest.raises(ValueError, match='SYSTEM_PROMPT_MODE'):
        normalize_browsecomp_sft({}, 'browsecomp-sft/data.parquet', 0, system_prompt_mode='typo')


def test_original_rows_load_through_the_training_dataset(tmp_path):
    from search_comp.data.openseeker import OpenSeekerDataset
    from test_original_sft_inference import Tokenizer
    row = normalize_browsecomp_sft(planner_row(), 'browsecomp-sft/syn_planner.parquet', 0,
                                  system_prompt_mode='original')
    filename = tmp_path / 'train.jsonl'
    filename.write_text(json.dumps(row) + '\n', encoding='utf-8')
    sample = OpenSeekerDataset(filename, Tokenizer())[0]
    text = bytes(sample['input_ids']).decode('utf-8')
    assert text.startswith('<|im_start|>system\nOriginal system<|im_end|>')
    assert '<tool_calls_begin>' not in text
    assert '"name":"web_search"' in text
    assert sample['n_searches'] == 2


def test_original_preprocessing_keeps_protection_deduplication_and_manifest(tmp_path, monkeypatch):
    from search_comp.workflows import prepare_openseeker as workflow
    from test_original_sft_inference import Tokenizer

    class MeasuredTokenizer(Tokenizer):
        name_or_path = 'mock'
        backend_tokenizer = SimpleNamespace(to_str=lambda: 'mock-backend')

        def __len__(self):
            return 256

    def initialize(models, offline, mode='normalized'):
        monkeypatch.setattr(workflow, 'TOKENIZERS', {'mock': MeasuredTokenizer()}, raising=False)
        monkeypatch.setattr(workflow, 'SYSTEM_PROMPT_MODE', mode, raising=False)

    class Pool:
        def __init__(self, workers, *, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, items, **kwargs):
            return map(function, items)

    monkeypatch.setattr(workflow, 'initialize', initialize)
    monkeypatch.setattr(workflow, 'ProcessPoolExecutor', Pool)
    raw = tmp_path / 'syn_planner.jsonl'
    raw.write_text(''.join(json.dumps(row) + '\n' for row in [
        planner_row(), planner_row(), planner_row('Protected question?')]), encoding='utf-8')
    protected = tmp_path / 'questions.jsonl'
    protected.write_text(json.dumps(dict(id='1', query='Protected question?', answer='Secret')) + '\n', encoding='utf-8')
    output = tmp_path / 'prepared'
    workflow.run(SimpleNamespace(source='browsecomp-sft', input=[raw], output=output,
        models=['mock'], workers=1, offline=True, system_prompt_mode='original', min_length=1,
        max_length=10000, exact_length=None, correctness='any', require_correct=False,
        observations='any', tokenizer=None, seed=42, val_fraction=0, test_fraction=0, limit=None,
        protected_questions=[str(protected)], canonical_questions=None, overlap_threshold=0.30))
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1
    assert rows[0]['messages'][0]['content'] == 'Original system'
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['system_prompt_mode'] == manifest['arguments']['system_prompt_mode'] == 'original'
    assert manifest['counts']['protected_question'] == 1
    assert manifest['counts']['duplicate_question'] == 1
    assert manifest['counts']['train'] == 1
    assert manifest['question_references'][str(protected)] == digest(protected)
