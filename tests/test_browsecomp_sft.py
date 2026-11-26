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
