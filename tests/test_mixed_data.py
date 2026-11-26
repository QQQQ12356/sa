"""Mixing ratios apply to train; every configured held-out split stays protected."""
import json

import pytest

from search_comp.data.openseeker import VERSION, question_key
from search_comp.experiments import mixed_data
from search_comp.experiments.runtime import read_json, write_json


def sample(question, length=10):
    return dict(id=question, question=question, question_id=question_key(question),
                lengths={"mock": {"total": length}})


def parent(tmp_path, source, **splits):
    directory = tmp_path / source
    directory.mkdir()
    write_json(directory / "manifest.json", dict(
        protocol=VERSION, arguments={"source": "polar" if source == "openseeker" else source},
        tokenizers={"mock": {"backend_sha256": "same-tokenizer"}}))
    for split in mixed_data.SPLITS:
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in splits.get(split, [])), encoding="utf-8")
    return directory


def mix(tmp_path, monkeypatch, first, second, ratio, third=None, profile="mixed"):
    monkeypatch.setattr(mixed_data, "ROOT", tmp_path)
    mixed_data.mix_datasets(openseeker_prepared=first, research_prepared=second,
                            browsecomp_sft_prepared=third, profile=profile, tokenizer="mock",
                            max_length=100, ratio=ratio, seed=42)
    return tmp_path / "artifacts/datasets/mixed/prepared" / profile


@pytest.mark.parametrize("ratio,expected", [("0:1", ["research"]), ("2:0", ["open"])])
def test_zero_weight_at_either_position(tmp_path, monkeypatch, ratio, expected):
    first = parent(tmp_path, "openseeker", train=[sample("open")])
    second = parent(tmp_path, "research", train=[sample("research")])
    output = mix(tmp_path, monkeypatch, first, second, ratio)
    assert [row["question"] for row in read_rows(output, "train")] == expected
    assert read_json(output / "manifest.json")["arguments"]["ratio"] == ratio


def read_rows(directory, split):
    return [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_zero_source_can_be_empty_but_still_protects_held_out_questions(tmp_path, monkeypatch):
    first = parent(tmp_path, "openseeker", train=[sample("keep"), sample("reserved"), sample("too long test")])
    second = parent(tmp_path, "research", test=[sample("reserved"), sample("too long test", 101)])
    output = mix(tmp_path, monkeypatch, first, second, "1:0")
    assert [row["question"] for row in read_rows(output, "train")] == ["keep"]
    assert [row["question"] for row in read_rows(output, "test")] == ["reserved"]
    manifest = read_json(output / "manifest.json")
    assert manifest["actual_train_ratio"] == "1:0"
    assert manifest["excluded"] == {"cross_split": 2, "over_length": 1}
    assert set(manifest["parents"]) == {"openseeker", "research"}


def test_three_sources_can_train_only_browsecomp_sft(tmp_path, monkeypatch):
    first = parent(tmp_path, "openseeker")
    second = parent(tmp_path, "research")
    third = parent(tmp_path, "browsecomp-sft", train=[sample("browse")])
    output = mix(tmp_path, monkeypatch, first, second, "0:0:3", third)
    assert [row["question"] for row in read_rows(output, "train")] == ["browse"]
    assert read_json(output / "manifest.json")["actual_train_ratio"] == "0:0:1"


def test_positive_ratios_remain_deterministic_without_replacement(tmp_path, monkeypatch):
    first = parent(tmp_path, "openseeker", train=[sample(f"open {i}") for i in range(5)])
    second = parent(tmp_path, "research", train=[sample(f"research {i}") for i in range(3)])
    output = mix(tmp_path, monkeypatch, first, second, "4:2")
    repeated = mix(tmp_path, monkeypatch, first, second, "4:2", profile="repeat")
    assert read_json(output / "manifest.json")["actual_train_ratio"] == "4:2"
    assert len({row["question"] for row in read_rows(output, "train")}) == 6
    assert (output / "train.jsonl").read_bytes() == (repeated / "train.jsonl").read_bytes()


@pytest.mark.parametrize("ratio", ["0:0", "0:0:0", "-1:1", "0.5:1", "1", "1:1:1", "１:0"])
def test_invalid_ratios_fail_before_reading_parents(tmp_path, monkeypatch, ratio):
    with pytest.raises(ValueError, match="RATIO"):
        mix(tmp_path, monkeypatch, tmp_path / "missing1", tmp_path / "missing2", ratio)
    assert not (tmp_path / "artifacts").exists()


def test_positive_source_without_train_samples_fails_clearly(tmp_path, monkeypatch):
    first = parent(tmp_path, "openseeker")
    second = parent(tmp_path, "research", train=[sample("research")])
    with pytest.raises(ValueError, match="可用训练样本不足"):
        mix(tmp_path, monkeypatch, first, second, "1:0")
