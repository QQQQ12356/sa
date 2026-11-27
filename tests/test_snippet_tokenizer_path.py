"""Callable official runs accept tokenizer paths from shell settings."""
from pathlib import Path

import pytest

from search_comp.experiments.benchmarks import offline_options
from search_comp.workflows.offline_benchmark import run


@pytest.mark.parametrize("path_type", [str, Path])
def test_official_tokenizer_path_is_normalized(tmp_path, path_type):
    args = offline_options(
        "browsecomp-plus", tmp_path, prompt_profile="official",
        snippet_tokenizer_path=path_type(tmp_path), kind="beacon", dry_run=True,
    )

    run(args)

    assert args.snippet_tokenizer_path == tmp_path.resolve()
    assert isinstance(args.snippet_tokenizer_path, Path)


@pytest.mark.parametrize("value,message", [
    (None, "requires a local Qwen3-0.6B"),
    ("", "requires a local Qwen3-0.6B"),
    ("missing", "must be an existing local Qwen3-0.6B directory"),
    ("file", "must be an existing local Qwen3-0.6B directory"),
])
def test_official_tokenizer_path_validation(tmp_path, value, message):
    if value == "file":
        (tmp_path / "file").write_text("not a directory", encoding="utf-8")
    tokenizer_path = str(tmp_path / value) if value else value
    args = offline_options(
        "browsecomp-plus", tmp_path, prompt_profile="official",
        snippet_tokenizer_path=tokenizer_path, kind="beacon", dry_run=True,
    )

    with pytest.raises(ValueError, match=message):
        run(args)
