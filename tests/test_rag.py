from types import SimpleNamespace

import pytest

from ai_toolkit.rag import build_messages, citation_warnings, generate_answer
from ai_toolkit.retrieval import Document, Hit


HITS = [Hit(Document("a", "LoRA 训练低秩参数。", "a.md"), 1.0)]


def test_prompt_contains_source_and_explicit_data_boundary():
    messages = build_messages("LoRA 是什么？", HITS)
    assert messages[0]["role"] == "system"
    assert "不可信数据" in messages[0]["content"]
    assert "[1] 文件：a.md" in messages[1]["content"]
    assert "LoRA 是什么？" in messages[1]["content"]


def test_citation_range_and_missing_citations():
    assert citation_warnings("低秩适配 [1]", 1) == []
    assert "无效" in citation_warnings("错误 [0] [2]", 1)[0]
    assert "没有" in citation_warnings("没有引用", 1)[0]


def test_no_hits_skips_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "不足" in generate_answer("问题", [])


def test_missing_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        generate_answer("LoRA", HITS)


def test_sdk_request_contract_without_external_call(monkeypatch):
    import sys

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["config"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="低秩参数 [1]"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    assert generate_answer("LoRA", HITS) == "低秩参数 [1]"
    assert captured["request"]["model"] == "test-model"
    assert captured["config"]["timeout"] == 45.0
    assert "a.md" in captured["request"]["messages"][1]["content"]
