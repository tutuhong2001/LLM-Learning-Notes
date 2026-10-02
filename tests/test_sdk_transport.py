"""Validate real SDK serialization/parsing without network or paid credentials."""

import json

import pytest

from ai_toolkit.rag import generate_answer
from ai_toolkit.retrieval import Document, Hit


def test_real_sdk_over_mock_http_transport(monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    real_client = openai.OpenAI

    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert "[1] 文件：note.md" in payload["messages"][1]["content"]
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "检索后生成回答 [1]"}}],
        })

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: real_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    hits = [Hit(Document("note", "RAG 先检索再生成。", "note.md"), 1.0)]
    assert generate_answer("RAG?", hits) == "检索后生成回答 [1]"
