"""Test model input conventions without downloading a model in CI."""

import sys
from types import SimpleNamespace

from ai_toolkit.retrieval import BGE_QUERY_PREFIX, DEFAULT_MODEL, Document, SemanticRetriever


def test_bge_prefix_normalization_and_score_order(monkeypatch):
    calls = []

    class Matrix:
        def __matmul__(self, vector):
            return [0.1, 0.9]

    class FakeModel:
        def __init__(self, name, device):
            assert name == DEFAULT_MODEL
            assert device == "cpu"

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return Matrix() if isinstance(texts, list) else [1, 0]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel))
    retriever = SemanticRetriever([Document("a", "甲", "a.md"), Document("b", "乙", "b.md")])
    assert retriever.search("问题", 1)[0].document.id == "b"
    assert calls[0][0] == ["甲", "乙"]
    assert calls[1][0] == BGE_QUERY_PREFIX + "问题"
    assert all(options["normalize_embeddings"] for _, options in calls)
