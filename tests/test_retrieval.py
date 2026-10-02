from pathlib import Path

import pytest

from ai_toolkit.retrieval import BM25Retriever, Document, load_documents


def test_rare_exact_term_ranks_first():
    retriever = BM25Retriever([
        Document("other", "文档检索和普通问答", "other.md"),
        Document("lora", "LoRA 冻结权重并训练低秩参数", "lora.md"),
    ])
    assert retriever.search("LoRA", 1)[0].document.id == "lora"
    assert retriever.search("qwertyunknown") == []


def test_chinese_matching_and_stable_ties():
    retriever = BM25Retriever([Document("b", "向量检索", "b.md"),
                               Document("a", "向量检索", "a.md")])
    assert [hit.document.id for hit in retriever.search("向量", 10)] == ["a", "b"]


@pytest.mark.parametrize("query,k", [("", 3), ("  ", 3), ("test", 0), ("test", -1)])
def test_invalid_search(query, k):
    with pytest.raises(ValueError):
        BM25Retriever([Document("a", "test", "a.md")]).search(query, k)


def test_nested_ids_and_blank_files(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "topic.md").write_text("知识", encoding="utf-8")
    (tmp_path / "topic.md").write_text("根文档", encoding="utf-8")
    (tmp_path / "blank.md").write_text("  ", encoding="utf-8")
    assert {doc.id for doc in load_documents(tmp_path)} == {"sub/topic", "topic"}


def test_missing_and_empty_corpus(tmp_path: Path):
    with pytest.raises(ValueError):
        load_documents(tmp_path)
    with pytest.raises(ValueError):
        load_documents(tmp_path / "missing")
