import json

import pytest

from ai_toolkit.evaluation import evaluate
from ai_toolkit.retrieval import Document, Hit


class FixedRetriever:
    documents = [Document(key, key, key + ".md") for key in "abc"]

    def search(self, query, top_k):
        return [Hit(doc, 1.0) for doc in self.documents[:top_k]]


def test_metrics_with_multiple_relevant_docs_and_miss(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([
        {"query": "partial", "relevant_ids": ["b", "c"]},
        {"query": "miss", "relevant_ids": ["c"]},
    ]), encoding="utf-8")
    report = evaluate(FixedRetriever(), dataset, 2)
    assert report["metrics"] == {"hit_rate@2": 0.5, "recall@2": 0.25, "mrr@2": 0.25}


@pytest.mark.parametrize("cases", [[], {}, [{"query": "q", "relevant_ids": ["missing"]}],
                                      [{"query": "q", "relevant_ids": []}],
                                      [{"query": " ", "relevant_ids": ["a"]}]])
def test_reject_invalid_labels(tmp_path, cases):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate(FixedRetriever(), dataset, 3)
