"""Retrieval metrics with explicit relevance labels and strict validation."""

import json
from pathlib import Path


def evaluate(retriever, dataset: Path, top_k: int) -> dict:
    if top_k < 1:
        raise ValueError("top-k 必须大于 0。")
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评估集必须是非空 JSON 数组。")
    available = {document.id for document in retriever.documents}
    details = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("每个评估条目必须有非空 query。")
        labels = case.get("relevant_ids")
        if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
            raise ValueError("relevant_ids 必须是非空字符串数组。")
        relevant = set(labels)
        if relevant - available:
            raise ValueError(f"评估集引用了不存在的文档：{sorted(relevant - available)}")
        retrieved = [hit.document.id for hit in retriever.search(case["query"], top_k)]
        matched = relevant.intersection(retrieved)
        first_rank = next((rank for rank, identifier in enumerate(retrieved, 1)
                           if identifier in relevant), None)
        details.append({
            "query": case["query"], "relevant_ids": sorted(relevant),
            "retrieved_ids": retrieved, "hit": int(bool(matched)),
            "recall": len(matched) / len(relevant),
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        })
    count = len(details)
    return {
        "top_k": top_k, "num_queries": count,
        "metrics": {
            f"hit_rate@{top_k}": sum(row["hit"] for row in details) / count,
            f"recall@{top_k}": sum(row["recall"] for row in details) / count,
            f"mrr@{top_k}": sum(row["reciprocal_rank"] for row in details) / count,
        },
        "details": details,
    }
