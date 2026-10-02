import json
from pathlib import Path

from ai_toolkit.cli import main


ROOT = Path(__file__).resolve().parents[1]
DOCS = str(ROOT / "data" / "knowledge")


def test_dry_run_is_explicit_and_works_without_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["ask", "RAG", "--docs", DOCS, "--dry-run"]) == 0
    assert "没有调用模型" in capsys.readouterr().out


def test_evaluation_writes_report(tmp_path):
    output = tmp_path / "nested" / "report.json"
    assert main(["evaluate", "--docs", DOCS, "--dataset", str(ROOT / "data/eval_questions.json"),
                 "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["num_queries"] == 20
    assert 0 <= report["metrics"]["recall@3"] <= 1


def test_invalid_top_k_fails(capsys):
    assert main(["search", "RAG", "--top-k", "0"]) == 1
    assert "top-k" in capsys.readouterr().err
