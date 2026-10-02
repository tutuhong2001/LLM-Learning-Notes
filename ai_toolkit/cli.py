"""Run from the repository root: python -m ai_toolkit --help."""

import argparse
import json
from pathlib import Path
import sys

from .evaluation import evaluate
from .rag import build_messages, citation_warnings, generate_answer
from .retrieval import BM25Retriever, DEFAULT_MODEL, SemanticRetriever, load_documents


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文 AI 笔记：检索 / RAG / 检索评估")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("search", "检索知识库"), ("ask", "检索后生成带来源的回答"),
                            ("evaluate", "评估 Hit Rate、Recall、MRR")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--backend", choices=["bm25", "semantic"], default="bm25")
        command.add_argument("--docs", type=Path, default=Path("data/knowledge"))
        command.add_argument("--top-k", type=int, default=3)
        command.add_argument("--embedding-model", default=DEFAULT_MODEL)
        if name in {"search", "ask"}:
            command.add_argument("question")
        if name == "ask":
            command.add_argument("--dry-run", action="store_true", help="仅显示资料和请求，不调用 API")
            command.add_argument("--model", help="覆盖 OPENAI_MODEL")
        if name == "evaluate":
            command.add_argument("--dataset", type=Path, default=Path("data/eval_questions.json"))
            command.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.top_k < 1:
        raise ValueError("top-k 必须大于 0。")
    if hasattr(args, "question") and not args.question.strip():
        raise ValueError("问题不能为空。")
    documents = load_documents(args.docs)
    retriever = (BM25Retriever(documents) if args.backend == "bm25"
                 else SemanticRetriever(documents, args.embedding_model))
    if args.command == "evaluate":
        report = evaluate(retriever, args.dataset, args.top_k)
        report.update({"backend": args.backend,
                       "embedding_model": args.embedding_model if args.backend == "semantic" else None})
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    hits = retriever.search(args.question, args.top_k)
    print(f"检索后端：{args.backend} | 命中文档：{len(hits)}")
    for index, hit in enumerate(hits, 1):
        print(f"\n[{index}] {hit.document.source} | score={hit.score:.4f}\n{hit.document.text}")
    if args.command == "ask":
        if args.dry_run:
            print("\n[DRY RUN] 以下是待发送的请求内容；没有调用模型，也没有生成回答。")
            print(json.dumps(build_messages(args.question, hits), ensure_ascii=False, indent=2))
        else:
            print("\n回答：")
            answer = generate_answer(args.question, hits, args.model)
            print(answer)
            for warning in citation_warnings(answer, len(hits)):
                print(f"提示：{warning}", file=sys.stderr)
    elif not hits:
        print("没有匹配资料，请调整问题或尝试 semantic 后端。")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows terminals and redirected files should both preserve Chinese output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = make_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Provider exception text can contain request details: only disclose the type.
        print(f"执行失败（{type(exc).__name__}）。请检查依赖、网络或 API 配置。", file=sys.stderr)
        return 1
