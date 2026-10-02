"""Reproduce delivery checks without network calls by default.

Run from the repository root: python scripts/verify.py [--semantic]
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tests and save reproducible demo reports.")
    parser.add_argument("--semantic", action="store_true", help="Also load/download the embedding model")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    commands = [
        ("tests.txt", ["-m", "pytest", "-q"]),
        ("search-demo.txt", ["-m", "ai_toolkit", "search", "LoRA 为什么能减少训练参数？", "--top-k", "2"]),
        ("rag-dry-run.txt", ["-m", "ai_toolkit", "ask", "RAG 与微调有什么区别？", "--dry-run"]),
    ]
    for backend in (["bm25", "semantic"] if args.semantic else ["bm25"]):
        for label, dataset in [("eval", "eval_questions.json"), ("paraphrases", "eval_paraphrases.json")]:
            commands.append((f"{backend}-{label}-run.txt", [
                "-m", "ai_toolkit", "evaluate", "--backend", backend,
                "--dataset", f"data/{dataset}", "--output", f"outputs/{backend}-{label}.json",
            ]))
    for filename, command in commands:
        print(f"Running: {filename}", flush=True)
        result = subprocess.run([sys.executable, *command], cwd=root, capture_output=True)
        (output / filename).write_bytes(result.stdout)
        if result.returncode:
            print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
            return result.returncode
    versions = {}
    for name in ["pytest", "openai", "sentence-transformers", "transformers", "torch", "numpy", "huggingface-hub"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = {"python": platform.python_version(), "platform": platform.system(),
                   "packages": versions, "semantic_requested": args.semantic}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print("Verification passed. Reports saved in outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
