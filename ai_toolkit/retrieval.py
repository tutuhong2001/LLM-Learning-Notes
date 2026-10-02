"""Lexical baseline and real Sentence-Transformers retrieval."""

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


def load_documents(directory: Path) -> list[Document]:
    """Each short Markdown file is one independently citable knowledge unit."""
    if not directory.is_dir():
        raise ValueError(f"知识库目录不存在：{directory}")
    documents = []
    for path in sorted(directory.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            relative = path.relative_to(directory).as_posix()
            documents.append(Document(relative.removesuffix(".md"), text, relative))
    if not documents:
        raise ValueError(f"知识库中没有非空 Markdown 文件：{directory}")
    return documents


def tokenize(text: str) -> list[str]:
    """English words plus Chinese characters/bigrams; no tokenizer download."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def validate_query(query: str, top_k: int) -> None:
    if not query.strip():
        raise ValueError("问题不能为空。")
    if top_k < 1:
        raise ValueError("top-k 必须大于 0。")


class BM25Retriever:
    """Dependency-free baseline; scores are not probabilities."""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("文档列表不能为空。")
        self.documents = documents
        self.counts = [Counter(tokenize(doc.text)) for doc in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average_length = sum(self.lengths) / len(documents) or 1.0
        self.df = Counter(token for count in self.counts for token in count)

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        validate_query(query, top_k)
        query_tokens = set(tokenize(query))
        hits = []
        for document, counts, length in zip(self.documents, self.counts, self.lengths):
            score = 0.0
            for token in sorted(query_tokens):
                frequency = counts[token]
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.documents) - self.df[token] + 0.5)
                               / (self.df[token] + 0.5))
                denominator = frequency + 1.5 * (0.25 + 0.75 * length / self.average_length)
                score += idf * frequency * 2.5 / denominator
            if score > 0:
                hits.append(Hit(document, score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.document.id))[:top_k]


class SemanticRetriever:
    """Normalized embeddings + dot product = cosine similarity."""

    def __init__(self, documents: list[Document], model_name: str = DEFAULT_MODEL):
        if not documents:
            raise ValueError("文档列表不能为空。")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('语义检索依赖缺失或导入失败。请在独立虚拟环境运行 pip install -e ".[semantic]"；'
                               '若已安装，请单独导入 sentence_transformers 排查依赖冲突。') from exc
        self.documents = documents
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = self.model.encode(
            [document.text for document in documents],
            normalize_embeddings=True, show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        validate_query(query, top_k)
        # This instruction belongs to this BGE model, not arbitrary embedding models.
        prefix = BGE_QUERY_PREFIX if self.model_name == DEFAULT_MODEL else ""
        vector = self.model.encode(prefix + query, normalize_embeddings=True,
                                   show_progress_bar=False)
        scores = self.embeddings @ vector
        hits = [Hit(document, float(score))
                for document, score in zip(self.documents, scores)]
        return sorted(hits, key=lambda hit: (-hit.score, hit.document.id))[:top_k]
