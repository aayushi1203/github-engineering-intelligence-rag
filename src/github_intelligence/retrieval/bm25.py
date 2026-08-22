import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from github_intelligence.models import DocumentChunk, RetrievalResult

_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class BM25Retriever:
    """In-memory BM25 index backed by a persistent canonical chunk file."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._chunks: list[DocumentChunk] = []
        self._corpus: list[list[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._average_length = 0.0
        if index_path.exists():
            self._load()

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._rebuild()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as output:
            for chunk in self._chunks:
                output.write(chunk.model_dump_json() + "\n")

    def search(
        self, query: str, *, top_k: int = 10, repository: str | None = None
    ) -> list[RetrievalResult]:
        if not query.strip() or top_k <= 0 or not self._corpus:
            return []
        query_tokens = tokenize(query)
        if not any(token in self._document_frequencies for token in query_tokens):
            return []
        scores = [self._score(tokens, query_tokens) for tokens in self._corpus]
        ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
        results: list[RetrievalResult] = []
        for index, score in ranked:
            chunk = self._chunks[index]
            if repository and chunk.repository != repository:
                continue
            if float(score) <= 0:
                continue
            results.append(_result(chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def _load(self) -> None:
        with self.index_path.open(encoding="utf-8") as source:
            self._chunks = [
                DocumentChunk.model_validate_json(line) for line in source if line.strip()
            ]
        self._rebuild()

    def _rebuild(self) -> None:
        self._corpus = [tokenize(chunk.text) for chunk in self._chunks]
        self._document_frequencies = Counter(
            token for document in self._corpus for token in set(document)
        )
        self._average_length = (
            sum(len(document) for document in self._corpus) / len(self._corpus)
            if self._corpus
            else 0.0
        )

    def _score(self, document: list[str], query: list[str]) -> float:
        frequencies = Counter(document)
        document_count = len(self._corpus)
        score = 0.0
        k1, b = 1.5, 0.75
        for token in query:
            frequency = frequencies[token]
            if not frequency:
                continue
            document_frequency = self._document_frequencies[token]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_ratio = len(document) / self._average_length if self._average_length else 0
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length_ratio))
            )
        return score


def _result(chunk: DocumentChunk, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        title=chunk.title,
        source_url=chunk.source_url,
        source_type=chunk.source_type,
        score=score,
        repository=chunk.repository,
        metadata=chunk.metadata,
    )
