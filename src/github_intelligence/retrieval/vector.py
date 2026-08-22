from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from github_intelligence.models import DocumentChunk, RetrievalResult


class Encoder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class VectorMatch:
    chunk: DocumentChunk
    score: float


class VectorStore(Protocol):
    def replace(
        self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]], dimension: int
    ) -> None: ...

    def search(
        self, vector: Sequence[float], *, top_k: int, repository: str | None
    ) -> list[VectorMatch]: ...

    def exists(self) -> bool: ...


class SentenceTransformerEncoder:
    """Lazy sentence-transformers adapter so BM25 usage has no model startup cost."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        dimension = self._load().get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report an output dimension")
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._load().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=len(texts) > 100
        )
        return [[float(value) for value in row] for row in embeddings]


class QdrantVectorStore:
    """Embedded Qdrant adapter; imports Qdrant only when vector storage is requested."""

    def __init__(self, path: Path | str, collection: str, batch_size: int = 64) -> None:
        from qdrant_client import QdrantClient, models

        self.client = QdrantClient(path=str(path))
        self.models = models
        self.collection = collection
        self.batch_size = batch_size

    def exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def replace(
        self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]], dimension: int
    ) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self.models.VectorParams(
                size=dimension, distance=self.models.Distance.COSINE
            ),
        )
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            batch_vectors = vectors[start : start + self.batch_size]
            points = [
                self.models.PointStruct(
                    id=chunk.chunk_id,
                    vector=list(vector),
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self, vector: Sequence[float], *, top_k: int, repository: str | None
    ) -> list[VectorMatch]:
        query_filter = None
        if repository:
            query_filter = self.models.Filter(
                must=[
                    self.models.FieldCondition(
                        key="repository", match=self.models.MatchValue(value=repository)
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            VectorMatch(DocumentChunk.model_validate(point.payload or {}), float(point.score))
            for point in response.points
        ]


class QdrantVectorRetriever:
    """Semantic retriever over a Qdrant-compatible vector store."""

    def __init__(self, store: VectorStore, encoder: Encoder) -> None:
        self.store = store
        self.encoder = encoder

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        vectors = self.encoder.encode([chunk.text for chunk in chunks]) if chunks else []
        self.store.replace(chunks, vectors, self.encoder.dimension)

    def search(
        self, query: str, *, top_k: int = 10, repository: str | None = None
    ) -> list[RetrievalResult]:
        if not query.strip() or top_k <= 0 or not self.store.exists():
            return []
        matches = self.store.search(
            self.encoder.encode([query])[0], top_k=top_k, repository=repository
        )
        return [
            RetrievalResult(
                chunk_id=match.chunk.chunk_id,
                text=match.chunk.text,
                title=match.chunk.title,
                source_url=match.chunk.source_url,
                source_type=match.chunk.source_type,
                score=match.score,
                repository=match.chunk.repository,
                metadata=match.chunk.metadata,
            )
            for match in matches
        ]
