"""Copy-on-write application storage, separate from destructive snapshot replace()."""

import hashlib
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from github_intelligence.models import DocumentChunk
from github_intelligence.retrieval.vector import QdrantVectorStore, VectorMatch


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def publish_lexical(path: Path, chunks: Sequence[DocumentChunk]) -> str:
    """Persist canonical chunks atomically; BM25 loads its existing scoring implementation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".chunks-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for chunk in chunks:
                output.write(chunk.model_dump_json() + "\n")
            output.flush()
            os.fsync(output.fileno())
        digest = file_hash(temporary)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


class IncrementalVectorStore(QdrantVectorStore):
    """Immutable points shared by revision membership, using real Qdrant operations."""

    def __init__(
        self,
        path: Path | str | None,
        collection: str,
        batch_size: int = 64,
        *,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 5,
    ) -> None:
        if (path is None) == (url is None):
            raise ValueError("Specify exactly one embedded path or server URL")
        if url is None:
            assert path is not None
            self.path = str(path)
            self.identity = str(Path(path).resolve())
            super().__init__(path, collection, batch_size)
        else:
            from qdrant_client import QdrantClient, models

            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Server URL must not contain credentials or query parameters")
            self.path, self.identity = url.rstrip("/"), url.rstrip("/")
            self.collection, self.batch_size, self.models = collection, batch_size, models
            self.client = QdrantClient(
                url=url, api_key=api_key, timeout=math.ceil(timeout), check_compatibility=False
            )

    def replace(
        self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]], dimension: int
    ) -> None:
        raise RuntimeError("Full replacement is forbidden for application stores")

    def search(
        self, vector: Sequence[float], *, top_k: int, repository: str | None
    ) -> list[VectorMatch]:
        raise RuntimeError("Application retrieval requires a pinned READY membership")

    def ensure(self, dimension: int) -> None:
        if not self.exists():
            from qdrant_client.http.exceptions import UnexpectedResponse

            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=self.models.VectorParams(
                        size=dimension, distance=self.models.Distance.COSINE
                    ),
                )
            except UnexpectedResponse as error:
                if error.status_code not in {400, 409} or not self.exists():
                    raise
                # Another repository may initialize the shared collection concurrently.

        config = self.client.get_collection(self.collection).config.params.vectors
        if not isinstance(config, self.models.VectorParams) or config.size != dimension:
            raise ValueError("Application collection embedding dimension mismatch")

    def lookup(self, ids: Sequence[str]) -> dict[str, tuple[DocumentChunk, list[float]]]:
        if not ids or not self.exists():
            return {}
        result: dict[str, tuple[DocumentChunk, list[float]]] = {}
        for start in range(0, len(ids), self.batch_size):
            records = self.client.retrieve(
                self.collection,
                ids=list(ids[start : start + self.batch_size]),
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                if not isinstance(record.vector, list) or any(
                    not isinstance(value, (float, int)) for value in record.vector
                ):
                    raise ValueError("Unexpected vector format")
                result[str(record.id)] = (
                    DocumentChunk.model_validate(record.payload),
                    [float(value) for value in record.vector if isinstance(value, (int, float))],
                )
        return result

    def upsert(
        self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]], dimension: int
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk/vector cardinality mismatch")
        self.ensure(dimension)
        for start in range(0, len(chunks), self.batch_size):
            points = [
                self.models.PointStruct(
                    id=chunk.chunk_id, vector=list(vector), payload=chunk.model_dump(mode="json")
                )
                for chunk, vector in zip(
                    chunks[start : start + self.batch_size],
                    vectors[start : start + self.batch_size],
                    strict=True,
                )
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def delete(self, ids: Sequence[str]) -> None:
        if not self.exists():
            return
        for start in range(0, len(ids), self.batch_size):
            self.client.delete(
                self.collection,
                points_selector=self.models.PointIdsList(
                    points=list(ids[start : start + self.batch_size])
                ),
                wait=True,
            )

    def search_membership(
        self, vector: Sequence[float], ids: Sequence[str], top_k: int
    ) -> list[VectorMatch]:
        if not ids or top_k <= 0:
            return []
        response = self.client.query_points(
            self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=self.models.Filter(must=[self.models.HasIdCondition(has_id=list(ids))]),
            with_payload=True,
        )
        return [
            VectorMatch(DocumentChunk.model_validate(point.payload), float(point.score))
            for point in response.points
        ]
