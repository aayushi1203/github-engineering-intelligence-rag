"""Shared construction for API/worker; legacy offline retrieval remains independent."""

from collections.abc import Callable, Sequence
from uuid import UUID

import httpx
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from github_intelligence.config import Settings
from github_intelligence.generation import GroundedAnswerService, OpenAIResponsesProvider
from github_intelligence.generation.provider import LLMProvider
from github_intelligence.github import GitHubClient
from github_intelligence.indexing.coordinator import IncrementalCoordinator, ReadyIndex
from github_intelligence.indexing.storage import IncrementalVectorStore
from github_intelligence.models import DocumentChunk, RetrievalResult
from github_intelligence.persistence.database import create_database_engine
from github_intelligence.retrieval import CrossEncoderReranker, DocumentChunker, RetrievalPipeline
from github_intelligence.retrieval.reranker import Reranker
from github_intelligence.retrieval.vector import Encoder, SentenceTransformerEncoder


class ReadyVectorRetriever:
    def __init__(self, ready: ReadyIndex, encoder: Encoder):
        self.ready, self.encoder = ready, encoder

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        raise RuntimeError("READY readers cannot mutate indexes")

    def search(
        self, query: str, *, top_k: int = 10, repository: str | None = None
    ) -> list[RetrievalResult]:
        matches = self.ready.vector_search(self.encoder.encode([query])[0], top_k=top_k)
        return [
            RetrievalResult(
                chunk_id=m.chunk.chunk_id,
                text=m.chunk.text,
                title=m.chunk.title,
                source_url=m.chunk.source_url,
                source_type=m.chunk.source_type,
                score=m.score,
                repository=m.chunk.repository,
                metadata=m.chunk.metadata,
            )
            for m in matches
        ]


class Runtime:
    def __init__(
        self,
        settings: Settings,
        *,
        engine: Engine | None = None,
        store: IncrementalVectorStore | None = None,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        provider: LLMProvider | None = None,
        github_transport: httpx.AsyncBaseTransport | None = None,
        allow_embedded_single_process: bool = False,
    ):
        if settings.github_webhooks_enabled and (
            not settings.github_webhook_secret or len(settings.github_webhook_secret.encode()) < 32
        ):
            raise ValueError("Enabled GitHub webhooks require a secret of at least 32 bytes")
        if not settings.application_embedding_version:
            raise ValueError("APPLICATION_EMBEDDING_VERSION is required")
        if settings.worker_heartbeat_seconds >= settings.worker_lease_seconds / 2:
            raise ValueError("Heartbeat interval must be less than half the lease duration")
        if settings.application_qdrant_mode not in {"server", "embedded"}:
            raise ValueError("Explicit application Qdrant mode is invalid")
        if settings.application_qdrant_mode == "embedded" and not allow_embedded_single_process:
            raise ValueError("API/worker separate processes require server-mode Qdrant")
        DocumentChunker(settings.chunk_max_words, settings.validated_chunk_overlap_words)
        if settings.application_root.resolve() in {
            settings.index_dir.resolve(),
            settings.qdrant_path.resolve(),
        }:
            raise ValueError("Application storage must be isolated from snapshot stores")
        self.settings = settings
        self.engine = engine or create_database_engine(settings.database_url)
        self.store = store or IncrementalVectorStore(
            settings.application_root / "qdrant"
            if settings.application_qdrant_mode == "embedded"
            else None,
            "application_chunks",
            url=settings.application_qdrant_url
            if settings.application_qdrant_mode == "server"
            else None,
            api_key=settings.application_qdrant_api_key,
            timeout=settings.application_qdrant_timeout_seconds,
        )
        self.encoder = encoder or SentenceTransformerEncoder(
            settings.embedding_model, settings.embedding_device
        )
        self.reranker = reranker or CrossEncoderReranker(
            settings.reranker_model, settings.reranker_device
        )
        self.provider = provider
        self.github_transport = github_transport

    def coordinator(
        self,
        *,
        boundary: Callable[[str], None] | None = None,
        guard: Callable[[Session, UUID], None] | None = None,
    ) -> IncrementalCoordinator:
        return IncrementalCoordinator(
            self.engine,
            self.store,
            self.encoder,
            self.settings.application_root,
            embedding_version=f"{self.settings.embedding_model}@{self.settings.application_embedding_version}",
            chunker=DocumentChunker(
                self.settings.chunk_max_words, self.settings.validated_chunk_overlap_words
            ),
            boundary=boundary,
            publication_guard=guard,
        )

    def github_client(self) -> GitHubClient:
        return GitHubClient(self.settings, transport=self.github_transport)

    def configuration(self) -> dict[str, object]:
        return dict(self.coordinator().configuration)

    def answers(self, ready: ReadyIndex) -> GroundedAnswerService:
        settings = self.settings
        provider = self.provider or OpenAIResponsesProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            api_url=settings.openai_api_url,
            max_output_tokens=settings.generation_max_output_tokens,
        )
        pipeline = RetrievalPipeline(
            bm25=ready.lexical,
            vector=ReadyVectorRetriever(ready, self.encoder),
            reranker=self.reranker,
            candidates_per_backend=settings.retrieval_candidates_per_backend,
            rerank_candidate_pool=settings.rerank_candidate_pool,
            rrf_k=settings.rrf_k,
        )
        return GroundedAnswerService(
            pipeline,
            provider,
            top_k=settings.generation_top_k,
            max_context_chars=settings.generation_max_context_chars,
            repair_attempts=settings.generation_citation_repair_attempts,
        )

    def close(self) -> None:
        self.store.client.close()
        self.engine.dispose()
