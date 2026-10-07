from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    application_root: Path = Path(".cache/application-index")
    application_embedding_version: str | None = None
    application_qdrant_mode: str = "server"
    application_qdrant_url: str = "http://127.0.0.1:6333"
    application_qdrant_api_key: str | None = Field(default=None, repr=False)
    application_qdrant_timeout_seconds: float = Field(default=5, gt=0, le=30)
    github_webhooks_enabled: bool = False
    github_webhook_secret: str | None = Field(default=None, repr=False)
    github_webhook_max_body_bytes: int = Field(default=1_000_000, ge=1024, le=25_000_000)
    worker_concurrency: int = Field(default=1, ge=1, le=8)
    worker_lease_seconds: int = Field(default=90, ge=3, le=3600)
    worker_heartbeat_seconds: float = Field(default=15, gt=0, le=300)
    worker_poll_seconds: float = Field(default=2, ge=0.1, le=60)
    worker_max_attempts: int = Field(default=4, ge=1, le=10)
    worker_retry_base_seconds: float = Field(default=5, ge=0, le=300)
    worker_retry_max_seconds: float = Field(default=300, ge=0, le=3600)
    github_token: str | None = Field(default=None, repr=False)
    database_url: str | None = Field(default=None, repr=False)
    github_api_url: str = "https://api.github.com"
    github_cache_dir: Path = Path(".cache/github")
    github_cache_ttl_seconds: int = Field(default=3600, ge=0)
    github_request_timeout_seconds: float = Field(default=30, gt=0)
    github_max_retries: int = Field(default=4, ge=1, le=10)
    github_max_retry_wait_seconds: float = Field(default=300, ge=0, le=3600)
    github_per_page: int = Field(default=100, ge=1, le=100)
    github_max_pages: int = Field(default=0, ge=0)
    github_doc_max_bytes: int = Field(default=1_000_000, ge=1)
    index_dir: Path = Path(".cache/index")
    qdrant_path: Path = Path(".cache/qdrant")
    qdrant_collection: str = "github_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    chunk_max_words: int = Field(default=220, ge=20)
    chunk_overlap_words: int = Field(default=30, ge=0)
    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    retrieval_candidates_per_backend: int = Field(default=25, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1)
    rerank_candidate_pool: int = Field(default=40, ge=1, le=500)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_api_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.6-sol"
    generation_top_k: int = Field(default=8, ge=1, le=50)
    generation_max_context_chars: int = Field(default=30_000, ge=1_000)
    generation_max_output_tokens: int = Field(default=1_200, ge=100)
    generation_citation_repair_attempts: int = Field(default=1, ge=0, le=3)
    investigation_max_steps: int = Field(default=6, ge=1, le=25)
    investigation_max_mcp_calls: int = Field(default=4, ge=0, le=20)
    investigation_max_sources: int = Field(default=12, ge=1, le=50)
    mcp_search_limit: int = Field(default=5, ge=1, le=20)
    log_level: str = "INFO"

    @property
    def validated_chunk_overlap_words(self) -> int:
        if self.chunk_overlap_words >= self.chunk_max_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_MAX_WORDS")
        return self.chunk_overlap_words
