from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    github_token: str | None = Field(default=None, repr=False)
    github_api_url: str = "https://api.github.com"
    github_cache_dir: Path = Path(".cache/github")
    github_cache_ttl_seconds: int = Field(default=3600, ge=0)
    github_request_timeout_seconds: float = Field(default=30, gt=0)
    github_max_retries: int = Field(default=4, ge=1, le=10)
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
    log_level: str = "INFO"

    @property
    def validated_chunk_overlap_words(self) -> int:
        if self.chunk_overlap_words >= self.chunk_max_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_MAX_WORDS")
        return self.chunk_overlap_words
