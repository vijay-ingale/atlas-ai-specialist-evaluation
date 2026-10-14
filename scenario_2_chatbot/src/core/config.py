from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")
    document_dir: Path = ROOT / "data/documents"
    index_dir: Path = ROOT / "index"
    openai_api_key: str = "replace_with_your_api_key"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    context_top_k: int = Field(default=4, ge=1, le=20)
    min_rerank_score: float = 0.0
    chunk_tokens: int = Field(default=180, ge=32, le=1000)
    overlap_tokens: int = Field(default=30, ge=0)
    max_document_bytes: int = Field(default=1_000_000, ge=1)
    max_context_characters: int = Field(default=12000, ge=1000, le=50000)
    provider_timeout_seconds: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_limits(self):
        if self.context_top_k > self.retrieval_top_k:
            raise ValueError("CONTEXT_TOP_K must not exceed RETRIEVAL_TOP_K")
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("OVERLAP_TOKENS must be smaller than CHUNK_TOKENS")
        return self
