"""Environment-backed application configuration with safe secret handling."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
ProviderName = Literal["local", "openai"]
ModelName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
ChunkSize = Annotated[int, Field(ge=128, le=8192)]
ChunkOverlap = Annotated[int, Field(ge=0, le=2048)]
TopK = Annotated[int, Field(ge=1, le=100)]
SourceByteLimit = Annotated[int, Field(ge=1024, le=1_073_741_824)]
RetrievalThreshold = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]


class AppSettings(BaseSettings):
    """Validated settings loaded from ``GROUNDED_RAG_*`` environment values."""

    model_config = SettingsConfigDict(
        env_prefix="GROUNDED_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Environment = "development"
    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes")
    embedding_provider: ProviderName = "local"
    generation_provider: ProviderName = "local"
    embedding_model: ModelName = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model: ModelName = "local-deterministic"
    openai_api_key: SecretStr | None = None
    chunk_size: ChunkSize = 1000
    chunk_overlap: ChunkOverlap = 150
    top_k: TopK = 5
    minimum_retrieval_score: RetrievalThreshold = 0.2
    max_source_bytes: SourceByteLimit = 10 * 1024 * 1024

    @model_validator(mode="after")
    def validate_pipeline_configuration(self) -> Self:
        """Reject combinations that cannot produce a safe working pipeline."""
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(msg)

        uses_openai = (
            self.embedding_provider == "openai"
            or self.generation_provider == "openai"
        )
        if uses_openai and self.openai_api_key is None:
            msg = "openai_api_key is required when an OpenAI provider is selected"
            raise ValueError(msg)
        return self
