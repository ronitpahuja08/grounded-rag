"""Tests for validated, environment-backed application settings."""

import os

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from grounded_rag import AppSettings


class IsolatedSettings(AppSettings):
    """App settings variant that never reads a developer's local .env file."""

    model_config = SettingsConfigDict(env_file=None)


@pytest.fixture(autouse=True)
def clear_grounded_rag_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep settings tests independent from the machine running them."""
    for name in tuple(os.environ):
        if name.startswith("GROUNDED_RAG_"):
            monkeypatch.delenv(name)


def test_settings_use_safe_local_defaults() -> None:
    settings = IsolatedSettings()

    assert settings.embedding_provider == "local"
    assert settings.generation_provider == "local"
    assert settings.top_k == 5
    assert settings.max_source_bytes == 10 * 1024 * 1024
    assert settings.openai_api_key is None


def test_settings_load_prefixed_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDED_RAG_ENVIRONMENT", "test")
    monkeypatch.setenv("GROUNDED_RAG_TOP_K", "9")
    monkeypatch.setenv("GROUNDED_RAG_MINIMUM_RETRIEVAL_SCORE", "0.35")

    settings = IsolatedSettings()

    assert settings.environment == "test"
    assert settings.top_k == 9
    assert settings.minimum_retrieval_score == 0.35


def test_settings_mask_provider_secrets() -> None:
    settings = IsolatedSettings(
        embedding_provider="openai",
        openai_api_key=SecretStr("test-secret-value"),
    )

    assert "test-secret-value" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-secret-value"


def test_settings_require_key_for_openai_provider() -> None:
    with pytest.raises(ValidationError, match="openai_api_key"):
        IsolatedSettings(generation_provider="openai")


def test_settings_reject_overlap_that_consumes_entire_chunk() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        IsolatedSettings(chunk_size=256, chunk_overlap=256)


def test_settings_validate_configurable_source_size_limit() -> None:
    assert IsolatedSettings(max_source_bytes=4096).max_source_bytes == 4096

    with pytest.raises(ValidationError, match="max_source_bytes"):
        IsolatedSettings(max_source_bytes=512)
