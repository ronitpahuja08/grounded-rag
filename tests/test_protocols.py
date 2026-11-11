"""Offline contract tests for provider-independent pipeline boundaries."""

from collections.abc import Sequence
from pathlib import Path

from grounded_rag import (
    AnswerGenerator,
    DocumentLoader,
    EmbeddingProvider,
    EmbeddingVector,
    GroundedAnswer,
    RetrievalHit,
    SourceDocument,
)


class FakeLoader:
    """Deterministic loader used to prove structural protocol compatibility."""

    def supports(self, source: Path) -> bool:
        return source.suffix == ".txt"

    def load(self, source: Path) -> Sequence[SourceDocument]:
        return (
            SourceDocument(
                id="fixture",
                content="Synthetic fixture content",
                source=source.as_posix(),
                media_type="text/plain",
            ),
        )


class FakeEmbeddingProvider:
    """Small deterministic embedder that never calls an external service."""

    @property
    def dimensions(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        return tuple((float(len(text)), 1.0) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return (float(len(text)), 0.0)


class FakeAnswerGenerator:
    """Deterministic generator that explicitly abstains without evidence."""

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievalHit],
    ) -> GroundedAnswer:
        del question
        if not evidence:
            return GroundedAnswer(
                answer="The available sources do not contain enough evidence.",
                status="insufficient_evidence",
            )
        raise NotImplementedError


def test_loader_protocol_accepts_structural_fake() -> None:
    loader: DocumentLoader = FakeLoader()

    assert isinstance(loader, DocumentLoader)
    assert loader.supports(Path("fixture.txt"))
    assert loader.load(Path("fixture.txt"))[0].id == "fixture"


def test_embedding_protocol_accepts_structural_fake() -> None:
    provider: EmbeddingProvider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimensions == 2
    assert provider.embed_documents(("a", "abcd")) == ((1.0, 1.0), (4.0, 1.0))
    assert provider.embed_query("abc") == (3.0, 0.0)


def test_generator_protocol_accepts_structural_fake() -> None:
    generator: AnswerGenerator = FakeAnswerGenerator()

    assert isinstance(generator, AnswerGenerator)
    answer = generator.generate("What is grounded RAG?", ())
    assert answer.status == "insufficient_evidence"
