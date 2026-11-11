# Grounded RAG

Grounded RAG is a from-scratch retrieval-augmented generation project for
answering questions over a local document collection with inspectable evidence.
Every answer will include the retrieved passages that supported it, making the
pipeline easier to evaluate, debug, and trust.

The project begins with the core ideas from freeCodeCamp's
[Learn RAG From Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) course
and the LangChain
[RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) reference
material. The implementation and product scope here are original and will grow
beyond the tutorial into a tested application.

## Planned capabilities

- Ingest Markdown, text, and PDF documents with stable source identifiers.
- Split documents using configurable, metadata-preserving chunking.
- Build a local vector index behind a provider-independent embedding interface.
- Retrieve, filter, and rerank relevant context for a question.
- Generate answers with passage-level citations and an explicit fallback when
  the evidence is insufficient.
- Evaluate retrieval quality and answer groundedness against a small benchmark.
- Expose the pipeline through a CLI and a local FastAPI service.

## Architecture

```text
documents -> parsing -> chunks -> embeddings -> vector index
                                                 |
question -> query embedding -> retrieval -> reranking -> grounded generation
                                      |                 |
                                      +---- evidence ---+
```

The retrieval and generation boundaries will use small Python protocols so the
core pipeline can be tested with deterministic fakes instead of network calls.

The complete user workflows, requirements, success measures, and version
boundaries are defined in [docs/product-scope.md](docs/product-scope.md).

## Repository layout

```text
docs/               Architecture and design decisions
src/grounded_rag/   Application package
tests/              Automated tests
ROADMAP.md          Eight-day build plan and acceptance criteria
```

## Development

Python 3.11 or newer is required. Create an isolated environment and install the
package with its development tools:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Activate the environment using `.venv/Scripts/Activate.ps1` on PowerShell or
`source .venv/bin/activate` on macOS and Linux before running project commands.

Copy `.env.example` to `.env` when overriding defaults. The local providers do
not need credentials. `GROUNDED_RAG_OPENAI_API_KEY` is required only when the
embedding or generation provider is explicitly set to `openai`; `.env` is
ignored by Git.

Run the complete Day 1 verification suite with:

```bash
pytest
ruff check .
mypy
python -m build
```

These checks are offline and deterministic. They cover package installation,
domain validation, environment configuration, stable hashing, and structural
provider contracts. Generated distributions are written to `dist/` and are
ignored by Git.

## Foundation API

Day 1 exposes the contracts later pipeline stages build on:

- `SourceDocument`, `Chunk`, and `RetrievalHit` preserve traceable evidence.
- `Citation` and `GroundedAnswer` prevent unsupported public responses.
- `AppSettings` validates environment-driven runtime configuration.
- `hash_text`, `hash_bytes`, and `hash_parts` create deterministic identities.
- `DocumentLoader`, `EmbeddingProvider`, and `AnswerGenerator` isolate external
  implementations behind typed interfaces.

The reasoning behind these choices and the dependency direction are recorded
in [docs/architecture.md](docs/architecture.md).

## Status

Day 1 establishes the tested foundation, data contracts, configuration, and
provider boundaries. The remaining ingestion, retrieval, generation,
evaluation, API, and demo milestones are documented in
[ROADMAP.md](ROADMAP.md).
