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

Python 3.11 or newer will be required. Day 1 establishes the install, test,
lint, and type-check commands before feature implementation begins. Provider
credentials will be read from environment variables; no keys or private
documents belong in Git.

## Status

Planning is complete for an eight-day, 80-commit build. Implementation begins
only after the plan is reviewed; the full scope and daily acceptance criteria
are documented in [ROADMAP.md](ROADMAP.md).
