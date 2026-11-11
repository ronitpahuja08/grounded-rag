# Product Scope

## Problem

Retrieval-augmented generation demos often hide the decisions that determine
whether an answer is trustworthy: how documents were parsed, why a passage was
retrieved, which claims are supported, and when the system should refuse to
answer. Grounded RAG will make those decisions inspectable from ingestion to
the final response.

## Intended users

- Developers learning how a complete RAG pipeline works beyond a notebook.
- Teams evaluating retrieval strategies before choosing hosted infrastructure.
- Maintainers who need traceable answers over local technical documentation.
- Reviewers who want reproducible evidence for retrieval and citation claims.

## Primary workflows

### Build a corpus

1. Select a directory containing text, Markdown, or PDF sources.
2. Inspect which files are accepted, skipped, changed, or removed.
3. Normalize and chunk supported documents while preserving source metadata.
4. Embed only new or changed chunks and persist the resulting index.

### Ask a grounded question

1. Convert the question into semantic and lexical retrieval queries.
2. Retrieve, filter, diversify, and optionally rerank candidate passages.
3. Generate an answer using only the bounded evidence context.
4. Validate every returned citation against the supplied passages.
5. Abstain when the evidence does not meet configured relevance requirements.

### Evaluate a configuration

1. Run a versioned question set against a selected pipeline configuration.
2. Measure retrieval quality, citation coverage, and abstention behavior.
3. Produce machine-readable and human-readable reports.
4. Compare a candidate configuration with the checked-in baseline.

## Functional requirements

- Every source, document, and chunk must have a stable identifier.
- Every retrieval hit must expose its score, rank, source, and location.
- Provider APIs must sit behind interfaces with deterministic offline fakes.
- The local index must survive process restarts and detect incompatible data.
- Answers must contain validated citations or an explicit abstention.
- CLI, HTTP API, and browser demo must share the same application services.
- Evaluation must run without paid provider credentials.

## Quality requirements

- **Traceability:** an answer citation resolves to the original source location.
- **Determinism:** offline tests and evaluation fixtures produce repeatable output.
- **Safety:** logs and errors do not expose credentials or private document text.
- **Observability:** retrieval decisions and request boundaries are structured.
- **Portability:** the default stack runs locally on Python 3.11 or newer.
- **Maintainability:** strict typing and narrow interfaces isolate providers.

## Success measures

The initial release is successful when:

- the complete offline test suite passes from a clean checkout;
- the evaluation runner reports hit rate, MRR, nDCG, citation coverage, and
  abstention accuracy;
- a user can ingest the demo corpus and query it through both CLI and HTTP;
- the browser demo shows the answer beside ranked evidence and source links;
- the same corpus can be re-ingested without recomputing unchanged chunks;
- documentation reproduces the published demo and evaluation baseline.

## Non-goals for version 0.1.0

- Multi-tenant authentication or hosted account management.
- Distributed vector databases or horizontal scaling.
- Training embedding, reranking, or generation models.
- Autonomous web crawling or ingestion of authenticated websites.
- Claiming that automated groundedness checks replace human review.
- Storing provider credentials or private corpora in the repository.

## Data and security boundaries

Local source content and generated indexes are ignored by Git by default.
Provider credentials are loaded from environment variables and are never part
of serialized configuration. Error responses may identify a source path but
must not dump document contents. Tests use synthetic fixtures that are safe to
publish.

## Release boundary

Version 0.1.0 includes the eight-day scope in the project roadmap: typed
contracts, ingestion, chunking, embeddings, persistent indexing, hybrid
retrieval, grounded generation, evaluation, CLI, API, browser demo, operations,
and documentation. Features outside the stated non-goals require a later
release decision.
