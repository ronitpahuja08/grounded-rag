# Eight-Day Build Plan

Grounded RAG is planned as an eight-day, production-minded build rather than a
single tutorial notebook. Each day ends with a usable vertical improvement,
automated verification, and ten focused commits. The complete plan contains 80
functional commits; commit boundaries can change if implementation evidence
shows that a different split is healthier.

## Product scope

The finished repository will include:

- A typed Python package with explicit domain and provider contracts.
- Text, Markdown, and PDF ingestion with stable source metadata.
- Deterministic, section-aware chunking and corpus manifests.
- Pluggable embedding and generation providers with offline test doubles.
- A persistent local vector index with semantic and lexical retrieval.
- Hybrid search, metadata filtering, diversity selection, and reranking hooks.
- Citation-backed generation with evidence validation and safe abstention.
- A versioned evaluation dataset and retrieval/groundedness reports.
- A CLI, FastAPI service, and small browser demo.
- Structured logs, health endpoints, Docker support, CI, and complete docs.

## Day 1 — Foundation, contracts, and configuration

Establish the architecture and make the core data contracts executable before
adding model providers.

Planned commits:

1. `docs: define grounded RAG product scope`
2. `chore: configure Python package and developer tooling`
3. `test: establish package verification baseline`
4. `feat: define source document model`
5. `feat: define chunk and retrieval hit models`
6. `feat: define citation and grounded answer models`
7. `feat: add validated application settings`
8. `feat: add stable content hashing utilities`
9. `feat: define loader embedding and generator protocols`
10. `docs: record architecture decisions and local setup`

Acceptance criteria:

- Domain models reject invalid identifiers, scores, and citations.
- Configuration loads from environment variables without exposing secrets.
- Provider contracts can be implemented by deterministic test doubles.
- Tests, linting, and strict type checking pass from a clean checkout.

## Day 2 — Document loading and normalization

Create a traceable ingestion layer that turns files into normalized documents
without losing their origin.

Planned commits:

1. `feat: add loader registry and file dispatch`
2. `feat: implement UTF-8 text loader`
3. `feat: implement Markdown loader with heading metadata`
4. `feat: implement PDF loader with page metadata`
5. `feat: normalize whitespace and line endings`
6. `feat: attach content hashes and source timestamps`
7. `feat: reject unsupported and oversized sources`
8. `feat: discover documents from directories`
9. `test: cover loading errors and metadata preservation`
10. `feat: add document inspection CLI command`

Acceptance criteria:

- Text, Markdown, and PDF fixtures produce normalized documents.
- Every document records its source, hash, media type, and location metadata.
- Binary, unsupported, and oversized files fail with actionable errors.
- Directory discovery is deterministic and ignores configured paths.

## Day 3 — Chunking and corpus lifecycle

Turn normalized documents into stable retrieval units and track incremental
corpus changes.

Planned commits:

1. `feat: add chunking configuration model`
2. `feat: implement token-aware overlapping splitter`
3. `feat: preserve Markdown section boundaries`
4. `feat: preserve PDF page provenance in chunks`
5. `feat: generate deterministic chunk identifiers`
6. `feat: suppress duplicate chunk content`
7. `feat: add corpus manifest persistence`
8. `feat: detect added changed and removed sources`
9. `feat: orchestrate incremental ingestion pipeline`
10. `test: cover chunk boundaries and manifest updates`

Acceptance criteria:

- Re-ingesting unchanged content produces identical chunk IDs.
- Overlap and maximum-size rules hold for boundary-heavy fixtures.
- Changed and deleted sources are reflected in the corpus manifest.
- Every chunk can be traced back to a file, section, and page where available.

## Day 4 — Embeddings and persistent vector index

Build the semantic storage layer behind provider-independent interfaces.

Planned commits:

1. `feat: add deterministic local embedding test double`
2. `feat: add OpenAI-compatible embedding provider`
3. `feat: batch embedding requests with bounded retries`
4. `feat: validate vector dimensions and finite values`
5. `feat: implement cosine similarity utilities`
6. `feat: implement in-memory vector index`
7. `feat: persist vector index and metadata atomically`
8. `feat: reload and validate saved indexes`
9. `feat: add incremental index update service`
10. `test: cover embedding failures and index persistence`

Acceptance criteria:

- All provider calls are replaceable with an offline deterministic embedder.
- Index save/load round trips preserve vectors and source metadata.
- Dimension mismatches, corrupt indexes, and non-finite vectors fail safely.
- Incremental updates avoid re-embedding unchanged chunks.

## Day 5 — Hybrid retrieval and reranking

Produce relevant, diverse evidence with inspectable scoring decisions.

Planned commits:

1. `feat: define retrieval query and result contracts`
2. `feat: implement semantic top-k retrieval`
3. `feat: implement BM25 lexical retrieval`
4. `feat: fuse lexical and semantic ranks`
5. `feat: add source and metadata filters`
6. `feat: add minimum relevance thresholds`
7. `feat: add maximal marginal relevance selection`
8. `feat: define optional reranker protocol`
9. `feat: emit structured retrieval traces`
10. `test: evaluate hybrid retrieval edge cases`

Acceptance criteria:

- Semantic, lexical, and hybrid modes are selectable through configuration.
- Filters and thresholds never leak excluded chunks.
- Diversity selection reduces near-duplicate context.
- Retrieval traces explain candidate scores, filtering, fusion, and final rank.

## Day 6 — Grounded generation and citations

Generate useful answers while making unsupported claims visible and rejectable.

Planned commits:

1. `feat: add OpenAI-compatible chat provider`
2. `feat: add deterministic generator test double`
3. `feat: assemble context within token budgets`
4. `feat: assign compact evidence citation labels`
5. `feat: build grounded answer prompt template`
6. `feat: parse structured generated answers`
7. `feat: validate cited evidence references`
8. `feat: reject uncited or unknown citations`
9. `feat: add insufficient-evidence abstention`
10. `test: cover grounded answer and citation policies`

Acceptance criteria:

- Every factual answer sentence is associated with known evidence labels.
- Invalid or invented citations never reach the public response model.
- Low-confidence retrieval produces a clear abstention instead of fabrication.
- Context assembly respects limits and retains original source metadata.

## Day 7 — Evaluation, CLI, and HTTP API

Measure the pipeline and expose its main workflows through stable interfaces.

Planned commits:

1. `feat: define versioned evaluation dataset schema`
2. `feat: add representative evaluation fixtures`
3. `feat: compute hit rate MRR and nDCG metrics`
4. `feat: compute citation coverage and abstention metrics`
5. `feat: generate JSON and Markdown evaluation reports`
6. `feat: compare retrieval configurations`
7. `feat: add ingest query and evaluate CLI commands`
8. `feat: add FastAPI application and request schemas`
9. `feat: expose corpus query and source endpoints`
10. `test: cover CLI and API workflows end to end`

Acceptance criteria:

- A checked-in dataset produces reproducible baseline metrics.
- Evaluation reports identify regressions and configuration differences.
- CLI and API share the same application services and response contracts.
- End-to-end tests run without external model credentials.

## Day 8 — Browser demo, operations, and release

Turn the pipeline into a polished showcase that another developer can run and
understand quickly.

Planned commits:

1. `feat: add browser query and evidence interface`
2. `feat: add source preview and citation navigation`
3. `feat: expose health readiness and corpus statistics`
4. `feat: add request IDs and structured logging`
5. `feat: add safe API error mapping and limits`
6. `chore: add Docker image and compose setup`
7. `ci: verify lint types tests and container build`
8. `docs: add architecture diagrams and configuration reference`
9. `docs: add reproducible demo and evaluation results`
10. `chore: prepare v0.1.0 release`

Acceptance criteria:

- The demo shows answers, retrieved passages, scores, and source locations.
- A new contributor can run the full stack from documented commands.
- CI verifies formatting, types, tests, and the container build.
- The release includes a clean repository, passing evaluation baseline, and no
  committed credentials or private documents.

## Final definition of done

- 80 focused commits across eight build days.
- Complete offline test path plus optional real-provider smoke tests.
- Public repository with setup, architecture, API, evaluation, and demo docs.
- Traceable citations from every answer back to stored source metadata.
- Reproducible evaluation results and a tagged `v0.1.0` release.
