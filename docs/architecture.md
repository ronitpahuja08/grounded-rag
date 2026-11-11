# Architecture Notes

## Design goals

1. Keep retrieval evidence visible from ingestion through the final answer.
2. Isolate external model providers behind narrow interfaces.
3. Make deterministic tests possible without API credentials.
4. Persist enough metadata to explain where every citation originated.

## Initial component boundaries

- **Loaders** turn source files into normalized documents.
- **Chunkers** produce stable chunks while preserving source metadata.
- **Embedders** map text to vectors through a provider-independent protocol.
- **Indexes** store vectors and return scored candidates.
- **Retrievers** apply filters, fusion, and reranking.
- **Generators** answer only from the supplied context.
- **Evaluators** measure retrieval and citation behavior against fixtures.

The first implementation will favor transparent local data structures over a
hosted vector database. A storage adapter can be added later without changing
the pipeline contracts.

## Day 1 decisions

### Validated, immutable boundary objects

Documents, chunks, retrieval hits, citations, and public answers are strict
Pydantic models. Unknown fields are rejected and instances are frozen after
validation. This makes failures happen where data enters the pipeline instead
of surfacing later during retrieval or response serialization.

Stable identifiers use lowercase letters, numbers, dots, colons, underscores,
and hyphens. Retrieval scores are finite values between zero and one. Citation
indexes must be contiguous from one, and citation spans must move forward
through the source. A grounded answer requires citations; an
`insufficient_evidence` answer cannot claim any.

### Reproducible content identity

SHA-256 is the content identity primitive. Text is normalized to NFC Unicode
and LF line endings before UTF-8 hashing, while all other whitespace remains
significant. Composite hashes use an eight-byte length prefix per value so
different value boundaries cannot produce the same byte stream.

### Environment-owned configuration

`AppSettings` reads only `GROUNDED_RAG_*` values and validates provider,
chunking, and retrieval combinations. Provider credentials use Pydantic's
`SecretStr`, which masks them in logs and object representations. Local
providers are the default, so tests and the initial development workflow do not
require network access or credentials.

### Structural provider interfaces

Document loading, embedding, and answer generation are `Protocol` interfaces.
The application depends on their behavior rather than a particular SDK. Small
deterministic fakes implement the same contracts in tests, while later provider
adapters can be selected from configuration without changing domain models.

## Day 1 dependency direction

```text
settings ───────────────────────────────┐
                                       v
loaders -> SourceDocument -> Chunk -> RetrievalHit -> AnswerGenerator
               |             |               |              |
               +------ stable hashing -------+              v
                                                   GroundedAnswer
                                                         |
                                                      Citation
```

Domain models and protocols do not import provider SDKs, storage engines, web
frameworks, or CLI code. This direction keeps the foundation reusable as those
adapters are added during later days.
