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
