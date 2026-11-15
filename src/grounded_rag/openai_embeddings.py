"""OpenAI-compatible embeddings behind the provider protocol."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.request import Request, urlopen

from grounded_rag.protocols import EmbeddingVector
from grounded_rag.vectors import validate_vector

EmbeddingTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any]],
    Mapping[str, Any],
]


class OpenAICompatibleEmbeddingProvider:
    """Use an OpenAI-compatible embeddings endpoint with explicit dimensions."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str = "https://api.openai.com/v1",
        transport: EmbeddingTransport | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip():
            msg = "api_key and model must not be empty"
            raise ValueError(msg)
        if dimensions < 1:
            msg = "dimensions must be positive"
            raise ValueError(msg)
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._endpoint = f"{base_url.rstrip('/')}/embeddings"
        self._transport = transport or _post_json

    @property
    def dimensions(self) -> int:
        """Return the configured embedding width."""
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        """Embed documents in input order using the compatible API schema."""
        if not texts:
            return ()
        payload = self._transport(
            self._endpoint,
            {"Authorization": f"Bearer {self._api_key}"},
            {"model": self._model, "input": list(texts)},
        )
        raw_data = payload.get("data")
        if not isinstance(raw_data, list) or len(raw_data) != len(texts):
            msg = "embedding response must contain one vector per input"
            raise ValueError(msg)
        vectors: list[EmbeddingVector] = []
        for item in raw_data:
            if not isinstance(item, dict) or not isinstance(
                item.get("embedding"), list
            ):
                msg = "embedding response contains an invalid vector"
                raise ValueError(msg)
            vectors.append(
                validate_vector(item["embedding"], dimensions=self._dimensions)
            )
        return tuple(vectors)

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one query through the document batch contract."""
        return self.embed_documents((text,))[0]


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        msg = "embedding response must be a JSON object"
        raise ValueError(msg)
    return result
