"""Role-aware encoding seam shared by contextual and static retrieval models."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class RetrievalEncoder(Protocol):
    """Return finite, L2-normalized float32 rows; empty input returns shape (0, dimension).

    Callers supply unprefixed text. An all-zero row is possible for an entirely
    out-of-vocabulary static input and must be counted during evaluation.
    """

    dimension: int
    max_length: int
    batch_size: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def count_truncated_passages(self, texts: Sequence[str]) -> int: ...


def _validate_vectors(vectors: object, count: int, dimension: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (count, dimension):
        raise ValueError(f"Expected {(count, dimension)} embeddings, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(array, axis=1)
    if not np.all((norms == 0) | np.isclose(norms, 1.0, atol=0.02)):
        raise ValueError("Nonzero embeddings must be L2-normalized")
    return array


def _validate_settings(dimension: int, max_length: int, batch_size: int) -> None:
    if dimension <= 0 or max_length <= 0 or batch_size <= 0:
        raise ValueError("dimension, max_length, and batch_size must be positive")


class E5Encoder:
    """Sentence Transformers adapter for asymmetric E5 passage retrieval.

    The injected model must have its max_seq_length set to ``max_length``.
    Loading a pinned local snapshot is handled separately from this seam.
    """

    def __init__(self, model: object, *, dimension: int = 768, max_length: int = 512, batch_size: int = 32):
        _validate_settings(dimension, max_length, batch_size)
        self.model = model
        self.dimension = dimension
        self.max_length = max_length
        self.batch_size = batch_size

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _validate_vectors(vectors, len(texts), self.dimension)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query: ")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def count_truncated_passages(self, texts: Sequence[str]) -> int:
        if len(texts) == 0:
            return 0
        lengths = self.model.tokenizer(
            ["passage: " + text for text in texts],
            add_special_tokens=True, truncation=False, return_length=True, verbose=False,
        )["length"]
        return sum(length > self.max_length for length in lengths)


class StaticEncoder:
    """Model2Vec StaticModel adapter; prefixes are explicit per model family.

    E5-derived students use E5's query/passage prefixes. Potion Retrieval uses
    empty prefixes. The adapter does not silently infer behavior from a name.
    """

    def __init__(
        self,
        model: object,
        *,
        dimension: int,
        max_length: int = 512,
        batch_size: int = 1024,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        _validate_settings(dimension, max_length, batch_size)
        self.model = model
        self.dimension = dimension
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_multiprocessing=False,
        )
        return _validate_vectors(vectors, len(texts), self.dimension)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def count_truncated_passages(self, texts: Sequence[str]) -> int:
        if len(texts) == 0:
            return 0
        prefixed = [self.passage_prefix + text for text in texts]
        lengths = [len(ids) for ids in self.model.tokenize(prefixed, max_length=None)]
        char_limit = self.max_length * self.model.median_token_length
        return sum(length > self.max_length or len(text) > char_limit for text, length in zip(prefixed, lengths))


def load_e5_encoder(path: str, *, max_length: int = 512, batch_size: int = 32) -> E5Encoder:
    """Load a pinned local teacher snapshot, never a floating Hub branch."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(path, device="cpu", local_files_only=True, trust_remote_code=False)
    model.max_seq_length = max_length
    dimension = model.get_embedding_dimension()
    if dimension is None:
        raise ValueError("Teacher did not report an embedding dimension")
    return E5Encoder(model, dimension=dimension, max_length=max_length, batch_size=batch_size)


def load_static_encoder(
    path: str,
    *,
    max_length: int = 512,
    batch_size: int = 1024,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> StaticEncoder:
    """Load a saved Model2Vec artifact; callers supply its frozen role prefixes."""
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(path, force_download=False)
    if not model.normalize:
        raise ValueError("Pinned static model must encode normalized vectors")
    return StaticEncoder(
        model,
        dimension=model.dim,
        max_length=max_length,
        batch_size=batch_size,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
