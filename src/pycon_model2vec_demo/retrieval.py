"""Exact, corpus-wide vector retrieval with stable IDs and tie-breaking."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .corpus import Corpus
from .encoders import RetrievalEncoder


@dataclass(frozen=True)
class RankedPassage:
    corpus_id: str
    similarity: float


def _best_indices(scores: np.ndarray, ids: Sequence[str], k: int) -> list[int]:
    """Select exact top-k, including all boundary ties before lexical sorting."""
    if len(scores) <= k:
        candidates = range(len(scores))
    else:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        candidates = np.flatnonzero(scores >= threshold)
    return sorted(candidates, key=lambda index: (-float(scores[index]), ids[index]))[:k]


def exact_top_k(
    query_vectors: np.ndarray,
    passage_vectors: np.ndarray,
    corpus_ids: Sequence[str],
    *,
    k: int = 10,
    corpus_block_size: int = 4096,
) -> tuple[tuple[RankedPassage, ...], ...]:
    """Score every passage, bounded by a corpus block; never use ANN or qrels."""
    if k <= 0 or corpus_block_size <= 0:
        raise ValueError("k and corpus_block_size must be positive")
    queries = np.asarray(query_vectors, dtype=np.float32)
    passages = passage_vectors
    if queries.ndim != 2 or passages.ndim != 2 or queries.shape[1] != passages.shape[1]:
        raise ValueError("Query and passage vectors must be 2D with matching dimensions")
    if passages.shape[0] != len(corpus_ids) or not corpus_ids:
        raise ValueError("Every passage vector needs one stable corpus ID")
    if not np.isfinite(queries).all():
        raise ValueError("Query vectors must be finite")
    output: list[list[RankedPassage]] = [[] for _ in range(len(queries))]
    for start in range(0, len(corpus_ids), corpus_block_size):
        stop = min(start + corpus_block_size, len(corpus_ids))
        block = np.asarray(passages[start:stop], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Passage vectors in block {start}:{stop} must be finite")
        similarities = queries @ block.T
        block_ids = corpus_ids[start:stop]
        for query_index, scores in enumerate(similarities):
            selected = _best_indices(scores, block_ids, min(k, len(block_ids)))
            candidates = output[query_index] + [
                RankedPassage(block_ids[index], float(scores[index])) for index in selected
            ]
            output[query_index] = sorted(candidates, key=lambda row: (-row.similarity, row.corpus_id))[:k]
    return tuple(tuple(rows) for rows in output)


def encode_corpus(
    corpus: Corpus,
    encoder: RetrievalEncoder,
    target: Path,
    *,
    corpus_sha256: str,
    model_fingerprint: str,
    batch_size: int | None = None,
) -> Path:
    """Encode the same source corpus into one model-specific, provenance-tagged mmap.

    The cache is published only after every row has been written. It is never
    shared between models or silently overwritten.
    """
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError("Corpus embedding cache already exists")
    if not model_fingerprint or not corpus_sha256:
        raise ValueError("Corpus checksum and model fingerprint are required")
    size = batch_size or encoder.batch_size
    if size <= 0:
        raise ValueError("batch_size must be positive")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        raise FileExistsError("A previous partial corpus encoding exists")
    vectors = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(corpus.ids), encoder.dimension)
    )
    zero_rows = 0
    truncated_rows = 0
    try:
        for start in range(0, len(corpus.ids), size):
            stop = min(start + size, len(corpus.ids))
            batch = encoder.encode_passages(corpus.texts[start:stop])
            truncated_rows += encoder.count_truncated_passages(corpus.texts[start:stop])
            if batch.shape != (stop - start, encoder.dimension):
                raise ValueError("Encoder returned an unexpected corpus batch shape")
            zero_rows += int(np.count_nonzero(np.linalg.norm(batch, axis=1) == 0))
            vectors[start:stop] = batch
        vectors.flush()
        del vectors
        vectors = None
        # A hard link publishes the finished cache only if the target is absent.
        # Keep the partial name too: no file is deleted or overwritten.
        os.link(temporary, target)
        manifest = {
            "corpus_sha256": corpus_sha256,
            "corpus_passage_count": len(corpus.ids),
            "model_fingerprint": model_fingerprint,
            "dimension": encoder.dimension,
            "dtype": "float32",
            "zero_vector_passages": zero_rows,
            "truncated_passages": truncated_rows,
            "source_ids_first_last": [corpus.ids[0], corpus.ids[-1]],
        }
        with target.with_suffix(".json").open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
    except Exception:
        if vectors is not None:
            del vectors
        # Preserve the partial cache for inspection; never delete data on failure.
        raise
    return target


def open_corpus_embeddings(path: Path, *, corpus: Corpus, corpus_sha256: str, model_fingerprint: str) -> np.ndarray:
    """Reject a cache from a different corpus, model artifact, or dimension."""
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["corpus_sha256"] != corpus_sha256 or manifest["model_fingerprint"] != model_fingerprint:
        raise ValueError("Corpus embedding cache provenance mismatch")
    vectors = np.load(path, mmap_mode="r")
    if vectors.shape != (len(corpus.ids), manifest["dimension"]):
        raise ValueError("Corpus embedding cache shape mismatch")
    if manifest["source_ids_first_last"] != [corpus.ids[0], corpus.ids[-1]]:
        raise ValueError("Corpus embedding cache ID order mismatch")
    return vectors
