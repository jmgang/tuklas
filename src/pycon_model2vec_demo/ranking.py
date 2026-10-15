"""Model-neutral judged-candidate ranking and nDCG."""

from dataclasses import dataclass
from math import isfinite, log2, sqrt
from typing import Protocol, Sequence

from .data import Candidate, QueryGroup


class TextEncoder(Protocol):
    """The only interface the ranking core needs from a model adapter."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    similarity: float


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embeddings must have matching nonzero dimensions")
    if not all(isfinite(value) for value in (*left, *right)):
        raise ValueError("Embeddings must contain finite values")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embeddings must have nonzero norm")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def rank_group(group: QueryGroup, encoder: TextEncoder) -> list[RankedCandidate]:
    """Rank only the passages judged for this query, never the full corpus."""
    query_vectors = encoder.encode([group.query])
    passage_vectors = encoder.encode([candidate.text for candidate in group.candidates])
    if len(query_vectors) != 1 or len(passage_vectors) != len(group.candidates):
        raise ValueError("Encoder returned an unexpected number of embeddings")
    ranked = [
        RankedCandidate(candidate, _cosine(query_vectors[0], vector))
        for candidate, vector in zip(group.candidates, passage_vectors)
    ]
    return sorted(ranked, key=lambda item: (-item.similarity, item.candidate.passage_id))


def ndcg_at_k(ranked: Sequence[RankedCandidate], k: int = 10) -> float | None:
    """Return None if no judged candidate has positive relevance."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranked:
        raise ValueError("ranked candidates cannot be empty")

    def dcg(relevances: Sequence[int]) -> float:
        return sum((2**rel - 1) / log2(index + 2) for index, rel in enumerate(relevances[:k]))

    ideal = dcg(sorted((item.candidate.relevance for item in ranked), reverse=True))
    if ideal == 0:
        return None
    return dcg([item.candidate.relevance for item in ranked]) / ideal
