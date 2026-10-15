"""Lossless text slices and explicit token-weighted normalized segment pooling.

The text is never decoded from token IDs: offsets index the original string.
Every slice is counted again with its role prefix and special tokens before use.
This is a changed retrieval policy, not native long-context inference.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class SegmentPolicy:
    count: Callable[[str], int]
    content_count: Callable[[str], int]
    token_limit: int
    character_limit: int | None = None
    prefix: str = ""

    def fits(self, text: str) -> bool:
        return (self.character_limit is None or len(self.prefix + text) <= self.character_limit) and self.count(self.prefix + text) <= self.token_limit

    def plan(self, text: str) -> list[list[int]]:
        if not text:
            raise ValueError("Empty source text requires explicit handling")
        result, start = [], 0
        while start < len(text):
            rest = text[start:]
            if self.fits(rest):
                end = len(text)
            else:
                lo, hi = 0, len(rest)
                if self.character_limit is not None:
                    hi = min(hi, self.character_limit - len(self.prefix))
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self.fits(rest[:mid]):
                        lo = mid
                    else:
                        hi = mid - 1
                if lo == 0:
                    raise ValueError("Even one source character cannot fit the frozen policy")
                # Prefer a word boundary without dropping delimiters. Recount
                # after boundary selection; tokenizer counts need not be monotone.
                cut = lo
                for offset in range(lo - 1, max(0, lo // 2) - 1, -1):
                    if rest[offset].isspace() and self.fits(rest[:offset + 1]):
                        cut = offset + 1
                        break
                end = start + cut
            piece = text[start:end]
            if not self.fits(piece):
                raise AssertionError("Segment exceeds frozen limit")
            weight = self.content_count(piece) if piece.strip() else 0
            result.append([start, end, max(1, weight) if piece.strip() else 0])
            start = end
        if ''.join(text[a:b] for a, b, _ in result) != text:
            raise AssertionError("Source content was lost or duplicated")
        return result


def pool(vectors: np.ndarray, weights: list[int]) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(weights) or not np.isfinite(values).all():
        raise ValueError("Invalid segment vectors")
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or not w.sum():
        raise ValueError("Pooling needs positive total token weight")
    mean = np.average(values.astype(np.float64), axis=0, weights=w)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm else mean).astype(np.float32)
