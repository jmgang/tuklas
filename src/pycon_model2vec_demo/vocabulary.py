"""Deterministic corpus-only vocabulary selection for the custom student."""

from __future__ import annotations

from collections import Counter
import gzip
from hashlib import sha256
import heapq
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Iterator


LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize_tokens(text: str) -> list[str]:
    """NFKC + casefold, then alphabetic tokens of at least two characters."""
    return [token for token in LETTERS.findall(unicodedata.normalize("NFKC", text).casefold()) if len(token) >= 2]


def load_stopwords(path: Path) -> tuple[frozenset[str], str]:
    raw = path.read_bytes()
    words = frozenset(line for line in raw.decode("utf-8").splitlines() if line and not line.startswith("#"))
    if not words or any(word != word.casefold() for word in words):
        raise ValueError("Versioned stopword list is empty or not casefolded")
    return words, sha256(raw).hexdigest()


def passage_terms(text: str, stopwords: frozenset[str], ngram_lengths: tuple[int, ...]) -> set[str]:
    tokens = normalize_tokens(text)
    terms: set[str] = set()
    for size in ngram_lengths:
        if size <= 0:
            raise ValueError("N-gram lengths must be positive")
        for offset in range(len(tokens) - size + 1):
            words = tokens[offset : offset + size]
            if any(word not in stopwords for word in words):
                terms.add(" ".join(words))
    return terms


def select_vocabulary(
    passages: Iterable[str], *, stopwords: frozenset[str], ngram_lengths: tuple[int, ...],
    min_passage_frequency: int, candidate_cap: int
) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """Count each term once per passage and sort by frequency, then lexical tie."""
    if min_passage_frequency <= 0 or candidate_cap <= 0:
        raise ValueError("Frequency threshold and candidate cap must be positive")
    frequencies: Counter[str] = Counter()
    passage_count = 0
    for text in passages:
        passage_count += 1
        frequencies.update(passage_terms(text, stopwords, ngram_lengths))
    eligible = ((term, count) for term, count in frequencies.items() if count >= min_passage_frequency)
    selected = heapq.nsmallest(candidate_cap, eligible, key=lambda item: (-item[1], item[0]))
    return selected, {"passage_count": passage_count, "distinct_term_count": len(frequencies)}


def iter_released_passages(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Invalid passage at corpus line {line_number}")
            yield row["text"]
