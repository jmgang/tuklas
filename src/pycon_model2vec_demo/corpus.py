"""Validated DisastIR corpus and audited judgment-to-corpus-ID mapping."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Mapping, Sequence

from .data import QueryGroup, _file_sha256


def _space_normalize(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Corpus:
    ids: tuple[str, ...]
    texts: tuple[str, ...]
    exact_index: Mapping[str, str]
    whitespace_index: Mapping[str, str]
    duplicate_text_count: int

    def resolve(self, text: str) -> tuple[str | None, str]:
        if text in self.exact_index:
            return self.exact_index[text], "exact"
        normalized = _space_normalize(text)
        if normalized in self.whitespace_index:
            return self.whitespace_index[normalized], "whitespace"
        return None, "unmatched"


@dataclass(frozen=True)
class MappedQuery:
    query_id: str
    query: str
    category: str | None
    judgments: Mapping[str, int]


def load_corpus(path: Path, *, expected_sha256: str | None = None) -> Corpus:
    """Read released passage IDs in file order; first duplicate text wins."""
    actual = _file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Corpus SHA-256 differs from the pinned release")
    opener = gzip.open if path.suffix == ".gz" else Path.open
    ids: list[str] = []
    texts: list[str] = []
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    seen_ids: set[str] = set()
    duplicates = 0
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corpus line {line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Corpus line {line_number}: expected object")
            corpus_id, text = record.get("corpus_id"), record.get("text")
            if not isinstance(corpus_id, str) or not corpus_id or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Corpus line {line_number}: missing ID or text")
            if corpus_id in seen_ids:
                raise ValueError(f"Corpus line {line_number}: duplicate ID {corpus_id}")
            seen_ids.add(corpus_id)
            ids.append(corpus_id)
            texts.append(text)
            if text in exact:
                duplicates += 1
            exact.setdefault(text, corpus_id)
            normalized.setdefault(_space_normalize(text), corpus_id)
    if not ids:
        raise ValueError("Corpus has no passages")
    return Corpus(tuple(ids), tuple(texts), exact, normalized, duplicates)


def map_query_groups(groups: Sequence[QueryGroup], corpus: Corpus) -> tuple[tuple[MappedQuery, ...], dict[str, object]]:
    """Map judged passage text to stable corpus IDs; report every lost judgment."""
    mapped: list[MappedQuery] = []
    total: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {}
    for group in groups:
        counts = categories.setdefault(group.category or "unknown", Counter())
        judgments: dict[str, int] = {}
        for candidate in group.candidates:
            corpus_id, match_kind = corpus.resolve(candidate.text)
            total[match_kind] += 1
            counts[match_kind] += 1
            if corpus_id is None:
                if candidate.relevance > 0:
                    total["unmatched_positive"] += 1
                    counts["unmatched_positive"] += 1
                continue
            if corpus_id in judgments:
                total["collapsed_duplicate_target"] += 1
                counts["collapsed_duplicate_target"] += 1
            judgments[corpus_id] = max(candidate.relevance, judgments.get(corpus_id, 0))
        if not any(score > 0 for score in judgments.values()):
            total["queries_without_mapped_positive"] += 1
            counts["queries_without_mapped_positive"] += 1
        mapped.append(MappedQuery(group.query_id, group.query, group.category, judgments))
    report: dict[str, object] = {
        "query_count": len(groups),
        "usable_judgment_count": sum(total[k] for k in ("exact", "whitespace", "unmatched")),
        "exact_matches": total["exact"],
        "whitespace_matches": total["whitespace"],
        "unmatched": total["unmatched"],
        "unmatched_positive": total["unmatched_positive"],
        "collapsed_duplicate_target": total["collapsed_duplicate_target"],
        "queries_without_mapped_positive": total["queries_without_mapped_positive"],
        "categories": {
            category: {
                "exact_matches": counts["exact"],
                "whitespace_matches": counts["whitespace"],
                "unmatched": counts["unmatched"],
                "unmatched_positive": counts["unmatched_positive"],
                "collapsed_duplicate_target": counts["collapsed_duplicate_target"],
                "queries_without_mapped_positive": counts["queries_without_mapped_positive"],
            }
            for category, counts in sorted(categories.items())
        },
    }
    return tuple(mapped), report
