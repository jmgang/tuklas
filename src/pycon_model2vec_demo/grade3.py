"""Grade-3 DisastIR cohorts and exact-retrieval metrics.

The strict cohort counts raw judgments before mapping. A missing corpus match
must never turn a multi-gold query into an apparent single-gold query.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence

import numpy as np

from .corpus import Corpus, MappedQuery
from .data import QueryGroup
from .retrieval import RankedPassage


METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_100", "mrr_at_10", "ndcg_at_10", "unjudged_at_10")
TOP10_METRICS = tuple(metric for metric in METRICS if metric != "recall_at_100")


@dataclass(frozen=True)
class Grade3Query:
    query_id: str
    query: str
    category: str
    raw_grade3_count: int
    gold_ids: frozenset[str]
    judged_ids: frozenset[str]


def select_cohorts(groups: Sequence[QueryGroup], mapped: Sequence[MappedQuery], corpus: Corpus) -> tuple[tuple[Grade3Query, ...], tuple[Grade3Query, ...], dict]:
    """Return strict single-gold and all mapped grade-3 queries with exclusions."""
    if len(groups) != len(mapped):
        raise ValueError("Raw and mapped query counts differ")
    corpus_ids = set(corpus.ids)
    strict: list[Grade3Query] = []
    broad: list[Grade3Query] = []
    exclusions: Counter[str] = Counter()
    category_audit: dict[str, Counter[str]] = defaultdict(Counter)
    for group, mapped_query in zip(groups, mapped):
        if group.query_id != mapped_query.query_id or group.query != mapped_query.query:
            raise ValueError("Raw and mapped query order differs")
        raw = [candidate for candidate in group.candidates if candidate.relevance == 3]
        gold = frozenset(corpus_id for corpus_id, grade in mapped_query.judgments.items() if grade == 3)
        if not gold.issubset(corpus_ids):
            raise ValueError("Mapped gold ID is outside the pinned corpus")
        if len(gold) > len(raw):
            raise ValueError("Mapping produced more grade-3 passages than the raw judgments")
        query = Grade3Query(group.query_id, group.query, group.category or "unknown", len(raw), gold, frozenset(mapped_query.judgments))
        counts = category_audit[query.category]
        counts["total"] += 1
        counts["raw_grade3_judgments"] += len(raw)
        counts["mapped_grade3_passages"] += len(gold)
        counts["unmapped_or_collapsed_grade3"] += len(raw) - len(gold)
        if len(raw) == 0:
            reason = "no_raw_grade3"
        elif len(raw) != 1:
            reason = "multiple_raw_grade3"
        elif len(gold) != 1:
            reason = "single_raw_grade3_unmapped"
        else:
            reason = "selected_single"
            strict.append(query)
        exclusions[reason] += 1
        counts[reason] += 1
        if gold:
            broad.append(query)
            counts["selected_all_mapped_grade3"] += 1
        else:
            counts["no_mapped_grade3"] += 1
    if not strict or not broad:
        raise ValueError("Grade-3 cohorts are empty")
    audit = {
        "heldout_queries": len(groups),
        "strict_single_queries": len(strict),
        "all_mapped_grade3_queries": len(broad),
        "strict_selection_counts": dict(sorted(exclusions.items())),
        "raw_grade3_judgments": sum(row["raw_grade3_judgments"] for row in category_audit.values()),
        "mapped_grade3_passages": sum(row["mapped_grade3_passages"] for row in category_audit.values()),
        "unmapped_or_collapsed_grade3": sum(row["unmapped_or_collapsed_grade3"] for row in category_audit.values()),
        "by_category": {name: dict(sorted(counts.items())) for name, counts in sorted(category_audit.items())},
    }
    return tuple(strict), tuple(broad), audit


def score_query(query: Grade3Query, ranking: Sequence[RankedPassage]) -> dict[str, float]:
    if not query.gold_ids or len(ranking) < 100:
        raise ValueError("Grade-3 scoring requires mapped gold and a top-100 ranking")
    ids = [row.corpus_id for row in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError("Ranking contains duplicate passage IDs")
    gold_count = len(query.gold_ids)
    positions = [index for index, corpus_id in enumerate(ids, 1) if corpus_id in query.gold_ids]
    first = positions[0] if positions else None
    ideal = sum(1 / log2(index + 1) for index in range(1, min(gold_count, 10) + 1))
    values = {f"recall_at_{k}": sum(position <= k for position in positions) / gold_count for k in (1, 3, 5, 10, 100)}
    values["mrr_at_10"] = 1 / first if first is not None and first <= 10 else 0.0
    values["ndcg_at_10"] = sum(1 / log2(position + 1) for position in positions if position <= 10) / ideal
    values["unjudged_at_10"] = sum(corpus_id not in query.judged_ids for corpus_id in ids[:10]) / 10
    return values


def score_top10_query(query: Grade3Query, ranking: Sequence[RankedPassage]) -> dict[str, float]:
    """Score all mapped grade-3 golds using exactly ten retrieved passages."""
    if not query.gold_ids or len(ranking) != 10:
        raise ValueError("Top-10 scoring requires mapped gold and exactly ten retrieved passages")
    ids = [row.corpus_id for row in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError("Ranking contains duplicate passage IDs")
    positions = [index for index, corpus_id in enumerate(ids, 1) if corpus_id in query.gold_ids]
    ideal = sum(1 / log2(index + 1) for index in range(1, min(len(query.gold_ids), 10) + 1))
    values = {f"recall_at_{k}": sum(position <= k for position in positions) / len(query.gold_ids) for k in (1, 3, 5, 10)}
    values["mrr_at_10"] = 1 / positions[0] if positions else 0.0
    values["ndcg_at_10"] = sum(1 / log2(position + 1) for position in positions) / ideal
    values["unjudged_at_10"] = sum(corpus_id not in query.judged_ids for corpus_id in ids) / 10
    return values


def summarize(rows: Sequence[Mapping[str, object]], *, seed: int = 20261001, repetitions: int = 2000, metrics: Sequence[str] = METRICS) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty cohort")
    rng = np.random.default_rng(seed)
    categories = sorted({row["category"] for row in rows})

    def means(selected: Sequence[Mapping[str, object]]) -> dict[str, float]:
        return {metric: float(np.mean([row["metrics"][metric] for row in selected])) for metric in metrics}

    intervals = {}
    for metric in metrics:
        values = np.asarray([row["metrics"][metric] for row in rows], dtype=np.float64)
        sampled = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
        low, high = np.quantile(sampled, (0.025, 0.975))
        intervals[metric] = {"low": float(low), "high": float(high)}
    return {
        "query_count": len(rows),
        "overall": means(rows),
        "bootstrap_95": {"method": "query-level percentile", "seed": seed, "repetitions": repetitions, "metrics": intervals},
        "by_category": {category: {"query_count": len(selected := [row for row in rows if row["category"] == category]), "metrics": means(selected)} for category in categories},
    }


def paired_with_teacher(teacher_rows: Sequence[Mapping[str, object]], student_rows: Sequence[Mapping[str, object]], *, seed: int = 20261001, repetitions: int = 2000, metrics: Sequence[str] = METRICS) -> dict:
    teacher = {row["query_id"]: row["metrics"] for row in teacher_rows}
    student = {row["query_id"]: row["metrics"] for row in student_rows}
    if not teacher or teacher.keys() != student.keys():
        raise ValueError("Paired comparison requires identical query IDs")
    rng = np.random.default_rng(seed)
    result = {}
    for metric in metrics:
        deltas = np.asarray([student[key][metric] - teacher[key][metric] for key in sorted(teacher)], dtype=np.float64)
        sampled = deltas[rng.integers(0, len(deltas), size=(repetitions, len(deltas)))].mean(axis=1)
        low, high = np.quantile(sampled, (0.025, 0.975))
        result[metric] = {"mean_difference": float(deltas.mean()), "low": float(low), "high": float(high)}
    return {"seed": seed, "repetitions": repetitions, "metrics": result}
