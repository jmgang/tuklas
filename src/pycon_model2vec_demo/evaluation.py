"""Standard IR measures and query-level uncertainty for full-corpus retrieval."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Mapping, Sequence

import ir_measures
import numpy as np

from .corpus import MappedQuery
from .retrieval import RankedPassage


MEASURES = (
    ir_measures.nDCG @ 10,
    ir_measures.R @ 10,
    ir_measures.R @ 100,
    ir_measures.RR @ 10,
    ir_measures.Judged @ 10,
)
QUALITY_METRICS = ("nDCG@10", "R@10", "R@100", "RR@10", "Hit@10")
METRIC_LABELS = {
    "nDCG@10": "nDCG@10",
    "R@10": "Recall@10 of known positives",
    "R@100": "Recall@100 of known positives",
    "RR@10": "MRR@10",
    "Hit@10": "Hit@10",
    "Unjudged@10": "Unjudged@10 (unknown relevance)",
}


def score_rankings(
    queries: Sequence[MappedQuery], rankings: Sequence[Sequence[RankedPassage]], *, positive_threshold: int = 1
) -> dict[str, object]:
    """Score known judgments; unjudged retrieved passages remain unknown, not negative facts."""
    if not queries or len(queries) != len(rankings) or len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Each unique query must have exactly one ranking")
    if positive_threshold != 1:
        raise ValueError("This frozen experiment uses a positive threshold of 1")
    if any(not any(grade >= positive_threshold for grade in query.judgments.values()) for query in queries):
        raise ValueError("A query has no mapped positive and needs an explicit exclusion report")
    qrels = [
        ir_measures.Qrel(query.query_id, corpus_id, grade)
        for query in queries for corpus_id, grade in query.judgments.items()
    ]
    run = [
        ir_measures.ScoredDoc(query.query_id, row.corpus_id, float(len(rows) - rank))
        for query, rows in zip(queries, rankings) for rank, row in enumerate(rows)
    ]
    rows_by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for result in ir_measures.iter_calc(MEASURES, qrels, run):
        rows_by_query[result.query_id][str(result.measure)] = float(result.value)
    expected_metrics = {str(metric) for metric in MEASURES}
    if set(rows_by_query) != {query.query_id for query in queries}:
        raise RuntimeError("The IR evaluator omitted a query")
    output_rows = []
    for query, ranked in zip(queries, rankings):
        values = rows_by_query[query.query_id]
        if set(values) != expected_metrics:
            raise RuntimeError(f"Missing IR measure for {query.query_id}")
        values["Hit@10"] = float(any(query.judgments.get(row.corpus_id, 0) >= positive_threshold for row in ranked[:10]))
        values["Unjudged@10"] = 1.0 - values["Judged@10"]
        output_rows.append({"query_id": query.query_id, "category": query.category, "metrics": values})
    categories = sorted({row["category"] for row in output_rows})
    def aggregate(selected):
        return {metric: mean(row["metrics"][metric] for row in selected) for metric in sorted(output_rows[0]["metrics"])}
    return {
        "query_count": len(output_rows),
        "overall": aggregate(output_rows),
        "by_category": {
            category: {"query_count": len(selected := [row for row in output_rows if row["category"] == category]), "metrics": aggregate(selected)}
            for category in categories
        },
        "per_query": output_rows,
    }


def paired_bootstrap_difference(
    first: Mapping[str, object], second: Mapping[str, object], *, metric: str = "nDCG@10", seed: int = 20261001, repetitions: int = 2000
) -> dict[str, float]:
    """Paired query-level interval for second minus first; same query IDs required."""
    first_rows = {row["query_id"]: row["metrics"][metric] for row in first["per_query"]}
    second_rows = {row["query_id"]: row["metrics"][metric] for row in second["per_query"]}
    if not first_rows or first_rows.keys() != second_rows.keys():
        raise ValueError("Paired bootstrap requires the same nonempty query IDs")
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    differences = np.asarray([second_rows[key] - first_rows[key] for key in sorted(first_rows)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    sampled = differences[sample_indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"mean_difference": float(differences.mean()), "ci_95_low": float(low), "ci_95_high": float(high), "repetitions": repetitions}


def bootstrap_mean_interval(
    report: Mapping[str, object], *, metric: str, seed: int = 20261001, repetitions: int = 2000,
    category: str | None = None,
) -> dict[str, float]:
    """Bootstrap over whole queries; do not resample individual judged pairs."""
    rows = [row for row in report["per_query"] if category is None or row["category"] == category]
    if not rows or repetitions <= 0:
        raise ValueError("Bootstrap needs queries and positive repetitions")
    values = np.asarray([row["metrics"][metric] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_95_low": float(low), "ci_95_high": float(high), "query_count": len(values), "repetitions": repetitions}


def quality_intervals(
    report: Mapping[str, object], *, seed: int = 20261001, repetitions: int = 2000
) -> dict[str, object]:
    """Overall and per-category query-bootstrap intervals for quality measures."""
    categories = sorted({row["category"] for row in report["per_query"]})
    return {
        "method": "percentile-bootstrap-over-queries",
        "seed": seed,
        "overall": {
            metric: bootstrap_mean_interval(report, metric=metric, seed=seed, repetitions=repetitions)
            for metric in QUALITY_METRICS
        },
        "by_category": {
            category: {
                metric: bootstrap_mean_interval(report, metric=metric, seed=seed, repetitions=repetitions, category=category)
                for metric in QUALITY_METRICS
            }
            for category in categories
        },
    }


def paired_quality_intervals(
    first: Mapping[str, object], second: Mapping[str, object], *, seed: int = 20261001, repetitions: int = 2000
) -> dict[str, object]:
    """Intervals for second minus first on exactly matched query IDs."""
    return {
        "method": "paired-percentile-bootstrap-over-queries",
        "seed": seed,
        "metrics": {
            metric: paired_bootstrap_difference(first, second, metric=metric, seed=seed, repetitions=repetitions)
            for metric in QUALITY_METRICS
        },
    }
