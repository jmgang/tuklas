"""Build an auditable retrieval report from already-scored model runs."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from .corpus import Corpus, MappedQuery
from .evaluation import QUALITY_METRICS, paired_quality_intervals, quality_intervals
from .retrieval import RankedPassage


LIMITATIONS = (
    "The released QA queries are LLM-generated, not observed production questions.",
    "Relevance judgments are pooled and LLM-rated; unjudged retrieved passages have unknown relevance.",
    "Some released judged passages do not map to the released corpus, including positive judgments.",
    "The corpus does not provide reliable source document, page, or URL provenance for cited answers.",
    "This measures passage retrieval, not generated answer quality or a deployed RAG system.",
    "The ready-made Potion baseline has different training data and teacher lineage from the E5 students.",
    "One host's latency and memory measurements are not a production service-level guarantee.",
)


def _metric_row(score: Mapping[str, object], query_id: str) -> Mapping[str, float]:
    return next(row["metrics"] for row in score["per_query"] if row["query_id"] == query_id)


def disagreement_examples(
    queries: Sequence[MappedQuery], corpus: Corpus,
    scores: Mapping[str, Mapping[str, object]],
    rankings: Mapping[str, Sequence[Sequence[RankedPassage]]],
    *, reference: str = "teacher", comparison: str = "custom_student", count: int = 3,
) -> dict[str, list[dict[str, object]]]:
    """Choose largest positive/negative per-query nDCG differences, not anecdotes by hand."""
    if reference not in scores or comparison not in scores or count <= 0:
        raise ValueError("Both compared models and a positive example count are required")
    if any(len(rows) != len(queries) for rows in rankings.values()):
        raise ValueError("Every ranking set must match query order")
    text_by_id = dict(zip(corpus.ids, corpus.texts))
    differences = []
    for index, query in enumerate(queries):
        delta = _metric_row(scores[comparison], query.query_id)["nDCG@10"] - _metric_row(scores[reference], query.query_id)["nDCG@10"]
        differences.append((float(delta), query.query_id, index))

    def example(item: tuple[float, str, int]) -> dict[str, object]:
        delta, _, index = item
        query = queries[index]
        return {
            "query_id": query.query_id,
            "category": query.category,
            "query": query.query,
            "comparison_minus_reference_ndcg_at_10": delta,
            "retrieved": {
                model: [
                    {
                        "rank": rank,
                        "corpus_id": passage.corpus_id,
                        "relevance_if_judged": query.judgments.get(passage.corpus_id),
                        "passage_excerpt": text_by_id[passage.corpus_id][:300],
                    }
                    for rank, passage in enumerate(rankings[model][index][:3], 1)
                ]
                for model in (reference, comparison)
            },
        }

    gains = sorted((row for row in differences if row[0] > 0), key=lambda row: (-row[0], row[1]))[:count]
    losses = sorted((row for row in differences if row[0] < 0), key=lambda row: (row[0], row[1]))[:count]
    return {"comparison_better": [example(row) for row in gains], "comparison_worse": [example(row) for row in losses]}


def illustrative_decisions(
    scores: Mapping[str, Mapping[str, object]], measurements: Mapping[str, Mapping[str, object]],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """Apply predeclared illustrative constraints; show every individual condition."""
    teacher = thresholds["teacher_reference"]
    if teacher not in scores or teacher not in measurements:
        raise ValueError("Decision requires teacher quality and operational measurements")
    reference = scores[teacher]["overall"]
    teacher_p95 = measurements[teacher]["warm"]["single_query_end_to_end"]["p95_ms"]
    if teacher_p95 <= 0:
        raise ValueError("Teacher p95 must be positive")
    quality = thresholds["quality"]
    operational = thresholds["operational"]
    decisions = {}
    for name, score in scores.items():
        if name == teacher:
            continue
        measurement = measurements[name]
        observed = score["overall"]
        p95 = measurement["warm"]["single_query_end_to_end"]["p95_ms"]
        values = {
            "ndcg_at_10_loss": reference["nDCG@10"] - observed["nDCG@10"],
            "known_positive_recall_at_100_loss": reference["R@100"] - observed["R@100"],
            "single_query_end_to_end_p95_ms": p95,
            "model_artifact_mib": measurement["model_artifact_size_bytes"] / 2**20,
            "peak_process_rss_gib": measurement["peak_rss_after_workload_bytes"] / 2**30,
            "end_to_end_p95_speedup_over_teacher": teacher_p95 / p95,
        }
        passed = {
            "ndcg_at_10_loss": values["ndcg_at_10_loss"] <= quality["maximum_absolute_ndcg_at_10_loss"],
            "known_positive_recall_at_100_loss": values["known_positive_recall_at_100_loss"] <= quality["maximum_absolute_known_positive_recall_at_100_loss"],
            "single_query_end_to_end_p95_ms": p95 <= operational["maximum_single_query_end_to_end_p95_ms"],
            "model_artifact_mib": values["model_artifact_mib"] <= operational["maximum_model_artifact_mib"],
            "peak_process_rss_gib": values["peak_process_rss_gib"] <= operational["maximum_peak_process_rss_gib"],
            "end_to_end_p95_speedup_over_teacher": values["end_to_end_p95_speedup_over_teacher"] >= operational["minimum_end_to_end_p95_speedup_over_teacher"],
        }
        decisions[name] = {"replacement_candidate": all(passed.values()), "observed": values, "passes": passed}
    return decisions


def build_report(
    *, split: str, queries: Sequence[MappedQuery], corpus: Corpus,
    scores: Mapping[str, Mapping[str, object]],
    rankings: Mapping[str, Sequence[Sequence[RankedPassage]]],
    coverage: Mapping[str, object], provenance: Mapping[str, object],
    measurements: Mapping[str, Mapping[str, object]] | None = None,
    thresholds: Mapping[str, object] | None = None,
    bootstrap_seed: int = 20261001, bootstrap_repetitions: int = 2000,
) -> dict[str, object]:
    if not queries or not scores or set(scores) != set(rankings):
        raise ValueError("Report requires aligned queries, scores, and rankings")
    expected_ids = [query.query_id for query in queries]
    for name, score in scores.items():
        if [row["query_id"] for row in score["per_query"]] != expected_ids or len(rankings[name]) != len(queries):
            raise ValueError(f"{name} query order differs from the frozen split")
    result: dict[str, object] = {
        "status": f"{split}-full-corpus-retrieval-quality",
        "scope": "exact full-corpus passage retrieval; judged-candidate ranking is not the primary metric",
        "query_count": len(queries),
        "corpus_passage_count": len(corpus.ids),
        "mapped_judgment_coverage": dict(coverage),
        "provenance": dict(provenance),
        "models": {
            name: {"scores": score, "query_bootstrap_intervals": quality_intervals(score, seed=bootstrap_seed, repetitions=bootstrap_repetitions)}
            for name, score in scores.items()
        },
        "paired_with_teacher": {
            name: paired_quality_intervals(scores["teacher"], score, seed=bootstrap_seed, repetitions=bootstrap_repetitions)
            for name, score in scores.items() if name != "teacher" and "teacher" in scores
        },
        "limitations": list(LIMITATIONS),
    }
    if "teacher" in scores and "custom_student" in scores:
        result["disagreement_examples"] = disagreement_examples(queries, corpus, scores, rankings)
    if measurements is not None:
        if set(measurements) != set(scores):
            raise ValueError("Operational measurements must cover every scored model")
        result["operational_measurements"] = dict(measurements)
        if any(value.get("corpus_embedding_build_backend") == "SageMaker GPU" for value in measurements.values()):
            result["limitations"].append(
                "The teacher corpus was encoded on a SageMaker GPU, while warm query/retrieval timings and other model builds ran on the local host; corpus build times are not same-hardware comparisons."
            )
    if thresholds is not None:
        if measurements is None:
            raise ValueError("Decision thresholds require operational measurements")
        result["illustrative_thresholds"] = dict(thresholds)
        result["illustrative_decisions"] = illustrative_decisions(scores, measurements, thresholds)
    return result


def render_markdown(report: Mapping[str, object]) -> str:
    """Compact human-readable companion to the complete machine-readable report."""
    def number(value: float) -> str:
        return f"{value:.3f}"

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        f"# {clean(report['status'])}", "",
        f"Exact retrieval over {report['corpus_passage_count']:,} passages and {report['query_count']:,} queries.", "",
        "## Overall", "",
        "| Model | nDCG@10 | Recall@10* | Recall@100* | MRR@10 | Hit@10 | Unjudged@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, model in report["models"].items():
        metrics = model["scores"]["overall"]
        lines.append("| " + clean(name) + " | " + " | ".join(number(metrics[key]) for key in (*QUALITY_METRICS, "Unjudged@10")) + " |")
    lines += ["", "*Recall denominators are mapped, known-positive judgments, not all truly relevant passages.", "", "## Query-bootstrap 95% intervals", ""]
    for name, model in report["models"].items():
        interval = model["query_bootstrap_intervals"]["overall"]["nDCG@10"]
        lines.append(f"- {clean(name)} nDCG@10: {number(interval['mean'])} [{number(interval['ci_95_low'])}, {number(interval['ci_95_high'])}]")
    lines += ["", "## Category breakdown", "", "| Category | Queries | " + " | ".join(clean(name) for name in report["models"]) + " |", "| --- | ---: | " + " | ".join("---:" for _ in report["models"]) + " |"]
    category_names = sorted(next(iter(report["models"].values()))["scores"]["by_category"])
    for category in category_names:
        counts = next(iter(report["models"].values()))["scores"]["by_category"][category]["query_count"]
        values = []
        for model in report["models"].values():
            interval = model["query_bootstrap_intervals"]["by_category"][category]["nDCG@10"]
            values.append(f"{number(interval['mean'])} [{number(interval['ci_95_low'])}, {number(interval['ci_95_high'])}]")
        lines.append(f"| {clean(category)} | {counts} | " + " | ".join(values) + " |")
    lines += ["", "## Judgment mapping", "", "```json", json.dumps(report["mapped_judgment_coverage"], indent=2, sort_keys=True), "```", ""]
    if report.get("paired_with_teacher"):
        lines += ["## Paired differences from teacher", ""]
        for name, paired in report["paired_with_teacher"].items():
            value = paired["metrics"]["nDCG@10"]
            lines.append(f"- {clean(name)} minus teacher nDCG@10: {number(value['mean_difference'])} [{number(value['ci_95_low'])}, {number(value['ci_95_high'])}]")
        lines.append("")
    if "operational_measurements" in report:
        lines += ["## Operational measurements", "", "| Model | Warm query+search p95 (ms) | Model MiB | Corpus vectors MiB | Peak process GiB | Corpus build seconds if measured | Build host |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, value in report["operational_measurements"].items():
            built = value.get("separate_fresh_corpus_embedding_build_seconds", value["full_corpus_embedding_build_seconds"])
            lines.append(f"| {clean(name)} | {number(value['warm']['single_query_end_to_end']['p95_ms'])} | {number(value['model_artifact_size_bytes'] / 2**20)} | {number(value['corpus_embedding_cache_size_bytes'] / 2**20)} | {number(value['peak_rss_after_workload_bytes'] / 2**30)} | {number(built) if built is not None else 'not measured (cache reused)'} | {clean(value.get('corpus_embedding_build_backend', 'not recorded'))} |")
        lines.append("")
    if "illustrative_decisions" in report:
        lines += ["## Illustrative replacement decisions", ""]
        for name, decision in report["illustrative_decisions"].items():
            failed = [key for key, passed in decision["passes"].items() if not passed]
            lines.append(f"- {clean(name)}: {'passes' if decision['replacement_candidate'] else 'fails'}; failed checks: {', '.join(failed) if failed else 'none'}")
        lines.append("")
    if "disagreement_examples" in report:
        lines += ["## Query disagreements: custom student versus teacher", ""]
        for direction, examples in report["disagreement_examples"].items():
            lines.append(f"### {clean(direction)}")
            lines.append("")
            for example in examples:
                lines.append(f"- {clean(example['category'])}: {clean(example['query'])} (nDCG delta {number(example['comparison_minus_reference_ndcg_at_10'])})")
                for name, retrieved in example["retrieved"].items():
                    if retrieved:
                        first = retrieved[0]
                        lines.append(f"  - {clean(name)} top passage, label {clean(first['relevance_if_judged'])}: {clean(first['passage_excerpt'])}")
            lines.append("")
    lines += ["## Limitations", ""] + [f"- {clean(item)}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"
