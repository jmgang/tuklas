"""Three-way Tagalog development partition for the v8 ranking protocol.

The frozen ``multiwikiqa_ranking_v5.prepare`` returns a two-way split in which
the 197-question validation cohort is used both to select ranking checkpoints
and to report the headline development metric.  That couples two jobs that must
stay separate: selecting on the reported cohort makes the reported number a
maximum over every evaluation the ranking loop performed, not an estimate of
generalisation.

This module keeps the v5 article-level hash predicate byte-for-byte so the
reported cohort stays exactly the 197 questions the earlier runs reported, and
splits the remaining development questions into a disjoint selection cohort and
a training cohort:

===========  ==================  =====  ====================================
cohort       ``component_id`` %5  count  role
===========  ==================  =====  ====================================
report       ``0``                 197  reported metric only; never selected
selection    ``1``                 129  ranking checkpoint selection only
train        ``2``, ``3``, ``4``   689  ranking supervision
===========  ==================  =====  ====================================

The v5 hygiene rules (held-out text overlap, ambiguous duplicate questions and
missing answer-bearing chunks) are applied to the training cohort exactly as
before.  They are additionally *audited* on the selection and report cohorts,
which v5 skipped entirely: the selection cohort must be completely clean, and
any unclean report-cohort question is recorded rather than silently retained,
so the one known held-out text duplicate stays visible in the receipt.
"""

from __future__ import annotations

import unicodedata
from hashlib import sha256

REPORT_RESIDUE = 0
SELECTION_RESIDUE = 1
TRAIN_RESIDUES = (2, 3, 4)

EXPECTED_TAGALOG = {"report": 197, "selection": 129, "train": 689}


def normalized_question(value: str) -> str:
    """Match the v5 ``key`` helper so hygiene decisions stay identical."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def article_residue(component_id: str) -> int:
    """The frozen v5 article-level split predicate."""

    digest = sha256(("ranking-v1:" + component_id).encode()).hexdigest()
    return int(digest[:16], 16) % 5


def prepare_three_way(queries: list[dict], chunks: list[dict], *, language: str = "tl") -> dict:
    """Split development questions into report/selection/train cohorts."""

    forbidden_ids = {row["query_id"] for row in queries if row["partition"] == "evaluation"}
    forbidden_text = {
        normalized_question(row["question"])
        for row in queries
        if row["partition"] == "evaluation"
    }
    references: dict[str, set[str]] = {}
    for row in queries:
        references.setdefault(normalized_question(row["question"]), set()).add(
            row["gold_parent_id"]
        )
    by_parent: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_parent.setdefault(chunk["parent_id"], []).append(chunk)

    def hygiene(row: dict) -> str | None:
        question = normalized_question(row["question"])
        if row["query_id"] in forbidden_ids or question in forbidden_text:
            return "evaluation question overlap"
        if len(references[question]) != 1:
            return "ambiguous duplicate question"
        if not answer_bearing(row):
            return "no fully answer-bearing chunk"
        return None

    def answer_bearing(row: dict) -> list[dict]:
        return [
            chunk
            for chunk in by_parent.get(row["gold_parent_id"], [])
            if any(
                chunk["start"] <= span["start"] and span["end"] <= chunk["end"]
                for span in row["answer_spans"]
            )
        ]

    development = [
        row
        for row in queries
        if row["partition"] == "development" and row["language"] == language
    ]

    report: list[dict] = []
    selection: list[dict] = []
    train: list[dict] = []
    skipped: list[dict] = []
    report_hygiene: list[dict] = []

    for row in development:
        residue = article_residue(row["component_id"])
        reason = hygiene(row)
        if residue == REPORT_RESIDUE:
            # Retained even when unclean so the cohort stays comparable with the
            # earlier runs; the reason is recorded for the receipt instead.
            if reason is not None:
                report_hygiene.append({"query_id": row["query_id"], "reason": reason})
            report.append(row)
            continue
        if residue == SELECTION_RESIDUE:
            if reason is not None:
                raise ValueError(
                    "selection cohort must be hygiene-clean: "
                    f"{row['query_id']} ({reason})"
                )
            selection.append(row)
            continue
        if reason is not None:
            skipped.append({"query_id": row["query_id"], "reason": reason})
            continue
        positive = answer_bearing(row)
        chosen = min(positive, key=lambda chunk: (len(chunk["text"]), chunk["start"], chunk["chunk_id"]))
        train.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "parent_id": row["gold_parent_id"],
            "component_id": row["component_id"],
            "chunk_id": chosen["chunk_id"],
            "positive": chosen["text"],
            "language": row["language"],
        })

    cohorts = {"report": report, "selection": selection, "train": train}
    for name, rows in cohorts.items():
        if not rows:
            raise ValueError(f"empty {name} cohort")
    if language == "tl":
        actual = {name: len(rows) for name, rows in cohorts.items()}
        if actual != EXPECTED_TAGALOG:
            raise ValueError(f"Tagalog cohort sizes changed: {actual}")

    articles = {
        name: {row["component_id"] for row in rows} for name, rows in cohorts.items()
    }
    for left in cohorts:
        for right in cohorts:
            if left < right and articles[left] & articles[right]:
                raise ValueError(f"article leakage between {left} and {right}")
    for name, rows in cohorts.items():
        if {row["query_id"] for row in rows} & forbidden_ids:
            raise ValueError(f"held-out question leaked into {name}")
    if {row["query_id"] for row in train} & {
        row["query_id"] for row in report + selection
    }:
        raise ValueError("training question also scored")

    return {
        "report": report,
        "selection": selection,
        "train": train,
        "skipped": skipped,
        "audit": {
            "language": language,
            "report_questions": len(report),
            "selection_questions": len(selection),
            "train_questions": len(train),
            "skipped_questions": len(skipped),
            "report_residue": REPORT_RESIDUE,
            "selection_residue": SELECTION_RESIDUE,
            "train_residues": list(TRAIN_RESIDUES),
            "report_cohort_matches_v5_validation": True,
            "report_hygiene_exceptions": report_hygiene,
            "selection_cohort_hygiene_clean": True,
            "report_cohort_used_for_selection": False,
            "heldout_labels_used": False,
        },
    }
