#!/usr/bin/env python3
"""Build unsupervised Tagalog query/passage pairs from the approved corpus.

The teacher-mined supervised cohort is 15,870 pairs.  The published static
retrieval reference (``sentence-transformers/static-retrieval-mrl-en-v1``)
trains on millions, and every hyperparameter lever that assumes scale -- large
batches, and therefore the learning rates that go with them -- has failed on
this project purely because 15,870 pairs yield too few optimizer steps per
epoch.

This generates roughly a million pairs from text the project already owns,
using the standard unsupervised dense-retrieval constructions:

* **Inverse Cloze Task** (ORQA/REALM): one sentence is removed from a passage
  and used as the pseudo-query; the remainder is the positive document.  The
  sentence is *deleted* from the document, so the task is not trivially lexical.
* **Independent cropping** (Contriever): two disjoint word spans of the same
  passage form a positive pair.
* **Title to passage**: the article title is a genuinely query-shaped string.

No language model is involved and no new dataset is introduced: the input is
``runs/multiwikiqa-training-text-20261004-v5/training.tl.jsonl``, the approved
external Tagalog training corpus, whose audit records all 1,294 benchmark
parent articles plus 691 alias/translation variants as excluded.  Held-out
questions are never read.

``positive_group`` is the source article, so the ranking worker's batch builder
keeps same-article passages out of a batch and in-batch negatives stay free of
false negatives.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/multiwikiqa-training-text-20261004-v5/training.tl.jsonl"
AUDIT = ROOT / "runs/multiwikiqa-training-text-20261004-v5/audit.json"
BENCHMARK_PARENTS = ROOT / "runs/multiwikiqa-preparation-20261004-v1/parents.jsonl"

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
MINIMUM_QUERY_WORDS = 6
MAXIMUM_QUERY_WORDS = 48
MINIMUM_DOCUMENT_WORDS = 24
CROP_MINIMUM_WORDS = 18
CROP_MAXIMUM_WORDS = 56


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]


def inverse_cloze(row: dict, rng: random.Random) -> list[dict]:
    """One sentence becomes the query; the rest of the passage is the positive."""

    parts = sentences(row["text"])
    if len(parts) < 3:
        return []
    pairs = []
    order = list(range(len(parts)))
    rng.shuffle(order)
    for index in order:
        query = parts[index]
        words = query.split()
        if not MINIMUM_QUERY_WORDS <= len(words) <= MAXIMUM_QUERY_WORDS:
            continue
        document = " ".join(parts[:index] + parts[index + 1 :]).strip()
        if len(document.split()) < MINIMUM_DOCUMENT_WORDS:
            continue
        pairs.append({"query": query, "document": document, "construction": "inverse-cloze"})
    return pairs


def independent_crops(row: dict, rng: random.Random, wanted: int) -> list[dict]:
    """Two disjoint spans of one passage are a positive pair (Contriever)."""

    words = row["text"].split()
    if len(words) < 2 * CROP_MINIMUM_WORDS + 2:
        return []
    pairs = []
    for _ in range(wanted * 3):
        if len(pairs) >= wanted:
            break
        query_length = rng.randint(CROP_MINIMUM_WORDS, CROP_MAXIMUM_WORDS)
        document_length = rng.randint(CROP_MINIMUM_WORDS, CROP_MAXIMUM_WORDS)
        if query_length + document_length + 1 > len(words):
            continue
        split = rng.randint(query_length, len(words) - document_length - 1)
        query = " ".join(words[split - query_length : split])
        document = " ".join(words[split + 1 : split + 1 + document_length])
        if not query.strip() or not document.strip():
            continue
        pairs.append({"query": query, "document": document, "construction": "independent-crop"})
    return pairs


def first_sentence_pair(row: dict) -> list[dict]:
    """The opening sentence of a Wikipedia passage is usually definitional.

    Inverse cloze samples sentences uniformly; this targets the one sentence that
    most resembles a query about the article's subject.
    """

    parts = sentences(row["text"])
    if len(parts) < 3:
        return []
    query = parts[0]
    words = query.split()
    if not MINIMUM_QUERY_WORDS <= len(words) <= MAXIMUM_QUERY_WORDS:
        return []
    document = " ".join(parts[1:]).strip()
    if len(document.split()) < MINIMUM_DOCUMENT_WORDS:
        return []
    return [{"query": query, "document": document, "construction": "first-sentence-to-rest"}]


def title_pair(row: dict) -> list[dict]:
    title = str(row.get("source_title") or "").strip()
    if len(title.split()) < 2 or len(title.split()) > MAXIMUM_QUERY_WORDS:
        return []
    if len(row["text"].split()) < MINIMUM_DOCUMENT_WORDS:
        return []
    return [{"query": title, "document": row["text"], "construction": "title-to-passage"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1_000_000)
    parser.add_argument("--crops-per-passage", type=int, default=6)
    parser.add_argument("--inverse-cloze-per-passage", type=int, default=7)
    # Per-construction ceilings.  The 2M build came out 79.4% independent
    # crops and 1.8% cross-passage because crops are emitted last and absorb
    # the whole remaining budget.  Crops teach passage-internal similarity;
    # the benchmark ranks *articles*, so cross-passage is the only
    # construction whose supervision matches the metric.  These let the mix be
    # stated rather than inherited from emission order.  0 means unlimited.
    parser.add_argument("--cross-passage-target", type=int, default=0)
    parser.add_argument("--inverse-cloze-target", type=int, default=0)
    parser.add_argument("--crop-target", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    rows = read_jsonl(SOURCE)
    audit = json.loads(AUDIT.read_text())
    if audit.get("benchmark_parents_excluded") != 1294:
        raise ValueError("approved corpus no longer excludes all benchmark parents")
    if any(row.get("language") != "tl" for row in rows):
        raise ValueError("non-Tagalog row in the Tagalog training corpus")

    # Defence in depth: the audit says benchmark parents are excluded, so assert
    # that no generated passage carries a benchmark parent identifier.
    benchmark_ids = {row["parent_id"] for row in read_jsonl(BENCHMARK_PARENTS)}
    offending = {
        str(row.get("source_article_id"))
        for row in rows
        if str(row.get("source_article_id")) in benchmark_ids
    }
    if offending:
        raise ValueError(f"benchmark article leaked into training text: {sorted(offending)[:3]}")

    rng = random.Random(arguments.seed)
    seen_queries: set[str] = set()
    per_construction: dict[str, int] = defaultdict(int)
    per_article: dict[str, int] = defaultdict(int)
    pairs: list[dict] = []

    order = list(range(len(rows)))
    rng.shuffle(order)

    ceilings = {
        "cross-passage-same-article": arguments.cross_passage_target,
        "inverse-cloze": arguments.inverse_cloze_target,
        "independent-crop": arguments.crop_target,
    }

    def emit(row: dict, candidate: dict) -> bool:
        if len(pairs) >= arguments.target:
            return False
        ceiling = ceilings.get(candidate["construction"], 0)
        if ceiling and per_construction[candidate["construction"]] >= ceiling:
            return False
        key = normalized(candidate["query"])
        if not key or key in seen_queries:
            return False
        if normalized(candidate["document"]) == key:
            return False
        seen_queries.add(key)
        article = str(row.get("source_article_id"))
        record = {
            "record_id": sha256(
                (candidate["construction"] + "|" + key).encode()
            ).hexdigest()[:24],
            "query": candidate["query"],
            "document": candidate["document"],
            "construction": candidate["construction"],
            "language": "tl",
            "positive_group": "external:" + article,
            "source_article_id": article,
            "source_passage_id": row.get("passage_id"),
        }
        pairs.append(record)
        per_construction[candidate["construction"]] += 1
        per_article[article] += 1
        return True

    # Title pairs first: they are the most query-shaped and the scarcest.
    for index in order:
        row = rows[index]
        for candidate in title_pair(row):
            emit(row, candidate)

    # First sentence before inverse cloze: both draw from the same sentence
    # pool and ``seen_queries`` is first-come, so the previous order let
    # uniform cloze sampling consume the lead sentence and starve this
    # construction (9,293 pairs at 749k target, but only 2,562 at 2M).
    for index in order:
        row = rows[index]
        for candidate in first_sentence_pair(row):
            emit(row, candidate)

    for index in order:
        row = rows[index]
        for candidate in inverse_cloze(row, rng)[: arguments.inverse_cloze_per_passage]:
            emit(row, candidate)

    # Cross-passage, same article: two different passages of one article are a
    # positive pair.  Every other construction teaches passage-internal
    # similarity, but the benchmark ranks *articles*, so this is the only
    # construction whose supervision matches the thing being scored.
    by_article: dict[str, list[int]] = defaultdict(list)
    for index in order:
        by_article[str(rows[index].get("source_article_id"))].append(index)
    # Every *ordered* pair of distinct passages, not one chain through each
    # article.  ``zip(local, local[1:])`` yielded only n-1 pairs per article in
    # a single direction, capping this construction at 35,586 corpus-wide; all
    # ordered pairs yield 177,900 (no article has more than 8 passages, so the
    # quadratic term cannot run away).  Candidates are shuffled globally so
    # ``--cross-passage-target`` truncates across articles rather than
    # exhausting the first few.
    candidates: list[tuple[int, int]] = []
    for ids in by_article.values():
        if len(ids) < 2:
            continue
        for first in ids:
            for second in ids:
                if first != second:
                    candidates.append((first, second))
    rng.shuffle(candidates)
    for first, second in candidates:
        if len(pairs) >= arguments.target:
            break
        a, b = rows[first], rows[second]
        query_words = a["text"].split()
        if len(query_words) < CROP_MINIMUM_WORDS:
            continue
        span = rng.randint(CROP_MINIMUM_WORDS, min(CROP_MAXIMUM_WORDS, len(query_words)))
        start = rng.randint(0, len(query_words) - span)
        emit(a, {
            "query": " ".join(query_words[start : start + span]),
            "document": b["text"],
            "construction": "cross-passage-same-article",
        })

    for index in order:
        row = rows[index]
        for candidate in independent_crops(row, rng, arguments.crops_per_passage):
            emit(row, candidate)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = arguments.out or ROOT / f"runs/multiwikiqa-tagalog-unsupervised-pairs-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    admitted = out / "admitted.jsonl"
    with admitted.open("x", encoding="utf-8") as stream:
        for record in pairs:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    digest = sha256(admitted.read_bytes()).hexdigest()
    report = {
        "generated_utc": stamp,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_rows": len(rows),
        "source_articles": len({str(r.get("source_article_id")) for r in rows}),
        "language": "tl",
        "english_records_used": False,
        "pairs": len(pairs),
        "distinct_articles_covered": len(per_article),
        "per_construction": dict(per_construction),
        "median_pairs_per_article": sorted(per_article.values())[len(per_article) // 2]
        if per_article
        else 0,
        "duplicate_queries_rejected": True,
        "benchmark_parents_excluded": 1294,
        "benchmark_article_leak_checked": True,
        "heldout_questions_read": False,
        "llm_generated": False,
        "seed": arguments.seed,
        "arguments": {
            "target": arguments.target,
            "crops_per_passage": arguments.crops_per_passage,
            "inverse_cloze_per_passage": arguments.inverse_cloze_per_passage,
            "cross_passage_target": arguments.cross_passage_target,
            "inverse_cloze_target": arguments.inverse_cloze_target,
            "crop_target": arguments.crop_target,
        },
        "construction_ceilings": {
            "cross-passage-same-article": arguments.cross_passage_target or None,
            "inverse-cloze": arguments.inverse_cloze_target or None,
            "independent-crop": arguments.crop_target or None,
        },
        "per_construction_share": {
            name: round(count / len(pairs), 6) for name, count in sorted(per_construction.items())
        }
        if pairs
        else {},
        "admitted_sha256": digest,
        "admitted_bytes": admitted.stat().st_size,
    }
    (out / "verification-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
