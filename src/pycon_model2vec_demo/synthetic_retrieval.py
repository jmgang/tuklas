"""Grounded bilingual synthetic-query preparation and validation.

The benchmark questions, answers and parents are never inputs to this stage.
Synthetic supervision comes only from the separately approved English/Tagalog
Wikipedia training corpus.  Each generated item must carry a verbatim support
quote and quality subscores before it can be admitted to training.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Mapping, Sequence
import unicodedata


QUERY_TYPES = ("natural_question", "keyword_query", "factual_claim")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def passage_quality_ok(text: str, *, minimum_characters: int = 400, maximum_characters: int = 1200) -> bool:
    text = normalize_text(text)
    if not minimum_characters <= len(text) <= maximum_characters:
        return False
    alphabetic = sum(character.isalpha() for character in text)
    if alphabetic / max(len(text), 1) < 0.55:
        return False
    if sum(text.count(mark) for mark in (".", "?", "!")) < 2:
        return False
    lowered = text.casefold()
    noisy_sections = ("references", "external links", "see also", "mga sanggunian", "panlabas na kawing")
    return not any(lowered.startswith(section) for section in noisy_sections)


def choose_balanced_passages(
    rows: Sequence[Mapping[str, object]],
    *,
    per_language: int,
    seed: str,
) -> list[dict[str, object]]:
    """Choose one deterministic passage per article, balanced by language."""

    if per_language <= 0 or not seed:
        raise ValueError("selection count and seed are required")
    best_by_article: dict[tuple[str, str], tuple[tuple[int, str], Mapping[str, object]]] = {}
    for row in rows:
        language = str(row["language"])
        if language not in {"en", "tl"}:
            raise ValueError(f"unsupported language: {language}")
        article_id = str(row["source_article_id"])
        passage_id = str(row["passage_id"])
        text = str(row["text"])
        if not article_id or not passage_id or not passage_quality_ok(text):
            continue
        key = (language, article_id)
        ordering = (int(row.get("start", 0)), passage_id)
        current = best_by_article.get(key)
        if current is None or ordering < current[0]:
            best_by_article[key] = (ordering, row)

    chosen: list[dict[str, object]] = []
    for language in ("en", "tl"):
        candidates = [
            (sha256(f"{seed}:{language}:{article_id}".encode()).hexdigest(), article_id, row)
            for (row_language, article_id), (_, row) in best_by_article.items()
            if row_language == language
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) < per_language:
            raise ValueError(f"only {len(candidates)} qualifying {language} articles")
        for stable_hash, article_id, row in candidates[:per_language]:
            chosen.append(
                {
                    "language": language,
                    "source_article_id": article_id,
                    "source_title": str(row["source_title"]),
                    "source_url": str(row["source_url"]),
                    "passage_id": str(row["passage_id"]),
                    "text": str(row["text"]),
                    "text_sha256": str(row["text_sha256"]),
                    "selection_sha256": stable_hash,
                }
            )
    return chosen


def choose_language_passages(
    rows: Sequence[Mapping[str, object]],
    *,
    languages: Sequence[str],
    per_language: int,
    seed: str,
    excluded_articles: set[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Select deterministic one-passage-per-article cohorts for requested languages."""

    requested = tuple(languages)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("languages must be unique and non-empty")
    if any(language not in {"en", "tl"} for language in requested):
        raise ValueError("only English and Tagalog source languages are supported")
    excluded = excluded_articles or set()
    filtered = [
        row for row in rows
        if str(row["language"]) in requested
        and (str(row["language"]), str(row["source_article_id"])) not in excluded
    ]
    # Reuse the original deterministic selection algorithm one language at a
    # time so existing bilingual receipts remain reproducible.
    chosen = []
    for language in requested:
        language_rows = [row for row in filtered if str(row["language"]) == language]
        # The balanced helper requires both languages, so mirror only for the
        # selection call and retain the requested side. IDs are language-scoped.
        mirrored = list(language_rows)
        for row in language_rows:
            mirror = dict(row)
            mirror["language"] = "en" if language == "tl" else "tl"
            mirror["source_article_id"] = "mirror:" + str(row["source_article_id"])
            mirror["passage_id"] = "mirror:" + str(row["passage_id"])
            mirrored.append(mirror)
        selected = choose_balanced_passages(mirrored, per_language=per_language, seed=seed)
        chosen.extend(row for row in selected if row["language"] == language)
    return chosen


def generation_prompt(record: Mapping[str, object]) -> str:
    language = str(record["language"])
    language_name = "English" if language == "en" else "Tagalog"
    title = str(record["source_title"])
    passage = str(record["text"])
    return f"""Create exactly three retrieval-training queries grounded only in the source passage below.

Output language: {language_name}. Do not translate into the other language.
Required query types, exactly once each: natural_question, keyword_query, factual_claim.

For every item:
- query must be answerable or verifiable from this passage alone;
- answer must not introduce information absent from the passage;
- evidence must be a verbatim quote copied from the passage;
- keep query at most 24 words, answer at most 35 words, and evidence at most 35 words;
- avoid referring to 'the passage', 'the text', or 'the article';
- avoid a query that is merely the title.

Return only one JSON object with this exact shape:
{{"items":[{{"query_type":"natural_question","query":"...","answer":"...","evidence":"..."}},{{"query_type":"keyword_query","query":"...","answer":"...","evidence":"..."}},{{"query_type":"factual_claim","query":"...","answer":"...","evidence":"..."}}]}}

Article title: {title}
Source passage:
{passage}"""


def generation_batch_record(record_id: str, record: Mapping[str, object]) -> dict[str, object]:
    if not record_id:
        raise ValueError("record id is required")
    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "system": "You create grounded bilingual retrieval data and output strict JSON.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": generation_prompt(record)}]}],
            "max_tokens": 700,
            "temperature": 0.1,
        },
    }


def parse_model_json(text: str) -> Mapping[str, object]:
    cleaned = _FENCE.sub("", text.strip()).strip()
    start, stop = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or stop <= start:
        raise ValueError("model response does not contain a JSON object")
    value = json.loads(cleaned[start : stop + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def validate_generated_items(
    payload: Mapping[str, object],
    source: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return admitted items or reject the complete response.

    Complete-response rejection prevents one malformed item from silently
    changing the intended English/Tagalog/query-type balance.
    """

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != len(QUERY_TYPES):
        raise ValueError("response must contain exactly three items")
    by_type: dict[str, Mapping[str, object]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("generated item must be an object")
        query_type = str(item.get("query_type", ""))
        if query_type not in QUERY_TYPES or query_type in by_type:
            raise ValueError("query types must appear exactly once")
        by_type[query_type] = item

    passage = normalize_text(str(source["text"]))
    title = normalize_text(str(source["source_title"])).casefold()
    admitted: list[dict[str, object]] = []
    normalized_queries: set[str] = set()
    for query_type in QUERY_TYPES:
        item = by_type[query_type]
        query = normalize_text(str(item.get("query", "")))
        answer = normalize_text(str(item.get("answer", "")))
        evidence = normalize_text(str(item.get("evidence", "")))
        if not query or not answer or len(evidence) < 12:
            raise ValueError("query, answer and substantive evidence are required")
        if len(query.split()) > 24 or len(answer.split()) > 35 or len(evidence.split()) > 35:
            raise ValueError("query, answer or evidence exceeds its frozen word limit")
        if evidence.casefold() not in passage.casefold():
            raise ValueError("evidence is not a verbatim normalized source substring")
        normalized_query = query.casefold()
        if normalized_query in normalized_queries or normalized_query == title:
            raise ValueError("query is duplicated or merely repeats the title")
        if any(phrase in normalized_query for phrase in ("the passage", "the text", "the article", "ang talata", "ang teksto", "ang artikulo")):
            raise ValueError("query contains a forbidden source-reference phrase")
        normalized_queries.add(normalized_query)
        admitted.append(
            {
                "language": str(source["language"]),
                "query_type": query_type,
                "query": query,
                "answer": answer,
                "evidence": evidence,
                "positive_passage_id": str(source["passage_id"]),
                "positive_article_id": str(source["source_article_id"]),
                "source_text_sha256": str(source["text_sha256"]),
            }
        )
    return admitted
