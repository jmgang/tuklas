"""Auditable MultiWikiQA preparation; no model calls or training input creation."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence


def digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def chunk_article(parent: Mapping, *, size: int, overlap: int) -> list[dict]:
    """Character boundaries depend on source text only, not any answer location."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require 0 <= overlap < positive chunk size")
    text = parent["text"]
    if not isinstance(text, str) or not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append({"chunk_id": f'{parent["parent_id"]}:{start}:{end}',
                       "parent_id": parent["parent_id"], "language": parent["language"],
                       "start": start, "end": end, "text": text[start:end]})
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _components(parents: Sequence[Mapping]) -> dict[str, str]:
    """Group articles with exactly equal raw contexts; retain all parent IDs."""
    by_content = defaultdict(list)
    for parent in parents:
        by_content[parent["text_sha256"]].append(parent["parent_id"])
    result = {}
    for ids in by_content.values():
        component = digest(sorted(ids))
        result.update({parent_id: component for parent_id in ids})
    return result


def prepare(rows_by_language: Mapping[str, Sequence[Mapping]], config: Mapping) -> dict:
    """Keep original rows externally; derive a structural-only candidate cohort."""
    if set(rows_by_language) != set(config["languages"]):
        raise ValueError("Unexpected or missing language file")
    development_fraction = config["split"]["development_fraction"]
    if not 0 < development_fraction < 1:
        raise ValueError("Development fraction must be between zero and one")
    parents_by_id, annotations = {}, []
    counts = Counter()
    seen = {}
    for language in config["languages"]:
        for row_index, row in enumerate(rows_by_language[language]):
            required = {"id", "title", "context", "question", "answers"}
            if not required <= row.keys():
                raise ValueError(f"Unexpected schema at {language} row {row_index}")
            if any(not isinstance(row[field], str) for field in required - {"answers"}):
                raise ValueError(f"Non-string text at {language} row {row_index}")
            url, text, question = row["id"], row["context"], row["question"]
            if not url.startswith(f"https://{language}.wikipedia.org/wiki/"):
                raise ValueError(f"Unexpected source URL at {language} row {row_index}")
            parent_id = f"{language}:{digest(url)}"
            content_hash = sha256(text.encode("utf-8")).hexdigest()
            parent = {"parent_id": parent_id, "language": language, "source_url": url,
                      "title": row["title"], "text": text, "text_sha256": content_hash}
            prior = parents_by_id.setdefault(parent_id, parent)
            if prior["text"] != text or prior["title"] != row["title"]:
                raise ValueError(f"One source identity has conflicting content: {url}")
            answers = row["answers"]
            if (not isinstance(answers, dict) or set(answers) != {"text", "answer_start"}
                    or not isinstance(answers["text"], list)
                    or not isinstance(answers["answer_start"], list)
                    or len(answers["text"]) != len(answers["answer_start"])):
                raise ValueError(f"Malformed answers at {language} row {row_index}")
            query_id = f"{language}:row-{row_index:06d}:{digest(row)}"
            reasons, spans = [], []
            if not question.strip():
                reasons.append("empty-question")
            if not text.strip():
                reasons.append("empty-context")
            if not answers["text"]:
                reasons.append("no-answer-reference")
            for answer, start in zip(answers["text"], answers["answer_start"]):
                if not isinstance(answer, str) or not isinstance(start, int) or isinstance(start, bool):
                    raise ValueError(f"Malformed answer type at {language} row {row_index}")
                if not answer.strip():
                    reasons.append("empty-answer-reference")
                elif start < 0 or text[start:start + len(answer)] != answer:
                    reasons.append("invalid-answer-offset")
                else:
                    spans.append({"text": answer, "start": start, "end": start + len(answer)})
            key = digest([language, parent_id, " ".join(question.split()).casefold(), spans,
                          answers if not spans else None])
            duplicate_of = seen.get(key)
            if duplicate_of:
                reasons.append("duplicate-qa-annotation")
            else:
                seen[key] = query_id
            annotations.append({"query_id": query_id, "language": language,
                                "source_row_index": row_index, "question": question,
                                "gold_parent_id": parent_id, "source_url": url,
                                "answer_spans": spans, "exclusion_reasons": sorted(set(reasons)),
                                "duplicate_of": duplicate_of, "semantic_review": "pending"})
            counts[f"raw_{language}"] += 1
    parents = sorted(parents_by_id.values(), key=lambda item: item["parent_id"])
    components = _components(parents)
    chunks, by_parent = [], {}
    for parent in parents:
        parent["component_id"] = components[parent["parent_id"]]
        key = digest([config["split"]["seed"], parent["component_id"]])
        # Identical components get identical deterministic assignments.
        parent["partition"] = ("development" if int(key, 16) / 2**256 < development_fraction
                               else "evaluation")
        pieces = chunk_article(parent, size=config["chunks"]["max_characters"],
                               overlap=config["chunks"]["overlap_characters"])
        chunks.extend(pieces)
        by_parent[parent["parent_id"]] = pieces
        counts[f'parents_{parent["language"]}'] += 1
        counts[f'parents_{parent["partition"]}_{parent["language"]}'] += 1
    parent_index = {parent["parent_id"]: parent for parent in parents}
    review_queue = []
    for query in annotations:
        parent = parent_index[query["gold_parent_id"]]
        query["partition"] = parent["partition"]
        query["component_id"] = parent["component_id"]
        supporting = [piece["chunk_id"] for piece in by_parent[query["gold_parent_id"]]
                      if any(piece["start"] <= span["start"] and span["end"] <= piece["end"]
                             for span in query["answer_spans"])]
        query["answer_bearing_chunk_ids"] = supporting
        for reason in query["exclusion_reasons"]:
            counts[f"reason_{reason}_{query['language']}"] += 1
        if query["exclusion_reasons"]:
            counts[f"excluded_{query['language']}"] += 1
            continue
        counts[f"eligible_{query['language']}"] += 1
        counts[f"eligible_{query['partition']}_{query['language']}"] += 1
        if not supporting:
            counts[f"answer_not_contained_in_single_chunk_{query['language']}"] += 1
        contexts = [piece["text"] for piece in by_parent[query["gold_parent_id"]]
                    if piece["chunk_id"] in supporting]
        review_queue.append({"query_id": query["query_id"], "language": query["language"],
                             "source_url": query["source_url"], "title": parent["title"],
                             "question": query["question"], "answer_spans": query["answer_spans"],
                             "evidence_chunks": contexts, "semantic_review": "pending"})
    eligible = [query for query in annotations if not query["exclusion_reasons"]]
    return {"parents": parents, "chunks": chunks, "annotations": annotations,
            "candidate_queries": eligible, "review_queue": review_queue,
            "audit": {"counts": dict(sorted(counts.items())), "raw_queries": len(annotations),
                      "candidate_queries": len(eligible), "parents": len(parents),
                      "chunks": len(chunks), "components": len(set(components.values())),
                      "semantically_approved_queries": 0,
                      "model_scoring_ready": False,
                      "structural_rules": "declared-in-config-and-source-no-model-results-used",
                      "limitations": ["synthetic questions require semantic review",
                                      "candidate character layout requires tokenizer preflight",
                                      "one known source parent, not exhaustive relevance",
                                      "separate languages, not natural Taglish",
                                      "near-duplicate and translated articles not fully audited"]}}


def score_single_gold(query: Mapping, ranked_parent_ids: Sequence[str],
                      returned_chunks: Mapping[str, Mapping]) -> dict[str, float]:
    """One source gold; evidence coverage refers to actually delivered chunks."""
    if len(ranked_parent_ids) > 10 or len(set(ranked_parent_ids)) != len(ranked_parent_ids):
        raise ValueError("Rankings require at most ten unique parents")
    if query.get("exclusion_reasons"):
        raise ValueError("Structurally excluded query is not a scoring candidate")
    gold = query["gold_parent_id"]
    rank = ranked_parent_ids.index(gold) + 1 if gold in ranked_parent_ids else None
    output = {f"Recall@{k}": float(rank is not None and rank <= k) for k in (1, 3, 5, 10)}
    output.update({"MRR@10": 1.0 / rank if rank else 0.0,
                   "nDCG@10": 1.0 / math.log2(rank + 1) if rank else 0.0})
    for parent_id, chunk in returned_chunks.items():
        if parent_id not in ranked_parent_ids or chunk["parent_id"] != parent_id:
            raise ValueError("Returned chunk does not belong to its ranked parent")
        if chunk["start"] < 0 or chunk["end"] <= chunk["start"]:
            raise ValueError("Invalid returned source offsets")
    piece = returned_chunks.get(gold)
    covered = bool(piece and any(piece["start"] <= span["start"]
                                and span["end"] <= piece["end"]
                                for span in query["answer_spans"]))
    for k in (1, 3, 5, 10):
        output[f"AnswerSpanCoverage@{k}"] = float(covered and rank is not None and rank <= k)
    return output
