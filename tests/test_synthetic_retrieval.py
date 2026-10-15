import json

import pytest

from pycon_model2vec_demo.synthetic_retrieval import (
    choose_balanced_passages,
    choose_language_passages,
    generation_batch_record,
    parse_model_json,
    validate_generated_items,
)


def _source(language="tl", passage_id="tl:a:0:500"):
    return {
        "language": language,
        "source_article_id": f"{language}:a",
        "source_title": "Halimbawa",
        "source_url": "https://example.invalid/a",
        "passage_id": passage_id,
        "text": "Si Maria ay nagtanim ng mangga sa Davao noong 1990. Lumaki ang puno sa tabi ng ilog. Namunga ito pagkaraan ng limang taon. " * 4,
        "text_sha256": "a" * 64,
        "start": 0,
    }


def test_selection_is_balanced_and_one_passage_per_article():
    rows = []
    for language in ("en", "tl"):
        for article in range(3):
            for part in range(2):
                row = _source(language, f"{language}:{article}:{part}")
                row["source_article_id"] = f"{language}:{article}"
                rows.append(row)
    selected = choose_balanced_passages(rows, per_language=2, seed="fixed")
    assert [row["language"] for row in selected].count("en") == 2
    assert [row["language"] for row in selected].count("tl") == 2
    assert len({(row["language"], row["source_article_id"]) for row in selected}) == 4


def test_tagalog_expansion_excludes_prior_articles():
    rows = []
    for article in range(5):
        row = _source("tl", f"tl:{article}:0")
        row["source_article_id"] = f"tl:{article}"
        rows.append(row)
    selected = choose_language_passages(
        rows,
        languages=["tl"],
        per_language=3,
        seed="expansion",
        excluded_articles={("tl", "tl:0"), ("tl", "tl:1")},
    )
    assert len(selected) == 3
    assert {row["language"] for row in selected} == {"tl"}
    assert not {row["source_article_id"] for row in selected} & {"tl:0", "tl:1"}


def test_batch_record_contains_no_benchmark_material_and_freezes_three_types():
    record = generation_batch_record("syn0000001", _source())
    encoded = json.dumps(record, ensure_ascii=False)
    assert record["recordId"] == "syn0000001"
    assert "natural_question" in encoded
    assert "keyword_query" in encoded
    assert "factual_claim" in encoded


def test_grounded_response_requires_exact_evidence_and_all_quality_scores():
    source = _source()
    payload = {
        "items": [
            {
                "query_type": kind,
                "query": f"Tanong {index}?",
                "answer": "Sa Davao noong 1990.",
                "evidence": "Si Maria ay nagtanim ng mangga sa Davao noong 1990.",
            }
            for index, kind in enumerate(("natural_question", "keyword_query", "factual_claim"), start=1)
        ]
    }
    admitted = validate_generated_items(payload, source)
    assert len(admitted) == 3
    assert {item["query_type"] for item in admitted} == {"natural_question", "keyword_query", "factual_claim"}

    payload["items"][0]["evidence"] = "Hindi ito nasa pinagmulan."
    with pytest.raises(ValueError, match="not a verbatim"):
        validate_generated_items(payload, source)


def test_json_parser_accepts_code_fence_but_not_prose_only():
    assert parse_model_json('```json\n{"items": []}\n```') == {"items": []}
    with pytest.raises(ValueError):
        parse_model_json("there is no object here")
