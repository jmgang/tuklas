#!/usr/bin/env python3
"""Prepare immutable Sonnet 4.5 batch input from approved external text only."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.synthetic_retrieval import choose_language_passages, generation_batch_record


SOURCES = {
    "en": (ROOT / "runs/multiwikiqa-training-text-20261004-v5/training.en.jsonl", "b5169afeea3a24b0eed905deef610c30d74b69426b0dc34a66d1d9b514c80a1e"),
    "tl": (ROOT / "runs/multiwikiqa-training-text-20261004-v5/training.tl.jsonl", "55d54eb2ad72d89dfeaafaf5f021049e51511173d271d0e4d0659d33445d2404"),
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-language", type=int, default=2000)
    parser.add_argument("--seed", default="multiwikiqa-synthetic-retrieval-v1")
    parser.add_argument("--languages", nargs="+", choices=("en", "tl"), default=("en", "tl"))
    parser.add_argument("--exclude-selected", action="append", type=Path, default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse existing output: {output}")

    rows = []
    source_receipts = {}
    for language, (path, expected) in SOURCES.items():
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"approved {language} source checksum changed")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["language"] != language:
                    raise ValueError("source language mismatch")
                rows.append(row)
                count += 1
        source_receipts[language] = {"path": str(path.relative_to(ROOT)), "sha256": actual, "rows": count, "bytes": path.stat().st_size}

    excluded_articles = set()
    exclusion_receipts = []
    for raw_path in args.exclude_selected:
        path = raw_path.resolve()
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                excluded_articles.add((str(row["language"]), str(row["source_article_id"])))
                count += 1
        exclusion_receipts.append({
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path), "rows": count,
        })
    selected = choose_language_passages(
        rows,
        languages=args.languages,
        per_language=args.per_language,
        seed=args.seed,
        excluded_articles=excluded_articles,
    )
    output.mkdir(parents=True, exist_ok=False)
    selected_path = output / "selected-passages.jsonl"
    batch_path = output / "claude-sonnet-4-5-generation-input.jsonl"
    with selected_path.open("x", encoding="utf-8") as selected_stream, batch_path.open("x", encoding="utf-8") as batch_stream:
        for index, source in enumerate(selected, start=1):
            record_id = f"synx{index:07d}"
            selected_stream.write(json.dumps({"record_id": record_id, **source}, ensure_ascii=False, sort_keys=True) + "\n")
            batch_stream.write(json.dumps(generation_batch_record(record_id, source), ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "protocol": "multiwikiqa-synthetic-retrieval-v1",
        "status": "prepared-not-uploaded-not-submitted",
        "benchmark_questions_answers_parents_used": False,
        "source_kind": "previously-approved-external-wikipedia-training-text",
        "seed": args.seed,
        "per_language": args.per_language,
        "selected_languages": list(args.languages),
        "source_passages": len(selected),
        "maximum_generated_pairs_after_validation": len(selected) * 3,
        "query_types": ["natural_question", "keyword_query", "factual_claim"],
        "quality_rules": {
            "verbatim_normalized_evidence_required": True,
            "second_model_review": False,
            "three_types_or_reject_generation_response": True,
            "teacher_positive_aware_negative_mining_pending": True,
        },
        "bedrock": {
            "generator_model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "api": "batch-inference-invoke-model",
            "region": "ap-southeast-1",
            "generator_maximum_output_tokens_per_source": 700,
            "generator_batch_input_usd_per_million_tokens": 1.50,
            "generator_batch_output_usd_per_million_tokens": 7.50,
            "maximum_output_only_cost_estimate_usd": len(selected) * 700 / 1_000_000 * 7.50,
            "estimated_input_tokens_from_utf8_bytes_divided_by_four": batch_path.stat().st_size / 4,
            "estimated_input_cost_usd": batch_path.stat().st_size / 4 / 1_000_000 * 1.50,
            "credits_verified": False,
        },
        "sources": source_receipts,
        "excluded_prior_selected_sources": exclusion_receipts,
        "artifacts": {
            selected_path.name: {"bytes": selected_path.stat().st_size, "sha256": digest(selected_path)},
            batch_path.name: {"bytes": batch_path.stat().st_size, "sha256": digest(batch_path)},
        },
    }
    write_new(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
