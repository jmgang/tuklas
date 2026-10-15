#!/usr/bin/env python3
"""Materialize replacement bilingual tokenizers and measured size budgets."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.compact_word_model import build_compact_word_tokenizer, unknown_fraction
from pycon_model2vec_demo.multiwikiqa_compact_retrieval import artifact_budget


SOURCE = ROOT / "runs/multiwikiqa-balanced-vocabulary-20261005-v1"
SOURCE_EXPECTED = {
    "vocabulary-quality.json": "e83ff2262da7fdf69d814029cdc5b867ccfae3ee0aff112acae22d19c7e49ac9",
    "vocabulary-preferred_30mb.json": "410c42269311e1fb17a319ec8b6240471b3271ad0e9ef4ba774a8ccbe0fffe79",
}
AUDIT_QUESTIONS = ROOT / "runs/multiwikiqa-preparation-20261004-v1/annotations.all.jsonl"
AUDIT_QUESTIONS_SHA = "5e4cb1d6550aa74ad5f3d9a1269b079a5077094311b1a5c752dd45f8037aef7d"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    if digest(AUDIT_QUESTIONS) != AUDIT_QUESTIONS_SHA:
        raise ValueError("question audit source changed")
    questions = []
    with AUDIT_QUESTIONS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            questions.append(row["question"])
    variants = {
        "quality": {"source": "vocabulary-quality.json", "dimension": 480},
        "preferred_30mb": {"source": "vocabulary-preferred_30mb.json", "dimension": 384},
    }
    reports = {}
    for name, settings in variants.items():
        source = SOURCE / settings["source"]
        if digest(source) != SOURCE_EXPECTED[settings["source"]]:
            raise ValueError(f"frozen vocabulary checksum changed: {source.name}")
        vocabulary = json.loads(source.read_text(encoding="utf-8"))
        tokenizer = build_compact_word_tokenizer(vocabulary)
        path = output / f"tokenizer-{name}.json"
        if path.exists():
            raise FileExistsError(path)
        tokenizer.save(str(path), pretty=False)
        missing, total, fraction = unknown_fraction(tokenizer, questions)
        budget = artifact_budget(
            vocab_size=tokenizer.get_vocab_size(with_added_tokens=True),
            dimension=int(settings["dimension"]),
            dtype="int8",
            tokenizer_bytes=path.stat().st_size,
            metadata_bytes=250_000,
            reference_artifact_bytes=131_200_180,
        )
        if budget.complete_artifact_bytes >= 100_000_000:
            raise ValueError(f"{name} cannot pass the strict 100MB package gate")
        reports[name] = {
            "vocabulary_source": str(source.relative_to(ROOT)),
            "vocabulary_source_sha256": digest(source),
            "tokenizer_path": str(path.relative_to(ROOT)),
            "tokenizer_sha256": digest(path),
            "tokenizer_bytes": path.stat().st_size,
            "vocabulary_size_including_specials": tokenizer.get_vocab_size(with_added_tokens=True),
            "trained_export_dimension": int(settings["dimension"]),
            "export_dtype": "int8",
            "question_tokens_audit_only": total,
            "question_unknown_tokens_audit_only": missing,
            "question_unknown_fraction_audit_only": fraction,
            "conservative_complete_artifact_budget_bytes": budget.complete_artifact_bytes,
            "hard_maximum_complete_artifact_bytes": 100_000_000,
        }
    write_new(output / "manifest.json", {
        "protocol": "multiwikiqa-compact-word-tokenizer-v2",
        "status": "prepared-not-trained",
        "supersedes_size_assumption_only": "multiwikiqa-balanced-vocabulary-v1 expected-final-token-ceiling; vocabulary lists remain unchanged",
        "model2vec_vocabulary_argument_not_used": True,
        "reason": "Model2Vec vocabulary expansion retains SEA-LION's 249999-token base and cannot satisfy the package gate",
        "question_text_used_for_coverage_audit_only": True,
        "question_text_used_for_vocabulary_selection_or_training": False,
        "variants": reports,
    })
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
