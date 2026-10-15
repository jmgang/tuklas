"""Create immutable bilingual vocabularies for the sub-100 MB students.

This does not train or evaluate a model.  It counts only the already-approved
external English/Tagalog text and the frozen MultiWikiQA candidate passages.
Questions are read solely for an out-of-vocabulary audit and never influence
selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import regex

from pycon_model2vec_demo.multiwikiqa_compact_retrieval import select_balanced_vocabulary


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = regex.compile(r"\w+|[^\w\s]+")
SOURCES = {
    "external_en": ROOT / "runs/multiwikiqa-training-text-20261004-v5/training.en.jsonl",
    "external_tl": ROOT / "runs/multiwikiqa-training-text-20261004-v5/training.tl.jsonl",
    "approved": ROOT / "runs/multiwikiqa-preparation-20261004-v1/chunks.candidate.jsonl",
    "questions_audit_only": ROOT / "runs/multiwikiqa-preparation-20261004-v1/annotations.all.jsonl",
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, sort_keys=True, indent=2)
        target.write("\n")


def count_sources() -> tuple[Counter[str], Counter[str], dict[str, int]]:
    counts = {"en": Counter(), "tl": Counter()}
    rows = Counter()
    for name in ("external_en", "external_tl", "approved"):
        with SOURCES[name].open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                language = row.get("language") or name.rsplit("_", 1)[-1]
                if language not in counts:
                    raise ValueError(f"unexpected language in {name}: {language}")
                counts[language].update(tokens(row["text"]))
                rows[name] += 1
    return counts["en"], counts["tl"], dict(rows)


def coverage(vocabulary: set[str]) -> dict[str, object]:
    totals = Counter()
    covered = Counter()
    by_partition: dict[str, Counter[str]] = {}
    with SOURCES["questions_audit_only"].open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = f"{row['partition']}:{row['language']}"
            by_partition.setdefault(key, Counter())
            values = tokens(row["question"])
            totals[key] += len(values)
            covered[key] += sum(value in vocabulary for value in values)
            by_partition[key]["questions"] += 1
    return {
        key: {
            "questions": by_partition[key]["questions"],
            "tokens": totals[key],
            "covered_tokens": covered[key],
            "token_coverage": covered[key] / totals[key] if totals[key] else 0.0,
        }
        for key in sorted(totals)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)

    english, tagalog, row_counts = count_sources()
    variants = {
        "quality": {"requested_custom_tokens": 170_000, "expected_final_token_ceiling": 180_000},
        "preferred_30mb": {"requested_custom_tokens": 56_000, "expected_final_token_ceiling": 64_000},
    }
    reports = {}
    for name, settings in variants.items():
        vocabulary = select_balanced_vocabulary(
            english,
            tagalog,
            size=settings["requested_custom_tokens"],
            shared_fraction=0.25,
        )
        path = output / f"vocabulary-{name}.json"
        write_new(path, vocabulary)
        reports[name] = {
            **settings,
            "actual_custom_tokens": len(vocabulary),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "question_coverage_audit_only": coverage(set(vocabulary)),
        }

    # The Tagalog-only experiment is a separate, explicitly monolingual arm.
    # Keep deterministic frequency ordering and never use English counts or
    # evaluation questions to select its vocabulary.
    ordered_tagalog = [
        token for token, _ in sorted(tagalog.items(), key=lambda item: (-item[1], item[0]))
    ]
    for size in (80_000, 160_000):
        tagalog_only = ordered_tagalog[:size]
        if len(tagalog_only) != size or len(set(tagalog_only)) != size:
            raise ValueError(f"Tagalog sources did not produce {size} unique tokens")
        tagalog_path = output / f"vocabulary-tagalog-only-{size}.json"
        write_new(tagalog_path, tagalog_only)
        reports[f"tagalog_only_{size}"] = {
            "requested_custom_tokens": size,
            "actual_custom_tokens": len(tagalog_only),
            "selection_languages": ["tl"],
            "english_counts_used": False,
            "questions_used_for_selection": False,
            "sha256": digest(tagalog_path),
            "bytes": tagalog_path.stat().st_size,
            "question_coverage_audit_only": {
                key: value
                for key, value in coverage(set(tagalog_only)).items()
                if key.endswith(":tl")
            },
        }

    manifest = {
        "protocol": "multiwikiqa-balanced-vocabulary-v1",
        "selection_inputs": ["external_en", "external_tl", "approved"],
        "questions_used_for_selection": False,
        "questions_used_for_coverage_audit_only": True,
        "source_rows": row_counts,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in SOURCES.items()
        },
        "english_unique_tokens": len(english),
        "tagalog_unique_tokens": len(tagalog),
        "variants": reports,
    }
    write_new(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
