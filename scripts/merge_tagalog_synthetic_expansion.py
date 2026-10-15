#!/usr/bin/env python3
"""Merge the preserved Tagalog control and expansion query-pair cohorts.

The output is immutable and auditable.  Exact duplicate queries are preserved
in a separate rejection receipt so the ranking worker never receives the same
query twice.  English rows and overlapping source articles are hard failures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalized_query(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def verified_rows(admitted: Path, report: Path) -> list[dict[str, object]]:
    receipt = json.loads(report.read_text(encoding="utf-8"))
    if digest(admitted) != receipt["admitted_sha256"]:
        raise ValueError(f"admitted checksum mismatch: {admitted}")
    rows = [json.loads(line) for line in admitted.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_rows = int(receipt.get("admitted_query_pairs", receipt.get("admitted_pairs", -1)))
    if len(rows) != expected_rows:
        raise ValueError(f"admitted row-count mismatch: {admitted}")
    return rows


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-admitted", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--expansion-admitted", type=Path, required=True)
    parser.add_argument("--expansion-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-safely-mined", type=int, default=12_000)
    args = parser.parse_args()

    control = verified_rows(args.control_admitted.resolve(), args.control_report.resolve())
    expansion = verified_rows(args.expansion_admitted.resolve(), args.expansion_report.resolve())
    control_tl = [row for row in control if row.get("language") == "tl"]
    if any(row.get("language") != "tl" for row in expansion):
        raise ValueError("expansion contains a non-Tagalog row")

    control_articles = {str(row["positive_article_id"]) for row in control_tl}
    expansion_articles = {str(row["positive_article_id"]) for row in expansion}
    overlap = control_articles & expansion_articles
    if overlap:
        raise ValueError(f"expansion reused {len(overlap)} control source articles")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    admitted_path = output / "admitted.jsonl"
    rejected_path = output / "rejected-duplicates.jsonl"
    seen: set[str] = set()
    admitted = Counter()
    rejected = Counter()
    with admitted_path.open("x", encoding="utf-8") as accepted, rejected_path.open("x", encoding="utf-8") as rejected_stream:
        for cohort, rows in (("control", control_tl), ("expansion", expansion)):
            for row in rows:
                key = normalized_query(str(row["query"]))
                if not key:
                    raise ValueError(f"empty normalized query in {cohort}")
                if key in seen:
                    rejected[cohort] += 1
                    rejected_stream.write(json.dumps({"cohort": cohort, "reason": "duplicate_normalized_query", "row": row}, ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                seen.add(key)
                accepted.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                admitted[cohort] += 1

    report = {
        "protocol": "multiwikiqa-tagalog-ranking-expansion-v1",
        "language": "tl",
        "control_input_pairs_all_languages": len(control),
        "control_input_pairs_tagalog": len(control_tl),
        "expansion_input_pairs_tagalog": len(expansion),
        "admitted_pairs": sum(admitted.values()),
        "admitted_by_cohort": dict(admitted),
        "duplicate_queries_rejected_by_cohort": dict(rejected),
        "control_source_articles": len(control_articles),
        "expansion_source_articles": len(expansion_articles),
        "source_article_overlap": 0,
        "admitted_sha256": digest(admitted_path),
        "rejected_sha256": digest(rejected_path),
        "minimum_safely_mined_pairs_required_downstream": args.minimum_safely_mined,
        "inputs": {
            "control_admitted": {"path": str(args.control_admitted.resolve()), "sha256": digest(args.control_admitted.resolve())},
            "control_report": {"path": str(args.control_report.resolve()), "sha256": digest(args.control_report.resolve())},
            "expansion_admitted": {"path": str(args.expansion_admitted.resolve()), "sha256": digest(args.expansion_admitted.resolve())},
            "expansion_report": {"path": str(args.expansion_report.resolve()), "sha256": digest(args.expansion_report.resolve())},
        },
    }
    write_json_new(output / "verification-report.json", report)
    print(json.dumps({"output": str(output), **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
