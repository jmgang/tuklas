"""Independent read-only integrity verification, with a new immutable receipt.

Original preparation artifacts are never changed. No cleanup or model execution.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq
from pycon_model2vec_demo.multiwikiqa import prepare


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def checksum(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(directory):
    manifest = load(directory / "manifest.json")
    config = load(directory / "settings.json")
    for relative, expected in manifest["artifacts_sha256"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Manifest path escapes run directory")
        if checksum(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    rows = {}
    for language, spec in config["files"].items():
        path = directory / "raw" / f"{language}.parquet"
        if path.stat().st_size != spec["bytes"] or checksum(path) != spec["sha256"]:
            raise ValueError(f"Raw source differs: {language}")
        rows[language] = pq.read_table(path).to_pylist()
    recomputed = prepare(rows, config)
    checks = {"parents": "parents.jsonl", "chunks": "chunks.candidate.jsonl",
              "annotations": "annotations.all.jsonl", "review_queue": "semantic-review.pending.jsonl"}
    for key, filename in checks.items():
        if recomputed[key] != lines(directory / filename):
            raise ValueError(f"Independent preparation mismatch: {filename}")
    if recomputed["audit"] != load(directory / "audit.json"):
        raise ValueError("Audit counts differ")
    for partition in ("development", "evaluation"):
        expected = [q for q in recomputed["candidate_queries"] if q["partition"] == partition]
        if lines(directory / f"queries.{partition}.candidate.jsonl") != expected:
            raise ValueError(f"Partition differs: {partition}")
    expected_exclusions = [q for q in recomputed["annotations"] if q["exclusion_reasons"]]
    if lines(directory / "structural-exclusions.jsonl") != expected_exclusions:
        raise ValueError("Exclusions differ")
    queries = recomputed["annotations"]
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Query identifiers are not unique")
    parents = {p["parent_id"]: p for p in recomputed["parents"]}
    by_parent = defaultdict(list)
    for chunk in recomputed["chunks"]:
        parent = parents[chunk["parent_id"]]
        if chunk["text"] != parent["text"][chunk["start"]:chunk["end"]]:
            raise ValueError("Chunk source text differs")
        by_parent[chunk["parent_id"]].append(chunk)
    for parent_id, parent in parents.items():
        covered_end = 0
        for chunk in by_parent[parent_id]:
            if chunk["start"] > covered_end:
                raise ValueError("Original article character was dropped")
            covered_end = max(covered_end, chunk["end"])
        if covered_end != len(parent["text"]):
            raise ValueError("Article tail was dropped")
    parts = defaultdict(set)
    for query in queries:
        parts[query["component_id"]].add(query["partition"])
    if any(len(values) != 1 for values in parts.values()):
        raise ValueError("Article/content component crosses partitions")
    repeated = defaultdict(list)
    for query in queries:
        key = (query["language"], query["gold_parent_id"],
               " ".join(query["question"].split()).casefold())
        repeated[key].append(query)
    conflicts = []
    for (language, parent_id, _), group in repeated.items():
        if len(group) < 2:
            continue
        answer_sets = {json.dumps(q["answer_spans"], ensure_ascii=False, sort_keys=True) for q in group}
        conflicts.append({"language": language, "parent_id": parent_id,
                          "question": group[0]["question"],
                          "query_ids": [q["query_id"] for q in group],
                          "source_row_indices": [q["source_row_index"] for q in group],
                          "different_references": len(answer_sets) > 1,
                          "answer_spans": [q["answer_spans"] for q in group],
                          "action": "review-pending-not-automatically-repaired-or-excluded"})
    uncovered = [{"query_id": q["query_id"], "language": q["language"],
                  "source_url": q["source_url"], "answer_spans": q["answer_spans"]}
                 for q in recomputed["candidate_queries"] if not q["answer_bearing_chunk_ids"]]
    return {"verified": True, "verified_artifacts": len(manifest["artifacts_sha256"]),
            "counts": recomputed["audit"], "raw_rows_preserved": True,
            "all_original_characters_covered": True, "one_source_parent_per_query": True,
            "article_partitions_disjoint": True, "no_model_results": True,
            "normalized_repeated_question_groups": conflicts,
            "candidate_chunks_not_covering_full_answer": uncovered,
            "model_scoring_ready": False,
            "remaining_gates": ["semantic QA review including conflicting references",
                                "multilingual teacher and tokenizer budget selection",
                                "separate training material selection"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Only inspect a project run directory")
    report = verify(directory)
    path = directory / ("verification-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        + "-" + uuid4().hex[:8] + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"verified": True, "receipt": str(path),
                      "counts": report["counts"],
                      "repeated_question_groups": len(report["normalized_repeated_question_groups"]),
                      "uncovered_full_answers": len(report["candidate_chunks_not_covering_full_answer"])}))


if __name__ == "__main__":
    main()
