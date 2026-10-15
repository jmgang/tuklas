"""Read-only corpus verification; writes only a new, exclusive verification receipt."""
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
from pycon_model2vec_demo.wikipedia_training import (
    article_key, url_key, ContentGuard, text_hash, stable_priority, article_passages,
)


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, checksum in manifest["artifacts_sha256"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory) or digest(path) != checksum:
            raise ValueError(f"Artifact checksum/path mismatch: {relative}")
    config = json.loads((directory / "settings.json").read_text())
    provenance = json.loads((directory / "provenance.start.json").read_text())
    for evidence in json.loads((directory / "reused-alias-evidence.json").read_text()):
        source = (ROOT / evidence["path"]).resolve()
        if not source.is_relative_to(ROOT / "runs") or digest(source) != evidence["sha256"]:
            raise ValueError("Reused immutable alias evidence differs")
    source_provenance = json.loads((directory / "source-provenance.json").read_text())
    for source in source_provenance["sources"]:
        for evidence in source.get("reused_byte_ranges", []):
            part = (ROOT / evidence["path"]).resolve()
            if not part.is_relative_to(ROOT / "runs") or digest(part) != evidence["sha256"]:
                raise ValueError("Preserved imported byte range differs")
    benchmark = ROOT / "runs" / config["benchmark_run"]
    if digest(benchmark / "parents.jsonl") != provenance["benchmark_parent_sha256"]:
        raise ValueError("Benchmark corpus differs")
    if digest(benchmark / "manifest.json") != provenance["benchmark_manifest_sha256"]:
        raise ValueError("Benchmark manifest differs")
    parents = list(records(benchmark / "parents.jsonl"))
    guard = ContentGuard(parents, config["overlap_guard"])
    original_keys = {key for p in parents for key in
                     (article_key(p["language"], p["title"]), url_key(p["source_url"]))}
    expected_keys = set(original_keys)
    for response in records(directory / "wikipedia-alias-responses.jsonl"):
        language = response["language"]
        query = response["response"]["query"]
        for category in ("normalized", "converted", "redirects"):
            for alias in query.get(category, []):
                expected_keys.add(article_key(language, alias["from"]))
                expected_keys.add(article_key(language, alias["to"]))
        for page in query["pages"]:
            expected_keys.add(article_key(language, page["title"]))
            for link in page.get("langlinks", []):
                if link["lang"] in {"en", "tl"}:
                    expected_keys.add(article_key(link["lang"], link.get("title", link.get("*", ""))))
    frozen_keys = {tuple(key) for key in json.loads((directory / "article-exclusion-keys.json").read_text())}
    if frozen_keys != expected_keys:
        raise ValueError("Frozen title/translation exclusions differ from saved API evidence")
    audit = json.loads((directory / "audit.json").read_text())
    counts = Counter()
    per_article = Counter()
    seen_ids, seen_hashes = set(), set()
    selected_by_source = defaultdict(lambda: defaultdict(list))
    for language in config["files"]:
        previous = None
        for row in records(directory / f"training.{language}.jsonl"):
            if row["language"] != language or "question" in row or "answer" in row or "answers" in row:
                raise ValueError("Training schema includes labels or wrong language")
            if row["passage_id"] in seen_ids:
                raise ValueError("Duplicate training passage identity")
            seen_ids.add(row["passage_id"])
            key = (language, row["text_sha256"])
            if key in seen_hashes or row["text_sha256"] != text_hash(row["text"]):
                raise ValueError("Duplicate or mismatched normalized training text hash")
            seen_hashes.add(key)
            if url_key(row["source_url"]) in frozen_keys or article_key(language, row["source_title"]) in frozen_keys:
                raise ValueError("Benchmark source/alias/translation appears in training")
            priority = (stable_priority(config["sampling"]["seed"], row["passage_id"]), row["passage_id"])
            if previous is not None and priority < previous:
                raise ValueError("Selected records not in stable-hash order")
            previous = priority
            if not (config["sampling"]["minimum_passage_characters"] <= len(row["text"].strip()) <=
                    config["sampling"]["maximum_passage_characters"]):
                raise ValueError("Training passage length outside frozen bounds")
            counts[f"selected_passages_{language}"] += 1
            counts[f"selected_characters_{language}"] += len(row["text"])
            counts[f"selected_whitespace_words_{language}"] += len(row["text"].split())
            per_article[row["source_article_id"]] += 1
            selected_by_source[row["source_file"]][row["source_row_index"]].append(row)
        counts[f"selected_articles_{language}"] = sum(identity.startswith(language + ":") for identity in per_article)
        if counts[f"selected_passages_{language}"] > config["sampling"]["maximum_passages_per_language"]:
            raise ValueError("Language sample cap exceeded")
        if counts[f"selected_passages_{language}"] == 0:
            raise ValueError("Missing selected language")
    if max(per_article.values()) > config["sampling"]["maximum_passages_per_article"]:
        raise ValueError("Article sample cap exceeded")
    verified_source_rows = 0
    source_paths = set()
    for language, files in config["files"].items():
        for spec in files:
            source_paths.add(spec["path"])
            path = directory / "raw" / spec["path"].replace("/", "--")
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError("Pinned raw Wikipedia source differs")
            pending = selected_by_source.get(spec["path"], {})
            verified = set()
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for source in batch.to_pylist():
                    selected = pending.get(index)
                    if selected:
                        if guard.find(source["text"]):
                            raise ValueError("Detected benchmark content overlap in selected source article")
                        candidates, _ = article_passages(language, source, spec["path"], index, config["sampling"])
                        expected = {record["passage_id"]: record for _, record in candidates}
                        offsets = sorted((r["start"], r["end"]) for r in selected)
                        if any(a[1] > b[0] for a, b in zip(offsets, offsets[1:])):
                            raise ValueError("Overlapping selected training windows")
                        for row in selected:
                            if expected.get(row["passage_id"]) != row:
                                raise ValueError("Selected record differs from original row/source slice/article cap")
                        verified.add(index)
                        verified_source_rows += 1
                    index += 1
            if verified != set(pending):
                raise ValueError("Selected source row index does not exist")
            if index != pq.ParquetFile(path).metadata.num_rows:
                raise ValueError("Source row scan incomplete")
            print(json.dumps({"stage": "source-slices-verified", "source": spec["path"],
                              "articles_scanned": index, "selected_articles": len(verified)}), flush=True)
    if not set(selected_by_source).issubset(source_paths):
        raise ValueError("Unexpected source file in selected training text")
    for key, value in counts.items():
        if audit["counts"].get(key) != value:
            raise ValueError(f"Audit count mismatch: {key}")
    return {"verified": True, "verified_artifacts": len(manifest["artifacts_sha256"]),
            "selected_counts": dict(counts), "verified_original_source_articles": verified_source_rows,
            "benchmark_parents_guarded": len(parents), "detected_benchmark_overlap": 0,
            "duplicates_per_language": 0, "original_source_offsets_verified": True,
            "linked_translation_evidence_verified": True, "article_caps_verified": True,
            "label_free_schema_verified": True, "no_training_performed": True,
            "selection_verification": "source slice and per-article hash selection verified; global sample order/caps verified, whole bottom-k pool not replayed",
            "limitations": audit["limitations"], "teacher_tokenizer_preflight_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Only verify within project runs")
    result = verify(directory)
    receipt = directory / ("verification-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"verified": True, "receipt": str(receipt), "counts": result["selected_counts"]}), flush=True)


if __name__ == "__main__":
    main()
