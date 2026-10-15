"""Prepare approved disjoint EN/TL training text, preserving immutable artifacts.

Downloads public source text only. No LLM calls, featurizing, training or AWS usage.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq
import requests
from pycon_model2vec_demo.wikipedia_training import (
    ContentGuard, HashSample, article_key, article_passages, url_key,
)


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_lines(path, records):
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def resolve_links(job):
    language, titles = job
    other = "en" if language == "tl" else "tl"
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "formatversion": 2,
              "prop": "langlinks|pageprops", "ppprop": "wikibase_item",
              "redirects": 1, "converttitles": 1, "lllang": other, "lllimit": "max",
              "titles": "\x1f" + "\x1f".join(titles)}
    headers = {"User-Agent": "PyConDavaoEmbeddingResearch/1.0 (noncommercial dataset audit; no editing)"}
    for attempt in range(4):
        response = requests.get(endpoint, params=params, headers=headers, timeout=(15, 60))
        if response.status_code == 429 and attempt < 3:
            delay = int(response.headers.get("Retry-After", 5 * (attempt + 1)))
            if delay > 300:
                raise ValueError(f"Wikipedia requests a {delay}s backoff; partial evidence preserved")
            print(json.dumps({"stage": "alias-lookup-rate-backoff", "language": language,
                              "seconds": max(1, delay)}), flush=True)
            remaining = max(1, delay)
            while remaining:
                pause = min(10, remaining)
                time.sleep(pause)
                remaining -= pause
            continue
        response.raise_for_status()
        break
    data = response.json()
    if "error" in data or "continue" in data:
        raise ValueError(f"Incomplete Wikipedia alias lookup: {data.get('error', data.get('continue'))}")
    if not isinstance(data.get("query", {}).get("pages"), list):
        raise ValueError("Wikipedia query page list missing")
    return {"language": language, "requested_titles": titles, "endpoint": endpoint,
            "parameters": params, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "response": data}


def exclusion_keys(parents, responses):
    keys = set()
    for parent in parents:
        keys.add(article_key(parent["language"], parent["title"]))
        keys.add(url_key(parent["source_url"]))
    linked, missing, resolved = 0, [], 0
    for response in responses:
        language = response["language"]
        query = response["response"]["query"]
        for category in ("normalized", "converted", "redirects"):
            for alias in query.get(category, []):
                keys.add(article_key(language, alias["from"]))
                keys.add(article_key(language, alias["to"]))
        for page in query["pages"]:
            keys.add(article_key(language, page["title"]))
            if page.get("missing") or page.get("invalid"):
                missing.append({"language": language, "title": page["title"]})
                continue
            resolved += 1
            for link in page.get("langlinks", []):
                if link["lang"] in {"en", "tl"}:
                    keys.add(article_key(link["lang"], link.get("title", link.get("*", ""))))
                    linked += 1
    return keys, {"resolved_live_pages": resolved, "linked_translation_entries": linked,
                  "unresolved_live_pages": missing,
                  "limitations": ["Live links can differ from the 2023 snapshot.",
                                  "Missing/deleted pages retain original title/URL exclusions.",
                                  "Unlinked translations and semantic paraphrases are not guaranteed detected."]}


def download(job):
    config, spec, path, reuse_root = job
    url = f'https://huggingface.co/datasets/{config["dataset"]}/resolve/{config["revision"]}/{spec["path"]}'
    checksum = sha256()
    size = 0
    old = reuse_root / "raw" / path.name if reuse_root else None
    if old and old.is_file() and old.stat().st_size == spec["bytes"] and digest(old) == spec["sha256"]:
        with old.open("rb") as source, path.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        method = "copied-verified-preserved-source"
        checksum_value = digest(path)
        size = path.stat().st_size
    else:
        parts_dir = path.parent / (path.name + ".range-parts")
        parts_dir.mkdir(exist_ok=False)
        width = 1024 * 1024
        intervals = [(start, min(start + width, spec["bytes"]) - 1)
                     for start in range(0, spec["bytes"], width)]
        old_ranges = []
        if old:
            old_parts = old.parent / (old.name + ".range-parts")
            if old_parts.is_dir():
                for part in sorted(old_parts.glob("*.bin")):
                    start, end = map(int, part.stem.split("-"))
                    if part.stat().st_size != end - start + 1:
                        raise ValueError("Saved byte range size differs")
                    old_ranges.append((start, end, part, digest(part)))
        def fetch(interval):
            start, end = interval
            expected_range = f'bytes {start}-{end}/{spec["bytes"]}'
            for old_start, old_end, old_part, _ in old_ranges:
                if old_start <= start and old_end >= end:
                    with old_part.open("rb") as source:
                        source.seek(start - old_start)
                        data = source.read(end - start + 1)
                    if len(data) != end - start + 1:
                        raise ValueError("Saved range does not contain requested bytes")
                    chunk = parts_dir / f"{start:012d}-{end:012d}.bin"
                    with chunk.open("xb") as stream:
                        stream.write(data)
                    return chunk
            for attempt in range(4):
                try:
                    # Query changes only the transport cache key, never the pinned source.
                    began = time.monotonic()
                    with requests.get(url + f"?download=true&range-start={start}&range-end={end}",
                            headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 20)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                            raise ValueError("Server returned a different byte range; refusing assembly")
                        blocks = []
                        for block in response.iter_content(65536):
                            if time.monotonic() - began > 30:
                                raise requests.Timeout("Range exceeded its wall-clock allowance")
                            blocks.append(block)
                        data = b"".join(blocks)
                        if len(data) != end - start + 1:
                            raise ValueError("Incomplete range response")
                    chunk = parts_dir / f"{start:012d}-{end:012d}.bin"
                    with chunk.open("xb") as stream:
                        stream.write(data)
                    return chunk
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(min(10, 2 * (attempt + 1)))
            raise AssertionError("Unreachable range-download state")
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(fetch, intervals))
        for _, _, old_part, old_hash in old_ranges:
            if digest(old_part) != old_hash:
                raise ValueError("Preserved byte range changed during import")
        with path.open("xb") as stream:
            for part in parts:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        stream.write(block)
                        checksum.update(block)
                        size += len(block)
        method = "verified-byte-range-assembly"
        checksum_value = checksum.hexdigest()
    if checksum_value != spec["sha256"] or size != spec["bytes"]:
        raise ValueError(f"Pinned source checksum/size differs: {spec['path']}; partial preserved")
    print(json.dumps({"stage": "source-verified", "file": spec["path"], "bytes": size}), flush=True)
    return {"path": spec["path"], "url": url, "bytes": size, "sha256": checksum_value,
            "transport": method, "preserved_source_run": str(reuse_root.relative_to(ROOT)) if reuse_root else None,
            "reused_byte_ranges": [{"path": str(p.relative_to(ROOT)), "sha256": h}
                                   for _, _, p, h in old_ranges] if method == "verified-byte-range-assembly" else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--reuse-alias-run", help="Read verified complete lookup batches from a preserved run; never modify it")
    parser.add_argument("--reuse-raw-run", help="Copy only fully checksummed source files from a preserved run")
    args = parser.parse_args()
    run_id = args.run_id or ("multiwikiqa-training-text-" +
                             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run identifier must be a single new directory name")
    config_path = ROOT / "configs/multiwikiqa-training-text-v1.json"
    config = json.loads(config_path.read_text())
    benchmark = ROOT / "runs" / config["benchmark_run"]
    manifest = json.loads((benchmark / "manifest.json").read_text())
    parents_path = benchmark / "parents.jsonl"
    if digest(parents_path) != manifest["artifacts_sha256"]["parents.jsonl"]:
        raise ValueError("Benchmark parent manifest differs")
    parents = read_lines(parents_path)
    alias_source = None
    if args.reuse_alias_run:
        if Path(args.reuse_alias_run).name != args.reuse_alias_run:
            raise ValueError("Alias source must be one project run name")
        alias_source = (ROOT / "runs" / args.reuse_alias_run).resolve()
        if not alias_source.is_relative_to(ROOT / "runs"):
            raise ValueError("Alias source escapes project runs")
        previous = json.loads((alias_source / "provenance.start.json").read_text())
        if previous["benchmark_parent_sha256"] != digest(parents_path):
            raise ValueError("Saved alias evidence used a different benchmark")
    raw_source = None
    if args.reuse_raw_run:
        if Path(args.reuse_raw_run).name != args.reuse_raw_run:
            raise ValueError("Raw source must be one project run name")
        raw_source = (ROOT / "runs" / args.reuse_raw_run).resolve()
        if not raw_source.is_relative_to(ROOT / "runs"):
            raise ValueError("Raw source escapes project runs")
    required = sum(spec["bytes"] for files in config["files"].values() for spec in files) + 2 * 1024**3
    if shutil.disk_usage(ROOT).free < required:
        raise ValueError("Not enough free space for preserved downloads and outputs; no cleanup attempted")
    directory = ROOT / "runs" / run_id
    directory.mkdir(exist_ok=False)
    (directory / "raw").mkdir()
    write_json(directory / "settings.json", config)
    write_json(directory / "provenance.start.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_authorization": "2026-10-04: yes to separate English/Tagalog Wikipedia training text excluding benchmark articles",
        "benchmark_parent_sha256": digest(parents_path),
        "benchmark_manifest_sha256": digest(benchmark / "manifest.json"),
        "config_sha256": digest(config_path), "runner_sha256": digest(Path(__file__)),
        "module_sha256": digest(ROOT / "src/pycon_model2vec_demo/wikipedia_training.py"),
        "no_aws_resources": True, "no_model_calls": True, "no_training": True,
        "reuse_alias_run": args.reuse_alias_run,
        "reuse_raw_run": args.reuse_raw_run,
    })
    jobs = []
    for language in config["files"]:
        titles = sorted({p["title"] for p in parents if p["language"] == language})
        jobs.extend((language, titles[start:start + 40]) for start in range(0, len(titles), 40))
    # Read-only public metadata; freeze responses before selecting training examples.
    responses = []
    reused = []
    (directory / "alias-responses").mkdir()
    for index, job in enumerate(jobs):
        source_path = alias_source / "alias-responses" / f"batch-{index:03d}.json" if alias_source else None
        if source_path and source_path.is_file():
            response = json.loads(source_path.read_text())
            if (response["language"], response["requested_titles"]) != job:
                raise ValueError("Saved alias batch differs from the frozen request")
            if "error" in response["response"] or "continue" in response["response"]:
                raise ValueError("Incomplete saved alias response")
            if not isinstance(response["response"].get("query", {}).get("pages"), list):
                raise ValueError("Saved alias page list missing")
            reused.append({"path": str(source_path.relative_to(ROOT)), "sha256": digest(source_path)})
        else:
            time.sleep(5)
            response = resolve_links(job)
        write_json(directory / "alias-responses" / f"batch-{index:03d}.json", response)
        responses.append(response)
        print(json.dumps({"stage": "alias-batch-saved", "completed": index + 1,
                          "total": len(jobs)}), flush=True)
    write_json(directory / "reused-alias-evidence.json", reused)
    write_lines(directory / "wikipedia-alias-responses.jsonl", responses)
    excluded, link_audit = exclusion_keys(parents, responses)
    write_json(directory / "article-exclusion-keys.json", [list(k) for k in sorted(excluded)])
    write_json(directory / "alias-audit.json", link_audit)
    print(json.dumps({"stage": "benchmark-exclusions-frozen", "benchmark_articles": len(parents),
                      "article_keys": len(excluded), "linked_entries": link_audit["linked_translation_entries"]}), flush=True)
    download_jobs = []
    for files in config["files"].values():
        for spec in files:
            filename = spec["path"].replace("/", "--")
            download_jobs.append((config, spec, directory / "raw" / filename, raw_source))
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(download, download_jobs))
    card = requests.get(f'https://huggingface.co/datasets/{config["dataset"]}/resolve/{config["revision"]}/README.md', timeout=(15, 60))
    card.raise_for_status()
    with (directory / "raw" / "README.source.md").open("xb") as stream:
        stream.write(card.content)
    write_json(directory / "source-provenance.json", {"sources": sources,
               "dataset": config["dataset"], "revision": config["revision"], "license": config["license"]})
    guard = ContentGuard(parents, config["overlap_guard"])
    counts = Counter()
    samples = {language: HashSample(config["sampling"]["maximum_passages_per_language"])
               for language in config["files"]}
    identities = set()
    content_seen = set()
    with (directory / "article-exclusions.jsonl").open("x", encoding="utf-8") as exclusions:
        for language, files in config["files"].items():
            for spec in files:
                path = directory / "raw" / spec["path"].replace("/", "--")
                row_index = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        index = row_index
                        row_index += 1
                        counts[f"source_articles_{language}"] += 1
                        if not all(isinstance(row.get(k), str) for k in ("id", "url", "title", "text")):
                            raise ValueError("Source schema differs")
                        identity = language + ":" + row["id"]
                        if identity in identities:
                            raise ValueError("Duplicate source article identity")
                        identities.add(identity)
                        key = url_key(row["url"])
                        if key[0] != language:
                            raise ValueError("Source language/URL differs")
                        reason, match = None, None
                        if key in excluded or article_key(language, row["title"]) in excluded:
                            reason = "benchmark-article-alias-or-translation"
                        elif not row["text"].strip():
                            reason = "empty-article"
                        else:
                            overlap = guard.find(row["text"])
                            if overlap:
                                reason, match = overlap
                        if reason:
                            counts[f"excluded_articles_{language}"] += 1
                            counts[f"reason_{reason}_{language}"] += 1
                            exclusions.write(json.dumps({"language": language, "source_article_id": identity,
                                "source_url": row["url"], "source_title": row["title"], "source_file": spec["path"],
                                "source_row_index": index, "reason": reason, "matched_benchmark_parent": match},
                                ensure_ascii=False, sort_keys=True) + "\n")
                            continue
                        counts[f"eligible_articles_{language}"] += 1
                        passages, before_cap = article_passages(language, row, spec["path"], index, config["sampling"])
                        counts[f"eligible_windows_before_article_cap_{language}"] += before_cap
                        counts[f"windows_after_article_cap_{language}"] += len(passages)
                        for priority, passage in passages:
                            dedup_key = (language, passage["text_sha256"])
                            if dedup_key in content_seen:
                                counts[f"duplicate_windows_{language}"] += 1
                                continue
                            content_seen.add(dedup_key)
                            samples[language].add(passage, priority)
                    if row_index % 10240 == 0:
                        print(json.dumps({"stage": "filtering", "language": language,
                                          "articles_processed": counts[f"source_articles_{language}"]}), flush=True)
                print(json.dumps({"stage": "source-filtered", "file": spec["path"], "rows": row_index}), flush=True)
    outputs = []
    for language, sample in samples.items():
        selected = sample.records()
        write_lines(directory / f"training.{language}.jsonl", selected)
        counts[f"selected_passages_{language}"] = len(selected)
        counts[f"selected_articles_{language}"] = len({p["source_article_id"] for p in selected})
        counts[f"selected_characters_{language}"] = sum(len(p["text"]) for p in selected)
        counts[f"selected_whitespace_words_{language}"] = sum(len(p["text"].split()) for p in selected)
        outputs.append({"language": language, "path": f"training.{language}.jsonl", "rows": len(selected)})
    audit = {"counts": dict(sorted(counts.items())), "benchmark_parents_excluded": len(parents),
             "benchmark_parent_sha256": digest(parents_path), "article_exclusion_keys": len(excluded),
             "aliases": link_audit, "outputs": outputs, "source_revision": config["revision"],
             "unlabeled_real_source_text": True, "qa_used_for_training": False,
             "existing_experiments_modified": False, "model_training_started": False,
             "tokenizer_preflight_complete": False, "teacher_selected": False,
             "sampling_limits": config["sampling"], "overlap_guard": config["overlap_guard"],
             "limitations": ["Wikipedia samples are not natural Taglish or parallel training pairs.",
                             "English uses three fixed shards, not a uniform sample of all English Wikipedia.",
                             "Lexical overlap and current translation links cannot rule out semantic leakage.",
                             "No claim that a pretrained teacher never saw these Wikipedia articles.",
                             "Training windows are character bounded; teacher tokenizer preflight remains required.",
                             "Benchmark QA semantic review and reader evaluation are not completed here."]}
    write_json(directory / "audit.json", audit)
    report = ["# Separate bilingual Wikipedia training text", "", "Data preparation only: no training, model calls or AWS resources.", "",
              "| Language | Source articles scanned | Selected articles | Training passages |",
              "| --- | ---: | ---: | ---: |"]
    for language in samples:
        report.append(f'| {language} | {counts[f"source_articles_{language}"]:,} | {counts[f"selected_articles_{language}"]:,} | {counts[f"selected_passages_{language}"]:,} |')
    report += ["", "All 1,294 benchmark source articles (both partitions) supplied the exclusion guard.",
               "Saved source URLs, original row indexes, source offsets and checksums accompany each selected passage.",
               "", "## Limitations", ""] + ["- " + item for item in audit["limitations"]]
    with (directory / "preparation-report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    artifacts = {str(path.relative_to(directory)): digest(path) for path in sorted(directory.rglob("*")) if path.is_file()}
    write_json(directory / "manifest.json", {"artifacts_sha256": artifacts,
               "completed_utc": datetime.now(timezone.utc).isoformat(), "data_preparation_only": True})
    print(json.dumps({"stage": "complete", "directory": str(directory), "counts": audit["counts"]}), flush=True)


if __name__ == "__main__":
    main()
