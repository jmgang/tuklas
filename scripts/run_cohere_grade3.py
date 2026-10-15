"""Add Bedrock Cohere embeddings to the pinned full-corpus grade-3 benchmark.

Every paid batch is saved exclusively and verified before reuse. All source
passages and all mapped grade-3 golds remain in the evaluation. Nothing is
deleted. Published vectors and reports are never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if os.environ.get("COHERE_BOTO3_PATH"):
    sys.path.insert(0, os.environ["COHERE_BOTO3_PATH"])

import numpy as np

from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere
from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.grade3 import TOP10_METRICS, paired_with_teacher, score_top10_query, select_cohorts, summarize
from pycon_model2vec_demo.measurement import _timing_summary, host_metadata, peak_rss_bytes
from pycon_model2vec_demo.model_registry import MODEL_NAMES, local_model_artifacts
from pycon_model2vec_demo.retrieval import exact_top_k


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing JSON differs: {path}")
        return
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def event(directory: Path, value: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    save(directory / "events" / f"{time.time_ns()}-{uuid4().hex[:8]}.json", {"time_utc": now, **value})
    print(json.dumps({"time_utc": now, **value}, sort_keys=True), flush=True)


def prepare(config: dict) -> tuple:
    data = read(ROOT / "configs/disastir-qa-all-categories.json")
    for key in ("dataset", "revision", "corpus_sha256"):
        if config[key] != data[key]:
            raise ValueError(f"Pinned dataset differs: {key}")
    reference = ROOT / config["reference_run"]
    cohort = read(reference / "cohort.json")
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=config["corpus_sha256"])
    prepared = load_qa_collection(data_dir=ROOT / "data/raw", revision=data["revision"], category_checksums=data["categories"], seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"])
    mapped, _ = map_query_groups(prepared.evaluation, corpus)
    _, queries, selection_audit = select_cohorts(prepared.evaluation, mapped, corpus)
    if [query.query_id for query in queries] != cohort["all_grade3_query_ids"]:
        raise ValueError("Primary grade-3 query cohort changed")
    if len(queries) != config["expected_query_count"] or len(corpus.ids) != config["expected_passage_count"]:
        raise ValueError("Pinned corpus/query counts changed")
    if config["primary_cohort"] != "all-mapped-grade3" or config["top_k"] != 10:
        raise ValueError("This runner requires all mapped grade-3 golds and top-10 retrieval")
    artifacts = local_model_artifacts(ROOT)
    old_report = read(reference / "all-report.json")
    rows = {}
    for name in MODEL_NAMES:
        result = read(reference / f"{name}-query-results.json")
        if result["model_fingerprint"] != artifacts[name].fingerprint or result["corpus_sha256"] != config["corpus_sha256"]:
            raise ValueError(f"Reference model provenance differs: {name}")
        if [row["query_id"] for row in result["per_query"]] != [query.query_id for query in queries]:
            raise ValueError(f"Reference query order differs: {name}")
        if result["cohort_sha256"] != old_report["provenance"]["cohort_sha256"]:
            raise ValueError(f"Reference cohort hash differs: {name}")
        rows[name] = [{**row, "metrics": {key: row["metrics"][key] for key in TOP10_METRICS}} for row in result["per_query"]]
    return corpus, queries, prepared, rows, selection_audit, old_report


def encode_batches(name: str, role: str, texts: tuple[str, ...], encoder: BedrockCohere, config: dict, directory: Path, provenance: dict) -> tuple[list[dict], list[Path]]:
    batch_dir = directory / name / role
    batch_dir.mkdir(parents=True, exist_ok=True)
    size = config["batch_size"]
    indices = list(range(0, len(texts), size))
    completed: dict[int, dict] = {}
    paths: dict[int, Path] = {}
    pending = []
    for start in indices:
        stop = min(start + size, len(texts))
        vector_path = batch_dir / f"{start:06d}-{stop:06d}.npy"
        meta_path = vector_path.with_suffix(".json")
        expected = {**provenance, "role": role, "start": start, "stop": stop, "texts_sha256": digest(texts[start:stop])}
        paths[start] = vector_path
        # Reuse the four initial successful v1 batches only when their actual
        # input strings fit the newly explicit v3 character limit. No changed
        # input may inherit a vector from the previous protocol.
        if not vector_path.exists() and not meta_path.exists() and config.get("reuse_verified_batches_from"):
            legacy_path = directory.parent / config["reuse_verified_batches_from"] / name / role / vector_path.name
            legacy_meta_path = legacy_path.with_suffix(".json")
            if legacy_path.is_file() and legacy_meta_path.is_file():
                legacy = read(legacy_meta_path)
                old = legacy["provenance"]
                old_settings = old["settings"]
                new_settings = {key: value for key, value in encoder.model.items() if key != "maximum_input_characters"}
                identical = all(old[key] == expected[key] for key in ("corpus_sha256", "query_ids_sha256", "model", "aws_profile", "aws_region", "role", "start", "stop", "texts_sha256"))
                limit = encoder.model.get("maximum_input_characters")
                identical = identical and old_settings == new_settings and (not limit or all(len(text) <= limit for text in texts[start:stop]))
                if identical and file_digest(legacy_path) == legacy["vectors_sha256"]:
                    os.link(legacy_path, vector_path)
                    save(meta_path, {"provenance": expected, "vectors_sha256": legacy["vectors_sha256"], "request": legacy["request"], "reused_from": str(legacy_path.relative_to(ROOT))})
        if meta_path.exists():
            metadata = read(meta_path)
            if metadata["provenance"] != expected or not vector_path.is_file() or file_digest(vector_path) != metadata["vectors_sha256"]:
                raise ValueError(f"Saved embedding batch differs: {meta_path}")
            vectors = np.load(vector_path, mmap_mode="r")
            if vectors.shape != (stop - start, encoder.model["dimension"]):
                raise ValueError(f"Saved embedding batch shape differs: {vector_path}")
            completed[start] = metadata
        elif vector_path.exists():
            raise FileExistsError(f"Unverified batch exists; preserved for inspection: {vector_path}")
        else:
            pending.append((start, stop, vector_path, meta_path, expected))

    price = float(encoder.model["input_price_usd_per_million_tokens"])
    # Includes already-saved batches on resume. Bound the remaining token estimate
    # before calls and stop if measured cost crosses the configured limit.
    known_tokens = sum(row["request"]["input_tokens"] or row["request"]["characters"] / 3 for row in completed.values())
    remaining_estimate = sum(sum(map(len, texts[start:stop])) / 3 for start, stop, *_ in pending)
    if (known_tokens + remaining_estimate) / 1e6 * price > config["estimated_maximum_cost_usd_per_model"]:
        raise RuntimeError("Estimated remaining embedding spend exceeds the configured cost guard")
    event(directory, {"stage": "encoding", "model": name, "role": role, "completed_rows": sum(row["provenance"]["stop"] - row["provenance"]["start"] for row in completed.values()), "total_rows": len(texts), "pending_batches": len(pending)})

    def one_batch(item: tuple) -> tuple[int, dict]:
        start, stop, vector_path, meta_path, expected = item
        vectors, request = encoder.encode(texts[start:stop], role)
        with vector_path.open("xb") as output:
            np.save(output, vectors, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        metadata = {"provenance": expected, "vectors_sha256": file_digest(vector_path), "request": request}
        save(meta_path, metadata)
        return start, metadata

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(one_batch, item) for item in pending]
        for future in as_completed(futures):
            try:
                start, metadata = future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            completed[start] = metadata
            known_tokens += metadata["request"]["input_tokens"] or metadata["request"]["characters"] / 3
            if known_tokens / 1e6 * price > config["estimated_maximum_cost_usd_per_model"]:
                for pending_future in futures:
                    pending_future.cancel()
                raise RuntimeError("Measured embedding token spend exceeds the configured cost guard")
            if len(completed) % 10 == 0 or len(completed) == len(indices):
                finished_rows = sum(row["provenance"]["stop"] - row["provenance"]["start"] for row in completed.values())
                event(directory, {"stage": "encoding", "model": name, "role": role, "completed_rows": finished_rows, "total_rows": len(texts), "percent": round(finished_rows / len(texts) * 100, 2), "input_tokens": int(known_tokens), "estimated_cost_usd": round(known_tokens / 1e6 * price, 4), "current_session_seconds": round(time.perf_counter() - started, 2)})
    return [completed[start] for start in indices], [paths[start] for start in indices]


def assemble(paths: list[Path], metadata: list[dict], target: Path, rows: int, dimension: int, provenance: dict) -> np.ndarray:
    manifest_path = target.with_suffix(".json")
    if target.exists():
        manifest = read(manifest_path)
        if manifest["provenance"] != provenance or manifest["shape"] != [rows, dimension] or file_digest(target) != manifest["vectors_sha256"]:
            raise ValueError(f"Published full-corpus vectors differ: {target}")
        return np.load(target, mmap_mode="r")
    partial = target.with_name(target.name + ".build")
    if partial.exists() or manifest_path.exists():
        raise FileExistsError(f"Existing unpublished assembly preserved: {partial}")
    target.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(rows, dimension))
    offset = 0
    for path, meta in zip(paths, metadata):
        batch = np.load(path, mmap_mode="r")
        start, stop = meta["provenance"]["start"], meta["provenance"]["stop"]
        if start != offset or batch.shape != (stop - start, dimension):
            raise ValueError("Embedding batches have a gap or invalid shape")
        norms = np.linalg.norm(batch, axis=1)
        if not np.isfinite(batch).all() or not np.allclose(norms, 1.0, atol=2e-5):
            raise ValueError("Saved vectors are nonfinite or not normalized")
        matrix[start:stop] = batch
        offset = stop
    if offset != rows:
        raise ValueError("Assembled vectors do not cover every corpus row")
    matrix.flush()
    del matrix
    os.link(partial, target)
    save(manifest_path, {"provenance": provenance, "shape": [rows, dimension], "vectors_sha256": file_digest(target), "batch_count": len(paths)})
    return np.load(target, mmap_mode="r")


def remote_measurement(name: str, encoder: BedrockCohere, corpus: object, vectors: np.ndarray, prepared: object, directory: Path, provenance: dict) -> dict:
    path = directory / name / "measurement.json"
    if path.exists():
        result = read(path)
        if result["provenance"] != provenance:
            raise ValueError("Existing operational measurement provenance differs")
        return result
    reference = next(ROOT.glob("runs/measurement-teacher-*.json"))
    ids = read(reference)["development_query_ids"]
    lookup = {query.query_id: query.query for query in prepared.development}
    texts = [lookup[query_id] for query_id in ids]
    query_samples, search_samples, total_samples, requests = [], [], [], []
    for repetition in range(7):
        for text in texts:
            vector, request = encoder.encode([text], "search_query")
            started = time.perf_counter()
            exact_top_k(vector, vectors, corpus.ids, k=10, corpus_block_size=4096)
            search_seconds = time.perf_counter() - started
            requests.append(request)
            if repetition >= 2:
                query_samples.append(request["api_seconds"])
                search_samples.append(search_seconds)
                total_samples.append(request["api_seconds"] + search_seconds)
        event(directory, {"stage": "operational-measurement", "model": name, "round": repetition + 1, "rounds": 7})
    result = {
        "provenance": provenance, "environment": host_metadata(), "development_query_ids": ids,
        "warm": {"query_count": len(texts), "warmup_rounds": 2, "repeated_rounds": 5, "single_query_encoding": _timing_summary(query_samples), "single_query_exact_search": _timing_summary(search_samples), "single_query_end_to_end": _timing_summary(total_samples)},
        "requests": requests, "peak_rss_client_bytes": peak_rss_bytes(), "model_artifact_size_bytes": None, "peak_rss_server_bytes": None,
        "note": "Query encoding includes the network round trip to AWS; exact search is local. Server model size and RAM are unavailable. Client RSS includes corpus structures and mapped vectors and is not model RAM.",
    }
    save(path, result)
    return result


def evaluate_model(name: str, model: dict, client: object, config: dict, directory: Path, corpus: object, queries: tuple, prepared: object, protocol_hash: str) -> tuple[dict, dict]:
    provenance = {"protocol_sha256": protocol_hash, "corpus_sha256": config["corpus_sha256"], "query_ids_sha256": digest([query.query_id for query in queries]), "model": name, "settings": model, "aws_profile": config["aws_profile"], "aws_region": config["aws_region"]}
    result_path = directory / name / "query-results.json"
    if result_path.exists():
        result = read(result_path)
        if result["provenance"] != provenance:
            raise ValueError("Existing Cohere quality-result provenance differs")
        return result, read(directory / name / "measurement.json")
    encoder = BedrockCohere(client, model, tokens_per_minute=config["tokens_per_minute"])
    # A single development query establishes access before full-corpus spending.
    try:
        _, request = encoder.encode([prepared.development[0].query], "search_query")
        event(directory, {"stage": "access-confirmed", "model": name, "request": request})
    except Exception as exc:
        event(directory, {"stage": "access-failed", "model": name, "error_type": type(exc).__name__, "message": str(exc)})
        raise
    build_started = time.perf_counter()
    corpus_meta, corpus_paths = encode_batches(name, "search_document", corpus.texts, encoder, config, directory, provenance)
    corpus_vectors = assemble(corpus_paths, corpus_meta, directory / name / "corpus-vectors.npy", len(corpus.ids), model["dimension"], provenance)
    query_texts = tuple(query.query for query in queries)
    query_meta, query_paths = encode_batches(name, "search_query", query_texts, encoder, config, directory, provenance)
    query_vectors = assemble(query_paths, query_meta, directory / name / "query-vectors.npy", len(queries), model["dimension"], provenance)
    build_finished = time.perf_counter()
    measurement = remote_measurement(name, encoder, corpus, corpus_vectors, prepared, directory, provenance)
    rows = []
    search_started = time.perf_counter()
    for start in range(0, len(queries), 16):
        batch = queries[start:start + 16]
        rankings = exact_top_k(query_vectors[start:start + len(batch)], corpus_vectors, corpus.ids, k=10, corpus_block_size=4096)
        for query, ranking in zip(batch, rankings):
            rows.append({"query_id": query.query_id, "category": query.category, "raw_grade3_count": query.raw_grade3_count, "mapped_grade3_count": len(query.gold_ids), "metrics": score_top10_query(query, ranking), "top10": [{"corpus_id": item.corpus_id, "similarity": item.similarity} for item in ranking]})
        if (start // 16 + 1) % 10 == 0 or len(rows) == len(queries):
            event(directory, {"stage": "quality-scoring", "model": name, "completed_queries": len(rows), "total_queries": len(queries)})
    requests = [item["request"] for item in corpus_meta + query_meta] + measurement["requests"]
    actual_tokens = sum(item["input_tokens"] or 0 for item in requests)
    missing_counts = sum(item["input_tokens"] is None for item in requests)
    result = {"model": name, "provenance": provenance, "per_query": rows, "corpus_passage_count": len(corpus.ids), "operational": {"build_session_seconds": build_finished - build_started, "corpus_api_seconds_sum": sum(item["request"]["api_seconds"] for item in corpus_meta), "quality_search_seconds": time.perf_counter() - search_started, "corpus_vector_size_bytes": (directory / name / "corpus-vectors.npy").stat().st_size, "dimension": model["dimension"], "api_successful_requests": len(requests), "input_tokens": actual_tokens, "requests_without_token_counts": missing_counts, "estimated_api_cost_usd": actual_tokens / 1e6 * model["input_price_usd_per_million_tokens"], "price_status": model["price_status"], "price_source": model["price_source"], "truncation": model["truncate"], "maximum_input_characters": model.get("maximum_input_characters"), "character_truncated_corpus_inputs": sum(item["request"].get("character_truncated_inputs", 0) for item in corpus_meta), "character_truncated_query_inputs": sum(item["request"].get("character_truncated_inputs", 0) for item in query_meta), "retry_count": sum(len(item["retries"]) for item in requests)}}
    save(result_path, result)
    event(directory, {"stage": "model-complete", "model": name, "metrics": summarize(rows, metrics=TOP10_METRICS)["overall"], "operational": result["operational"]})
    return result, measurement


def render(report: dict) -> str:
    lines = ["# Primary DisastIR evaluation: all mapped grade-3 golds", "", f"{report['query_count']:,} questions; {report['corpus_passage_count']:,} passages; exact full-corpus retrieval; maximum cutoff 10.", "", "| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Encoding p50 ms | Encoding p95 ms | Model MiB | Peak model-process RSS MiB | Corpus vectors MiB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, model in report["models"].items():
        if model["status"] != "completed":
            lines.append(f"| {name} (blocked) | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        m = model["quality"]["overall"]
        measurement = model["measurement"]
        warm = measurement["warm"]["single_query_encoding"]
        size = measurement.get("model_artifact_size_bytes")
        rss = measurement.get("peak_rss_after_workload_bytes")
        vectors = model["corpus_vector_size_bytes"]
        scores = " | ".join(f"{m[key] * 100:.2f}%" for key in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"))
        cells = [name] + [f"{m[key] * 100:.2f}%" for key in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")]
        cells += [f"{warm['median_ms']:.2f}", f"{warm['p95_ms']:.2f}", f"{size / 2**20:.2f}" if size is not None else "remote / unavailable", f"{rss / 2**20:.2f}" if rss is not None else "remote / unavailable", f"{vectors / 2**20:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Recall divides by **all mapped grade-3 gold passages for each question**, then averages across the 1,127 questions. No single gold is chosen for questions with multiple grade-3 passages.", "", "Local encoding measurements use a CPU. Cohere encoding includes the AWS network round trip. All latency measurements use the same 32 development questions, two warm-up rounds and five measured rounds. Server model size and memory are not exposed by Bedrock.", "", "Quality includes category breakdowns, query-bootstrap 95% intervals and paired differences in the JSON report. The grade-3 selection was chosen after viewing the original held-out results, so this is exploratory. DisastIR judgments are pooled and LLM-rated; unjudged passages have unknown relevance."]
    for name, model in report["models"].items():
        if model["status"] == "blocked":
            lines += ["", f"**{name} access failure:** `{model['error']}`"]
        elif name.startswith("cohere"):
            operation = model["operational"]
            lines += ["", f"**{name}:** {operation['input_tokens']:,} reported input tokens; estimated API cost ${operation['estimated_api_cost_usd']:.2f}. This is a rate-based estimate, not a confirmed AWS invoice or credit application."]
    lines += ["", f"Corpus SHA-256: `{report['corpus_sha256']}`.", f"Protocol SHA-256: `{report['protocol_sha256']}`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["cohere_v3", "cohere_v4"])
    parser.add_argument("--config", default="configs/cohere-grade3-v2.json")
    args = parser.parse_args()
    path = ROOT / args.config
    config = read(path)
    protocol_hash = file_digest(path)
    if any(name not in config["models"] for name in args.models):
        raise ValueError("Unknown Cohere model")
    directory = ROOT / "runs" / f"cohere-grade3-{protocol_hash[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(directory / "protocol.json", config)
    event(directory, {"stage": "preparing", "pid": os.getpid(), "models": args.models})
    corpus, queries, prepared, rows, selection_audit, original = prepare(config)
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=config["aws_profile"])
    client = session.client("bedrock-runtime", region_name=config["aws_region"], config=Config(connect_timeout=10, read_timeout=90, max_pool_connections=config["workers"] + 2, retries={"max_attempts": 0}))
    models = {}
    for name in MODEL_NAMES:
        measurement = read(next(ROOT.glob(f"runs/measurement-{name}-*.json")))
        models[name] = {"status": "completed", "quality": summarize(rows[name], metrics=TOP10_METRICS), "measurement": measurement, "corpus_vector_size_bytes": measurement["corpus_embedding_cache_size_bytes"]}
    for name in args.models:
        try:
            result, measurement = evaluate_model(name, config["models"][name], client, config, directory, corpus, queries, prepared, protocol_hash)
            rows[name] = result["per_query"]
            models[name] = {"status": "completed", "quality": summarize(rows[name], metrics=TOP10_METRICS), "measurement": measurement, "corpus_vector_size_bytes": result["operational"]["corpus_vector_size_bytes"], "operational": result["operational"]}
        except Exception as exc:
            models[name] = {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
            event(directory, {"stage": "model-blocked", "model": name, **models[name]})
    report = {"status": "completed" if all(model["status"] == "completed" for model in models.values()) else "completed-with-blocked-models", "primary_cohort": config["primary_cohort"], "query_count": len(queries), "corpus_passage_count": len(corpus.ids), "top_k": 10, "corpus_sha256": config["corpus_sha256"], "protocol_sha256": protocol_hash, "reference_cohort_sha256": original["provenance"]["cohort_sha256"], "query_ids": [query.query_id for query in queries], "selection_audit": selection_audit, "models": models, "paired_with_teacher": {name: paired_with_teacher(rows["teacher"], model_rows, metrics=TOP10_METRICS) for name, model_rows in rows.items() if name != "teacher"}}
    report_path = directory / "reports" / f"{time.time_ns()}-report.json"
    save(report_path, report)
    with report_path.with_suffix(".md").open("x", encoding="utf-8") as output:
        output.write(render(report))
    event(directory, {"stage": "report-complete", "report": str(report_path.relative_to(ROOT)), "status": report["status"], "models": {name: model["status"] for name, model in models.items()}})


if __name__ == "__main__":
    main()
