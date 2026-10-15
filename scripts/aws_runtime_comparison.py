#!/usr/bin/env python3
"""Speed, memory and cost: Tuklas vs the SEA-LION teacher vs Cohere v3.

Every system runs on ONE host (ml.g4dn.xlarge: 4 vCPU, 16 GiB, one T4), each
in a fresh child process so resident memory is not shared between them:

  tuklas-pruned     static, CPU   the released 38.6 MB package
  tuklas-unpruned   static, CPU   the 137 MB package behind 0.935994 held-out
  teacher-gpu       transformer, T4
  teacher-cpu       transformer, the same 4 vCPUs the static models use
  cohere-v3         Bedrock API, called from this host in ap-southeast-1

The index is the shared window-192 / stride-128 layout (40,152 chunks over
1,294 articles) every quality number in this project was measured on.  Timing
queries are the 197 DEVELOPMENT questions; held-out questions are never read.

Each child's measurements are uploaded to S3 the moment the child exits.
Gates are judged only after that, and a failed gate marks the report instead
of discarding measurements.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/runtime-comparison")
sys.path.insert(0, str(INPUT))

WINDOW, STRIDE = 192, 128
SAMPLE_STEP = 20            # every 20th chunk: the fixed throughput sample
WARMUP_QUERIES = 8
LOCAL_REPETITIONS = 3
COHERE_REPETITIONS = 1
# Known development Recall@10 at window 192 (project notes); the harness gates.
STUDENT_DEVELOPMENT = 0.9441624365482234
TEACHER_DEVELOPMENT = 1.0
TEACHER_QUERY_PROMPT = "Panuto: Maghanap ng kaugnay na talata na sumasagot sa tanong.\nTanong: "
COHERE_MODEL = {"model_id": "cohere.embed-multilingual-v3", "dimension": 1024,
                "truncate": "NONE", "maximum_input_characters": 2048, "full_content": True}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def windows(text: str) -> list[tuple[int, int]]:
    starts = list(range(0, max(1, len(text) - WINDOW + 1), STRIDE))
    if starts and starts[-1] + WINDOW < len(text):
        starts.append(len(text) - WINDOW)
    out, seen = [], set()
    for begin in starts:
        begin = max(0, begin)
        end = min(len(text), begin + WINDOW)
        if end <= begin or (begin, end) in seen:
            continue
        seen.add((begin, end))
        out.append((begin, end))
    return out


def build_index(layout: dict) -> tuple[list[dict], list[int]]:
    frozen, parents = layout["chunks"], layout["parents"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in frozen:
        grouped[chunk["parent_id"]].append(chunk)
    texts: dict[str, str] = {}
    for parent_id, group in grouped.items():
        group = sorted(group, key=lambda c: c["start"])
        buffer: list[str | None] = [None] * max(c["end"] for c in group)
        for chunk in group:
            if len(chunk["text"]) != chunk["end"] - chunk["start"]:
                raise ValueError(f"{parent_id}: chunk text length disagrees with its span")
            for offset, character in enumerate(chunk["text"]):
                position = chunk["start"] + offset
                if buffer[position] is not None and buffer[position] != character:
                    raise ValueError(f"{parent_id}: overlapping windows disagree at {position}")
                buffer[position] = character
        if any(c is None for c in buffer):
            raise ValueError(f"{parent_id}: reconstruction hole")
        texts[parent_id] = "".join(buffer)  # type: ignore[arg-type]
    if set(texts) != set(parents):
        raise ValueError("reconstructed parent set differs from the frozen layout")
    records, owners = [], []
    for index, parent_id in enumerate(parents):
        text = texts[parent_id]
        for begin, end in windows(text) or [(0, len(text))]:
            records.append({"parent_id": parent_id, "text": text[begin:end],
                            "start": begin, "end": end})
            owners.append(index)
    if len(records) != 40_152 or len(parents) != 1_294:
        raise ValueError(f"index changed: {len(records)} chunks over {len(parents)} articles")
    return records, owners


# --------------------------------------------------------------------------- child

def child(name: str) -> None:
    import numpy as np
    import psutil
    import torch

    from multiwikiqa_evaluation import normalized, rank_parents, score

    plan = json.loads((WORK / "plan.json").read_text())
    spec = plan["systems"][name]
    corpus = json.loads((WORK / "corpus.json").read_text())
    records, owners, parents = corpus["records"], corpus["owners"], corpus["parents"]
    texts = [r["text"] for r in records]
    development = json.loads((WORK / "development.json").read_text())
    questions = [q["question"] for q in development]
    sample = [texts[i] for i in corpus["sample_indices"]]
    starts = np.r_[0, np.flatnonzero(np.diff(np.asarray(owners)) != 0) + 1]
    if len(starts) != len(parents):
        raise ValueError("chunks are not grouped contiguously by article")

    threads = int(plan["protocol"]["cpu_threads"])
    torch.set_num_threads(threads)
    process = psutil.Process()
    peak = {"rss": process.memory_info().rss}
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(0.005):
            try:
                peak["rss"] = max(peak["rss"], process.memory_info().rss)
            except psutil.Error:
                return

    runtime = spec["runtime"]
    cuda = runtime == "transformer-gpu"
    if cuda:
        # Initialise the CUDA context first so the baseline includes it and the
        # model increment does not.
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    baseline_rss = process.memory_info().rss
    peak["rss"] = baseline_rss
    threading.Thread(target=watch, daemon=True).start()

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    # ---- load --------------------------------------------------------------
    extra: dict = {}
    started = time.perf_counter_ns()
    if runtime == "static-cpu":
        from model2vec import StaticModel
        model = StaticModel.from_pretrained(str(WORK / "models" / name))
        prefix = json.loads((WORK / "models" / name / "config.json").read_text()).get("query_prefix", "")

        def encode_queries(values):
            return model.encode([prefix + v for v in values], batch_size=32,
                                max_length=None, use_multiprocessing=False)

        def encode_documents(values):
            return model.encode(values, batch_size=1024, max_length=None,
                                use_multiprocessing=False)
    elif runtime in ("transformer-gpu", "transformer-cpu"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(WORK / "models" / "teacher"),
                                    device="cuda" if cuda else "cpu", trust_remote_code=False)
        model.max_seq_length = 512
        extra["torch_dtype"] = str(next(model.parameters()).dtype)

        def encode_queries(values):
            out = model.encode(values, batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False,
                               prompt=TEACHER_QUERY_PROMPT)
            sync()
            return out

        def encode_documents(values):
            out = model.encode(values, batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            sync()
            return out
    elif runtime == "api":
        import boto3
        from bedrock_embeddings import BedrockCohere
        client = boto3.client("bedrock-runtime", region_name=plan["region"])
        model = BedrockCohere(client, COHERE_MODEL)
        receipts: list[dict] = []

        def encode_queries(values):
            vectors, receipt = model.encode(values, "search_query")
            receipts.append(receipt)
            return vectors

        def encode_documents(values):
            parts = []
            for begin in range(0, len(values), 96):
                vectors, receipt = model.encode(values[begin:begin + 96], "search_document")
                receipts.append(receipt)
                parts.append(vectors)
            return np.concatenate(parts)
    else:
        raise ValueError("unknown runtime " + runtime)
    sync()
    load_ms = (time.perf_counter_ns() - started) / 1e6
    loaded_rss = process.memory_info().rss

    # ---- indexing throughput on the fixed sample (every system) ------------
    encode_documents(sample[:64])   # warm kernels / connection before timing
    if runtime == "api":
        receipts.clear()
    started = time.perf_counter_ns()
    sample_vectors = normalized(np.asarray(encode_documents(sample), dtype=np.float32))
    sample_seconds = (time.perf_counter_ns() - started) / 1e9
    sample_receipts = list(receipts) if runtime == "api" else []
    if runtime == "api":
        receipts.clear()

    # ---- the full index: encoded where affordable, borrowed where not ------
    full_index = None
    documents = None
    if spec["full_index"]:
        started = time.perf_counter_ns()
        documents = normalized(np.asarray(encode_documents(texts), dtype=np.float32))
        full_index = {"chunks": len(texts),
                      "seconds": (time.perf_counter_ns() - started) / 1e9}
        full_index["chunks_per_second"] = len(texts) / full_index["seconds"]
        if name == "teacher-gpu":
            # Write-then-rename: a timeout mid-save must not leave a truncated
            # matrix for teacher-cpu and cohere-v3 to load.
            for target, value in (("teacher-documents.npy", documents),
                                  ("teacher-gpu-sample.npy", sample_vectors)):
                with (WORK / (target + ".tmp")).open("wb") as stream:
                    np.save(stream, value)
                os.replace(WORK / (target + ".tmp"), WORK / target)
    else:
        # Exact search time depends only on the matrix shape, so a system that
        # is too slow or costly to index fully is searched over the teacher's
        # 40,152 x 1024 matrix (Cohere v3 is also 1024-d).  If the teacher-gpu
        # child failed, a random matrix of the same shape keeps the timing.
        borrowed = WORK / "teacher-documents.npy"
        if borrowed.exists():
            documents = np.load(borrowed)
            extra["search_matrix"] = "teacher-gpu document vectors (same shape; search time only)"
        else:
            generator = np.random.default_rng(42)
            documents = normalized(generator.standard_normal(
                (len(texts), sample_vectors.shape[1]), dtype=np.float32))
            extra["search_matrix"] = "random matrix of the same shape (teacher-gpu vectors missing)"
        if documents.shape != (len(texts), sample_vectors.shape[1]):
            raise ValueError("borrowed search matrix has the wrong shape")
    corpus_rss = process.memory_info().rss

    def search(vector):
        parent_scores = np.maximum.reduceat(documents @ vector, starts)
        top = np.argpartition(-parent_scores, 9)[:10]
        return top[np.argsort(-parent_scores[top], kind="stable")]

    # ---- single-query latency ---------------------------------------------
    for text in questions[:WARMUP_QUERIES]:
        search(normalized(np.asarray(encode_queries([text]), dtype=np.float32))[0])
    if runtime == "api":
        receipts.clear()
    encode_ms, search_ms, total_ms = [], [], []
    repetitions = COHERE_REPETITIONS if runtime == "api" else LOCAL_REPETITIONS
    for _ in range(repetitions):
        for text in questions:
            t0 = time.perf_counter_ns()
            vector = normalized(np.asarray(encode_queries([text]), dtype=np.float32))[0]
            t1 = time.perf_counter_ns()
            search(vector)
            t2 = time.perf_counter_ns()
            encode_ms.append((t1 - t0) / 1e6)
            search_ms.append((t2 - t1) / 1e6)
            total_ms.append((t2 - t0) / 1e6)
    query_receipts = list(receipts) if runtime == "api" else []

    # Serving memory ends here: the gate below builds float32 copies of the
    # table and a second index, which no deployment would hold.
    stopped.set()
    peak["rss"] = max(peak["rss"], process.memory_info().rss)
    serving_peak_rss = peak["rss"]

    # ---- quality gate inputs (development only) ----------------------------
    def development_recall(query_vectors, document_vectors):
        rows = []
        for begin in range(0, len(development), 64):
            ranking = rank_parents(query_vectors[begin:begin + 64], document_vectors, owners, parents)
            for query, ranked in zip(development[begin:begin + 64], ranking, strict=True):
                rows.append(score(query, ranked, records))
        return {"development_recall_at_10": float(np.mean([r["Recall@10"] for r in rows])),
                "questions": len(rows)}

    gate = None
    if runtime == "static-cpu":
        # The gate uses the exact encoder that produced 0.944162 (the shootout
        # harness), not the timed library call, so a library difference cannot
        # masquerade as a regression.  Their agreement is recorded separately.
        from model2vec.train.utils import get_probable_pad_token_id
        from multiwikiqa_compact_retrieval import strip_padding_and_unknown
        weights = np.asarray(model.embedding, dtype=np.float32)
        pad_id, unknown_id = get_probable_pad_token_id(model.tokenizer), model.unk_token_id

        def harness_encode(values):
            out = np.zeros((len(values), weights.shape[1]), dtype=np.float32)
            for begin in range(0, len(values), 2048):
                for offset, encoding in enumerate(model.tokenizer.encode_batch_fast(
                        values[begin:begin + 2048], add_special_tokens=False)):
                    ids = strip_padding_and_unknown(encoding.ids, pad_id=pad_id, unknown_id=unknown_id)
                    if ids:
                        out[begin + offset] = weights[ids].mean(axis=0)
            return normalized(out)

        harness_documents = harness_encode(texts)
        harness_queries = harness_encode([prefix + q for q in questions])
        gate = development_recall(harness_queries, harness_documents)
        library_queries = normalized(np.asarray(encode_queries(questions), dtype=np.float32))
        extra["library_vs_harness_min_cosine"] = {
            "queries": float(np.min(np.sum(library_queries * harness_queries, axis=1))),
            "documents": float(np.min(np.sum(documents * harness_documents, axis=1)))}
        del harness_documents
    elif runtime.startswith("transformer"):
        query_vectors = normalized(np.asarray(encode_queries(questions), dtype=np.float32))
        if spec["full_index"] or "search_matrix" in extra and extra["search_matrix"].startswith("teacher-gpu"):
            gate = development_recall(query_vectors, documents)
    if name == "teacher-cpu":
        reference_path = WORK / "teacher-gpu-sample.npy"
        if reference_path.exists():
            reference = np.load(reference_path)
            extra["cpu_vs_gpu_sample_min_cosine"] = float(np.min(np.sum(reference * sample_vectors, axis=1)))

    def distribution(values):
        array = np.asarray(values, dtype=np.float64)
        return {"mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
                "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)),
                "minimum_ms": float(array.min()), "maximum_ms": float(array.max()),
                "samples": len(values)}

    result = {
        "system": name, "runtime": runtime, "device": spec["device"],
        "dimensions": int(sample_vectors.shape[1]),
        "load_ms": load_ms,
        "index_sample": {"chunks": len(sample), "seconds": sample_seconds,
                         "chunks_per_second": len(sample) / sample_seconds},
        "index_full": full_index,
        "single_query_encode": distribution(encode_ms),
        "exact_search_40152": distribution(search_ms),
        "end_to_end": distribution(total_ms),
        "memory": {"baseline_rss_bytes": baseline_rss, "rss_after_load_bytes": loaded_rss,
                   "rss_after_index_bytes": corpus_rss,
                   "peak_rss_bytes": serving_peak_rss,
                   "peak_rss_scope": "load, indexing, timed queries; excludes the quality gate",
                   "model_rss_increment_bytes": loaded_rss - baseline_rss},
        "development_gate": gate,
        **extra,
    }
    if runtime == "static-cpu":
        weights = WORK / "models" / name / "model.safetensors"
        result["weights_bytes"] = weights.stat().st_size
        result["weights_sha256"] = digest(weights)
        table = np.asarray(model.embedding)
        result["all_zero_rows"] = int(np.all(table == 0, axis=1).sum())
        result["table_shape"] = list(table.shape)
        result["artifact_bytes"] = sum(p.stat().st_size for p in (WORK / "models" / name).rglob("*") if p.is_file())
    elif runtime.startswith("transformer"):
        folder = WORK / "models" / "teacher"
        result["weights_bytes"] = sum(p.stat().st_size for p in folder.rglob("*.safetensors"))
        result["artifact_bytes"] = sum(p.stat().st_size for p in folder.rglob("*")
                                       if p.is_file() and ".cache" not in p.parts)
        if cuda:
            result["memory"]["gpu"] = {
                "device": torch.cuda.get_device_name(0),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    else:
        def tokens(rows):
            return sum(int(r["input_tokens"]) for r in rows if r.get("input_tokens") is not None)

        def service(rows):
            values = [r["service_latency_ms"] for r in rows if r.get("service_latency_ms") is not None]
            return distribution(values) if values else None

        result["cohere"] = {
            "region": plan["region"], "model_id": COHERE_MODEL["model_id"],
            "query_calls": len(query_receipts), "query_tokens": tokens(query_receipts),
            "query_tokens_missing": sum(r.get("input_tokens") is None for r in query_receipts),
            "query_service_latency": service(query_receipts),
            "document_calls": len(sample_receipts), "document_tokens": tokens(sample_receipts),
            "document_tokens_missing": sum(r.get("input_tokens") is None for r in sample_receipts),
            "retries": sum(len(r.get("retries", [])) for r in query_receipts + sample_receipts),
        }
    target = WORK / "results" / (name + ".json")
    target.parent.mkdir(exist_ok=True)
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


# --------------------------------------------------------------------- coordinator

CHILD_TIMEOUT_SECONDS = {"tuklas-pruned": 600, "tuklas-unpruned": 600, "teacher-gpu": 1500,
                         "teacher-cpu": 2100, "cohere-v3": 900}
NEEDS = {"tuklas-pruned": "tuklas-pruned", "tuklas-unpruned": "tuklas-unpruned",
         "teacher-gpu": "teacher", "teacher-cpu": "teacher", "cohere-v3": None}


def multipart_etag(path: Path, part_bytes: int) -> str:
    from hashlib import md5
    parts = []
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(part_bytes), b""):
            parts.append(md5(block).digest())
    if len(parts) == 1:
        return parts[0].hex()
    return md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"


def coordinator() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
         "model2vec==0.9.0", "sentence-transformers==5.7.0", "transformers==5.3.0",
         "huggingface-hub==1.33.0", "numpy==2.5.3", "boto3==1.43.108", "psutil==7.1.3"],
        check=True,
    )
    import importlib.metadata as metadata

    import boto3
    import psutil
    from huggingface_hub import snapshot_download

    from multiwikiqa_ranking_v5 import prepare

    plan = json.loads((INPUT / "plan.json").read_text())
    for filename, expected in plan["staged_sha256"].items():
        if digest(INPUT / filename) != expected:
            raise ValueError("staged input changed: " + filename)
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "models").mkdir()
    (WORK / "plan.json").write_bytes((INPUT / "plan.json").read_bytes())
    bucket, output = plan["bucket"], plan["output_prefix"].rstrip("/") + "/"
    s3 = boto3.client("s3", region_name=plan["region"])

    def publish(path: Path, relative: str) -> None:
        s3.upload_file(str(path), bucket, output + relative)

    def event(stage: str, **details) -> None:
        print(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                          "stage": stage, **details}), flush=True)

    layout = json.loads((INPUT / "shared-layout.json").read_text())
    records, owners = build_index(layout)
    annotations = [json.loads(line) for line in
                   (INPUT / "annotations.all.jsonl").read_text().splitlines() if line.strip()]
    tagalog = [row for row in annotations if row["language"] == "tl"]
    _, development, _ = prepare(tagalog, layout["chunks"])
    if len(development) != 197:
        raise ValueError(f"development cohort changed: {len(development)}")
    del annotations, tagalog  # held-out rows are never written for the children
    sample_indices = list(range(0, len(records), SAMPLE_STEP))
    (WORK / "corpus.json").write_text(json.dumps({
        "records": records, "owners": owners,
        "parents": layout["parents"], "sample_indices": sample_indices}))
    (WORK / "development.json").write_text(json.dumps(development))
    event("index-built", chunks=len(records), sample=len(sample_indices), development=len(development))

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    staged: dict[str, str] = {}

    # A staging failure disables only the systems that need that artifact.
    for name, package in plan["static_packages"].items():
        try:
            target = WORK / "models" / name
            target.mkdir()
            prefix = package["prefix"].rstrip("/") + "/"
            for row in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
                relative = row["Key"][len(prefix):]
                if relative:
                    s3.download_file(bucket, row["Key"], str(target / relative))
            weights = target / "model.safetensors"
            if "weights_sha256" in package and digest(weights) != package["weights_sha256"]:
                raise ValueError("weights sha256 differs from the expected package")
            if "weights_etag" in package and multipart_etag(weights, package["etag_part_bytes"]) != package["weights_etag"]:
                raise ValueError("weights multipart ETag differs from the S3 object")
            staged[name] = "ok"
            event("package-staged", system=name)
        except Exception as error:
            staged[name] = repr(error)[:300]
            event("package-failed", system=name, error=staged[name])
    try:
        snapshot_download(repo_id=plan["teacher"]["id"], revision=plan["teacher"]["revision"],
                          local_dir=WORK / "models" / "teacher")
        staged["teacher"] = "ok"
        event("teacher-staged")
    except Exception as error:
        staged["teacher"] = repr(error)[:300]
        event("teacher-failed", error=staged["teacher"])

    versions = {}
    for package in ("model2vec", "sentence-transformers", "transformers", "tokenizers",
                    "torch", "numpy", "boto3", "huggingface-hub"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    def write_report() -> dict:
        gates = {}
        expected = {"tuklas-pruned": STUDENT_DEVELOPMENT, "tuklas-unpruned": STUDENT_DEVELOPMENT,
                    "teacher-gpu": TEACHER_DEVELOPMENT, "teacher-cpu": TEACHER_DEVELOPMENT}
        for name, target in expected.items():
            if name in results:
                value = (results[name]["development_gate"] or {}).get("development_recall_at_10")
                gates[name] = {"observed": value, "expected": target,
                               "passed": value is not None and abs(value - target) < 1e-9}
        if "teacher-cpu" in results:
            cosine = results["teacher-cpu"].get("cpu_vs_gpu_sample_min_cosine")
            gates["teacher-cpu-matches-gpu"] = {"observed": cosine, "minimum": 0.999,
                                                "passed": cosine is not None and cosine >= 0.999}

        rates = plan["usd_per_hour"]
        costs = {}
        for name, result in results.items():
            if result["runtime"] == "api":
                c = result["cohere"]
                per_query = c["query_tokens"] / max(1, c["query_calls"])
                per_chunk = c["document_tokens"] / result["index_sample"]["chunks"]
                price = plan["cohere_usd_per_1k_tokens"]
                costs[name] = {"basis": f"Bedrock tokens at ${price}/1K; excludes the host that runs the search",
                               "tokens_per_query": per_query, "tokens_per_chunk": per_chunk,
                               "usd_per_million_queries": per_query * 1e6 / 1000 * price,
                               "usd_per_million_chunks_indexed": per_chunk * 1e6 / 1000 * price,
                               "usd_to_index_40152_chunks": per_chunk * 40_152 / 1000 * price}
            else:
                kind = "gpu" if result["device"] == "cuda" else "cpu"
                rate = rates[kind]
                mean_s = result["end_to_end"]["mean_ms"] / 1000
                source = result["index_full"] or result["index_sample"]
                throughput = source["chunks_per_second"]
                costs[name] = {"basis": f"{plan['instances'][kind]} at ${rate}/h, one sequential stream kept busy",
                               "throughput_source": "full 40,152-chunk index" if result["index_full"] else "2,008-chunk sample",
                               "usd_per_million_queries": 1e6 * mean_s / 3600 * rate,
                               "usd_per_million_chunks_indexed": 1e6 / throughput / 3600 * rate,
                               "usd_to_index_40152_chunks": 40_152 / throughput / 3600 * rate}

        complete = (not failures and len(results) == len(plan["order"])
                    and all(g["passed"] for g in gates.values()))
        report = {
            "status": "complete" if complete else "needs-review",
            "systems": results, "failures": failures, "staging": staged, "gates": gates, "costs": costs,
            "hardware": {"instance_type": plan["instance_type"], "platform": platform.platform(),
                         "cpu_model": cpu_model,
                         "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                         "ram_bytes": psutil.virtual_memory().total,
                         "cpu_threads": plan["protocol"]["cpu_threads"]},
            "library_versions": versions,
            "protocol": plan["protocol"],
            "heldout_use": "none; timing and gates use the 197 development questions only",
            "limitations": [
                "One host: CPU numbers come from the g4dn.xlarge's 4 vCPUs; the cost model prices them at a 4-vCPU CPU instance (ml.m5.xlarge)",
                "Teacher runs in its checkpoint dtype (see torch_dtype, fp32); an fp16 T4 deployment would be faster",
                "Cohere latency includes the network round trip from ap-southeast-1 to Bedrock; documents are sent sequentially in batches of 96",
                "Cohere peak RSS includes a borrowed 40,152 x 1024 search matrix; Cohere cost excludes the host that runs the search",
                "Teacher-CPU and Cohere index throughput are measured on a 2,008-chunk sample (every 20th chunk), not the full 40,152; static and teacher-GPU costs use the full index",
                "Per-query cost assumes one sequential stream kept fully busy; batching or concurrency changes it",
                "Exact search, no ANN index; RSS includes the Python runtime and imported libraries",
                "Catalog prices; credit coverage unverified",
            ],
        }
        path = WORK / "runtime-comparison.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        publish(path, "runtime-comparison.json")
        return report

    environment = dict(os.environ)
    threads = str(plan["protocol"]["cpu_threads"])
    environment.update({"PYTHONPATH": os.pathsep.join(filter(None, [str(INPUT), os.environ.get("PYTHONPATH")])),
                        "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                        "OPENBLAS_NUM_THREADS": threads})
    try:
        for name in plan["order"]:
            needed = NEEDS[name]
            if needed is not None and staged.get(needed) != "ok":
                failures[name] = f"not run: {needed} failed to stage"
                event("system-skipped", system=name, reason=failures[name])
                continue
            event("system-start", system=name)
            try:
                completed = subprocess.run(
                    [sys.executable, str(INPUT / "aws_runtime_comparison.py"), "--child", name],
                    env=environment, timeout=CHILD_TIMEOUT_SECONDS[name])
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            path = WORK / "results" / (name + ".json")
            if code != 0 or not path.exists():
                failures[name] = f"child exited {code}"
                event("system-failed", system=name, code=code)
                continue
            publish(path, "systems/" + name + ".json")      # preserve before judging
            results[name] = json.loads(path.read_text())
            event("system-complete", system=name,
                  p50_end_to_end_ms=results[name]["end_to_end"]["p50_ms"],
                  peak_rss_bytes=results[name]["memory"]["peak_rss_bytes"])
    finally:
        report = write_report()
    event("complete", status=report["status"],
          gates={k: v["passed"] for k, v in report["gates"].items()}, failures=failures)
    if report["status"] != "complete":
        raise SystemExit("measurements preserved, but a system failed or a gate did not pass")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    elif len(sys.argv) == 1:
        coordinator()
    else:
        raise SystemExit("usage: aws_runtime_comparison.py [--child SYSTEM]")
