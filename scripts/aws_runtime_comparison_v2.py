#!/usr/bin/env python3
"""Speed, memory and cost for every comparator in the 21-system table.

Same host type, index, timing protocol and child-process isolation as
`aws_runtime_comparison.py` (job pycon-runtime-20261013-112910-a01fd7), which
measured Tuklas, the teacher and Cohere v3.  This run adds the other dense
systems; BM25 and TF-IDF are excluded by instruction.

Runtimes:
  static-cpu      model2vec StaticModel from a local package or pinned HF snapshot
  transformer-gpu SentenceTransformer on the T4, full 40,152-chunk index
  transformer-cpu SentenceTransformer on the 4 vCPUs; index throughput from the
                  fixed 2,008-chunk sample, search over its GPU twin's matrix
  st-static-cpu   SentenceTransformer whose only module is a StaticEmbedding
  gensim-cpu      Word2Vec / fastText retrained in this job with the recipe of
                  pycon-classical-classic2 (the scored weights were never saved)

Gates: each system's development Recall@10 is compared with the value its
scoring job recorded.  Measurements are uploaded before any gate is judged,
and an interim report is published after every system.  Held-out questions
are never written for the children.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import md5, sha256
import gzip
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/runtime-comparison")
HF_CACHE = WORK / "hf-cache"
sys.path.insert(0, str(INPUT))

WINDOW, STRIDE = 192, 128
SAMPLE_STEP = 20
WARMUP_QUERIES = 8
LOCAL_REPETITIONS = 3
ONE_QUESTION = 1 / 197
WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    # Identical to aws_classical_baselines.py.
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def multipart_etag(path: Path, part_bytes: int) -> str:
    parts = []
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(part_bytes), b""):
            parts.append(md5(block).digest())
    if len(parts) == 1:
        return parts[0].hex()
    return md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"


def tree_bytes(folder: Path) -> int:
    # HF cache snapshots are symlinks into blobs/; follow them.
    seen, total = set(), 0
    for path in folder.rglob("*"):
        if ".cache" in path.parts or not path.is_file():
            continue
        real = path.resolve()
        if real not in seen:
            seen.add(real)
            total += real.stat().st_size
    return total


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


def load_sentence_transformer(spec: dict, device: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device=device,
                                cache_folder=str(HF_CACHE), trust_remote_code=False)
    # A StaticEmbedding module has no sequence limit and no setter.
    try:
        model.max_seq_length = int(spec.get("max_seq_length", 512))
    except AttributeError:
        pass
    return model


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
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    baseline_rss = process.memory_info().rss
    peak["rss"] = baseline_rss
    threading.Thread(target=watch, daemon=True).start()

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    extra: dict = {}
    started = time.perf_counter_ns()
    if runtime == "static-cpu":
        from model2vec import StaticModel
        folder = WORK / "models" / name
        model = StaticModel.from_pretrained(str(folder))
        prefix = (json.loads((folder / "config.json").read_text()).get("query_prefix", "")
                  if spec.get("query_prefix_from_config") else "")

        def encode_queries(values):
            return model.encode([prefix + v for v in values], batch_size=32,
                                max_length=None, use_multiprocessing=False)

        def encode_documents(values):
            return model.encode(values, batch_size=1024, max_length=None,
                                use_multiprocessing=False)
    elif runtime in ("transformer-gpu", "transformer-cpu", "st-static-cpu"):
        model = load_sentence_transformer(spec["model"], "cuda" if cuda else "cpu")
        extra["torch_dtype"] = str(next(model.parameters()).dtype)
        query_prefix = spec["model"].get("query_prefix", "")
        document_prefix = spec["model"].get("document_prefix", "")
        batch = int(spec["model"].get("batch_size", 64))

        def encode_queries(values):
            out = model.encode(values, batch_size=batch, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False,
                               **({"prompt": query_prefix} if query_prefix else {}))
            sync()
            return out

        def encode_documents(values):
            out = model.encode(values, batch_size=batch, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False,
                               **({"prompt": document_prefix} if document_prefix else {}))
            sync()
            return out
    elif runtime == "gensim-cpu":
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(str(WORK / "models" / name / "vectors.kv"), mmap=None)
        dimension = int(kv.vector_size)

        def embed(values):
            # The exact function that produced the scored numbers.
            out = np.zeros((len(values), dimension), dtype=np.float32)
            for i, text in enumerate(values):
                vectors = [kv[w] for w in words(text) if w in kv]
                if vectors:
                    out[i] = np.mean(vectors, axis=0)
            n = np.linalg.norm(out, axis=1, keepdims=True)
            return out / np.maximum(n, 1e-12)

        encode_queries = encode_documents = embed
    else:
        raise ValueError("unknown runtime " + runtime)
    sync()
    load_ms = (time.perf_counter_ns() - started) / 1e6
    loaded_rss = process.memory_info().rss

    encode_documents(sample[:64])
    started = time.perf_counter_ns()
    sample_vectors = normalized(np.asarray(encode_documents(sample), dtype=np.float32))
    sample_seconds = (time.perf_counter_ns() - started) / 1e9

    full_index = None
    if spec["full_index"]:
        started = time.perf_counter_ns()
        documents = normalized(np.asarray(encode_documents(texts), dtype=np.float32))
        full_index = {"chunks": len(texts), "seconds": (time.perf_counter_ns() - started) / 1e9}
        full_index["chunks_per_second"] = len(texts) / full_index["seconds"]
        if cuda:
            for target, value in ((name + "-documents.npy", documents),
                                  (name + "-sample.npy", sample_vectors)):
                with (WORK / (target + ".tmp")).open("wb") as stream:
                    np.save(stream, value)
                os.replace(WORK / (target + ".tmp"), WORK / target)
    else:
        borrowed = WORK / (spec["borrow_from"] + "-documents.npy")
        if borrowed.exists():
            documents = np.load(borrowed)
            extra["search_matrix"] = spec["borrow_from"] + " document vectors (same model; search time only)"
        else:
            generator = np.random.default_rng(42)
            documents = normalized(generator.standard_normal(
                (len(texts), sample_vectors.shape[1]), dtype=np.float32))
            extra["search_matrix"] = "random matrix of the same shape (GPU twin missing)"
        if documents.shape != (len(texts), sample_vectors.shape[1]):
            raise ValueError("borrowed search matrix has the wrong shape")
    corpus_rss = process.memory_info().rss

    def search(vector):
        parent_scores = np.maximum.reduceat(documents @ vector, starts)
        top = np.argpartition(-parent_scores, 9)[:10]
        return top[np.argsort(-parent_scores[top], kind="stable")]

    for text in questions[:WARMUP_QUERIES]:
        search(normalized(np.asarray(encode_queries([text]), dtype=np.float32))[0])
    encode_ms, search_ms, total_ms = [], [], []
    for _ in range(LOCAL_REPETITIONS):
        for text in questions:
            t0 = time.perf_counter_ns()
            vector = normalized(np.asarray(encode_queries([text]), dtype=np.float32))[0]
            t1 = time.perf_counter_ns()
            search(vector)
            t2 = time.perf_counter_ns()
            encode_ms.append((t1 - t0) / 1e6)
            search_ms.append((t2 - t1) / 1e6)
            total_ms.append((t2 - t0) / 1e6)

    stopped.set()
    peak["rss"] = max(peak["rss"], process.memory_info().rss)
    serving_peak_rss = peak["rss"]

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
        # The shootout's encoder: strip pad/unknown, mean, normalise.
        from model2vec.train.utils import get_probable_pad_token_id
        from multiwikiqa_compact_retrieval import strip_padding_and_unknown
        table = np.asarray(model.embedding, dtype=np.float32)
        pad_id, unknown_id = get_probable_pad_token_id(model.tokenizer), model.unk_token_id

        def harness_encode(values):
            out = np.zeros((len(values), table.shape[1]), dtype=np.float32)
            for begin in range(0, len(values), 2048):
                for offset, encoding in enumerate(model.tokenizer.encode_batch_fast(
                        values[begin:begin + 2048], add_special_tokens=False)):
                    ids = strip_padding_and_unknown(encoding.ids, pad_id=pad_id, unknown_id=unknown_id)
                    if ids:
                        out[begin + offset] = table[ids].mean(axis=0)
            return normalized(out)

        harness_documents = harness_encode(texts)
        harness_queries = harness_encode([prefix + q for q in questions])
        gate = development_recall(harness_queries, harness_documents)
        library_queries = normalized(np.asarray(encode_queries(questions), dtype=np.float32))
        extra["library_vs_harness_min_cosine"] = {
            "queries": float(np.min(np.sum(library_queries * harness_queries, axis=1))),
            "documents": float(np.min(np.sum(documents * harness_documents, axis=1)))}
        del harness_documents, table
    elif runtime in ("transformer-gpu", "st-static-cpu", "gensim-cpu") or (
            runtime == "transformer-cpu" and extra["search_matrix"].startswith(spec["borrow_from"])):
        query_vectors = normalized(np.asarray(encode_queries(questions), dtype=np.float32))
        gate = development_recall(query_vectors, documents)
    if runtime == "transformer-cpu":
        twin = WORK / (spec["borrow_from"] + "-sample.npy")
        if twin.exists():
            reference = np.load(twin)
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
                   "rss_after_index_bytes": corpus_rss, "peak_rss_bytes": serving_peak_rss,
                   "peak_rss_scope": "load, indexing, timed queries; excludes the quality gate",
                   "model_rss_increment_bytes": loaded_rss - baseline_rss},
        "development_gate": gate,
        **extra,
    }
    if runtime == "static-cpu":
        folder = WORK / "models" / name
        result["weights_bytes"] = (folder / "model.safetensors").stat().st_size
        result["artifact_bytes"] = tree_bytes(folder)
    elif runtime == "gensim-cpu":
        folder = WORK / "models" / name
        result["artifact_bytes"] = tree_bytes(folder)
        result["vocabulary"] = len(kv)
        result["weights"] = "retrained in this job with the pycon-classical-classic2 recipe"
    else:
        snapshot = Path(json.loads((WORK / "snapshots.json").read_text())[spec["model"]["id"]])
        result["artifact_bytes"] = tree_bytes(snapshot)
        result["weights_bytes"] = sum(p.resolve().stat().st_size for p in snapshot.rglob("*.safetensors")) or \
            sum(p.resolve().stat().st_size for p in snapshot.rglob("*.bin"))
        if cuda:
            result["memory"]["gpu"] = {
                "device": torch.cuda.get_device_name(0),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    target = WORK / "results" / (name + ".json")
    target.parent.mkdir(exist_ok=True)
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


# ------------------------------------------------------------------ staging helpers

def prefetch(model_id: str, revision: str) -> None:
    """Download exactly what SentenceTransformer needs, then exit."""
    load_sentence_transformer({"id": model_id, "revision": revision}, "cpu")


def train_gensim(name: str) -> None:
    from gensim.models import FastText, Word2Vec
    plan = json.loads((WORK / "plan.json").read_text())
    recipe = plan["systems"][name]["recipe"]
    sentences = []
    with gzip.open(INPUT / "training.tl.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                sentences.append(words(json.loads(line)["text"]))
    if len(sentences) != recipe["expected_source_rows"]:
        raise ValueError(f"corpus changed: {len(sentences)} rows")
    builder = FastText if recipe["kind"] == "fasttext" else Word2Vec
    model = builder(sentences, vector_size=recipe["dimension"], sg=recipe["sg"],
                    window=recipe["window"], min_count=recipe["min_count"],
                    workers=recipe["workers"], epochs=recipe["epochs"], seed=recipe["seed"])
    folder = WORK / "models" / name
    folder.mkdir(parents=True, exist_ok=False)
    model.wv.save(str(folder / "vectors.kv"))


# --------------------------------------------------------------------- coordinator

def coordinator() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
         "model2vec==0.9.0", "sentence-transformers==5.7.0", "transformers==5.3.0",
         "huggingface-hub==1.33.0", "numpy==2.5.3", "boto3==1.43.108", "psutil==7.1.3",
         "gensim==4.4.0"],
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
    HF_CACHE.mkdir()
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
    (WORK / "corpus.json").write_text(json.dumps({
        "records": records, "owners": owners, "parents": layout["parents"],
        "sample_indices": list(range(0, len(records), SAMPLE_STEP))}))
    (WORK / "development.json").write_text(json.dumps(development))
    event("index-built", chunks=len(records), development=len(development))

    environment = dict(os.environ)
    threads = str(plan["protocol"]["cpu_threads"])
    environment.update({"PYTHONPATH": os.pathsep.join(filter(None, [str(INPUT), os.environ.get("PYTHONPATH")])),
                        "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                        "OPENBLAS_NUM_THREADS": threads, "HF_HOME": str(HF_CACHE)})
    script = str(INPUT / "aws_runtime_comparison_v2.py")

    # ---- staging: each failure disables only the systems that need it ------
    staged: dict[str, str] = {}
    snapshots: dict[str, str] = {}
    for name, spec in plan["systems"].items():
        key = spec["stage_key"]
        if key in staged:
            continue
        try:
            if spec["runtime"] == "static-cpu" and "s3_prefix" in spec:
                target = WORK / "models" / name
                target.mkdir()
                prefix = spec["s3_prefix"].rstrip("/") + "/"
                for row in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
                    relative = row["Key"][len(prefix):]
                    if relative:
                        s3.download_file(bucket, row["Key"], str(target / relative))
                if multipart_etag(target / "model.safetensors", spec["etag_part_bytes"]) != spec["weights_etag"]:
                    raise ValueError("weights multipart ETag differs from the S3 object")
            elif spec["runtime"] == "static-cpu":
                # Only the files model2vec serves from; repos also ship ONNX
                # exports and duplicate tables that would inflate the size.
                snapshot_download(repo_id=spec["model"]["id"], revision=spec["model"]["revision"],
                                  local_dir=WORK / "models" / name,
                                  allow_patterns=["config.json", "model.safetensors",
                                                  "tokenizer.json", "modules.json", "README.md"])
            elif spec["runtime"] == "gensim-cpu":
                subprocess.run([sys.executable, script, "--train-gensim", name], env=environment,
                               check=True, timeout=1800)
            else:
                subprocess.run([sys.executable, script, "--prefetch", spec["model"]["id"],
                                spec["model"]["revision"]], env=environment, check=True, timeout=1800)
                snapshot = (HF_CACHE / ("models--" + spec["model"]["id"].replace("/", "--"))
                            / "snapshots" / spec["model"]["revision"])
                if not snapshot.is_dir():
                    raise FileNotFoundError(f"prefetch left no snapshot at {snapshot}")
                snapshots[spec["model"]["id"]] = str(snapshot)
            staged[key] = "ok"
            event("staged", key=key)
        except Exception as error:
            staged[key] = repr(error)[:300]
            event("stage-failed", key=key, error=staged[key])
    (WORK / "snapshots.json").write_text(json.dumps(snapshots))

    versions = {}
    for package in ("model2vec", "sentence-transformers", "transformers", "tokenizers",
                    "torch", "numpy", "gensim", "huggingface-hub"):
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

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    def write_report() -> dict:
        gates = {}
        for name, result in results.items():
            spec = plan["systems"][name]
            expected = spec.get("expected_development")
            observed = (result["development_gate"] or {}).get("development_recall_at_10")
            if expected is None:
                continue
            difference = None if observed is None else abs(observed - expected)
            exact = difference is not None and difference < 1e-9
            if spec["gate_mode"] == "exact":
                passed = exact
            elif spec["gate_mode"] == "one-question":
                passed = difference is not None and difference <= ONE_QUESTION + 1e-9
            else:  # "band": retrained models, not bit-reproducible; catches a broken pipeline
                passed = difference is not None and difference <= spec["gate_band"]
            gates[name] = {"observed": observed, "expected": expected, "mode": spec["gate_mode"],
                           "exact": exact, "passed": passed}
            if "cpu_vs_gpu_sample_min_cosine" in result:
                cosine = result["cpu_vs_gpu_sample_min_cosine"]
                gates[name + "-matches-gpu"] = {"observed": cosine, "minimum": 0.999, "mode": "exact",
                                                "exact": cosine >= 0.999, "passed": cosine >= 0.999}
        rates = plan["usd_per_hour"]
        costs = {}
        for name, result in results.items():
            kind = "gpu" if result["device"] == "cuda" else "cpu"
            rate = rates[kind]
            source = result["index_full"] or result["index_sample"]
            costs[name] = {"basis": f"{plan['instances'][kind]} at ${rate}/h, one sequential stream kept busy",
                           "throughput_source": "full 40,152-chunk index" if result["index_full"] else "2,008-chunk sample",
                           "usd_per_million_queries": 1e6 * result["end_to_end"]["mean_ms"] / 1000 / 3600 * rate,
                           "usd_per_million_chunks_indexed": 1e6 / source["chunks_per_second"] / 3600 * rate,
                           "usd_to_index_40152_chunks": 40_152 / source["chunks_per_second"] / 3600 * rate}
        complete = (not failures and len(results) == len(plan["order"])
                    and all(g["passed"] for g in gates.values()))
        report = {
            "status": "complete" if complete else "needs-review",
            "systems": results, "failures": failures, "staging": staged, "gates": gates, "costs": costs,
            "hardware": {"instance_type": plan["instance_type"], "platform": platform.platform(),
                         "cpu_model": cpu_model, "logical_cpus": psutil.cpu_count(),
                         "physical_cpus": psutil.cpu_count(logical=False),
                         "ram_bytes": psutil.virtual_memory().total,
                         "cpu_threads": plan["protocol"]["cpu_threads"]},
            "library_versions": versions, "protocol": plan["protocol"],
            "heldout_use": "none; timing and gates use the 197 development questions only",
            "limitations": plan["limitations"],
        }
        path = WORK / "runtime-comparison-v2.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        publish(path, "runtime-comparison-v2.json")
        return report

    try:
        for name in plan["order"]:
            spec = plan["systems"][name]
            needed = spec["stage_key"]
            if staged.get(needed) != "ok":
                failures[name] = f"not run: {needed} failed to stage"
                event("system-skipped", system=name, reason=failures[name])
                continue
            event("system-start", system=name)
            try:
                completed = subprocess.run([sys.executable, script, "--child", name], env=environment
                                           | {"HF_HUB_OFFLINE": "1"}, timeout=spec["timeout_seconds"])
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            path = WORK / "results" / (name + ".json")
            if code != 0 or not path.exists():
                failures[name] = f"child exited {code}"
                event("system-failed", system=name, code=code)
            else:
                publish(path, "systems/" + name + ".json")      # preserve before judging
                results[name] = json.loads(path.read_text())
                event("system-complete", system=name,
                      p50_end_to_end_ms=results[name]["end_to_end"]["p50_ms"])
            if spec["runtime"] == "transformer-cpu":
                # The GPU twin's matrix is no longer needed; free the disk.
                for suffix in ("-documents.npy", "-sample.npy"):
                    (WORK / (spec["borrow_from"] + suffix)).unlink(missing_ok=True)
            write_report()                                       # interim, every system
    finally:
        report = write_report()
    event("complete", status=report["status"], failures=failures,
          gates={k: v["passed"] for k, v in report["gates"].items()})
    if report["status"] != "complete":
        raise SystemExit("measurements preserved, but a system failed or a gate did not pass")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "--prefetch":
        prefetch(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "--train-gensim":
        train_gensim(sys.argv[2])
    elif len(sys.argv) == 1:
        coordinator()
    else:
        raise SystemExit("usage: aws_runtime_comparison_v2.py [--child SYSTEM | --prefetch ID REV | --train-gensim SYSTEM]")
