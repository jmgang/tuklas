"""Fresh AWS-only DisastIR benchmark; no old run imports or cache reuse.

Coordinator builds shared boundaries once, executes all fresh corpus builds,
then measures each model in isolation. Every result is exclusively created.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(os.environ.get("PYCON_FRESH_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.grade3 import select_cohorts, score_top10_query, summarize, paired_with_teacher, TOP10_METRICS
from pycon_model2vec_demo.parent_retrieval import shared_slices, parent_top_k
from pycon_model2vec_demo.segmented_encoding import SegmentPolicy
from pycon_model2vec_demo.measurement import host_metadata, peak_rss_bytes, _timing_summary
from pycon_model2vec_demo.encoders import load_e5_encoder, load_static_encoder, _validate_vectors
from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere

LOCAL = ("potion_retrieval", "plain_student", "custom_student", "tokenlearn_custom_student", "teacher")
NAMES = (*LOCAL, "cohere_v3")
COHERE = {"model_id": "cohere.embed-english-v3", "dimension": 1024,
          "truncate": "NONE", "full_content": True, "maximum_input_characters": 2048}


def read(path):
    return json.loads(path.read_text())


def file_digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(piece)
    return result.hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, indent=2, ensure_ascii=False)


def event(directory, **value):
    payload = {"time": datetime.now(timezone.utc).isoformat(), **value}
    path = directory / "events" / f"{time.time_ns()}-{uuid4().hex}.json"
    save(path, payload)
    print(json.dumps(payload), flush=True)
    if os.environ.get("PYCON_RESULT_BUCKET"):
        import boto3
        boto3.client("s3", region_name="ap-southeast-1").put_object(
            Bucket=os.environ["PYCON_RESULT_BUCKET"],
            Key=os.environ["PYCON_RESULT_PREFIX"] + "/events/" + path.name,
            Body=path.read_bytes(), IfNoneMatch="*", ServerSideEncryption="AES256")


def prepare():
    data = read(ROOT / "configs/disastir-qa-all-categories.json")
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(data_dir=ROOT / "data/raw", revision=data["revision"],
        category_checksums=data["categories"], seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"])
    mapped, mapping = map_query_groups(prepared.evaluation, corpus)
    _, queries, selection = select_cohorts(prepared.evaluation, mapped, corpus)
    dev = tuple(q for category in sorted(data["categories"]) for q in
        sorted((q for q in prepared.development if q.category == category), key=lambda q: q.query_id)[:4])
    if len(corpus.ids) != 239704 or len(queries) != 1127 or len(dev) != 32:
        raise ValueError("Approved corpus/query counts changed")
    return corpus, queries, dev, mapping, selection


def entries():
    return read(ROOT / "fresh-input-manifest.json")["frozen_models"]


def policies(name, manifest):
    from tokenizers import Tokenizer
    row = manifest["cohere_v3_tokenizer" if name == "cohere_v3" else name]
    tokenizer = Tokenizer.from_file(str(ROOT / row["path"] / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    special = name in ("teacher", "cohere_v3")
    unknown = tokenizer.token_to_id("[UNK]")

    def counts(texts, add_special=special):
        values = tokenizer.encode_batch(texts, add_special_tokens=add_special)
        return [len(v.ids) if add_special else sum(i != unknown for i in v.ids) for v in values]

    char_limit = (2048 if name == "cohere_v3" else None if name == "teacher" else
                  512 * int(np.median([len(t) for t in tokenizer.get_vocab()])))
    role_policies = {}
    for role in ("corpus", "queries", "development"):
        prefix = "" if name == "cohere_v3" else row["settings"]["passage_prefix" if role == "corpus" else "query_prefix"]
        role_policies[role] = SegmentPolicy(lambda text: counts([text])[0],
            lambda text: counts([text], False)[0], 480 if name == "cohere_v3" else 512,
            char_limit, prefix)
    return role_policies, counts


def make_catalog(directory):
    corpus, queries, dev, mapping, selection = prepare()
    manifest = entries()
    specs = {name: policies(name, manifest) for name in NAMES}
    for name, (rules, _) in specs.items():
        if any(not rules["queries"].fits(q.query) for q in queries):
            raise ValueError(f"Evaluation query exceeds {name} budget")
        if any(not rules["development"].fits(q.query) for q in dev):
            raise ValueError(f"Timing query exceeds {name} budget")
    segments, boundaries, split, totals = [], [], 0, {name: 0 for name in NAMES}
    for start in range(0, len(corpus.ids), 512):
        block = corpus.texts[start:start+512]
        lengths = {name: count([rules["corpus"].prefix+t for t in block])
                   for name, (rules, count) in specs.items()}
        for offset, text in enumerate(block):
            fits = all(lengths[name][offset] <= rules["corpus"].token_limit and
                (rules["corpus"].character_limit is None or len(rules["corpus"].prefix+text) <= rules["corpus"].character_limit)
                for name, (rules, _) in specs.items())
            slices = [[0, len(text)]] if fits else shared_slices(text, [rules["corpus"] for rules, _ in specs.values()])
            assert "".join(text[a:b] for a, b in slices) == text
            boundaries.append(slices)
            split += len(slices) > 1
            for a, b in slices:
                if text[a:b].strip():
                    segments.append([start+offset, a, b])
                    for name, (rules, _) in specs.items():
                        totals[name] += lengths[name][offset] if fits else rules["corpus"].count(rules["corpus"].prefix+text[a:b])
        if start % 20480 == 0:
            event(directory, stage="shared-preflight", passages_checked=start+len(block), total_passages=len(corpus.ids))
    catalog = {"segments": segments, "boundaries": boundaries, "split_passages": split,
        "passage_count": len(corpus.ids), "segment_count": len(segments),
        "query_ids": [q.query_id for q in queries], "development_query_ids": [q.query_id for q in dev],
        "corpus_ids_sha256": digest(corpus.ids), "corpus_texts_sha256": digest(corpus.texts),
        "query_texts_sha256": digest([q.query for q in queries]),
        "token_totals": totals, "mapping": mapping, "selection": selection,
        "policy": "fresh identical lossless segments; max similarity per unique original parent; top10; no pooling or old vectors"}
    if totals["cohere_v3"] / 1e6 * .10 > 15:
        raise ValueError("Cohere corpus API estimate exceeds $15 guard")
    save(directory / "shared-catalog.json", catalog)
    save(directory / "shared-catalog.sha256.json", {"sha256": file_digest(directory / "shared-catalog.json")})
    event(directory, stage="shared-preflight-complete", passage_count=len(corpus.ids),
          segment_count=len(segments), query_count=len(queries), split_passages=split)


class Encoder:
    def __init__(self, name):
        self.name = name
        manifest = entries()
        self.rules, _ = policies(name, manifest)
        self.requests = []
        if name == "cohere_v3":
            import boto3
            from botocore.config import Config
            self.base = BedrockCohere(boto3.client("bedrock-runtime", region_name="ap-southeast-1",
                config=Config(connect_timeout=10, read_timeout=90, retries={"max_attempts": 0})), COHERE)
            self.dimension = 1024
            self.artifact_bytes = None
        else:
            row = manifest[name]
            path, settings = ROOT / row["path"], row["settings"]
            for relative, expected in row["files_sha256"].items():
                if file_digest(path / relative) != expected:
                    raise ValueError(f"Frozen model changed: {name}/{relative}")
            self.artifact_bytes = sum((path / relative).stat().st_size for relative in row["files_sha256"])
            self.base = load_e5_encoder(str(path)) if name == "teacher" else load_static_encoder(
                str(path), query_prefix=settings["query_prefix"], passage_prefix=settings["passage_prefix"])
            self.dimension = self.base.dimension

    def encode(self, texts, role):
        policy = self.rules[role]
        if any(not policy.fits(text) for text in texts):
            raise ValueError(f"Unsegmented over-budget input for {self.name}/{role}")
        if self.name == "cohere_v3":
            vectors, request = self.base.encode(texts, "search_document" if role == "corpus" else "search_query")
            assert request["character_truncated_inputs"] == 0
            self.requests.append(request)
        elif self.name == "teacher":
            vectors = (self.base.encode_passages if role == "corpus" else self.base.encode_queries)(texts)
        else:
            vectors = self.base.model.encode([policy.prefix+text for text in texts], max_length=None,
                use_multiprocessing=False, batch_size=1024)
        return _validate_vectors(vectors, len(texts), self.dimension)


def worker(directory, name, phase):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=8)
    corpus, queries, dev, _, _ = prepare()
    catalog = read(directory / "shared-catalog.json")
    if catalog["corpus_ids_sha256"] != digest(corpus.ids) or catalog["query_ids"] != [q.query_id for q in queries]:
        raise ValueError("Worker dataset order changed")
    owners = np.asarray([row[0] for row in catalog["segments"]], dtype=np.int64)
    started = time.perf_counter()
    encoder = Encoder(name)
    load_seconds = time.perf_counter() - started
    model_dir = directory / name
    model_dir.mkdir(exist_ok=True)
    vector_path = model_dir / "segment-vectors.npy"
    if phase == "build":
        if vector_path.exists():
            raise FileExistsError("Fresh build cannot resume or replace vectors")
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32,
            shape=(len(owners), encoder.dimension))
        size = 96 if name == "cohere_v3" else 32 if name == "teacher" else 1024
        starts = range(0, len(owners), size)
        def batch(start):
            rows = catalog["segments"][start:start+size]
            texts = [corpus.texts[owner][a:b] for owner, a, b in rows]
            return start, encoder.encode(texts, "corpus")
        event(directory, stage="corpus-build-start", model=name, total_segments=len(owners))
        build_started = time.perf_counter()
        if name == "cohere_v3":
            # Limit outstanding futures as well as running API calls.
            with ThreadPoolExecutor(max_workers=2) as pool:
                for group_start in range(0, len(owners), size*8):
                    jobs = [pool.submit(batch, a) for a in range(group_start, min(group_start+size*8, len(owners)), size)]
                    for job in as_completed(jobs):
                        a, values = job.result()
                        vectors[a:a+len(values)] = values
                    done = min(group_start+size*8, len(owners))
                    vectors.flush()
                    event(directory, stage="corpus-build", model=name, completed_segments=done, total_segments=len(owners))
                    if sum(r["input_tokens"] or r["characters"] / 3 for r in encoder.requests) / 1e6 * .10 > 15:
                        raise RuntimeError("Cohere observed API cost estimate exceeds guard")
        else:
            for a in starts:
                _, values = batch(a)
                vectors[a:a+len(values)] = values
                if (a // size) % 100 == 0 or a+len(values) == len(owners):
                    vectors.flush()
                    event(directory, stage="corpus-build", model=name, completed_segments=a+len(values), total_segments=len(owners))
        vectors.flush()
        save(model_dir / "corpus-build.json", {"vectors_sha256": file_digest(vector_path),
            "fresh_segment_count": len(owners), "old_rows_reused": 0,
            "build_seconds": time.perf_counter()-build_started,
            "cold_load_seconds": load_seconds, "requests": encoder.requests,
            "peak_rss_bytes": peak_rss_bytes(), "vector_bytes": vector_path.stat().st_size})
        event(directory, stage="corpus-build-complete", model=name, completed_segments=len(owners))
        return
    record = read(model_dir / "corpus-build.json")
    if file_digest(vector_path) != record["vectors_sha256"]:
        raise ValueError("Fresh vector checksum mismatch")
    vectors = np.load(vector_path, mmap_mode="r")
    query_vectors = np.concatenate([encoder.encode([q.query for q in queries[a:a+96]], "queries")
        for a in range(0, len(queries), 96)])
    with (model_dir / "query-vectors.npy").open("xb") as stream:
        np.save(stream, query_vectors, allow_pickle=False)
    rows = []
    for a in range(0, len(queries), 16):
        rankings = parent_top_k(query_vectors[a:a+16], vectors, owners, corpus.ids, k=10)
        for query, ranking in zip(queries[a:a+16], rankings):
            rows.append({"query_id": query.query_id, "category": query.category,
                "metrics": score_top10_query(query, ranking), "top10": [r.__dict__ for r in ranking]})
        if a % 160 == 0:
            event(directory, stage="quality-scoring", model=name, completed_queries=len(rows), total_queries=len(queries))
    encode_times, searches, total = [], [], []
    for repetition in range(7):
        for q in dev:
            before = time.perf_counter()
            values = encoder.encode([q.query], "development")
            middle = time.perf_counter()
            parent_top_k(values, vectors, owners, corpus.ids, k=10)
            after = time.perf_counter()
            if repetition >= 2:
                encode_times.append(middle-before)
                searches.append(after-middle)
                total.append(after-before)
        event(directory, stage="isolated-measurement", model=name, round=repetition+1)
    result = {"model": name, "per_query": rows, "summary": summarize(rows, metrics=TOP10_METRICS),
        "fresh_vectors_sha256": record["vectors_sha256"], "shared_catalog_sha256": file_digest(directory / "shared-catalog.json"),
        "environment": host_metadata(), "threads": 8, "cold_load_seconds": load_seconds,
        "measurement": {"query_encode": _timing_summary(encode_times), "search": _timing_summary(searches),
            "end_to_end": _timing_summary(total), "warmup_rounds": 2, "measured_rounds": 5,
            "development_query_ids": [q.query_id for q in dev], "peak_rss_bytes": peak_rss_bytes(),
            "model_bytes": encoder.artifact_bytes, "vector_bytes": vector_path.stat().st_size},
        "requests": encoder.requests, "corpus_build": record,
        "memory_note": "Peak worker RSS includes dataset, tokenizers, runtime and touched vector pages; Cohere server RAM unknown"}
    save(model_dir / "query-results.json", result)
    event(directory, stage="model-evaluation-complete", model=name, metrics=result["summary"]["overall"])


def coordinator(directory):
    directory.mkdir(exist_ok=False)
    with (directory / "runner.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = read(ROOT / "aws-input-verification.json")["private_bedrock_dns_ips"]
        resolved = {r[4][0] for r in socket.getaddrinfo("bedrock-runtime.ap-southeast-1.amazonaws.com", 443, family=socket.AF_INET)}
        if not resolved or not resolved <= set(expected):
            raise ValueError("Private Bedrock DNS verification failed")
        import boto3
        identity = boto3.client("sts", region_name="ap-southeast-1").get_caller_identity()
        save(directory / "protocol.json", {"dataset": read(ROOT / "configs/disastir-qa-all-categories.json"),
            "frozen_models": entries(), "identity": identity, "private_bedrock_ips": sorted(resolved),
            "source_sha256": file_digest(Path(__file__)), "top_k": 10, "threads": 8,
            "fresh_outputs_only": True, "old_outputs_reused": False,
            "libraries": {n: version(n) for n in ("numpy", "torch", "model2vec", "sentence-transformers", "transformers")},
            "limitations": "Grade-3 selection was post-hoc to earlier experiments; pooled relevance labels are not exhaustive; corpus-trained students are transductive; v4 remains billing-blocked; no ANN."})
        event(directory, stage="benchmark-started", pid=os.getpid(), root=str(ROOT))
        make_catalog(directory)
        # Bulk builds may overlap. Timed operational measurements never do.
        def launch(name, phase):
            log = (directory / f"{name}-{phase}.log").open("x")
            return subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--directory", str(directory), "--worker", name, "--phase", phase], stdout=log, stderr=subprocess.STDOUT), log
        remote, remote_log = launch("cohere_v3", "build")
        try:
            for name in LOCAL:
                process, log = launch(name, "build")
                code = process.wait()
                log.close()
                if code:
                    raise RuntimeError(f"Fresh {name} build failed, see preserved log")
            code = remote.wait()
            if code:
                raise RuntimeError("Fresh Cohere build failed, see preserved log")
        finally:
            remote_log.close()
            # Stop only our active child on coordinator failure; never delete outputs.
            if remote.poll() is None:
                remote.terminate()
                remote.wait()
        event(directory, stage="all-builds-complete; isolated-measurements-starting")
        for name in NAMES:
            process, log = launch(name, "evaluate")
            code = process.wait()
            log.close()
            if code:
                raise RuntimeError(f"Fresh {name} evaluation failed, see preserved log")
        results = {name: read(directory / name / "query-results.json") for name in NAMES}
        report = {"protocol": read(directory / "protocol.json"), "models": results,
            "paired_with_teacher": {name: paired_with_teacher(results["teacher"]["per_query"], results[name]["per_query"], metrics=TOP10_METRICS) for name in NAMES if name != "teacher"},
            "cohere_v4": {"status": "blocked", "reason": "Previously verified Marketplace INVALID_PAYMENT_INSTRUMENT; no billing changes authorized"}}
        save(directory / "reports" / "combined-report.json", report)
        heading = "| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Encode p95 ms | Search p95 ms | Peak RSS MiB | Model MiB | Vectors MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        lines = []
        for name, result in results.items():
            m, op = result["summary"]["overall"], result["measurement"]
            cells = [name, *(f"{100*m[k]:.2f}%" for k in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10")),
                f"{m['mrr_at_10']:.4f}", f"{m['ndcg_at_10']:.4f}", f"{op['query_encode']['p95_ms']:.2f}",
                f"{op['search']['p95_ms']:.2f}", f"{op['peak_rss_bytes']/2**20:.1f}",
                "unknown" if op["model_bytes"] is None else f"{op['model_bytes']/2**20:.1f}", f"{op['vector_bytes']/2**20:.1f}"]
            lines.append("| " + " | ".join(cells) + " |")
        with (directory / "reports" / "combined-report.md").open("x") as output:
            output.write(heading + "\n".join(lines) + "\n\nAll results generated on AWS; whole-worker RSS, not weights-only RAM. v4 billing-blocked.\n")
        if os.environ.get("PYCON_RESULT_BUCKET"):
            client = boto3.client("s3", region_name="ap-southeast-1")
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".md", ".log") and "events" not in path.parts:
                    client.put_object(Bucket=os.environ["PYCON_RESULT_BUCKET"], Key=os.environ["PYCON_RESULT_PREFIX"]+"/"+path.relative_to(directory).as_posix(), Body=path.read_bytes(), IfNoneMatch="*", ServerSideEncryption="AES256")
        event(directory, stage="benchmark-complete", report=str(directory / "reports/combined-report.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--worker", choices=NAMES)
    parser.add_argument("--phase", choices=("build", "evaluate"))
    args = parser.parse_args()
    try:
        worker(args.directory, args.worker, args.phase) if args.worker else coordinator(args.directory)
    except Exception as error:
        if args.directory.exists():
            event(args.directory, stage="failed", model=args.worker, error=repr(error))
        raise
