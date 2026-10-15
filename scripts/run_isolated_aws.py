"""Replacement phase workers: fresh remote encoding, isolated timing and quality.

The controller must invoke each worker on its designated host. This file does
not provision compute, resume outputs or reuse rejected benchmark data.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ["PYCON_FRESH_ROOT"])
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import pycon_model2vec_demo  # Fix package ownership before the shared input helper adds its path.
import numpy as np
import run_fresh_aws_benchmark as shared  # Only pure input/catalog functions; never its workers/coordinator.
from pycon_model2vec_demo.isolated_client import RemoteEncoder, private_http_transport, sagemaker_transport
from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere, request_body, decode_vectors
from pycon_model2vec_demo.parent_retrieval import parent_top_k
from pycon_model2vec_demo.grade3 import score_top10_query, summarize, TOP10_METRICS
from pycon_model2vec_demo.measurement import _timing_summary, host_metadata, peak_rss_bytes


def latency_summary(samples):
    result = _timing_summary(samples)
    result.update(p50_ms=float(np.percentile(samples, 50))*1000,
                  p99_ms=float(np.percentile(samples, 99))*1000)
    return result


def session_clients():
    import boto3
    from botocore.config import Config
    # Timed requests must not hide SDK retries. Bulk Cohere retries are recorded
    # by its existing bounded adapter and never mixed into measured phases.
    cfg = Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 0})
    return boto3.client("sagemaker-runtime", region_name="ap-southeast-1", config=cfg), boto3.client(
        "bedrock-runtime", region_name="ap-southeast-1", config=cfg)


class CohereEncoder:
    dimension = 1024

    def __init__(self, client, *, measured=False):
        self.client = client
        self.measured = measured
        self.bulk = BedrockCohere(client, shared.COHERE)
        self.identity = {"model": "cohere_v3", "resource_id": "bedrock:cohere.embed-english-v3",
            "device": "managed-provider", "server_memory": None, "model_bytes": None}

    def encode(self, texts, role):
        role = "search_document" if role == "corpus" else "search_query"
        if not self.measured:
            return self.bulk.encode(texts, role)
        started = time.perf_counter()
        reply = self.client.invoke_model(modelId=shared.COHERE["model_id"],
            body=json.dumps(request_body(texts, role, shared.COHERE)),
            contentType="application/json", accept="application/json")
        try:
            value = json.loads(reply["body"].read())
        finally:
            reply["body"].close()
        elapsed = time.perf_counter() - started
        headers = reply["ResponseMetadata"]["HTTPHeaders"]
        provider = headers.get("x-amzn-bedrock-invocation-latency")
        return decode_vectors(value, texts, 1024), {"client_round_trip_seconds": elapsed,
            "provider_service_seconds": float(provider) / 1000 if provider else None,
            "request_id": reply["ResponseMetadata"]["RequestId"], "server_rss_bytes": None}


def encoder(name, settings, *, measured=False):
    sm, bedrock = session_clients()
    if name == "cohere_v3":
        return CohereEncoder(bedrock, measured=measured)
    send = sagemaker_transport(sm, settings["teacher_endpoint"]) if name == "teacher" else private_http_transport(
        f"http://{settings['server_private_ip']}:8080/encode")
    return RemoteEncoder(send, client_id=settings["client_instance"], model=name,
        fingerprint=shared.entries()[name]["fingerprint"])


def publish(directory, path, settings):
    import boto3
    key = settings["prefix"] + "/benchmark/" + path.relative_to(directory).as_posix()
    with path.open("rb") as stream:
        boto3.client("s3", region_name="ap-southeast-1").put_object(Bucket=settings["bucket"], Key=key,
            Body=stream, IfNoneMatch="*", ServerSideEncryption="AES256",
            Metadata={"sha256": shared.file_digest(path)})


def build(directory, name, settings):
    if settings["local_resource_id"] != settings["client_instance"]:
        raise ValueError("Build controller must run on the declared client host")
    corpus, queries, dev, _, _ = shared.prepare()
    catalog = shared.read(directory / "shared-catalog.json")
    if catalog["corpus_texts_sha256"] != shared.digest(corpus.texts):
        raise ValueError("Catalog provenance differs")
    model_dir = directory / name
    model_dir.mkdir(exist_ok=False)
    enc = encoder(name, settings)
    size = 96 if name == "cohere_v3" else 32 if name == "teacher" else 1024
    path = model_dir / "segment-vectors.npy"
    # Directory creation above rejects existing/partial output before w+.
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
        shape=(len(catalog["segments"]), enc.dimension))
    started = time.perf_counter()
    requests = []
    for a in range(0, len(values), size):
        block = catalog["segments"][a:a+size]
        texts = [corpus.texts[owner][start:end] for owner, start, end in block]
        encoded, metadata = enc.encode(texts, "corpus")
        values[a:a+len(block)] = encoded
        if name == "cohere_v3":
            requests.append(metadata)
            estimate = sum(x["input_tokens"] or x["characters"] / 3 for x in requests) / 1e6 * .1
            if estimate > 15:
                raise RuntimeError("Cohere observed estimate exceeded $15 guard")
        if a % (size*100) == 0 or a+len(block) == len(values):
            values.flush()
            shared.event(directory, stage="fresh-remote-build", model=name,
                completed_segments=a+len(block), total_segments=len(values))
    values.flush()
    build_seconds = time.perf_counter()-started
    for group_name, group in (("queries", queries), ("development", dev)):
        blocks = [enc.encode([q.query for q in group[a:a+size]], group_name)[0]
                  for a in range(0, len(group), size)]
        with (model_dir / f"{group_name}-vectors.npy").open("xb") as out:
            np.save(out, np.concatenate(blocks), allow_pickle=False)
    files = {p.name: shared.file_digest(p) for p in model_dir.glob("*.npy")}
    shared.save(model_dir / "build.json", {"model": name, "identity": enc.identity,
        "catalog_sha256": shared.file_digest(directory / "shared-catalog.json"),
        "files_sha256": files, "vector_bytes": path.stat().st_size,
        "fresh_rows": len(values), "old_rows_reused": 0, "build_seconds": build_seconds,
        "bulk_requests": requests})
    for path in sorted(model_dir.iterdir()):
        publish(directory, path, settings)
    shared.event(directory, stage="fresh-remote-build-complete", model=name,
                 completed_segments=len(values), build_seconds=build_seconds)


def measure(directory, name, settings):
    import psutil
    if settings["local_resource_id"] != settings["client_instance"]:
        raise ValueError("Timed client must run on its separate EC2")
    gate = shared.read(directory / "all-builds-complete.json")
    if sorted(gate["models"]) != sorted(shared.NAMES):
        raise ValueError("Not all fresh builds complete")
    # This process loads only the 32 timing queries, not a model or vector matrix.
    texts = shared.read(directory / "timing-queries.json")
    if len(texts) != 32:
        raise ValueError("Timing cohort changed")
    enc = encoder(name, settings, measured=True)
    samples, failures = [], []
    cpu_before = psutil.Process().cpu_times()
    started = time.perf_counter()
    for round_number in range(7):
        for row in texts:
            try:
                _, telemetry = enc.encode([row["query"]], "development")
                if round_number >= 2:
                    samples.append({"query_id": row["query_id"], "round": round_number,
                                    **telemetry})
            except Exception as error:
                failures.append({"query_id": row["query_id"], "round": round_number, "error": repr(error)})
    elapsed = time.perf_counter()-started
    cpu_after = psutil.Process().cpu_times()
    result = {"model": name, "server": enc.identity, "client_environment": host_metadata(),
        "protocol": settings["measurement_protocol"], "samples": samples, "failures": failures,
        "round_trip": latency_summary([x["client_round_trip_seconds"] for x in samples]) if samples else None,
        "server_encode": latency_summary([x["encode_seconds"] for x in samples]) if samples and
                         all("encode_seconds" in x for x in samples) else None,
        "client_cpu_seconds": cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system,
        "phase_wall_seconds": elapsed, "server_max_rss_bytes": max(
            (x["server_rss_bytes"] for x in samples if x.get("server_rss_bytes") is not None), default=None),
        "gpu_allocated_bytes": max((x.get("gpu_allocated_bytes", 0) for x in samples), default=None),
        "valid_successful_measurements": len(samples), "expected_measured_requests": 160}
    path = directory / name / "measurement.json"
    shared.save(path, result); publish(directory, path, settings)
    shared.event(directory, stage="isolated-network-measurement-complete", model=name,
                 samples=len(samples), failures=len(failures))


def quality(directory, name, settings):
    from threadpoolctl import threadpool_limits
    if settings["local_resource_id"] != settings["server_instance"]:
        raise ValueError("Quality search must not run on the load-generator host")
    threadpool_limits(limits=8)
    corpus, queries, dev, _, _ = shared.prepare()
    catalog = shared.read(directory / "shared-catalog.json")
    record = shared.read(directory / name / "build.json")
    if record["catalog_sha256"] != shared.file_digest(directory / "shared-catalog.json"):
        raise ValueError("Quality catalog changed")
    for filename, expected in record["files_sha256"].items():
        if shared.file_digest(directory / name / filename) != expected:
            raise ValueError("Fresh vector checksum mismatch")
    values = np.load(directory / name / "segment-vectors.npy", mmap_mode="r")
    qvalues = np.load(directory / name / "queries-vectors.npy", mmap_mode="r")
    dvalues = np.load(directory / name / "development-vectors.npy", mmap_mode="r")
    if len(qvalues) != len(queries) or len(dvalues) != len(dev):
        raise ValueError("Encoded query count differs")
    owners = [s[0] for s in catalog["segments"]]
    rows = []
    for a in range(0, len(queries), 16):
        rankings = parent_top_k(qvalues[a:a+16], values, owners, corpus.ids, k=10)
        for q, ranking in zip(queries[a:a+16], rankings):
            rows.append({"query_id": q.query_id, "category": q.category,
                "metrics": score_top10_query(q, ranking), "top10": [r.__dict__ for r in ranking]})
        if a % 160 == 0:
            shared.event(directory, stage="fresh-quality", model=name, completed_queries=len(rows))
    times = []
    for r in range(7):
        for q in dvalues:
            before = time.perf_counter()
            parent_top_k(q[None], values, owners, corpus.ids, k=10)
            if r >= 2:
                times.append(time.perf_counter()-before)
    path = directory / name / "quality.json"
    shared.save(path, {"model": name, "per_query": rows,
        "summary": summarize(rows, metrics=TOP10_METRICS), "fresh_build": record,
        "search": _timing_summary(times), "search_host": host_metadata(),
        "search_worker_peak_rss_bytes": peak_rss_bytes(), "search_threads": 8,
        "catalog_sha256": shared.file_digest(directory / "shared-catalog.json")})
    publish(directory, path, settings)
    shared.event(directory, stage="fresh-quality-complete", model=name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--phase", choices=("catalog", "build", "measure", "quality"), required=True)
    p.add_argument("--model", choices=shared.NAMES)
    a = p.parse_args(); settings = shared.read(a.settings)
    if a.phase == "catalog":
        a.directory.mkdir(exist_ok=False)
        shared.make_catalog(a.directory)
        _, _, dev, _, _ = shared.prepare()
        shared.save(a.directory / "timing-queries.json", [{"query_id": q.query_id, "query": q.query} for q in dev])
        for filename in ("shared-catalog.json", "shared-catalog.sha256.json", "timing-queries.json"):
            publish(a.directory, a.directory / filename, settings)
    else:
        if not a.model:
            raise ValueError("Model required")
        {"build": build, "measure": measure, "quality": quality}[a.phase](a.directory, a.model, settings)
