"""Measure one model in a fresh process on frozen development inputs only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.measurement import artifact_size_bytes, host_metadata, measure_warm_workload, peak_rss_bytes
from pycon_model2vec_demo.model_registry import MODEL_NAMES, local_model_artifacts
from pycon_model2vec_demo.retrieval import encode_corpus, open_corpus_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--fresh-embedding-build", action="store_true", help="Create a separate full-corpus cache to measure build time")
    args = parser.parse_args()

    dataset_config_path = ROOT / "configs/disastir-qa-all-categories.json"
    retrieval_config_path = ROOT / "configs/retrieval-v1.json"
    measurement_config_path = ROOT / "configs/measurement-v1.json"
    data = json.loads(dataset_config_path.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_config_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_config_path.read_text(encoding="utf-8"))
    artifact = local_model_artifacts(ROOT, require_all=False)[args.model]
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(
        data_dir=ROOT / "data/raw", revision=data["revision"], category_checksums=data["categories"],
        seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"],
    )
    per_category = {}
    for group in prepared.development:
        per_category.setdefault(group.category, []).append(group)
    query_groups = [
        group for category in sorted(per_category)
        for group in sorted(per_category[category], key=lambda item: item.query_id)[:measurement["development_queries_per_category"]]
    ]
    if len(query_groups) != len(data["categories"]) * measurement["development_queries_per_category"]:
        raise RuntimeError("Frozen development sample is incomplete")
    indices = np.linspace(0, len(corpus.ids) - 1, measurement["passage_batch_size"], dtype=int)
    probe_passages = tuple(corpus.texts[int(index)] for index in indices)
    probe_ids = tuple(corpus.ids[int(index)] for index in indices)

    baseline_rss = peak_rss_bytes()
    started = time.perf_counter()
    encoder = artifact.load_encoder()
    load_seconds = time.perf_counter() - started
    after_load_rss = peak_rss_bytes()
    if args.fresh_embedding_build:
        cache = ROOT / "runs/measurement-caches" / f"{args.model}-{artifact.fingerprint[:12]}.npy"
        if cache.exists():
            raise FileExistsError("A prior fresh measurement cache exists; refusing to overwrite")
    else:
        cache = ROOT / "runs/embeddings" / f"{args.model}-{artifact.fingerprint[:12]}.npy"
    build_seconds = None
    if not cache.exists():
        started = time.perf_counter()
        encode_corpus(corpus, encoder, cache, corpus_sha256=data["corpus_sha256"], model_fingerprint=artifact.fingerprint)
        build_seconds = time.perf_counter() - started
    vectors = open_corpus_embeddings(cache, corpus=corpus, corpus_sha256=data["corpus_sha256"], model_fingerprint=artifact.fingerprint)
    cache_manifest = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    external_build_seconds = cache_manifest.get("external_build_seconds")
    if external_build_seconds is not None and build_seconds is not None:
        raise ValueError("Cannot label a locally built cache as an external build")
    reported_build_seconds = build_seconds if build_seconds is not None else external_build_seconds
    build_backend = "SageMaker GPU" if external_build_seconds is not None else "local measurement host"
    warm = measure_warm_workload(
        encoder,
        query_texts=[group.query for group in query_groups], passage_texts=probe_passages,
        passage_vectors=vectors, corpus_ids=corpus.ids,
        top_k=measurement["search_top_k"], search_block_size=retrieval["corpus_block_size"],
        warmup=measurement["warmup_rounds"], repetitions=measurement["repeated_rounds"],
    )
    output = {
        "status": "development-input-operational-measurement; no-held-out-queries-or-quality-scores",
        "model": args.model,
        "model_fingerprint": artifact.fingerprint,
        "model_files_sha256": artifact.files_sha256,
        "model_artifact_size_bytes": artifact_size_bytes(artifact.path),
        "corpus_embedding_cache": str(cache.relative_to(ROOT)),
        "corpus_embedding_cache_size_bytes": cache.stat().st_size,
        "corpus_embedding_manifest": cache_manifest,
        "fresh_embedding_build_requested": args.fresh_embedding_build,
        "full_corpus_embedding_build_seconds": None if reported_build_seconds is None else round(reported_build_seconds, 3),
        "corpus_embedding_build_backend": build_backend,
        "corpus_embedding_build_environment": cache_manifest.get("external_build_environment", host_metadata()),
        "fresh_process_model_load_seconds": round(load_seconds, 3),
        "peak_rss_before_load_bytes": baseline_rss,
        "peak_rss_after_load_bytes": after_load_rss,
        "peak_rss_after_workload_bytes": peak_rss_bytes(),
        "warm": warm,
        "development_query_ids": [group.query_id for group in query_groups],
        "probe_corpus_ids": probe_ids,
        "config_sha256": {
            "dataset": sha256(dataset_config_path.read_bytes()).hexdigest(),
            "retrieval": sha256(retrieval_config_path.read_bytes()).hexdigest(),
            "measurement": sha256(measurement_config_path.read_bytes()).hexdigest(),
        },
        "environment": {**host_metadata(), "python_implementation": platform.python_implementation()},
        "interpretation": {
            "cold_load": measurement["cold_load_note"],
            "indexing": measurement["exact_search_note"],
            "cache_reused": build_seconds is None,
            "external_build": external_build_seconds is not None,
            "latency": "repeated warm samples on a single machine; not a production SLA",
        },
    }
    label = "measurement-build" if args.fresh_embedding_build else "measurement"
    target = ROOT / "runs" / f"{label}-{args.model}-{artifact.fingerprint[:12]}.json"
    with target.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(target)


if __name__ == "__main__":
    main()
