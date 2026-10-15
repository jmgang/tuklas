"""Score full-corpus retrieval; held-out mode opens only after five-model preflight."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.evaluation import score_rankings
from pycon_model2vec_demo.experiment import file_sha256, final_run_preflight, judged_candidate_rankings
from pycon_model2vec_demo.model_registry import MODEL_NAMES, local_model_artifacts
from pycon_model2vec_demo.reporting import build_report, render_markdown
from pycon_model2vec_demo.retrieval import encode_corpus, exact_top_k, open_corpus_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="Development only; defaults to existing four models")
    parser.add_argument("--development-queries-per-category", type=int, default=None, help="Development smoke limit; never used for held-out")
    args = parser.parse_args()
    if args.split == "heldout" and (args.models is not None or args.development_queries_per_category is not None):
        parser.error("Held-out mode always evaluates all five models and all queries")
    if args.development_queries_per_category is not None and args.development_queries_per_category <= 0:
        parser.error("Development query limit must be positive")

    # This preflight MUST precede load_qa_collection and any held-out query access.
    if args.split == "heldout":
        artifacts, measurements = final_run_preflight(ROOT)
        chosen = MODEL_NAMES
    else:
        available = local_model_artifacts(ROOT, require_all=False)
        chosen = tuple(args.models or ("teacher", "potion_retrieval", "plain_student", "custom_student"))
        if len(set(chosen)) != len(chosen):
            parser.error("Model names must not repeat")
        artifacts = {name: available[name] for name in chosen}
        measurements = None

    data_path = ROOT / "configs/disastir-qa-all-categories.json"
    retrieval_path = ROOT / "configs/retrieval-v1.json"
    decision_path = ROOT / "configs/decision-v1.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(
        data_dir=ROOT / "data/raw", revision=data["revision"], category_checksums=data["categories"],
        seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"],
    )
    groups = prepared.evaluation if args.split == "heldout" else prepared.development
    if args.split == "development" and args.development_queries_per_category is not None:
        groups = tuple(
            group for category in sorted(data["categories"])
            for group in sorted((row for row in groups if row.category == category), key=lambda row: row.query_id)[:args.development_queries_per_category]
        )
    queries, coverage = map_query_groups(groups, corpus)
    excluded = [query.query_id for query in queries if not any(grade > 0 for grade in query.judgments.values())]
    if excluded:
        raise ValueError(f"Mapped-positive exclusions require review before scoring: {excluded[:5]} ({len(excluded)} total)")
    if args.split == "heldout" and len(queries) != len(prepared.manifest["evaluation_query_ids"]):
        raise RuntimeError("Held-out query count differs from frozen split")
    scores = {}
    rankings = {}
    judged_pool = {}
    cache_manifests = {}
    for name in chosen:
        artifact = artifacts[name]
        encoder = artifact.load_encoder()
        cache = ROOT / "runs/embeddings" / f"{name}-{artifact.fingerprint[:12]}.npy"
        if not cache.exists():
            encode_corpus(corpus, encoder, cache, corpus_sha256=data["corpus_sha256"], model_fingerprint=artifact.fingerprint)
        vectors = open_corpus_embeddings(cache, corpus=corpus, corpus_sha256=data["corpus_sha256"], model_fingerprint=artifact.fingerprint)
        cache_manifests[name] = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        model_rankings = []
        candidate_rankings = []
        for start in range(0, len(queries), 8):
            batch = queries[start:start + 8]
            query_vectors = encoder.encode_queries([query.query for query in batch])
            model_rankings.extend(exact_top_k(
                query_vectors, vectors, corpus.ids,
                k=retrieval["secondary_top_k"], corpus_block_size=retrieval["corpus_block_size"],
            ))
            candidate_rankings.extend(judged_candidate_rankings(batch, query_vectors, vectors, corpus))
        rankings[name] = tuple(model_rankings)
        scores[name] = score_rankings(queries, rankings[name], positive_threshold=data["positive_relevance_threshold"])
        judged_pool[name] = score_rankings(queries, candidate_rankings, positive_threshold=data["positive_relevance_threshold"])
        print(f"{name}: nDCG@10={scores[name]['overall']['nDCG@10']:.4f}; Unjudged@10={scores[name]['overall']['Unjudged@10']:.4f}", flush=True)

    provenance = {
        "dataset_revision": data["revision"],
        "config_sha256": {
            "dataset": file_sha256(data_path), "retrieval": file_sha256(retrieval_path),
            "decision": file_sha256(decision_path),
            "models": file_sha256(ROOT / "configs/models-v1.json"),
            "tokenlearn": file_sha256(ROOT / "configs/tokenlearn-v1.json"),
        },
        "model_fingerprints": {name: artifacts[name].fingerprint for name in chosen},
        "corpus_embedding_manifests": cache_manifests,
        "frozen_split_sha256": sha256("\n".join(query.query_id for query in queries).encode()).hexdigest(),
        "excluded_queries_without_mapped_positive": excluded,
    }
    report = build_report(
        split=args.split, queries=queries, corpus=corpus, scores=scores, rankings=rankings,
        coverage=coverage, provenance=provenance,
        measurements=measurements,
        thresholds=json.loads(decision_path.read_text(encoding="utf-8")) if args.split == "heldout" else None,
    )
    report["judged_candidate_ranking_secondary_diagnostic"] = judged_pool
    report["interpretation"] = "The primary metric searches all released passages. Judged-candidate scores are only an easier secondary diagnostic."
    if args.split == "heldout":
        target = ROOT / "runs" / f"heldout-five-model-{provenance['frozen_split_sha256'][:12]}.json"
    else:
        key = sha256(json.dumps({"models": list(chosen), "ids": [query.query_id for query in queries], "fingerprints": provenance["model_fingerprints"]}, sort_keys=True).encode()).hexdigest()[:12]
        target = ROOT / "runs" / f"development-quality-{key}.json"
    markdown = target.with_suffix(".md")
    if target.exists() or markdown.exists():
        raise FileExistsError("This exact experiment report already exists; refusing to overwrite")
    with target.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    with markdown.open("x", encoding="utf-8") as destination:
        destination.write(render_markdown(report))
    print(target)
    print(markdown)


if __name__ == "__main__":
    main()
