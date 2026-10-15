"""Compare untrained and TokenLearn-trained custom students on development QA."""

from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.distillation import file_hashes
from pycon_model2vec_demo.encoders import load_static_encoder
from pycon_model2vec_demo.evaluation import paired_bootstrap_difference, score_rankings
from pycon_model2vec_demo.retrieval import encode_corpus, exact_top_k, open_corpus_embeddings


def main() -> None:
    data = json.loads((ROOT / "configs/disastir-qa-all-categories.json").read_text(encoding="utf-8"))
    models = json.loads((ROOT / "configs/models-v1.json").read_text(encoding="utf-8"))
    retrieval = json.loads((ROOT / "configs/retrieval-v1.json").read_text(encoding="utf-8"))
    tokenlearn = json.loads((ROOT / "configs/tokenlearn-v1.json").read_text(encoding="utf-8"))
    suffix = models["teacher"]["revision"][:12]
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(
        data_dir=ROOT / "data/raw", revision=data["revision"], category_checksums=data["categories"],
        seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"],
    )
    queries, coverage = map_query_groups(prepared.development, corpus)
    if len(queries) != 335:
        raise RuntimeError("The frozen development query split changed")
    student_settings = models["students"]["plain"]
    model_paths = {
        "custom_student": ROOT / "models" / f"custom-e5-{suffix}",
        "tokenlearn_custom_student": ROOT / tokenlearn["output_model"],
    }
    provenance_paths = {
        "custom_student": ROOT / "runs" / f"custom-e5-{suffix}-provenance.json",
        "tokenlearn_custom_student": ROOT / "runs" / f"tokenlearn-custom-e5-{suffix}-provenance.json",
    }
    scores = {}
    fingerprints = {}
    operational_diagnostics = {}
    for name, path in model_paths.items():
        provenance = json.loads(provenance_paths[name].read_text(encoding="utf-8"))
        hashes = file_hashes(path)
        if hashes != provenance["student"]["files_sha256"]:
            raise RuntimeError(f"{name} artifact does not match its provenance")
        fingerprint = sha256(json.dumps({"files": hashes, "settings": student_settings}, sort_keys=True).encode()).hexdigest()
        fingerprints[name] = fingerprint
        started = time.perf_counter()
        encoder = load_static_encoder(
            str(path), max_length=student_settings["max_length"], batch_size=student_settings["batch_size"],
            query_prefix=student_settings["query_prefix"], passage_prefix=student_settings["passage_prefix"],
        )
        load_seconds = time.perf_counter() - started
        cache = ROOT / "runs/embeddings" / f"{name}-{fingerprint[:12]}.npy"
        corpus_encoding_seconds = None
        if not cache.exists():
            started = time.perf_counter()
            encode_corpus(corpus, encoder, cache, corpus_sha256=data["corpus_sha256"], model_fingerprint=fingerprint)
            corpus_encoding_seconds = time.perf_counter() - started
        vectors = open_corpus_embeddings(cache, corpus=corpus, corpus_sha256=data["corpus_sha256"], model_fingerprint=fingerprint)
        started = time.perf_counter()
        query_vectors = encoder.encode_queries([query.query for query in queries])
        query_encoding_seconds = time.perf_counter() - started
        started = time.perf_counter()
        ranked = exact_top_k(
            query_vectors, vectors, corpus.ids,
            k=retrieval["secondary_top_k"], corpus_block_size=retrieval["corpus_block_size"],
        )
        exact_search_seconds = time.perf_counter() - started
        scores[name] = score_rankings(queries, ranked, positive_threshold=data["positive_relevance_threshold"])
        operational_diagnostics[name] = {
            "artifact_bytes": sum(file.stat().st_size for file in path.rglob("*") if file.is_file()),
            "load_seconds_single_run": round(load_seconds, 3),
            "corpus_encoding_seconds_if_new_cache": None if corpus_encoding_seconds is None else round(corpus_encoding_seconds, 3),
            "development_query_encoding_seconds_single_run": round(query_encoding_seconds, 3),
            "exact_search_seconds_single_run": round(exact_search_seconds, 3),
        }
    result = {
        "status": "development-only-two-student-full-corpus-quality-comparison; not-held-out-final",
        "query_count": len(queries),
        "corpus_passage_count": len(corpus.ids),
        "mapped_judgment_coverage": coverage,
        "model_fingerprints": fingerprints,
        "models": scores,
        "operational_diagnostics_not_repeated_benchmarks": operational_diagnostics,
        "tokenlearn_minus_untrained": paired_bootstrap_difference(scores["custom_student"], scores["tokenlearn_custom_student"]),
        "limitations": ["released queries are LLM-generated", "qrels are pooled and LLM-rated", "unjudged retrieved passages have unknown relevance"],
    }
    target = ROOT / "runs" / f"tokenlearn-development-comparison-{fingerprints['tokenlearn_custom_student'][:12]}.json"
    with target.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(target)
    for name, value in scores.items():
        print(name, {key: round(number, 4) for key, number in value["overall"].items()})
    print("paired nDCG@10 difference", result["tokenlearn_minus_untrained"])


if __name__ == "__main__":
    main()
