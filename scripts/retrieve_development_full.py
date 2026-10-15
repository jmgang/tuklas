"""Exact full-corpus retrieval for development queries only; no relevance scoring."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.distillation import file_hashes
from pycon_model2vec_demo.encoders import load_e5_encoder, load_static_encoder
from pycon_model2vec_demo.retrieval import encode_corpus, exact_top_k, open_corpus_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("teacher", "potion_retrieval", "plain_student", "custom_student", "tokenlearn_custom_student"))
    parser.add_argument("--dev-query-limit", type=int, default=2)
    args = parser.parse_args()
    if args.dev_query_limit <= 0:
        parser.error("--dev-query-limit must be positive")

    data = json.loads((ROOT / "configs/disastir-qa-all-categories.json").read_text())
    models = json.loads((ROOT / "configs/models-v1.json").read_text())
    retrieval = json.loads((ROOT / "configs/retrieval-v1.json").read_text())
    teacher_dir = ROOT / f"models/e5-base-v2-{models['teacher']['revision'][:12]}"
    potion_dir = ROOT / f"models/potion-retrieval-{models['ready_made_baseline']['revision'][:12]}"
    plain_dir = ROOT / f"models/plain-e5-{models['teacher']['revision'][:12]}"
    custom_dir = ROOT / f"models/custom-e5-{models['teacher']['revision'][:12]}"
    tokenlearn_dir = ROOT / f"models/tokenlearn-custom-e5-{models['teacher']['revision'][:12]}"
    student = models["students"]["plain"]
    choices = {
        "teacher": (teacher_dir, models["teacher"], lambda path: load_e5_encoder(str(path), max_length=models["teacher"]["max_length"], batch_size=models["teacher"]["batch_size"])),
        "potion_retrieval": (potion_dir, models["ready_made_baseline"], lambda path: load_static_encoder(str(path), max_length=models["ready_made_baseline"]["max_length"], batch_size=models["ready_made_baseline"]["batch_size"])),
        "plain_student": (plain_dir, student, lambda path: load_static_encoder(str(path), max_length=student["max_length"], batch_size=student["batch_size"], query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"])),
        "custom_student": (custom_dir, student, lambda path: load_static_encoder(str(path), max_length=student["max_length"], batch_size=student["batch_size"], query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"])),
        "tokenlearn_custom_student": (tokenlearn_dir, student, lambda path: load_static_encoder(str(path), max_length=student["max_length"], batch_size=student["batch_size"], query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"])),
    }
    model_dir, settings, loader = choices[args.model]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Pinned model artifact has not been prepared: {model_dir}")
    hashes = file_hashes(model_dir)
    fingerprint = sha256(json.dumps({"files": hashes, "settings": settings}, sort_keys=True).encode()).hexdigest()
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(
        data_dir=ROOT / "data/raw", revision=data["revision"], category_checksums=data["categories"],
        seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"],
    )
    dev_queries = sorted(prepared.development, key=lambda item: item.query_id)[:args.dev_query_limit]
    if len(dev_queries) != args.dev_query_limit:
        raise ValueError("Not enough development queries")
    encoder = loader(model_dir)
    cache = ROOT / "runs/embeddings" / f"{args.model}-{fingerprint[:12]}.npy"
    if not cache.exists():
        encode_corpus(corpus, encoder, cache, corpus_sha256=data["corpus_sha256"], model_fingerprint=fingerprint)
    passage_vectors = open_corpus_embeddings(
        cache, corpus=corpus, corpus_sha256=data["corpus_sha256"], model_fingerprint=fingerprint,
    )
    queries = encoder.encode_queries([group.query for group in dev_queries])
    ranked = exact_top_k(
        queries, passage_vectors, corpus.ids,
        k=retrieval["primary_top_k"], corpus_block_size=retrieval["corpus_block_size"],
    )
    output = {
        "status": "development-query-full-corpus-retrieval; no relevance metrics",
        "model": args.model,
        "model_fingerprint": fingerprint,
        "corpus_sha256": data["corpus_sha256"],
        "corpus_passage_count": len(corpus.ids),
        "cache": str(cache.relative_to(ROOT)),
        "results": [
            {"query_id": group.query_id, "ranked": [row.__dict__ for row in rows]}
            for group, rows in zip(dev_queries, ranked)
        ],
    }
    target = ROOT / "runs" / f"dev-full-{args.model}-{fingerprint[:12]}-{args.dev_query_limit}q.json"
    with target.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(target)


if __name__ == "__main__":
    main()
