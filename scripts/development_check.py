"""Five-model, development-query-only preflight; never computes relevance metrics."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.distillation import file_hashes
from pycon_model2vec_demo.encoders import load_e5_encoder, load_static_encoder
from pycon_model2vec_demo.retrieval import exact_top_k


POTION_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "modules.json",
    "special_tokens_map.json", "tokenizer_config.json", "vocab.txt",
)


def _fingerprint(hashes: dict[str, str]) -> str:
    return sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main() -> None:
    os.environ["HF_HOME"] = str(ROOT / "models/.hf-cache")
    os.environ["HF_XET_CACHE"] = str(ROOT / "models/.hf-cache/xet")
    from huggingface_hub import snapshot_download

    data_config_path = ROOT / "configs/disastir-qa-all-categories.json"
    model_config_path = ROOT / "configs/models-v1.json"
    retrieval_config_path = ROOT / "configs/retrieval-v1.json"
    data = json.loads(data_config_path.read_text())
    models = json.loads(model_config_path.read_text())
    retrieval = json.loads(retrieval_config_path.read_text())
    plain_provenance = json.loads((ROOT / "runs/plain-e5-f52bf8ec8c71-provenance.json").read_text())
    custom_provenance = json.loads((ROOT / "runs/custom-e5-f52bf8ec8c71-provenance.json").read_text())
    tokenlearn_provenance = json.loads((ROOT / "runs/tokenlearn-custom-e5-f52bf8ec8c71-provenance.json").read_text())
    vocabulary_record = json.loads((ROOT / "runs/custom-vocabulary-v1.json").read_text())

    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(
        data_dir=ROOT / "data/raw", revision=data["revision"],
        category_checksums=data["categories"], seed=data["split_seed"],
        evaluation_fraction=data["evaluation_fraction"],
    )
    dev_by_category = {}
    for group in prepared.development:
        dev_by_category.setdefault(group.category, []).append(group)
    per_category = retrieval["development_probe"]["queries_per_category"]
    dev_queries = [
        group
        for category in sorted(dev_by_category)
        for group in sorted(dev_by_category[category], key=lambda item: item.query_id)[:per_category]
    ]
    if len(dev_queries) != per_category * len(data["categories"]):
        raise RuntimeError("Development split lacks requested probe queries")
    size = retrieval["development_probe"]["evenly_spaced_corpus_passages"]
    corpus_indices = np.linspace(0, len(corpus.ids) - 1, size, dtype=int)
    probe_ids = tuple(corpus.ids[index] for index in corpus_indices)
    probe_texts = tuple(corpus.texts[index] for index in corpus_indices)

    potion = models["ready_made_baseline"]
    potion_dir = ROOT / "models" / f"potion-retrieval-{potion['revision'][:12]}"
    if any(not (potion_dir / name).is_file() for name in POTION_FILES):
        snapshot_download(
            repo_id=potion["repo"], revision=potion["revision"],
            local_dir=potion_dir, allow_patterns=list(POTION_FILES), local_files_only=True,
        )
    if any(not (potion_dir / name).is_file() for name in POTION_FILES):
        raise RuntimeError("Pinned Potion snapshot is incomplete")

    teacher_dir = ROOT / "models/e5-base-v2-f52bf8ec8c71"
    plain_dir = ROOT / "models/plain-e5-f52bf8ec8c71"
    custom_dir = ROOT / "models/custom-e5-f52bf8ec8c71"
    tokenlearn_dir = ROOT / "models/tokenlearn-custom-e5-f52bf8ec8c71"
    if file_hashes(teacher_dir) != plain_provenance["teacher"]["files_sha256"]:
        raise RuntimeError("Teacher artifact changed")
    if file_hashes(plain_dir) != plain_provenance["student"]["files_sha256"]:
        raise RuntimeError("Plain artifact changed")
    if file_hashes(custom_dir) != custom_provenance["student"]["files_sha256"]:
        raise RuntimeError("Custom artifact changed")
    if file_hashes(tokenlearn_dir) != tokenlearn_provenance["student"]["files_sha256"]:
        raise RuntimeError("TokenLearn artifact changed")
    if tokenlearn_provenance["custom_student_files_sha256"] != custom_provenance["student"]["files_sha256"]:
        raise RuntimeError("TokenLearn student was trained from a different custom baseline")
    if vocabulary_record["requested_terms_sha256"] != custom_provenance["vocabulary"]["requested_terms_sha256"]:
        raise RuntimeError("Custom vocabulary changed")

    student = models["students"]["plain"]
    candidates = (
        ("teacher", load_e5_encoder(str(teacher_dir), max_length=models["teacher"]["max_length"], batch_size=models["teacher"]["batch_size"]), file_hashes(teacher_dir)),
        ("potion_retrieval", load_static_encoder(str(potion_dir), max_length=potion["max_length"], batch_size=potion["batch_size"]), file_hashes(potion_dir)),
        ("plain_student", load_static_encoder(str(plain_dir), max_length=student["max_length"], batch_size=student["batch_size"], query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"]), file_hashes(plain_dir)),
        ("custom_student", load_static_encoder(str(custom_dir), max_length=student["max_length"], batch_size=student["batch_size"], query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"]), file_hashes(custom_dir)),
        ("tokenlearn_custom_student", load_static_encoder(str(tokenlearn_dir), max_length=student["max_length"], batch_size=student["batch_size"], query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"]), file_hashes(tokenlearn_dir)),
    )
    checks = {}
    for name, encoder, hashes in candidates:
        query_vectors = encoder.encode_queries([group.query for group in dev_queries])
        passage_vectors = encoder.encode_passages(probe_texts)
        ranked = exact_top_k(
            query_vectors, passage_vectors, probe_ids,
            k=retrieval["primary_top_k"], corpus_block_size=64,
        )
        if len(ranked) != len(dev_queries) or any(len(rows) != retrieval["primary_top_k"] for rows in ranked):
            raise RuntimeError(f"{name} returned incomplete exact top-k results")
        checks[name] = {
            "artifact_fingerprint": _fingerprint(hashes),
            "artifact_files_sha256": hashes,
            "dimension": encoder.dimension,
            "query_count": len(query_vectors),
            "probe_passage_count": len(passage_vectors),
            "query_zero_vectors": int(np.count_nonzero(np.linalg.norm(query_vectors, axis=1) == 0)),
            "passage_zero_vectors": int(np.count_nonzero(np.linalg.norm(passage_vectors, axis=1) == 0)),
            "probe_passages_truncated": encoder.count_truncated_passages(probe_texts),
            "top_k_complete": True,
        }
    output = {
        "status": "development-only-configuration-preflight; not-retrieval-quality-evaluation",
        "development_query_ids": [group.query_id for group in dev_queries],
        "held_out_query_count_not_scored": len(prepared.evaluation),
        "probe_corpus_ids": probe_ids,
        "source_corpus_passage_count": len(corpus.ids),
        "dataset_revision": data["revision"],
        "dataset_config_sha256": sha256(data_config_path.read_bytes()).hexdigest(),
        "model_config_sha256": sha256(model_config_path.read_bytes()).hexdigest(),
        "retrieval_config_sha256": sha256(retrieval_config_path.read_bytes()).hexdigest(),
        "vocabulary_record_sha256": sha256((ROOT / "runs/custom-vocabulary-v1.json").read_bytes()).hexdigest(),
        "models": checks,
    }
    target = ROOT / "runs/development-check-v3.json"
    with target.open("x", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(target)


if __name__ == "__main__":
    main()
