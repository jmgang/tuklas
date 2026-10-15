"""Train a fifth, TokenLearn-adapted copy of the custom Model2Vec student.

Only released *unlabeled passage text* enters the training stream. No queries,
qrels, or evaluation scores are read by either stage.
"""

from __future__ import annotations

import gzip
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np

from .data import _file_sha256
from .distillation import file_hashes
from .encoders import load_static_encoder


def _read_settings(project_dir: Path, config_path: Path) -> tuple[dict, dict, dict]:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    models = json.loads((project_dir / "configs/models-v1.json").read_text(encoding="utf-8"))
    dataset = json.loads((project_dir / "configs/disastir-qa-all-categories.json").read_text(encoding="utf-8"))
    if settings["source"] != "complete-released-unlabeled-disastir-passage-corpus":
        raise ValueError("TokenLearn training source must be the full unlabeled passage corpus")
    if settings["teacher_input_prefix"] != models["teacher"]["passage_prefix"]:
        raise ValueError("TokenLearn teacher passage prefix differs from the frozen E5 setting")
    if settings["target_dimension"] != models["students"]["plain"]["pca_dims"]:
        raise ValueError("TokenLearn target dimension must match the saved custom student")
    if version("tokenlearn") != settings["tokenlearn_version"]:
        raise RuntimeError("Installed TokenLearn version differs from frozen config")
    corpus_path = project_dir / "data/raw/corpus.jsonl.gz"
    if _file_sha256(corpus_path) != dataset["corpus_sha256"]:
        raise ValueError("Unlabeled corpus differs from the pinned release")
    return settings, models, dataset


def _passages(path: Path, prefix: str):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                raise ValueError("Corpus contains an invalid passage")
            yield {"text": prefix + record["text"]}


def _feature_files(directory: Path) -> list[Path]:
    json_files = sorted(directory.glob("feature_*.json"))
    for path in json_files:
        if not path.with_suffix(".npy").is_file():
            raise RuntimeError(f"Missing target embeddings for {path.name}")
    if len(json_files) != len(list(directory.glob("feature_*.npy"))):
        raise RuntimeError("Feature text and target-shard counts differ")
    return json_files


def featurize_corpus(project_dir: Path, config_path: Path) -> Path:
    """Use TokenLearn's featurizer with the pinned E5 and all released passages."""
    os.environ["HF_HOME"] = str(project_dir / "models/.hf-cache")
    settings, models, dataset = _read_settings(project_dir, config_path)
    from tokenlearn.featurize import featurize
    from .encoders import load_e5_encoder

    output_dir = project_dir / settings["features_dir"]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Completed feature manifest exists; refusing to overwrite it")
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = project_dir / "models" / f"e5-base-v2-{models['teacher']['revision'][:12]}"
    plain_record = json.loads((project_dir / "runs" / f"plain-e5-{models['teacher']['revision'][:12]}-provenance.json").read_text())
    if file_hashes(teacher_dir) != plain_record["teacher"]["files_sha256"]:
        raise ValueError("Teacher artifact differs from the original distillation")
    teacher = load_e5_encoder(str(teacher_dir), max_length=models["teacher"]["max_length"], batch_size=settings["feature_batch_size"])
    teacher.model.to(settings["teacher_device"])
    started = time.monotonic()
    featurize(
        dataset=_passages(project_dir / "data/raw/corpus.jsonl.gz", settings["teacher_input_prefix"]),
        model=teacher.model,
        output_dir=str(output_dir),
        max_means=settings["expected_passage_count"],
        batch_size=settings["feature_batch_size"],
        text_key="text",
    )
    files = _feature_files(output_dir)
    row_count = 0
    for path in files:
        texts = json.loads(path.read_text(encoding="utf-8"))
        vectors = np.load(path.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
        if len(texts) != len(vectors) or vectors.shape[1] != models["teacher"]["dimension"]:
            raise RuntimeError(f"Invalid TokenLearn feature shard {path.name}")
        row_count += len(texts)
    if row_count != settings["expected_passage_count"]:
        raise RuntimeError(f"Expected {settings['expected_passage_count']} passages, featurized {row_count}")
    manifest = {
        "status": "tokenlearn-features-complete",
        "corpus_sha256": dataset["corpus_sha256"],
        "teacher_files_sha256": plain_record["teacher"]["files_sha256"],
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "passage_count": row_count,
        "target_dimension": models["teacher"]["dimension"],
        "feature_files_sha256": file_hashes(output_dir),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    with manifest_path.open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
    return manifest_path


def train_custom_student(project_dir: Path, config_path: Path) -> Path:
    """Train the saved custom-vocabulary student using TokenLearn's API."""
    settings, models, dataset = _read_settings(project_dir, config_path)
    from model2vec import StaticModel
    import torch
    from tokenlearn.train import train_model
    from tokenlearn.utils import collect_means_and_texts

    features_dir = project_dir / settings["features_dir"]
    feature_manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
    if feature_manifest["config_sha256"] != sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("TokenLearn configuration changed after featurization")
    if feature_manifest["corpus_sha256"] != dataset["corpus_sha256"]:
        raise ValueError("TokenLearn features came from a different corpus")
    feature_hashes = file_hashes(features_dir)
    feature_hashes.pop("manifest.json", None)
    if feature_hashes != feature_manifest["feature_files_sha256"]:
        raise ValueError("TokenLearn feature shards changed after featurization")

    suffix = models["teacher"]["revision"][:12]
    custom_dir = project_dir / "models" / f"custom-e5-{suffix}"
    custom_record = json.loads((project_dir / "runs" / f"custom-e5-{suffix}-provenance.json").read_text(encoding="utf-8"))
    if file_hashes(custom_dir) != custom_record["student"]["files_sha256"]:
        raise ValueError("Custom-vocabulary student changed after distillation")
    output_dir = project_dir / settings["output_model"]
    provenance_path = project_dir / "runs" / f"tokenlearn-custom-e5-{suffix}-provenance.json"
    if output_dir.exists() or provenance_path.exists():
        raise FileExistsError("TokenLearn model or provenance already exists; refusing to overwrite")

    paths = _feature_files(features_dir)
    train_txt, train_vec = collect_means_and_texts(paths)
    if len(train_txt) != settings["expected_passage_count"] or train_vec.shape != (
        settings["expected_passage_count"], models["teacher"]["dimension"]
    ):
        raise RuntimeError("Feature row count or dimension differs from the frozen configuration")
    rng = np.random.default_rng(settings["shuffle_seed"])
    order = rng.permutation(len(train_txt))
    train_txt = [train_txt[int(index)] for index in order]
    train_vec = train_vec[order]
    np.random.seed(settings["shuffle_seed"])
    torch.manual_seed(settings["shuffle_seed"])

    base = StaticModel.from_pretrained(str(custom_dir), force_download=False)
    if base.tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("TokenLearn requires the tokenizer padding ID to be zero")
    if base.dim != settings["target_dimension"]:
        raise ValueError("Saved custom student has an unexpected dimension")
    # The saved inference artifact is float16; TokenLearn's torch optimizer
    # requires float32 parameters. This copy never changes the baseline files.
    trainable_base = StaticModel(vectors=base.embedding.astype(np.float32), tokenizer=base.tokenizer, config=base.config)
    started = time.monotonic()
    trained = train_model(
        trainable_base, train_txt, train_vec,
        device=settings["training_device"], pca_dims=settings["target_dimension"],
    )
    elapsed = time.monotonic() - started
    if trained.dim != base.dim or trained.tokens != base.tokens:
        raise RuntimeError("TokenLearn changed the student dimension or vocabulary")
    output_dir.mkdir(parents=True, exist_ok=False)
    trained.save_pretrained(output_dir)
    student = models["students"]["plain"]
    reloaded = load_static_encoder(
        str(output_dir), max_length=student["max_length"], batch_size=student["batch_size"],
        query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"],
    )
    queries = reloaded.encode_queries(["Where is the evacuation shelter?"])
    passages = reloaded.encode_passages(["An evacuation shelter is open after the flood."])
    if queries.shape != (1, base.dim) or passages.shape != (1, base.dim):
        raise RuntimeError("Saved TokenLearn student failed reload smoke test")
    provenance = {
        "status": "tokenlearn-custom-student-created; retrieval-quality-not-yet-evaluated",
        "training_source": "complete-released-unlabeled-passage-text-only",
        "corpus_sha256": dataset["corpus_sha256"],
        "passage_count": len(train_txt),
        "validation_count": min(10_000, len(train_txt) // 10),
        "shuffle_seed": settings["shuffle_seed"],
        "teacher_files_sha256": feature_manifest["teacher_files_sha256"],
        "custom_student_files_sha256": custom_record["student"]["files_sha256"],
        "features_manifest_sha256": sha256((features_dir / "manifest.json").read_bytes()).hexdigest(),
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "tokenlearn_version": version("tokenlearn"),
        "model2vec_version": version("model2vec"),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
        "scikit_learn_version": version("scikit-learn"),
        "training_elapsed_seconds": round(elapsed, 3),
        "student": {
            "path": str(output_dir.relative_to(project_dir)),
            "dimension": reloaded.dimension,
            "vocabulary_entries": len(reloaded.model.tokens),
            "files_sha256": file_hashes(output_dir),
        },
        "smoke_test": {"queries": 1, "passages": 1, "reloaded": True},
    }
    with provenance_path.open("x", encoding="utf-8") as target:
        json.dump(provenance, target, indent=2, sort_keys=True)
        target.write("\n")
    return provenance_path
