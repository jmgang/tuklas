"""Pinned, plain-vocabulary Model2Vec distillation and provenance capture."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np

from .encoders import StaticEncoder, load_static_encoder


SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
)


def file_hashes(directory: Path) -> dict[str, str]:
    """Hash regular artifact files, excluding Hub's mutable download metadata."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(directory).parts:
            digest = sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.relative_to(directory).as_posix()] = digest.hexdigest()
    return result


def _check_versions(expected: dict[str, str]) -> dict[str, str]:
    actual = {package: version(package) for package in expected}
    mismatches = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
    if mismatches:
        raise RuntimeError(f"Installed library versions differ from pinned config: {mismatches}")
    return actual


def distill_plain(project_dir: Path, config_path: Path) -> Path:
    """Download pinned E5 files, distill default vocabulary, reload, and record evidence.

    Never reads DisastIR queries, qrels, or passages. Refuses to replace an
    existing student or provenance record.
    """
    # Hugging Face resolves its cache paths at import time. Keep every download
    # and xet log inside the explicitly authorized project directory.
    os.environ["HF_HOME"] = str(project_dir / "models" / ".hf-cache")
    os.environ["HF_XET_CACHE"] = str(project_dir / "models" / ".hf-cache" / "xet")
    from huggingface_hub import snapshot_download
    from model2vec.distill import distill

    config = json.loads(config_path.read_text(encoding="utf-8"))
    teacher = config["teacher"]
    student = config["students"]["plain"]
    if student["vocabulary"] is not None:
        raise ValueError("Task 2.3 requires the teacher's default vocabulary")
    actual_versions = _check_versions(config["library_versions"])
    required_free = config["machine_preflight"]["minimum_free_disk_gib_for_distillation"] * 1024**3
    if shutil.disk_usage(project_dir).free < required_free:
        raise RuntimeError("Insufficient free disk for model download and distillation")

    suffix = teacher["revision"][:12]
    teacher_dir = project_dir / "models" / f"e5-base-v2-{suffix}"
    student_dir = project_dir / "models" / f"plain-e5-{suffix}"
    provenance_path = project_dir / "runs" / f"plain-e5-{suffix}-provenance.json"
    if provenance_path.exists():
        raise FileExistsError("Provenance already exists; refusing to overwrite the completed run")
    resuming_saved_artifact = student_dir.exists()
    if resuming_saved_artifact:
        required_student_files = {"config.json", "model.safetensors", "tokenizer.json", "modules.json"}
        missing_student_files = required_student_files - set(file_hashes(student_dir))
        if missing_student_files:
            raise RuntimeError(f"Existing student artifact is incomplete: {sorted(missing_student_files)}")
    teacher_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=teacher["repo"],
        revision=teacher["revision"],
        local_dir=teacher_dir,
        allow_patterns=list(SNAPSHOT_FILES),
    )
    missing = [name for name in SNAPSHOT_FILES if not (teacher_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Pinned teacher snapshot is incomplete: {missing}")

    distilled = None
    if not resuming_saved_artifact:
        distilled = distill(
            model_name=str(teacher_dir),
            vocabulary=None,
            device=student["device"],
            pca_dims=student["pca_dims"],
            sif_coefficient=student["sif_coefficient"],
            token_remove_pattern=student["token_remove_pattern"],
            trust_remote_code=False,
            quantize_to=student["quantize_to"],
            vocabulary_quantization=student["vocabulary_quantization"],
            pooling=student["pooling"],
        )
        if distilled.dim != student["pca_dims"]:
            raise RuntimeError(f"Unexpected distilled dimension: {distilled.dim}")
        student_dir.mkdir(parents=True, exist_ok=False)
        distilled.save_pretrained(student_dir)

    reloaded: StaticEncoder = load_static_encoder(
        str(student_dir),
        max_length=student["max_length"],
        batch_size=student["batch_size"],
        query_prefix=student["query_prefix"],
        passage_prefix=student["passage_prefix"],
    )
    smoke_queries = ["evacuation route", "flood warning"]
    smoke_passages = ["Road closures after flooding", "Emergency shelter is open"]
    if reloaded.dimension != student["pca_dims"] or not reloaded.model.normalize:
        raise RuntimeError("Reloaded student has unexpected dimension or normalization setting")
    for texts, prefix, encode_role in (
        (smoke_queries, student["query_prefix"], reloaded.encode_queries),
        (smoke_passages, student["passage_prefix"], reloaded.encode_passages),
    ):
        restored = encode_role(texts)
        direct = reloaded.model.encode([prefix + text for text in texts])
        if restored.shape != (len(texts), student["pca_dims"]) or not np.allclose(direct, restored, atol=0.002):
            raise RuntimeError("Reloaded student failed the role-aware smoke test")
        if distilled is not None:
            originals = distilled.encode([prefix + text for text in texts])
            if not np.allclose(originals, restored, atol=0.002):
                raise RuntimeError("Reloaded student differs from the in-memory distillation")

    provenance = {
        "status": "plain-student-created; retrieval-quality-not-yet-evaluated",
        "teacher": {"repo": teacher["repo"], "revision": teacher["revision"], "files_sha256": file_hashes(teacher_dir)},
        "student": {
            "path": str(student_dir.relative_to(project_dir)),
            "dimension": reloaded.dimension,
            "vocabulary_entries": len(reloaded.model.tokens),
            "settings": student,
            "files_sha256": file_hashes(student_dir),
        },
        "libraries": actual_versions,
        "host": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()},
        "smoke_test": {"queries": len(smoke_queries), "passages": len(smoke_passages), "reloaded": True},
        "resumed_from_saved_artifact": resuming_saved_artifact,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("x", encoding="utf-8") as target:
        json.dump(provenance, target, indent=2, sort_keys=True)
        target.write("\n")
    return provenance_path


def distill_custom(project_dir: Path, config_path: Path, vocabulary_path: Path) -> Path:
    """Distill the same pinned E5 with only the frozen corpus vocabulary changed."""
    os.environ["HF_HOME"] = str(project_dir / "models" / ".hf-cache")
    os.environ["HF_XET_CACHE"] = str(project_dir / "models" / ".hf-cache" / "xet")
    from model2vec.distill import distill

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocabulary_record = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    teacher = config["teacher"]
    student = config["students"]["plain"]
    rule = config["students"]["custom_vocabulary"]
    requested = [item["term"] for item in vocabulary_record["requested_terms"]]
    vocabulary_bytes = json.dumps(vocabulary_record["requested_terms"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if sha256(vocabulary_bytes).hexdigest() != vocabulary_record["requested_terms_sha256"]:
        raise ValueError("Frozen vocabulary terms changed after selection")
    if len(requested) != rule["candidate_cap"] or len(requested) != len(set(requested)):
        raise ValueError("Frozen vocabulary has the wrong count or duplicate entries")
    if vocabulary_record["source"] != "released-unlabeled-passage-text-only":
        raise ValueError("Custom vocabulary must come only from the released corpus")
    if vocabulary_record["rule"] != rule:
        raise ValueError("Frozen vocabulary rule differs from model config")
    actual_versions = _check_versions(config["library_versions"])

    suffix = teacher["revision"][:12]
    teacher_dir = project_dir / "models" / f"e5-base-v2-{suffix}"
    plain_dir = project_dir / "models" / f"plain-e5-{suffix}"
    plain_provenance_path = project_dir / "runs" / f"plain-e5-{suffix}-provenance.json"
    custom_dir = project_dir / "models" / f"custom-e5-{suffix}"
    provenance_path = project_dir / "runs" / f"custom-e5-{suffix}-provenance.json"
    plain_provenance = json.loads(plain_provenance_path.read_text(encoding="utf-8"))
    if file_hashes(teacher_dir) != plain_provenance["teacher"]["files_sha256"]:
        raise ValueError("Teacher files differ from those used for the plain student")
    if file_hashes(plain_dir) != plain_provenance["student"]["files_sha256"]:
        raise ValueError("Plain student files changed after provenance was recorded")
    if provenance_path.exists():
        raise FileExistsError("Custom-student provenance already exists")
    resuming_saved_artifact = custom_dir.exists()
    if not resuming_saved_artifact:
        if shutil.disk_usage(project_dir).free < config["machine_preflight"]["minimum_free_disk_gib_for_distillation"] * 1024**3:
            raise RuntimeError("Insufficient free disk for custom distillation")
        distilled = distill(
            model_name=str(teacher_dir),
            vocabulary=requested,
            device=student["device"],
            pca_dims=student["pca_dims"],
            sif_coefficient=student["sif_coefficient"],
            token_remove_pattern=student["token_remove_pattern"],
            trust_remote_code=False,
            quantize_to=student["quantize_to"],
            vocabulary_quantization=student["vocabulary_quantization"],
            pooling=student["pooling"],
        )
        if distilled.dim != student["pca_dims"]:
            raise RuntimeError("Custom student has an unexpected dimension")
        custom_dir.mkdir(parents=True, exist_ok=False)
        distilled.save_pretrained(custom_dir)
    reloaded = load_static_encoder(
        str(custom_dir), max_length=student["max_length"], batch_size=student["batch_size"],
        query_prefix=student["query_prefix"], passage_prefix=student["passage_prefix"],
    )
    if reloaded.dimension != student["pca_dims"] or not reloaded.model.normalize:
        raise RuntimeError("Reloaded custom student has unexpected settings")
    queries = ["flood warning", "evacuation route"]
    passages = ["Flood warning in the coastal region", "Emergency evacuation is underway"]
    for vectors, count in ((reloaded.encode_queries(queries), len(queries)), (reloaded.encode_passages(passages), len(passages))):
        if vectors.shape != (count, student["pca_dims"]) or not np.isfinite(vectors).all():
            raise RuntimeError("Reloaded custom student failed smoke test")

    plain_tokens = set(load_static_encoder(str(plain_dir)).model.tokens)
    custom_tokens = set(reloaded.model.tokens)
    present = [term for term in requested if term in custom_tokens]
    new = [term for term in present if term not in plain_tokens]
    absent = [term for term in requested if term not in custom_tokens]
    provenance = {
        "status": "custom-student-created; retrieval-quality-not-yet-evaluated",
        "teacher": {"repo": teacher["repo"], "revision": teacher["revision"], "files_sha256": file_hashes(teacher_dir)},
        "plain_student_files_sha256": plain_provenance["student"]["files_sha256"],
        "vocabulary": {
            "selection_record": str(vocabulary_path.relative_to(project_dir)),
            "requested_terms_sha256": vocabulary_record["requested_terms_sha256"],
            "requested_count": len(requested),
            "present_count": len(present),
            "new_vs_plain_count": len(new),
            "absent_count": len(absent),
            "present_terms": present,
            "new_vs_plain_terms": new,
            "absent_terms": absent,
        },
        "student": {
            "path": str(custom_dir.relative_to(project_dir)),
            "dimension": reloaded.dimension,
            "vocabulary_entries": len(reloaded.model.tokens),
            "settings_except_vocabulary": student,
            "files_sha256": file_hashes(custom_dir),
        },
        "libraries": actual_versions,
        "smoke_test": {"queries": len(queries), "passages": len(passages), "reloaded": True},
        "resumed_from_saved_artifact": resuming_saved_artifact,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("x", encoding="utf-8") as target:
        json.dump(provenance, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
    return provenance_path
