"""Guards and ranking helpers for the frozen retrieval experiment."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .corpus import Corpus, MappedQuery
from .model_registry import MODEL_NAMES, ModelArtifact, local_model_artifacts
from .retrieval import RankedPassage


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def final_run_preflight(project_dir: Path) -> tuple[dict[str, ModelArtifact], dict[str, dict]]:
    """Do not even open held-out queries until all five models/measurements match."""
    artifacts = local_model_artifacts(project_dir, require_all=True)
    if set(artifacts) != set(MODEL_NAMES):
        raise RuntimeError("Final evaluation requires all five frozen models")
    if list((project_dir / "runs").glob("heldout-five-model-*.json")):
        raise FileExistsError("A final held-out report already exists; do not rerun or tune against it")
    config_hashes = {
        name: file_sha256(project_dir / "configs" / filename)
        for name, filename in (
            ("dataset", "disastir-qa-all-categories.json"),
            ("retrieval", "retrieval-v1.json"),
            ("measurement", "measurement-v1.json"),
            ("decision", "decision-v1.json"),
            ("models", "models-v1.json"),
            ("tokenlearn", "tokenlearn-v1.json"),
        )
    }
    for name in ("dataset", "retrieval", "measurement", "decision"):
        value = json.loads((project_dir / "configs" / {
            "dataset": "disastir-qa-all-categories.json", "retrieval": "retrieval-v1.json",
            "measurement": "measurement-v1.json", "decision": "decision-v1.json",
        }[name]).read_text(encoding="utf-8"))
        if name != "dataset" and value.get("status") != "frozen-before-held-out-evaluation":
            raise RuntimeError(f"{name} config is not marked frozen")
    measurements = {}
    for name, artifact in artifacts.items():
        path = project_dir / "runs" / f"measurement-{name}-{artifact.fingerprint[:12]}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fresh-process measurement for {name}: {path}")
        measurement = json.loads(path.read_text(encoding="utf-8"))
        if measurement.get("model_fingerprint") != artifact.fingerprint:
            raise ValueError(f"{name} measurement fingerprint mismatch")
        if measurement.get("config_sha256") != {key: config_hashes[key] for key in ("dataset", "retrieval", "measurement")}:
            raise ValueError(f"{name} measurement config hash mismatch")
        if measurement.get("status") != "development-input-operational-measurement; no-held-out-queries-or-quality-scores":
            raise ValueError(f"{name} measurement was not development-only")
        build_path = project_dir / "runs" / f"measurement-build-{name}-{artifact.fingerprint[:12]}.json"
        if build_path.is_file():
            build = json.loads(build_path.read_text(encoding="utf-8"))
            if build.get("model_fingerprint") != artifact.fingerprint or build.get("config_sha256") != measurement["config_sha256"]:
                raise ValueError(f"{name} fresh corpus build provenance mismatch")
            if build.get("full_corpus_embedding_build_seconds") is None or not build.get("fresh_embedding_build_requested"):
                raise ValueError(f"{name} fresh corpus build timing is missing")
            measurement["separate_fresh_corpus_embedding_build_seconds"] = build["full_corpus_embedding_build_seconds"]
        measurements[name] = measurement
    return artifacts, measurements


def judged_candidate_rankings(
    queries: Sequence[MappedQuery], query_vectors: np.ndarray,
    passage_vectors: np.ndarray, corpus: Corpus,
) -> tuple[tuple[RankedPassage, ...], ...]:
    """Secondary, easier ranking restricted to each query's mapped judged pool."""
    if len(query_vectors) != len(queries):
        raise ValueError("Query vector count mismatch")
    index = {corpus_id: position for position, corpus_id in enumerate(corpus.ids)}
    output = []
    for query, vector in zip(queries, query_vectors):
        ids = sorted(query.judgments)
        selected = np.asarray(passage_vectors[[index[corpus_id] for corpus_id in ids]], dtype=np.float32)
        similarities = selected @ vector
        output.append(tuple(sorted(
            (RankedPassage(corpus_id, float(similarity)) for corpus_id, similarity in zip(ids, similarities)),
            key=lambda row: (-row.similarity, row.corpus_id),
        )))
    return tuple(output)
