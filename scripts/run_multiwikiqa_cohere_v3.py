#!/usr/bin/env python3
"""Preflight or execute the frozen MultiWikiQA Cohere multilingual v3 baseline.

The default mode is offline preflight. Bedrock is invoked only with --execute.
Batch artifacts are immutable and checksum-verified so an interrupted execution
can resume without silently replacing paid outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere  # noqa: E402
from pycon_model2vec_demo.multiwikiqa_cohere_v3 import (  # noqa: E402
    file_sha256,
    load_frozen_inputs,
    preflight_summary,
)
from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score, summarize  # noqa: E402


def digest_json(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def load_or_encode(
    encoder: BedrockCohere,
    texts: list[str],
    role: str,
    directory: Path,
    *,
    dimension: int,
    batch_size: int,
) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        stem = f"{start:06d}-{start + len(batch):06d}"
        vector_path = directory / f"{stem}.npy"
        receipt_path = directory / f"{stem}.json"
        input_sha256 = digest_json(batch)
        if vector_path.exists() or receipt_path.exists():
            if not vector_path.is_file() or not receipt_path.is_file():
                raise RuntimeError(f"incomplete immutable Cohere batch: {stem}")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("input_sha256") != input_sha256
                or receipt.get("role") != role
                or receipt.get("rows") != len(batch)
                or receipt.get("vector_sha256") != file_sha256(vector_path)
            ):
                raise ValueError(f"existing Cohere batch provenance changed: {stem}")
            vectors = np.load(vector_path, allow_pickle=False)
        else:
            vectors, provider_receipt = encoder.encode(batch, role)
            if vectors.shape != (len(batch), dimension):
                raise ValueError("Cohere batch dimension changed")
            with vector_path.open("xb") as stream:
                np.save(stream, vectors, allow_pickle=False)
            write_new(receipt_path, {
                "role": role,
                "rows": len(batch),
                "input_sha256": input_sha256,
                "vector_sha256": file_sha256(vector_path),
                "provider": provider_receipt,
            })
        if vectors.shape != (len(batch), dimension) or vectors.dtype != np.float32:
            raise ValueError(f"invalid stored Cohere vectors: {stem}")
        parts.append(vectors)
    result = np.concatenate(parts)
    if result.shape != (len(texts), dimension) or not np.isfinite(result).all():
        raise ValueError("assembled Cohere matrix is invalid")
    return result


def execute(config: dict, output: Path) -> dict:
    import boto3
    from botocore.config import Config

    layout, queries = load_frozen_inputs(ROOT, config)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run-manifest.json"
    manifest = {
        "protocol": config["protocol"],
        "config_sha256": file_sha256(ROOT / "configs/multiwikiqa-cohere-multilingual-v3-eval-v1.json"),
        "model_id": config["model"]["model_id"],
        "region": config["aws"]["region"],
        "source_chunks": layout["segmentation"]["original_chunks"],
        "searchable_segments": len(layout["chunks"]),
        "split_chunks": layout["segmentation"]["split_chunks"],
        "safe_token_limit": layout["segmentation"]["safe_token_limit"],
        "tokenizer_revision": layout["segmentation"]["tokenizer_revision"],
        "queries": len(queries),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "latency_benchmark": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "protocol", "config_sha256", "model_id", "region", "source_chunks",
            "searchable_segments", "split_chunks", "safe_token_limit",
            "tokenizer_revision", "queries",
        ):
            if previous.get(key) != manifest[key]:
                raise ValueError("existing run manifest differs")
    else:
        write_new(manifest_path, manifest)

    session = boto3.Session(
        profile_name=config["aws"]["profile"],
        region_name=config["aws"]["region"],
    )
    catalog = {
        row["modelId"]
        for row in session.client("bedrock").list_foundation_models(byProvider="Cohere")["modelSummaries"]
    }
    if config["model"]["model_id"] not in catalog:
        raise RuntimeError("Cohere multilingual v3 is not listed in the selected AWS region")
    client = session.client(
        "bedrock-runtime",
        config=Config(retries={"max_attempts": 0, "mode": "standard"}),
    )
    encoder = BedrockCohere(client, config["model"])
    batch_size = int(config["model"]["maximum_texts_per_request"])
    dimension = int(config["model"]["dimension"])
    documents = load_or_encode(
        encoder,
        [row["text"] for row in layout["chunks"]],
        config["model"]["document_input_type"],
        output / "document-batches",
        dimension=dimension,
        batch_size=batch_size,
    )
    query_vectors = load_or_encode(
        encoder,
        [row["question"] for row in queries],
        config["model"]["query_input_type"],
        output / "query-batches",
        dimension=dimension,
        batch_size=batch_size,
    )

    parents = layout["parents"]
    parent_positions = {parent: index for index, parent in enumerate(parents)}
    owners = [parent_positions[row["parent_id"]] for row in layout["chunks"]]
    scored = []
    for start in range(0, len(queries), 64):
        rankings = rank_parents(
            query_vectors[start : start + 64], documents, owners, parents
        )
        for query, ranking in zip(queries[start : start + 64], rankings, strict=True):
            scored.append({
                **query,
                "metrics": score(query, ranking, layout["chunks"]),
                "top10": [
                    {"parent_id": parent, "chunk_index": index, "score": value}
                    for parent, index, value in ranking
                ],
            })
    report = {
        "protocol": config["protocol"],
        "model": config["model"],
        "cohort": {"language": "tl", "partition": "evaluation", "questions": len(queries)},
        "quality": summarize(scored)["evaluation_partition"],
        "per_query": scored,
        "retrieval": config["retrieval"],
        "segmentation": layout["segmentation"],
        "latency": {"status": "deferred-not-measured"},
        "server_memory": {"status": "provider-does-not-expose"},
        "server_model_size": {"status": "provider-does-not-expose"},
    }
    write_new(output / "quality-result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/multiwikiqa-cohere-multilingual-v3-eval-v1.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    summary = preflight_summary(ROOT, config)
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if not args.output:
        raise SystemExit("--output is required with --execute")
    report = execute(config, Path(args.output).resolve())
    print(json.dumps({"quality": report["quality"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
