"""Read-only development probe between a frozen static student and its ranked checkpoint.

This is a diagnostic, not a held-out evaluation or a model-selection shortcut.
It downloads existing immutable artifacts into RAM, checks their SHA256 hashes,
and writes only a new local diagnostic report.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import boto3
import numpy as np
from model2vec import StaticModel
from safetensors.numpy import load as load_safetensors
from tokenizers import Tokenizer

from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score
from pycon_model2vec_demo.multiwikiqa_ranking_v3 import prepare


BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
JOB = "pycon-multiwikiqa-ranked-v3-20261004-165739-f68b9a"
PREFIX = f"pycon-davao-model2vec/{JOB}/results"


def read_verified(s3, key, expected_hash=None, expected_bytes=None):
    result = s3.get_object(Bucket=BUCKET, Key=key)
    raw = result["Body"].read()
    result["Body"].close()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"SHA256 mismatch: {key}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError(f"Byte count mismatch: {key}")
    metadata_hash = result.get("Metadata", {}).get("sha256")
    if metadata_hash and metadata_hash != digest:
        raise ValueError(f"S3 metadata SHA256 mismatch: {key}")
    return raw, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "runs" / JOB / "mixed-plan.json").read_text())
    s3 = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1").client("s3")

    def spec_json(spec):
        data, _ = read_verified(s3, spec["key"], spec["sha256"], spec["bytes"])
        return json.loads(data)

    layout = spec_json(plan["evaluation_layout"])
    annotations, _ = read_verified(s3, plan["annotations"]["key"],
                                   plan["annotations"]["sha256"], plan["annotations"]["bytes"])
    queries = [json.loads(line) for line in annotations.splitlines()]
    train, validation, skipped = prepare(queries, layout["chunks"])
    del annotations, queries
    if len(train) != 1734 or len(validation) != 338 or len(skipped) != 6:
        raise ValueError("Development partition changed")
    assert not any(q["partition"] == "evaluation" for q in validation)
    print(f"Validated {len(validation)} development queries and {len(layout['chunks'])} chunks", flush=True)

    def epoch_file(epoch, filename):
        key = f"{PREFIX}/models/approved_articles_ranked/epoch-{epoch:02d}/{filename}"
        data, digest = read_verified(s3, key)
        return data, digest

    config_bytes, config_hash = epoch_file(0, "config.json")
    tokenizer_bytes, tokenizer_hash = epoch_file(0, "tokenizer.json")
    config = json.loads(config_bytes)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    del config_bytes, tokenizer_bytes
    raw0, hash0 = epoch_file(0, "model.safetensors")
    tensor0 = load_safetensors(raw0)
    raw3, hash3 = epoch_file(3, "model.safetensors")
    tensor3 = load_safetensors(raw3)
    if tensor0.keys() != tensor3.keys() or len(tensor0) != 1:
        raise ValueError("Checkpoint tensor layouts differ")
    key = next(iter(tensor0))
    base = np.array(tensor0[key], dtype=np.float32, copy=True)
    delta = np.array(tensor3[key] - base, dtype=np.float32, copy=True)
    del raw0, raw3, tensor0, tensor3
    gc.collect()
    if base.shape != (249999, 1024) or delta.shape != base.shape:
        raise ValueError("Checkpoint dimensions changed")
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError("Nonfinite checkpoint")
    print("Verified both 1,024-dimensional float32 checkpoints", flush=True)

    chunks = layout["chunks"]
    ids = layout["parents"]
    positions = {name: i for i, name in enumerate(ids)}
    owners = [positions[chunk["parent_id"]] for chunk in chunks]
    results = []
    for alpha in (0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1):
        vectors = base.copy()
        if alpha:
            vectors += np.float32(alpha) * delta
        model = StaticModel(vectors=vectors, tokenizer=Tokenizer.from_str(tokenizer.to_str()),
                            config=config, normalize=True)
        documents = model.encode([chunk["text"] for chunk in chunks], batch_size=512,
                                 max_length=None, use_multiprocessing=False)
        questions = model.encode([q["question"] for q in validation], batch_size=512,
                                 max_length=None, use_multiprocessing=False)
        per_query = []
        for start in range(0, len(validation), 64):
            rankings = rank_parents(questions[start:start + 64], documents, owners, ids)
            for question, ranking in zip(validation[start:start + 64], rankings):
                per_query.append({"query_id": question["query_id"],
                                  "language": question["language"],
                                  "gold_parent_id": question["gold_parent_id"],
                                  "gold_rank": next((i + 1 for i, row in enumerate(ranking)
                                                     if row[0] == question["gold_parent_id"]), None),
                                  "metrics": score(question, ranking, chunks)})
        grouped = {}
        for language in ("all", "en", "tl"):
            rows = per_query if language == "all" else [r for r in per_query if r["language"] == language]
            grouped[language] = {metric: float(np.mean([r["metrics"][metric] for r in rows]))
                                 for metric in ("Recall@10", "nDCG@10", "MRR@10")}
            grouped[language]["questions"] = len(rows)
            grouped[language]["hits@10"] = int(sum(r["metrics"]["Recall@10"] for r in rows))
        results.append({"alpha": alpha, "summary": grouped, "per_query": per_query})
        print(f"alpha={alpha}: {grouped}", flush=True)
        del model, vectors, documents, questions, per_query
        gc.collect()

    baseline = results[0]
    for result in results:
        result["versus_epoch0"] = {
            language: {"gained": [r["query_id"] for old, r in
                                   zip(baseline["per_query"], result["per_query"])
                                   if r["language"] == language and
                                   old["metrics"]["Recall@10"] == 0 and r["metrics"]["Recall@10"] == 1],
                       "lost": [r["query_id"] for old, r in
                                zip(baseline["per_query"], result["per_query"])
                                if r["language"] == language and
                                old["metrics"]["Recall@10"] == 1 and r["metrics"]["Recall@10"] == 0]}
            for language in ("en", "tl")}
        result["passes_original_guard"] = (result["alpha"] > 0 and
            result["summary"]["all"]["nDCG@10"] > baseline["summary"]["all"]["nDCG@10"] and
            all(result["summary"][lang][metric] >= baseline["summary"][lang][metric]
                for lang in ("en", "tl") for metric in ("Recall@10", "nDCG@10")))
        result["no_individual_recall_losses"] = all(
            not result["versus_epoch0"][lang]["lost"] for lang in ("en", "tl"))
    output = {"purpose": "development-only diagnostic; no heldout scoring or checkpoint selection",
              "job": JOB, "checkpoint_hashes": {"epoch0": hash0, "epoch3": hash3,
                                               "tokenizer": tokenizer_hash, "config": config_hash},
              "cohort": {"development_questions": 338, "chunks": 6546, "parents": 1294,
                         "layout_sha256": plan["evaluation_layout"]["sha256"],
                         "annotations_sha256": plan["annotations"]["sha256"]},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Preserved diagnostic: {args.output}", flush=True)


if __name__ == "__main__":
    main()
