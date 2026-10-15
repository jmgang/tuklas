#!/usr/bin/env python3
"""Independently verify one completed compact MultiWikiQA SageMaker run.

The verifier downloads the exact compact model, shared evaluation layout and
annotations into a new directory, checks every byte against immutable receipts,
then performs a fresh local exhaustive retrieval evaluation.  It does not
measure latency or peak memory and it never mutates the cloud run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score


RATE_USD_PER_HOUR = 1.315
BASELINE = (
    ROOT
    / "runs/pycon-multiwikiqa-eval-20261004-131724-611940"
    / "verified-20261004T134000Z/verification-and-results.json"
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download_verified(s3, bucket: str, key: str, destination: Path, expected: dict | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = s3.get_object(Bucket=bucket, Key=key)
    metadata_sha = response.get("Metadata", {}).get("sha256")
    value = sha256()
    size = 0
    with destination.open("xb") as stream:
        while True:
            chunk = response["Body"].read(8 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            value.update(chunk)
            size += len(chunk)
    actual = value.hexdigest()
    if expected is not None:
        if size != int(expected["bytes"]) or actual != expected["sha256"]:
            raise ValueError(f"receipt mismatch for {key}")
    if metadata_sha is not None and actual != metadata_sha:
        raise ValueError(f"S3 metadata checksum mismatch for {key}")
    return {"key": key, "bytes": size, "sha256": actual, "s3_metadata_sha256": metadata_sha}


def verify_existing(s3, bucket: str, key: str, path: Path, expected: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = digest(path)
    size = path.stat().st_size
    head = s3.head_object(Bucket=bucket, Key=key)
    metadata_sha = head.get("Metadata", {}).get("sha256")
    if size != int(expected["bytes"]) or actual != expected["sha256"]:
        raise ValueError(f"local receipt mismatch for {path}")
    if head["ContentLength"] != size or (metadata_sha is not None and metadata_sha != actual):
        raise ValueError(f"S3 receipt mismatch for {key}")
    return {"key": key, "bytes": size, "sha256": actual, "s3_metadata_sha256": metadata_sha}


def grouped_summary(rows: list[dict]) -> dict:
    groups = {
        "en": [row for row in rows if row["language"] == "en"],
        "tl": [row for row in rows if row["language"] == "tl"],
        "combined": rows,
    }
    result = {}
    for language, selected in groups.items():
        metrics = {}
        for metric in (
            "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
            "AnswerSpanCoverage@1", "AnswerSpanCoverage@3",
            "AnswerSpanCoverage@5", "AnswerSpanCoverage@10",
        ):
            values = [row["metrics"][metric] for row in selected if row["metrics"][metric] is not None]
            metrics[metric] = {"mean": float(np.mean(values)), "denominator": len(values)}
        result[language] = {"questions": len(selected), "metrics": metrics}
    return result


def baseline_slice(model: dict, language: str) -> dict:
    key = {"en": "English", "tl": "Tagalog", "combined": "evaluation_partition"}[language]
    return model["summary"][key]["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reuse-verified-download", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    record = json.loads((run / "staging-record.json").read_text())
    session = boto3.Session(profile_name=record["profile"], region_name=record["region"])
    s3, sm = session.client("s3"), session.client("sagemaker")
    state = sm.describe_training_job(TrainingJobName=record["job"])
    if state["TrainingJobStatus"] != "Completed" or state.get("FailureReason"):
        raise RuntimeError("training job is not terminal Completed")

    if args.reuse_verified_download:
        output = args.reuse_verified_download.resolve()
        if not output.is_dir() or (output / "verification.json").exists():
            raise ValueError("reuse directory must be an incomplete verification directory")
    else:
        stamp = datetime.now(timezone.utc).strftime("verification-%Y%m%dT%H%M%SZ")
        output = run / stamp
        output.mkdir(parents=False, exist_ok=False)
    prefix = record["prefix"] + "/results/"
    completion_key = prefix + "reports/compact-sota-completion.json"
    completion_path = output / "reports/compact-sota-completion.json"
    if args.reuse_verified_download:
        completion_receipt = verify_existing(
            s3, record["bucket"], completion_key, completion_path,
            {"bytes": completion_path.stat().st_size, "sha256": digest(completion_path)},
        )
    else:
        completion_receipt = download_verified(s3, record["bucket"], completion_key, completion_path)
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "complete":
        raise ValueError("completion report is not complete")

    compact_prefix = "models/selected-compact-int8-480/"
    compact_dir = output / "model"
    compact_receipts = []
    for name, size in sorted(completion["compact_artifact"]["files"].items()):
        relative = compact_prefix + name
        expected = completion["artifacts"][relative]
        if int(size) != int(expected["bytes"]):
            raise ValueError("compact artifact manifests disagree")
        if args.reuse_verified_download:
            receipt = verify_existing(s3, record["bucket"], prefix + relative, compact_dir / name, expected)
        else:
            receipt = download_verified(s3, record["bucket"], prefix + relative, compact_dir / name, expected)
        compact_receipts.append(receipt)
    measured_bytes = sum(path.stat().st_size for path in compact_dir.rglob("*") if path.is_file())
    if measured_bytes != completion["compact_artifact"]["complete_bytes"] or measured_bytes >= 100_000_000:
        raise ValueError("compact artifact size gate did not reproduce")

    plan = json.loads((ROOT / "runs/pycon-multiwikiqa-mixed-20261004-143440-0eb179/mixed-plan.json").read_text())
    layout_path = output / "evaluation/shared-layout.json"
    annotation_path = output / "evaluation/annotations.all.jsonl"
    if args.reuse_verified_download:
        layout_receipt = verify_existing(
            s3, record["bucket"], plan["evaluation_layout"]["key"], layout_path,
            plan["evaluation_layout"],
        )
        annotation_receipt = verify_existing(
            s3, record["bucket"], plan["annotations"]["key"], annotation_path,
            plan["annotations"],
        )
    else:
        layout_receipt = download_verified(
            s3, record["bucket"], plan["evaluation_layout"]["key"], layout_path,
            plan["evaluation_layout"],
        )
        annotation_receipt = download_verified(
            s3, record["bucket"], plan["annotations"]["key"], annotation_path,
            plan["annotations"],
        )
    layout = json.loads(layout_path.read_text())
    annotations = [json.loads(line) for line in annotation_path.open()]
    heldout = [row for row in annotations if row["partition"] == "evaluation"]
    if len(heldout) != 7_922 or len(layout["chunks"]) != 6_546 or len(layout["parents"]) != 1_294:
        raise ValueError("evaluation cohort changed")

    from model2vec import StaticModel

    model = StaticModel.from_pretrained(str(compact_dir))
    if model.embedding.shape != (170_003, 480) or model.embedding.dtype != np.int8:
        raise ValueError("compact model shape or dtype changed")
    document_vectors = model.encode(
        [row["text"] for row in layout["chunks"]], batch_size=1024,
        max_length=None, use_multiprocessing=False, show_progress_bar=True,
    )
    query_vectors = model.encode(
        ["[QRY] " + row["question"] for row in heldout], batch_size=1024,
        max_length=None, use_multiprocessing=False, show_progress_bar=True,
    )
    parent_positions = {parent: index for index, parent in enumerate(layout["parents"])}
    owners = [parent_positions[row["parent_id"]] for row in layout["chunks"]]
    rows = []
    for start in range(0, len(heldout), 64):
        rankings = rank_parents(
            query_vectors[start : start + 64], document_vectors, owners, layout["parents"]
        )
        for query, ranking in zip(heldout[start : start + 64], rankings, strict=True):
            if len(ranking) != 10 or len({item[0] for item in ranking}) != 10:
                raise ValueError("ranking does not contain ten unique parents")
            rows.append({
                "query_id": query["query_id"],
                "language": query["language"],
                "gold_parent_id": query["gold_parent_id"],
                "metrics": score(query, ranking, layout["chunks"]),
            })
    summary = grouped_summary(rows)
    reported_metric_differences = []
    for language in ("en", "tl", "combined"):
        for metric in ("Recall@10", "nDCG@10"):
            reported = completion["heldout"]["metrics"]["compact"][language][metric]
            reproduced = summary[language]["metrics"][metric]["mean"]
            if abs(reported - reproduced) > 1e-12:
                reported_metric_differences.append({
                    "language": language,
                    "metric": metric,
                    "cloud_report": reported,
                    "exported_artifact": reproduced,
                    "delta_export_minus_cloud": reproduced - reported,
                })

    baselines = json.loads(BASELINE.read_text())["results"]["models"]
    comparison = defaultdict(dict)
    for name in ("teacher", "potion_retrieval"):
        for language in ("en", "tl", "combined"):
            previous = baseline_slice(baselines[name], language)
            comparison[name][language] = {
                metric: {
                    "compact": summary[language]["metrics"][metric]["mean"],
                    "baseline": previous[metric]["mean"],
                    "delta": summary[language]["metrics"][metric]["mean"] - previous[metric]["mean"],
                }
                for metric in ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")
            }

    billable_seconds = int(state["BillableTimeInSeconds"])
    verification = {
        "status": "verified",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "job": record["job"],
        "managed_compute_released": True,
        "training_job_status": state["TrainingJobStatus"],
        "billable_seconds": billable_seconds,
        "estimated_compute_usd": billable_seconds / 3600 * RATE_USD_PER_HOUR,
        "credits_verified": False,
        "completion_report": completion_receipt,
        "compact_artifact": {
            "files": compact_receipts,
            "complete_bytes": measured_bytes,
            "dimension": int(model.embedding.shape[1]),
            "vocabulary": int(model.embedding.shape[0]),
            "dtype": str(model.embedding.dtype),
        },
        "evaluation": {
            "layout": layout_receipt,
            "annotations": annotation_receipt,
            "heldout_questions": len(heldout),
            "answer_span_denominator": sum(bool(row["answer_spans"]) for row in heldout),
            "ten_unique_parents_per_query": True,
            "summary": summary,
            "cloud_report_vs_exported_artifact_differences": reported_metric_differences,
        },
        "heldout_used_for_training_or_checkpoint_selection": completion["heldout_used_for_training_or_checkpoint_selection"],
        "comparison": dict(comparison),
        "verification_scope": {
            "compact_model_files_streamed_and_sha256_checked": True,
            "fresh_local_encoding_and_exact_reranking": True,
            "all_exported_artifact_metrics_recomputed": True,
            "cloud_reported_compact_metrics_reproduced_exactly": not reported_metric_differences,
            "full_width_model_files_downloaded": False,
            "latency_measured": False,
            "peak_memory_measured": False,
        },
    }
    result_path = output / "verification.json"
    result_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Verified compact SEA-LION student", "",
        f"Run: `{record['job']}`", "",
        f"Artifact: {measured_bytes:,} bytes (480-dimensional int8, 170,003 tokens).", "",
        "| Language | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Answer coverage@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for language in ("en", "tl", "combined"):
        values = summary[language]["metrics"]
        lines.append(
            "| " + language + " | " + " | ".join(
                f"{100 * values[name]['mean']:.2f}%" for name in
                ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "AnswerSpanCoverage@10")
            ) + " |"
        )
    lines.extend(["", "Latency and peak loaded memory were deliberately not measured in this run."])
    (output / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "verification": str(result_path),
        "results": str(output / "results.md"),
        "artifact_bytes": measured_bytes,
        "summary": summary,
        "billable_seconds": billable_seconds,
        "estimated_compute_usd": verification["estimated_compute_usd"],
    }, indent=2))


if __name__ == "__main__":
    main()
