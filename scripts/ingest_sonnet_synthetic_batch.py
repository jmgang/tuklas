#!/usr/bin/env python3
"""Verify and ingest a completed Sonnet 4.5 synthetic-query batch.

The raw Bedrock output is preserved byte-for-byte. Invalid responses are
recorded with reasons and excluded; there is no model-based review stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import boto3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.synthetic_retrieval import parse_model_json, validate_generated_items


BATCH_INPUT_USD_PER_MILLION = 1.50
BATCH_OUTPUT_USD_PER_MILLION = 7.50


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        stream.write("\n")


def extract_text(model_output: dict[str, object]) -> str:
    content = model_output.get("content")
    if not isinstance(content, list):
        raise ValueError("modelOutput.content is absent")
    texts = [str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"]
    if not texts or not "".join(texts).strip():
        raise ValueError("modelOutput contains no assistant text")
    return "".join(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    resource = json.loads((run / "resource-record.json").read_text(encoding="utf-8"))
    launch = json.loads((run / "create-job-response.json").read_text(encoding="utf-8"))
    session = boto3.Session(profile_name=resource["profile"], region_name=resource["region"])
    bedrock, s3 = session.client("bedrock"), session.client("s3")
    status = bedrock.get_model_invocation_job(jobIdentifier=launch["jobArn"])
    if status["status"] != "Completed":
        raise RuntimeError(f"batch is not complete: {status['status']} {status.get('message', '')}")
    expected_records = int(resource["source_passages"])
    if int(status.get("processedRecordCount", 0)) != expected_records:
        raise RuntimeError(f"unexpected processed record count: {status.get('processedRecordCount')}")

    output = args.output.resolve() if args.output else run / f"verified-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(parents=True, exist_ok=False)
    source_run = Path(resource["prepared_path"]).resolve()
    if digest(source_run / "manifest.json") != resource["prepared_manifest_sha256"]:
        raise ValueError("prepared manifest changed after batch launch")
    if digest(source_run / "selected-passages.jsonl") != resource["selected_passages_sha256"]:
        raise ValueError("selected source passages changed after batch launch")
    sources: dict[str, dict[str, object]] = {}
    with (source_run / "selected-passages.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            record_id = row.pop("record_id")
            if record_id in sources:
                raise ValueError(f"duplicate source record: {record_id}")
            sources[record_id] = row
    if len(sources) != expected_records:
        raise ValueError(f"unexpected source count: {len(sources)}")

    prefix = f"{resource['prefix']}/output/"
    listed = s3.list_objects_v2(Bucket=resource["bucket"], Prefix=prefix)
    # Bedrock writes both the JSONL responses and a separate manifest.json.out
    # beneath the output prefix.  Only the former contains recordId/modelOutput
    # rows; preserve the manifest in S3, but never parse it as inference output.
    objects = [item for item in listed.get("Contents", []) if item["Key"].endswith((".jsonl", ".jsonl.out"))]
    if not objects:
        raise RuntimeError("completed job has no JSONL output object")

    raw_dir = output / "raw"
    raw_dir.mkdir(parents=False, exist_ok=False)
    seen: set[str] = set()
    admitted_count = Counter()
    rejection_count = Counter()
    total_input_tokens = 0
    total_output_tokens = 0
    raw_receipts = []
    with (output / "admitted.jsonl").open("x", encoding="utf-8") as admitted_stream, (output / "rejected.jsonl").open("x", encoding="utf-8") as rejected_stream:
        for index, item in enumerate(sorted(objects, key=lambda value: value["Key"]), start=1):
            raw_path = raw_dir / f"output-{index:03d}.jsonl"
            body = s3.get_object(Bucket=resource["bucket"], Key=item["Key"])["Body"]
            with raw_path.open("xb") as raw_stream:
                for block in iter(lambda: body.read(1024 * 1024), b""):
                    raw_stream.write(block)
            raw_receipts.append({"key": item["Key"], "bytes": raw_path.stat().st_size, "sha256": digest(raw_path)})
            with raw_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    row = json.loads(line)
                    record_id = str(row.get("recordId", ""))
                    if record_id not in sources:
                        raise ValueError(f"unknown recordId {record_id!r} in {item['Key']}:{line_number}")
                    if record_id in seen:
                        raise ValueError(f"duplicate output recordId: {record_id}")
                    seen.add(record_id)
                    try:
                        if row.get("error"):
                            raise ValueError(f"Bedrock record error: {row['error']}")
                        model_output = row.get("modelOutput")
                        if not isinstance(model_output, dict):
                            raise ValueError("modelOutput is absent")
                        usage = model_output.get("usage", {})
                        if isinstance(usage, dict):
                            total_input_tokens += int(usage.get("input_tokens", 0))
                            total_output_tokens += int(usage.get("output_tokens", 0))
                        items = validate_generated_items(parse_model_json(extract_text(model_output)), sources[record_id])
                        for query in items:
                            full = {"record_id": record_id, **query}
                            admitted_stream.write(json.dumps(full, ensure_ascii=False, sort_keys=True) + "\n")
                            admitted_count[(query["language"], query["query_type"])] += 1
                    except Exception as error:  # preserve every rejected response and its exact reason
                        reason = f"{type(error).__name__}: {error}"
                        rejection_count[reason] += 1
                        rejected_stream.write(json.dumps({"record_id": record_id, "reason": reason, "raw": row}, ensure_ascii=False, sort_keys=True, default=str) + "\n")

    missing = sorted(set(sources) - seen)
    if missing:
        raise RuntimeError(f"completed output is missing {len(missing)} source records")
    if len(seen) != int(status.get("successRecordCount", 0)) + int(status.get("errorRecordCount", 0)):
        raise RuntimeError("output record count does not match Bedrock status counters")
    cost = total_input_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MILLION + total_output_tokens / 1_000_000 * BATCH_OUTPUT_USD_PER_MILLION
    report = {
        "protocol": "multiwikiqa-synthetic-retrieval-v1-sonnet45-ingestion",
        "job_arn": status["jobArn"],
        "job_status": status["status"],
        "source_records": len(sources),
        "output_records_seen": len(seen),
        "bedrock_success_records": int(status.get("successRecordCount", 0)),
        "bedrock_error_records": int(status.get("errorRecordCount", 0)),
        "admitted_source_records": sum(admitted_count.values()) // 3,
        "admitted_query_pairs": sum(admitted_count.values()),
        "admitted_by_language_and_type": {f"{language}:{query_type}": count for (language, query_type), count in sorted(admitted_count.items())},
        "rejected_source_records": sum(rejection_count.values()),
        "rejection_reasons": dict(rejection_count),
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "batch_input_rate_usd_per_million": BATCH_INPUT_USD_PER_MILLION,
            "batch_output_rate_usd_per_million": BATCH_OUTPUT_USD_PER_MILLION,
            "estimated_batch_inference_usd": cost,
            "credits_verified": False,
        },
        "raw_output_receipts": raw_receipts,
        "admitted_sha256": digest(output / "admitted.jsonl"),
        "rejected_sha256": digest(output / "rejected.jsonl"),
        "second_model_review_used": False,
    }
    write_new_json(output / "verification-report.json", report)
    print(json.dumps({"verified_output": str(output), **report}, indent=2, default=str))


if __name__ == "__main__":
    main()
