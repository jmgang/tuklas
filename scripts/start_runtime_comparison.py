#!/usr/bin/env python3
"""Launch the Tuklas / teacher / Cohere v3 speed, memory and cost comparison.

GPU job: it occupies the ml.g4dn.xlarge slot and counts against the
three-concurrent-GPU limit, which this launcher checks before creating it.

RUNTIME_DRY_RUN=1 prints the plan and request and stages nothing.

Large inputs (the shared layout and annotations) are server-side copied from
the shootout job's staged inputs and checked against their recorded sha256;
only the small worker and helper modules are uploaded from this machine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import boto3

ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-southeast-1"
PROFILE = "admin-general"
ACCOUNT = "<aws-account-id>"
BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
ROLE = "arn:aws:iam::<aws-account-id>:role/service-role/AmazonSageMaker-ExecutionRole-20220924T130717"
IMAGE = "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
INSTANCE = "ml.g4dn.xlarge"
# AWS Price List API, ap-southeast-1, read 2026-10-13.  The training rate is
# what this job costs; the hosting rates price the measured throughput.
TRAINING_USD_PER_HOUR = 1.03
HOSTING = {"cpu": ("ml.m5.xlarge", 0.288), "gpu": ("ml.g4dn.xlarge", 1.0304)}
COHERE_USD_PER_1K_TOKENS = 0.0001   # same constant as the Cohere v3 receipts
MAX_RUNTIME_SECONDS = 7_200
GPU_PREFIXES = ("ml.g4dn.", "ml.g5.", "ml.g6.", "ml.p")
SHOOTOUT_INPUT = "pycon-davao-model2vec/pycon-shootout-filipino-20261008-053701-be01db/input/"
LARGE_INPUTS = {
    "shared-layout.json": "740b0b97e406e13be18af14c440116f34e17cd3b1c85bcf540d0ed2424b81141",
    "annotations.all.jsonl": "5e4cb1d6550aa74ad5f3d9a1269b079a5077094311b1a5c752dd45f8037aef7d",
}
STATIC_PACKAGES = {
    "tuklas-pruned": {
        "prefix": "pycon-davao-model2vec/pycon-score-prunebest-20261007-182144-87966f/results/pruned-model/",
        # No sha256 was recorded at upload.  The local demo/model copy is NOT a
        # reference: its download was cut short and 2,560 rows are zero
        # (1,535 in part 3, 1,023 in part 4, 2 in part 1).  The
        # S3 object's own multipart ETag (5 x 8 MiB parts) is the identity.
        "weights_etag": "37a2a90c873cdacbdc76327082c5f909-5",
        "etag_part_bytes": 8_388_608,
    },
    "tuklas-unpruned": {
        "prefix": "pycon-davao-model2vec/pycon-sealion-frontier-512-20261007-110019-70972b/results/model/",
        "weights_sha256": "3528ef13d9d357ebe054243a94864e53a173e6356c1b1ef3bfa968f44552662c",
    },
}
TEACHER = {"id": "aisingapore/SEA-LION-E5-Embedding-600M",
           "revision": "6a97cedd22714f3d4eb7c6fd1a452bf310899683"}
SMALL_SOURCES = {
    "aws_runtime_comparison.py": ROOT / "scripts/aws_runtime_comparison.py",
    "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
    "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
    "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
    "bedrock_embeddings.py": ROOT / "src/pycon_model2vec_demo/bedrock_embeddings.py",
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    dry_run = os.environ.get("RUNTIME_DRY_RUN") == "1"
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    s3 = session.client("s3")
    sagemaker = session.client("sagemaker")

    running = []
    for page in sagemaker.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        for row in page["TrainingJobSummaries"]:
            detail = sagemaker.describe_training_job(TrainingJobName=row["TrainingJobName"])
            running.append((row["TrainingJobName"], detail["ResourceConfig"]["InstanceType"]))
    gpu = [job for job in running if job[1].startswith(GPU_PREFIXES)]
    if any(kind == INSTANCE for _, kind in gpu):
        raise RuntimeError(f"{INSTANCE} slot is busy: {gpu}")
    if len(gpu) >= 3:
        raise RuntimeError(f"three GPU jobs already running: {gpu}")

    # Large inputs: verify the shootout copies by recorded sha256, then copy
    # server-side.  Nothing large leaves or enters this machine.
    large = {}
    for name, expected in LARGE_INPUTS.items():
        head = s3.head_object(Bucket=BUCKET, Key=SHOOTOUT_INPUT + name)
        recorded = head.get("Metadata", {}).get("sha256", "")
        if recorded != expected:
            raise ValueError(f"{name}: recorded sha256 {recorded} does not match {expected}")
        large[name] = {"key": SHOOTOUT_INPUT + name, "sha256": recorded, "bytes": head["ContentLength"]}
    for name, package in STATIC_PACKAGES.items():
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=package["prefix"]).get("Contents", [])
        names = {row["Key"][len(package["prefix"]):] for row in listing}
        if not {"model.safetensors", "tokenizer.json", "config.json"} <= names:
            raise ValueError(f"{name}: incomplete package under {package['prefix']}")
        key = package["prefix"] + "model.safetensors"
        if "weights_sha256" in package:
            recorded = s3.head_object(Bucket=BUCKET, Key=key).get("Metadata", {}).get("sha256")
            if recorded != package["weights_sha256"]:
                raise ValueError(f"{name}: S3 sha256 metadata {recorded} != expected")
        if "weights_etag" in package:
            attributes = s3.get_object_attributes(Bucket=BUCKET, Key=key,
                                                  ObjectAttributes=["ETag", "ObjectParts"])
            parts = attributes["ObjectParts"]["Parts"] if "Parts" in attributes.get("ObjectParts", {}) else []
            if attributes["ETag"].strip('"') != package["weights_etag"]:
                raise ValueError(f"{name}: S3 ETag {attributes['ETag']} != expected")
            if any(part["Size"] != package["etag_part_bytes"] for part in parts[:-1]):
                raise ValueError(f"{name}: S3 part size differs from the ETag recipe")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-runtime-{timestamp}-{uuid4().hex[:6]}"
    prefix = f"pycon-davao-model2vec/{job}"
    plan = {
        "bucket": BUCKET, "region": REGION, "instance_type": INSTANCE,
        "output_prefix": f"{prefix}/results/",
        "order": ["tuklas-pruned", "tuklas-unpruned", "teacher-gpu", "teacher-cpu", "cohere-v3"],
        "systems": {
            "tuklas-pruned": {"runtime": "static-cpu", "device": "cpu", "full_index": True},
            "tuklas-unpruned": {"runtime": "static-cpu", "device": "cpu", "full_index": True},
            "teacher-gpu": {"runtime": "transformer-gpu", "device": "cuda", "full_index": True},
            "teacher-cpu": {"runtime": "transformer-cpu", "device": "cpu", "full_index": False},
            "cohere-v3": {"runtime": "api", "device": "bedrock", "full_index": False},
        },
        "static_packages": STATIC_PACKAGES,
        "teacher": TEACHER,
        "protocol": {"cpu_threads": 4, "window": 192, "stride": 128, "chunks": 40_152,
                     "timing_queries": "197 development questions", "heldout_read": False,
                     "local_repetitions": 3, "cohere_repetitions": 1, "sample_step": 20},
        "usd_per_hour": {"cpu": HOSTING["cpu"][1], "gpu": HOSTING["gpu"][1]},
        "instances": {"cpu": HOSTING["cpu"][0], "gpu": HOSTING["gpu"][0]},
        "cohere_usd_per_1k_tokens": COHERE_USD_PER_1K_TOKENS,
        "large_inputs": large,
        "staged_sha256": {**{name: digest(path) for name, path in SMALL_SOURCES.items()},
                          **{name: spec["sha256"] for name, spec in large.items()}},
    }
    request = {
        "TrainingJobName": job,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE, "TrainingInputMode": "File",
            "ContainerArguments": ["python", "/opt/ml/input/data/input/aws_runtime_comparison.py"],
        },
        "InputDataConfig": [{"ChannelName": "input", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/"},
        "ResourceConfig": {"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 50},
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": "runtime-comparison"}],
    }
    summary = {"job": job, "instance": INSTANCE, "usd_per_hour": TRAINING_USD_PER_HOUR,
               "maximum_compute_usd": round(TRAINING_USD_PER_HOUR * MAX_RUNTIME_SECONDS / 3600, 4),
               "credit_coverage": "unverified", "running_jobs": running,
               "results": f"s3://{BUCKET}/{prefix}/results/"}
    if dry_run:
        print(json.dumps({"dry_run": True, "plan": plan, "request": request, **summary}, indent=2))
        return

    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    plan_path = run / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    for name, path in {**SMALL_SOURCES, "plan.json": plan_path}.items():
        with path.open("rb") as stream:
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/input/{name}", Body=stream,
                          ContentLength=path.stat().st_size, IfNoneMatch="*",
                          ServerSideEncryption="AES256", Metadata={"sha256": digest(path)})
    for name, spec in large.items():
        s3.copy_object(Bucket=BUCKET, Key=f"{prefix}/input/{name}",
                       CopySource={"Bucket": BUCKET, "Key": spec["key"]},
                       MetadataDirective="COPY", ServerSideEncryption="AES256")
    (run / "create-training-job-request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    response = sagemaker.create_training_job(**request)
    (run / "create-training-job-response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({**summary, "run": str(run)}, indent=2))


if __name__ == "__main__":
    main()
