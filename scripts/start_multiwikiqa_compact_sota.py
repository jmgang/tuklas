#!/usr/bin/env python3
"""Stage and launch one immutable compact SEA-LION training/evaluation job."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
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


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        stream.write("\n")


def main() -> None:
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    sagemaker, s3 = session.client("sagemaker"), session.client("s3")
    active = sagemaker.list_training_jobs(StatusEquals="InProgress", MaxResults=100)["TrainingJobSummaries"]
    if active:
        raise RuntimeError("another SageMaker training job is active; refusing duplicate paid compute")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-compact-sota-{timestamp}-{uuid4().hex[:6]}"
    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{job}"
    verified = ROOT / "runs/multiwikiqa-sonnet45-20261005-041012-bb7ee5/verified-20261005T042822Z"
    sources = {
        "aws_multiwikiqa_compact_sota.py": ROOT / "scripts/aws_multiwikiqa_compact_sota.py",
        "aws_sealion_train_repair_v3.py": ROOT / "scripts/aws_sealion_train_repair_v3.py",
        "compact_word_model.py": ROOT / "src/pycon_model2vec_demo/compact_word_model.py",
        "full_rank_pca.py": ROOT / "src/pycon_model2vec_demo/full_rank_pca.py",
        "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
        "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
        "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
        "sealion_runtime.py": ROOT / "src/pycon_model2vec_demo/sealion_runtime_v3.py",
        "sealion_training.py": ROOT / "src/pycon_model2vec_demo/sealion_training.py",
        "sealion_source_integrity.py": ROOT / "src/pycon_model2vec_demo/sealion_source_integrity.py",
        "settings.json": ROOT / "runs/pycon-multiwikiqa-mixed-20261004-143440-0eb179/settings.json",
        "mixed-plan.json": ROOT / "runs/pycon-multiwikiqa-mixed-20261004-143440-0eb179/mixed-plan.json",
        "sota-repair.json": ROOT / "configs/multiwikiqa-sota-repair-v1.json",
        "tokenizer-quality.json": ROOT / "runs/multiwikiqa-compact-tokenizers-20261005-v2/tokenizer-quality.json",
        "synthetic-admitted.jsonl": verified / "admitted.jsonl",
        "synthetic-verification-report.json": verified / "verification-report.json",
    }
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing staged source {name}: {path}")
    records = {
        name: {"local_path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        for name, path in sources.items()
    }
    input_manifest = {
        "protocol": "multiwikiqa-compact-sota-v1",
        "job": job,
        "files": records,
        "teacher": {
            "model": "aisingapore/SEA-LION-E5-Embedding-600M",
            "revision": "6a97cedd22714f3d4eb7c6fd1a452bf310899683",
            "dimension": 1024,
        },
        "hard_size_gate_bytes": 100_000_000,
        "compact_export": {"vocabulary": 170_003, "dimension": 480, "dtype": "int8"},
        "synthetic": {"admitted_queries": 10_827, "second_model_review": False},
        "heldout_policy": "7922 evaluation questions unavailable to training/checkpoint selection",
    }
    manifest_path = run / "input-manifest.json"
    write_new(manifest_path, input_manifest)
    records["input-manifest.json"] = {
        "local_path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": digest(manifest_path),
    }

    for name, spec in records.items():
        path = Path(spec["local_path"])
        key = f"{prefix}/input/{name}"
        with path.open("rb") as stream:
            s3.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=stream,
                ContentLength=path.stat().st_size,
                IfNoneMatch="*",
                ServerSideEncryption="AES256",
                Metadata={"sha256": spec["sha256"]},
            )
        head = s3.head_object(Bucket=BUCKET, Key=key)
        if head["ContentLength"] != spec["bytes"] or head["Metadata"].get("sha256") != spec["sha256"]:
            raise ValueError("S3 staging verification failed: " + name)
        print("STAGED", name, flush=True)

    request = {
        "TrainingJobName": job,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
            "ContainerArguments": ["python", "/opt/ml/input/data/input/aws_multiwikiqa_compact_sota.py"],
        },
        "InputDataConfig": [{
            "ChannelName": "input",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
                "S3DataDistributionType": "FullyReplicated",
            }},
            "InputMode": "File",
        }],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/",
        },
        "ResourceConfig": {
            "InstanceType": "ml.g4dn.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 28_800},
        "EnableNetworkIsolation": False,
        "EnableInterContainerTrafficEncryption": False,
        "EnableManagedSpotTraining": False,
        "Environment": {
            "SEALION_OUTPUT_BUCKET": BUCKET,
            "SEALION_OUTPUT_PREFIX": f"{prefix}/results",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
        "Tags": [
            {"Key": "Project", "Value": "pycon-davao-model2vec-demo"},
            {"Key": "Protocol", "Value": "compact-sota-v1"},
        ],
    }
    write_new(run / "create-training-job-request.json", request)
    write_new(run / "staging-record.json", {
        "job": job,
        "profile": PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "prefix": prefix,
        "files": records,
        "managed_training": True,
        "no_endpoint": True,
        "temporary_training_volume": True,
        "runtime_ceiling_seconds": 28_800,
        "credits_verified": False,
    })
    response = sagemaker.create_training_job(**request)
    write_new(run / "create-training-job-response.json", response)
    state = sagemaker.describe_training_job(TrainingJobName=job)
    write_new(run / "initial-state.json", state)
    print(json.dumps({
        "job": job,
        "status": state["TrainingJobStatus"],
        "instance": "ml.g4dn.2xlarge",
        "results": f"s3://{BUCKET}/{prefix}/results/",
        "run": str(run),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
