#!/usr/bin/env python3
"""Launch a GPU job comparing off-the-shelf multilingual embedding models.

This is a GPU job: four of the five comparators are transformers and the
worker hardcodes device="cuda".  It therefore CONSUMES the single
ml.g4dn.4xlarge quota slot and counts against the three-concurrent-GPU
constraint.  Confirm that slot is free before launching -- this launcher
carries no concurrency guard.

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
APPROVED_INSTANCES = {
    "ml.g4dn.xlarge": 0.736, "ml.g4dn.2xlarge": 1.315,
    "ml.g4dn.4xlarge": 2.106, "ml.g4dn.8xlarge": 2.72,
    "ml.r8i.2xlarge": 0.6732, "ml.r8i.4xlarge": 1.3464,
}
EVALUATION = ROOT / "runs/pycon-compact-sota-20261005-055331-d48482/verification-20261005T090654Z/evaluation"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    label = os.environ["SCORE_LABEL"]
    source_job = os.environ.get("SCORE_SOURCE_JOB", "")
    instance_type = os.environ.get("SCORE_INSTANCE_TYPE", "ml.g4dn.4xlarge")
    if instance_type not in APPROVED_INSTANCES:
        raise ValueError(f"unapproved scoring instance: {instance_type}")

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    s3 = session.client("s3")
    sagemaker = session.client("sagemaker")

    student_prefix = os.environ["SHOOTOUT_STUDENT_PREFIX"].rstrip("/") + "/"
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=student_prefix)
    package = [(row["Key"], row["Size"]) for row in listing.get("Contents", [])]
    if not package:
        raise ValueError("no student package objects under " + student_prefix)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-shootout-{label}-{timestamp}-{uuid4().hex[:6]}"[:63]
    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{job}"

    plan = {
        "bucket": BUCKET,
        "label": label,
        "source_job": source_job or None,
        "output_prefix": f"{prefix}/results/",
        "student_prefix": student_prefix,
        # The worker owns the chunking; recorded literally so the receipt
        # cannot contradict what ran.
        "window": 192,
        "stride": 128,
        "scored_on_cpu": False,
        "device": os.environ.get("SHOOTOUT_DEVICE", "cuda"),
        # Each model's own convention.  multilingual-e5 REQUIRES these prefixes
        # and loses several points without them; BGE-M3 and LaBSE take none.
        "static_models": json.loads(os.environ["SHOOTOUT_STATIC_MODELS"]),
        "transformer_models": json.loads(os.environ["SHOOTOUT_TRANSFORMER_MODELS"]),
        "student_package_objects": {key.split("/")[-1]: size for key, size in package},
        "student_package_bytes": sum(size for _, size in package),
        "cohorts": {"development": 197, "heldout": 3984},
        # No selection happens here: window and stride are worker constants and
        # every model is scored on the one shared index.
        "selection_cohort": None,
        "retrained": False,
    }
    plan_path = run / "shootout-plan.json"
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")

    sources = {
        "aws_model_shootout.py": ROOT / "scripts/aws_model_shootout.py",
        "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
        "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
        "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
        "shared-layout.json": EVALUATION / "shared-layout.json",
        "annotations.all.jsonl": EVALUATION / "annotations.all.jsonl",
        "shootout-plan.json": plan_path,
    }
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing staged source {name}: {path}")
        key = f"{prefix}/input/{name}"
        with path.open("rb") as stream:
            s3.put_object(
                Bucket=BUCKET, Key=key, Body=stream,
                ContentLength=path.stat().st_size, IfNoneMatch="*",
                ServerSideEncryption="AES256", Metadata={"sha256": digest(path)},
            )
        print("STAGED", name, flush=True)

    request = {
        "TrainingJobName": job,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
            "ContainerArguments": [
                "python", "/opt/ml/input/data/input/aws_model_shootout.py",
            ],
        },
        "InputDataConfig": [{
            "ChannelName": "input",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
                "S3DataDistributionType": "FullyReplicated",
            }},
        }],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/"},
        "ResourceConfig": {
            "InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 50,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 10_800},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": f"shootout-{label}"}],
    }
    with (run / "create-training-job-request.json").open("x", encoding="utf-8") as stream:
        json.dump(request, stream, indent=2, sort_keys=True)
        stream.write("\n")
    response = sagemaker.create_training_job(**request)
    with (run / "create-training-job-response.json").open("x", encoding="utf-8") as stream:
        json.dump(response, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    print(json.dumps({
        "job": job,
        "instance": instance_type,
        "usd_per_hour": APPROVED_INSTANCES[instance_type],
        "maximum_compute_usd": round(
            APPROVED_INSTANCES[instance_type]
            * request["StoppingCondition"]["MaxRuntimeInSeconds"] / 3600, 4),
        "credit_coverage": "unverified",
        "student_package_bytes": plan["student_package_bytes"],
        "results": f"s3://{BUCKET}/{prefix}/results/",
        "run": str(run),
    }, indent=2))


if __name__ == "__main__":
    main()
