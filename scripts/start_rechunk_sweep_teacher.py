#!/usr/bin/env python3
"""Launch a GPU SageMaker job that sweeps chunk size for the TEACHER.

Scoring a static lookup table needs no GPU, so this runs on an ``ml.r8i`` CPU
instance.  Those carry their own quota of 15 and therefore never compete with
the four single-quota ``g4dn`` GPU slots -- a scoring job can run alongside a
full complement of training jobs.
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
}
EVALUATION = ROOT / "runs/pycon-compact-sota-20261005-055331-d48482/verification-20261005T090654Z/evaluation"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    model_prefix = ""
    windows = json.loads(os.environ.get("RECHUNK_WINDOWS", "[]"))
    if not windows:
        raise ValueError("set RECHUNK_WINDOWS to a JSON list of {window,stride}")
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

    package = []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-rechunkT-{label}-{timestamp}-{uuid4().hex[:6]}"[:63]
    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{job}"

    plan = {
        "bucket": BUCKET,
        "label": label,
        "source_job": source_job or None,
        "output_prefix": f"{prefix}/results/",
        "teacher_id": "aisingapore/SEA-LION-E5-Embedding-600M",
        "teacher_revision": "6a97cedd22714f3d4eb7c6fd1a452bf310899683",
        "package_objects": {key.split("/")[-1]: size for key, size in package},
        "package_bytes": sum(size for _, size in package),
        "cohorts": {"development": 197, "heldout": 3984},
        "windows": windows,
        "selection_cohort": "development",
        "scored_on_cpu": True,
        "retrained": False,
    }
    plan_path = run / "rechunk-plan.json"
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")

    sources = {
        "aws_rechunk_sweep_teacher.py": ROOT / "scripts/aws_rechunk_sweep_teacher.py",
        "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
        "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
        "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
        "shared-layout.json": EVALUATION / "shared-layout.json",
        "annotations.all.jsonl": EVALUATION / "annotations.all.jsonl",
        "rechunk-plan.json": plan_path,
    }
    keepset = os.environ.get("SCORE_KEEP_TOKEN_IDS", "")
    if keepset:
        sources["keep-token-ids.json"] = Path(keepset)
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
                "python", "/opt/ml/input/data/input/aws_rechunk_sweep_teacher.py",
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 7_200},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": f"rechunk-{label}"}],
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
        "maximum_compute_usd": round(APPROVED_INSTANCES[instance_type], 4),
        "package_bytes": plan["package_bytes"],
        "results": f"s3://{BUCKET}/{prefix}/results/",
        "run": str(run),
    }, indent=2))


if __name__ == "__main__":
    main()
