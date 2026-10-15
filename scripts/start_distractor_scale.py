#!/usr/bin/env python3
"""Launch a CPU job that scales the retrieval haystack with distractors.

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
    "ml.r8i.large": 0.1683, "ml.r8i.xlarge": 0.3366, "ml.r8i.2xlarge": 0.6732,
    "ml.r8i.4xlarge": 1.3464, "ml.r8i.8xlarge": 2.6928,
}
EVALUATION = ROOT / "runs/pycon-compact-sota-20261005-055331-d48482/verification-20261005T090654Z/evaluation"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    model_prefix = os.environ["SCORE_MODEL_PREFIX"]
    corpus_key = os.environ["SCALE_CORPUS_KEY"]
    label = os.environ["SCORE_LABEL"]
    source_job = os.environ.get("SCORE_SOURCE_JOB", "")
    instance_type = os.environ.get("SCORE_INSTANCE_TYPE", "ml.r8i.2xlarge")
    if instance_type not in APPROVED_INSTANCES:
        raise ValueError(f"unapproved scoring instance: {instance_type}")

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    s3 = session.client("s3")
    sagemaker = session.client("sagemaker")

    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=model_prefix.rstrip("/") + "/")
    package = [(row["Key"], row["Size"]) for row in listing.get("Contents", [])]
    if not package:
        raise ValueError("no package objects under " + model_prefix)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-scale-{label}-{timestamp}-{uuid4().hex[:6]}"[:63]
    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{job}"

    plan = {
        "bucket": BUCKET,
        "label": label,
        "source_job": source_job or None,
        "model_prefix": model_prefix,
        "output_prefix": f"{prefix}/results/",
        "model2vec_version": "0.9.0",
        "package_objects": {key.split("/")[-1]: size for key, size in package},
        "package_bytes": sum(size for _, size in package),
        "cohorts": {"development": 197, "heldout": 3984},
        "expected_source_rows": 75_658,
        "corpus_key": corpus_key,
        "scored_on_cpu": True,
        "retrained": False,
    }
    plan_path = run / "scale-plan.json"
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")

    sources = {
        "aws_distractor_scale.py": ROOT / "scripts/aws_distractor_scale.py",
        "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
        "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
        "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
        "shared-layout.json": EVALUATION / "shared-layout.json",
        "annotations.all.jsonl": EVALUATION / "annotations.all.jsonl",
        "scale-plan.json": plan_path,
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

    s3.copy_object(
        Bucket=BUCKET, Key=f"{prefix}/input/training.tl.jsonl.gz",
        CopySource={"Bucket": BUCKET, "Key": corpus_key},
        ServerSideEncryption="AES256", MetadataDirective="COPY",
    )
    print("COPIED corpus server-side", flush=True)

    request = {
        "TrainingJobName": job,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
            "ContainerArguments": [
                "python", "/opt/ml/input/data/input/aws_distractor_scale.py",
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
        "StoppingCondition": {"MaxRuntimeInSeconds": 14_400},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": f"scale-{label}"}],
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
