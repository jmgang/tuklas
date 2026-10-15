#!/usr/bin/env python3
"""Launch a CPU job that builds Tagalog pair sets in AWS and writes them to S3.

Only the small inputs are uploaded (audit.json 4KB, parents.jsonl 5.8MB); the
92MB corpus is already in the bucket as a 35MB gzip and is referenced by key
with a server-side copy.  The 0.7-1.6GB pair payloads never touch local disk.

Runs on ``ml.r8i`` (CPU quota 15), so it does not consume a g4dn GPU slot.
"""

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
INSTANCE = "ml.r8i.2xlarge"
USD_PER_HOUR = 0.6732
# Already in the bucket; identical content to the v5 Tagalog corpus (75,658
# rows, verified byte-equivalent before the local duplicate was removed).
CORPUS_KEY = "pycon-davao-model2vec/pycon-sealion-20261004-073934-2eccba/input/training.tl.jsonl.gz"
PAIR_SETS_PREFIX = "pycon-davao-model2vec/pair-sets/"
EXPECTED_SOURCE_ROWS = 75_658

ARMS = [
    {
        # Volume-matched counterpart to nocrop-v15: the SAME 548,619 pairs but
        # crop-heavy and with cross-passage held down to the v14 share (1.8%).
        # Without this arm the comparison varies crop share and total volume
        # together and cannot attribute a result to either.
        "name": "cropmatched-v15",
        "target": 548_619,
        "crops_per_passage": 6,
        "inverse_cloze_per_passage": 7,
        "cross_passage_target": 9_875,
        "seed": 42,
    },
    {
        # Built once locally, so the digest is a reproducibility assertion.
        "name": "balancedfull-v15",
        "target": 2_000_000,
        "crops_per_passage": 21,
        "inverse_cloze_per_passage": 7,
        "cross_passage_target": 0,
        "seed": 42,
        "expected_sha256": "16194407661574ebf51dd5c032e69802a53871308b9b923ae792f8a0a4a1b588",
        "expected_pairs": 1_922_846,
    },
]


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    s3 = session.client("s3")
    sagemaker = session.client("sagemaker")

    corpus_bytes = int(
        s3.head_object(Bucket=BUCKET, Key=CORPUS_KEY)["ContentLength"]
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-buildpairs-{timestamp}-{uuid4().hex[:6]}"[:63]
    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{job}"

    plan = {
        "bucket": BUCKET,
        "arms": ARMS,
        "pair_sets_prefix": PAIR_SETS_PREFIX,
        "expected_source_rows": EXPECTED_SOURCE_ROWS,
        "corpus_key": CORPUS_KEY,
        "corpus_bytes": corpus_bytes,
        "built_in_aws": True,
        "local_payload_written": False,
        "heldout_questions_read": False,
        "llm_generated": False,
    }
    plan_path = run / "build-plan.json"
    with plan_path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")

    sources = {
        "aws_build_tagalog_pairs.py": ROOT / "scripts/aws_build_tagalog_pairs.py",
        "build_tagalog_unsupervised_pairs.py": ROOT / "scripts/build_tagalog_unsupervised_pairs.py",
        "audit.json": ROOT / "runs/multiwikiqa-training-text-20261004-v5/audit.json",
        "parents.jsonl": ROOT / "runs/multiwikiqa-preparation-20261004-v1/parents.jsonl",
        "build-plan.json": plan_path,
    }
    uploaded = 0
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing staged source {name}: {path}")
        with path.open("rb") as stream:
            s3.put_object(
                Bucket=BUCKET, Key=f"{prefix}/input/{name}", Body=stream,
                ContentLength=path.stat().st_size, IfNoneMatch="*",
                ServerSideEncryption="AES256", Metadata={"sha256": digest(path)},
            )
        uploaded += path.stat().st_size
        print("STAGED", name, path.stat().st_size, flush=True)

    # Server-side copy: the corpus is never downloaded or re-uploaded.
    s3.copy_object(
        Bucket=BUCKET, Key=f"{prefix}/input/training.tl.jsonl.gz",
        CopySource={"Bucket": BUCKET, "Key": CORPUS_KEY},
        ServerSideEncryption="AES256", MetadataDirective="COPY",
    )
    print("COPIED corpus server-side", corpus_bytes, flush=True)

    request = {
        "TrainingJobName": job,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
            "ContainerArguments": [
                "python", "/opt/ml/input/data/input/aws_build_tagalog_pairs.py",
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
            "InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3_600},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": "build-pairs"}],
    }
    with (run / "create-training-job-request.json").open("x", encoding="utf-8") as stream:
        json.dump(request, stream, indent=2, sort_keys=True)
        stream.write("\n")
    sagemaker.create_training_job(**request)
    print(json.dumps({
        "job": job,
        "instance": INSTANCE,
        "maximum_compute_usd": round(USD_PER_HOUR, 4),
        "uploaded_bytes": uploaded,
        "corpus_copied_server_side_bytes": corpus_bytes,
        "arms": [arm["name"] for arm in ARMS],
        "pair_sets": f"s3://{BUCKET}/{PAIR_SETS_PREFIX}",
        "run": str(run),
    }, indent=2))


if __name__ == "__main__":
    main()
