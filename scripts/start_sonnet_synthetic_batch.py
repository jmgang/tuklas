#!/usr/bin/env python3
"""Launch one immutable Sonnet 4.5 Bedrock batch-generation job.

This launcher deliberately has no resume or overwrite mode.  It verifies the
prepared input receipts, uploads them under a fresh S3 prefix, creates a
resource-scoped Bedrock service role, and records the exact AWS request and
response locally for the conference-demo audit trail.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import boto3
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-southeast-1"
ACCOUNT = "<aws-account-id>"
PROFILE = "admin-general"
BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
PREPARED = ROOT / "runs/multiwikiqa-synthetic-20261005-v4"


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def upload_new(s3, path: Path, key: str) -> dict[str, object]:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        raise FileExistsError(f"refusing to overwrite s3://{BUCKET}/{key}")
    checksum = digest(path)
    with path.open("rb") as stream:
        response = s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=stream,
            ContentType="application/jsonl" if path.suffix == ".jsonl" else "application/json",
            Metadata={"sha256": checksum},
        )
    receipt = s3.head_object(Bucket=BUCKET, Key=key)
    if receipt.get("Metadata", {}).get("sha256") != checksum:
        raise RuntimeError(f"S3 checksum metadata mismatch for {key}")
    return {
        "bucket": BUCKET,
        "key": key,
        "bytes": path.stat().st_size,
        "sha256": checksum,
        "etag": str(response.get("ETag", "")).strip('"'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    prepared = args.prepared.resolve()
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "prepared-not-uploaded-not-submitted":
        raise ValueError("prepared manifest is not in the frozen pre-launch state")
    if manifest["bedrock"]["generator_model"] != MODEL_ID:
        raise ValueError("prepared generator model changed")
    if int(manifest["source_passages"]) <= 0 or int(manifest["per_language"]) <= 0:
        raise ValueError("prepared passage cohort is empty")
    if not manifest.get("selected_languages"):
        raise ValueError("prepared language cohort is absent")
    if manifest["quality_rules"].get("second_model_review") is not False:
        raise ValueError("a second-model review stage is not authorized")
    expected_hashes = {
        filename: str(spec["sha256"])
        for filename, spec in manifest["artifacts"].items()
    }
    for filename, expected in expected_hashes.items():
        path = prepared / filename
        actual = digest(path)
        if actual != expected or manifest["artifacts"][filename]["sha256"] != expected:
            raise ValueError(f"frozen checksum mismatch: {filename}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = sha256(
        f"{timestamp}:{expected_hashes['claude-sonnet-4-5-generation-input.jsonl']}".encode()
    ).hexdigest()[:6]
    run_id = args.run_id or f"multiwikiqa-sonnet45-{timestamp}-{suffix}"
    if len(run_id) > 63:
        raise ValueError("run id is too long for Bedrock")
    local = ROOT / "runs" / run_id
    local.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{run_id}"
    input_key = f"{prefix}/input/claude-sonnet-4-5-generation-input.jsonl"
    output_prefix = f"{prefix}/output/"
    role_name = f"{run_id}-bedrock"[:64].rstrip("-")

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    sts, bedrock, iam, s3 = (session.client(name) for name in ("sts", "bedrock", "iam", "s3"))
    identity = sts.get_caller_identity()
    if identity["Account"] != ACCOUNT:
        raise RuntimeError(f"wrong AWS account: {identity['Account']}")
    profile = bedrock.get_inference_profile(inferenceProfileIdentifier=MODEL_ID)
    if profile["status"] != "ACTIVE":
        raise RuntimeError(f"inference profile is not active: {profile['status']}")

    uploads = []
    for path, key in (
        (prepared / "claude-sonnet-4-5-generation-input.jsonl", input_key),
        (prepared / "selected-passages.jsonl", f"{prefix}/source/selected-passages.jsonl"),
        (manifest_path, f"{prefix}/source/prepared-manifest.json"),
        (ROOT / "src/pycon_model2vec_demo/synthetic_retrieval.py", f"{prefix}/source/synthetic_retrieval.py"),
        (ROOT / "scripts/prepare_synthetic_retrieval_batch.py", f"{prefix}/source/prepare_synthetic_retrieval_batch.py"),
        (Path(__file__).resolve(), f"{prefix}/source/start_sonnet_synthetic_batch.py"),
    ):
        uploads.append(upload_new(s3, path, key))
    write_new(local / "upload-receipts.json", uploads)

    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": ACCOUNT},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{REGION}:{ACCOUNT}:model-invocation-job/*"},
            },
        }],
    }
    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="Scoped Bedrock batch role for PyCon Davao synthetic retrieval data",
        Tags=[{"Key": "Project", "Value": "pycon-davao-model2vec"}, {"Key": "RunId", "Value": run_id}],
    )["Role"]
    allowed_models = [profile["inferenceProfileArn"], *[entry["modelArn"] for entry in profile["models"]]]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadOnlyThisInput",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{BUCKET}/{prefix}/input/*",
            },
            {
                "Sid": "WriteOnlyThisOutput",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                "Resource": f"arn:aws:s3:::{BUCKET}/{prefix}/output/*",
            },
            {
                "Sid": "ListOnlyThisRun",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": f"arn:aws:s3:::{BUCKET}",
                "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*", prefix]}},
            },
            {
                "Sid": "InvokeOnlyFrozenSonnet",
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": allowed_models,
            },
        ],
    }
    iam.put_role_policy(RoleName=role_name, PolicyName="BedrockBatchRunScope", PolicyDocument=json.dumps(policy))
    write_new(local / "role-trust.json", trust)
    write_new(local / "role-policy.json", policy)

    request = {
        "jobName": run_id,
        "roleArn": role["Arn"],
        "clientRequestToken": sha256(run_id.encode()).hexdigest(),
        "modelId": MODEL_ID,
        "modelInvocationType": "InvokeModel",
        "inputDataConfig": {"s3InputDataConfig": {
            "s3Uri": f"s3://{BUCKET}/{input_key}",
            "s3InputFormat": "JSONL",
            "s3BucketOwner": ACCOUNT,
        }},
        "outputDataConfig": {"s3OutputDataConfig": {
            "s3Uri": f"s3://{BUCKET}/{output_prefix}",
            "s3BucketOwner": ACCOUNT,
        }},
        "timeoutDurationInHours": 24,
        "tags": [{"key": "Project", "value": "pycon-davao-model2vec"}, {"key": "RunId", "value": run_id}],
    }
    write_new(local / "create-job-request.json", request)
    write_new(local / "resource-record.json", {
        "run_id": run_id,
        "region": REGION,
        "profile": PROFILE,
        "account": ACCOUNT,
        "caller_arn": identity["Arn"],
        "bucket": BUCKET,
        "prefix": prefix,
        "input_uri": request["inputDataConfig"]["s3InputDataConfig"]["s3Uri"],
        "output_uri": request["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"],
        "role_name": role_name,
        "role_arn": role["Arn"],
        "model_id": MODEL_ID,
        "prepared_manifest_sha256": digest(manifest_path),
        "prepared_path": str(prepared),
        "source_passages": int(manifest["source_passages"]),
        "selected_languages": list(manifest["selected_languages"]),
        "selected_passages_sha256": expected_hashes["selected-passages.jsonl"],
        "status": "role-and-input-created-job-not-yet-accepted",
    })

    # IAM role propagation is eventually consistent. The idempotency token makes
    # every retry the same launch rather than a duplicate paid job.
    response = None
    for attempt in range(12):
        try:
            response = bedrock.create_model_invocation_job(**request)
            break
        except ClientError as error:
            message = error.response.get("Error", {}).get("Message", "")
            if "role" not in message.casefold() or attempt == 11:
                raise
            time.sleep(5)
    assert response is not None
    write_new(local / "create-job-response.json", response)
    status = bedrock.get_model_invocation_job(jobIdentifier=response["jobArn"])
    write_new(local / "first-status.json", status)
    print(json.dumps({
        "run_id": run_id,
        "job_arn": response["jobArn"],
        "status": status["status"],
        "input_uri": request["inputDataConfig"]["s3InputDataConfig"]["s3Uri"],
        "output_uri": request["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"],
        "local_record": str(local),
    }, indent=2))


if __name__ == "__main__":
    main()
