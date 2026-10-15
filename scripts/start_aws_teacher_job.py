"""Stage pinned teacher inputs and start one capped SageMaker GPU job.

The job uses new names and S3 keys on every invocation. Existing local
embeddings, partial files, model artifacts, and reports are never modified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.model_registry import local_model_artifacts


def file_sha256(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = sha256()
    opener = gzip.open if decompress_gzip else Path.open
    with opener(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_new(s3: object, bucket: str, key: str, path: Path) -> None:
    with path.open("rb") as source:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=source,
            ContentLength=path.stat().st_size,
            ServerSideEncryption="AES256",
            IfNoneMatch="*",
        )
    print(f"staged={path.name} bytes={path.stat().st_size}", flush=True)


def ensure_private_bucket(s3: object, name: str) -> None:
    try:
        s3.head_bucket(Bucket=name)
    except ClientError as exc:
        if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
            raise
        s3.create_bucket(
            Bucket=name,
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-1"},
        )
        s3.put_public_access_block(
            Bucket=name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        s3.put_bucket_encryption(
            Bucket=name,
            ServerSideEncryptionConfiguration={"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
        )
        print("created private project S3 bucket", flush=True)


def main() -> None:
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    account_id = session.client("sts").get_caller_identity()["Account"]
    bucket = f"sagemaker-ap-southeast-1-{account_id}"
    now = datetime.now(timezone.utc)
    job_name = f"pycon-e5-teacher-{now:%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    prefix = f"pycon-davao-model2vec/{job_name}"

    config = json.loads((ROOT / "configs/models-v1.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "configs/disastir-qa-all-categories.json").read_text(encoding="utf-8"))
    teacher = local_model_artifacts(ROOT, require_all=True)["teacher"]
    cache = ROOT / "runs/embeddings" / f"teacher-{teacher.fingerprint[:12]}.npy"
    if cache.exists() or cache.with_suffix(".json").exists():
        raise FileExistsError("Teacher cache already published; refusing another build")
    teacher_dir = teacher.path
    corpus = ROOT / "data/raw/corpus.jsonl.gz"
    if file_sha256(corpus, decompress_gzip=True) != data["corpus_sha256"]:
        raise ValueError("Pinned corpus SHA256 mismatch")
    with gzip.open(corpus, "rt", encoding="utf-8") as source:
        count = sum(1 for _ in source)
    if count != 239704:
        raise ValueError(f"Unexpected corpus passage count: {count}")

    staging = ROOT / "runs/aws-input"
    staging.mkdir(parents=True, exist_ok=True)
    archive_path = staging / f"{job_name}-teacher.tar.gz"
    with archive_path.open("xb") as destination:
        with tarfile.open(fileobj=destination, mode="w:gz") as archive:
            archive.add(teacher_dir, arcname=teacher_dir.name)
    manifest = {
        "corpus_sha256": data["corpus_sha256"],
        "teacher_archive_sha256": file_sha256(archive_path),
        "teacher_directory": teacher_dir.name,
        "model_fingerprint": teacher.fingerprint,
        "expected_passage_count": count,
        "dimension": config["teacher"]["dimension"],
        "max_length": config["teacher"]["max_length"],
        "batch_size": config["teacher"]["batch_size"],
    }
    manifest_path = staging / f"{job_name}-input-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, sort_keys=True, indent=2)
        output.write("\n")

    s3 = session.client("s3")
    ensure_private_bucket(s3, bucket)
    for name, path in (
        ("corpus.jsonl.gz", corpus),
        ("teacher.tar.gz", archive_path),
        ("input-manifest.json", manifest_path),
        ("aws_encode_teacher.py", ROOT / "scripts/aws_encode_teacher.py"),
    ):
        upload_new(s3, bucket, f"{prefix}/input/{name}", path)

    role_name = "AmazonSageMaker-ExecutionRole-20220924T130717"
    role_arn = session.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]
    image = "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
    sagemaker = session.client("sagemaker")
    sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-lc"],
            "ContainerArguments": ["python /opt/ml/input/data/input/aws_encode_teacher.py"],
        },
        RoleArn=role_arn,
        InputDataConfig=[{
            "ChannelName": "input",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{bucket}/{prefix}/input/",
                "S3DataDistributionType": "FullyReplicated",
            }},
            "InputMode": "File",
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{prefix}/native-output/"},
        ResourceConfig={"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 10800},
        Environment={"PYCON_OUTPUT_BUCKET": bucket, "PYCON_OUTPUT_PREFIX": f"{prefix}/gpu-vectors"},
        Tags=[
            {"Key": "Project", "Value": "pycon-davao-model2vec-demo"},
            {"Key": "Purpose", "Value": "one-off-teacher-corpus-encoding"},
        ],
        EnableNetworkIsolation=False,
    )
    record = {
        "job_name": job_name,
        "region": "ap-southeast-1",
        "instance_type": "ml.g4dn.xlarge",
        "max_runtime_seconds": 10800,
        "s3_input": f"s3://{bucket}/{prefix}/input/",
        "s3_gpu_output": f"s3://{bucket}/{prefix}/gpu-vectors/",
        "image": image,
        "input_manifest": manifest,
        "status": "launched",
    }
    record_path = staging / f"{job_name}-launch.json"
    with record_path.open("x", encoding="utf-8") as output:
        json.dump(record, output, sort_keys=True, indent=2)
        output.write("\n")
    print(f"LAUNCHED {job_name}", flush=True)


if __name__ == "__main__":
    main()
