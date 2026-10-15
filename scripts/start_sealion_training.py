"""Verify/stage existing corpora and launch one ephemeral two-student GPU job."""
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys
from uuid import uuid4

import boto3

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
REGION = "ap-southeast-1"
IMAGE = "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
ROLE = "arn:aws:iam::<aws-account-id>:role/service-role/AmazonSageMaker-ExecutionRole-20220924T130717"


def checksum(path):
    value = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, indent=2, default=str)
        target.write("\n")


def main():
    session = boto3.Session(profile_name="admin-general", region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != "<aws-account-id>":
        raise ValueError("Wrong AWS account")
    config_path = ROOT / "configs/sealion-two-students-v1.json"
    settings = json.loads(config_path.read_text())
    if settings["teacher"]["dimension"] != 1024 or settings["initialization"]["pca_dims"] is not None:
        raise ValueError("Unapproved dimensionality/PCA")
    sm = session.client("sagemaker")
    existing = sm.list_training_jobs(NameContains="pycon-sealion", StatusEquals="InProgress")["TrainingJobSummaries"]
    if existing:
        raise RuntimeError("Existing SEA-LION training job; refusing duplicate: " + str(existing))
    name = "pycon-sealion-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    prefix = "pycon-davao-model2vec/" + name
    directory = ROOT / "runs" / name
    directory.mkdir(exist_ok=False)
    files = {
        "settings.json": config_path,
        "aws_sealion_train.py": ROOT / "scripts/aws_sealion_train.py",
        "sealion_training.py": ROOT / "src/pycon_model2vec_demo/sealion_training.py",
    }
    for arm in settings["arms"].values():
        for filename, spec in arm["files"].items():
            path = ROOT / spec["path"]
            if checksum(path) != spec["sha256"]:
                raise ValueError("Input SHA256 mismatch: " + str(path))
            with path.open() as source:
                rows = sum(1 for _ in source)
            if rows != spec["rows"]:
                raise ValueError("Unexpected row count: " + str(path))
            compressed = directory / (filename + ".gz")
            with path.open("rb") as source, compressed.open("xb") as target:
                with gzip.GzipFile(filename=filename, fileobj=target, mode="wb", mtime=0, compresslevel=3) as archive:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        archive.write(block)
            check = sha256()
            with gzip.open(compressed, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    check.update(block)
            if check.hexdigest() != spec["sha256"]:
                raise ValueError("Lossless input packaging checksum failed")
            files[filename + ".gz"] = compressed
    records = {filename: {"local_path": str(path), "bytes": path.stat().st_size, "sha256": checksum(path)}
               for filename, path in files.items()}
    write_new(directory / "staging-plan.json", {"job_name": name, "prefix": prefix, "files": records,
                                               "settings": settings, "phase": "verified-awaiting-upload"})
    s3 = session.client("s3")
    for filename, path in files.items():
        with path.open("rb") as source:
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/input/{filename}", Body=source,
                          ContentLength=path.stat().st_size, IfNoneMatch="*",
                          ServerSideEncryption="AES256", Metadata={"sha256": records[filename]["sha256"]})
        head = s3.head_object(Bucket=BUCKET, Key=f"{prefix}/input/{filename}")
        if head["ContentLength"] != records[filename]["bytes"] or head["Metadata"].get("sha256") != records[filename]["sha256"]:
            raise ValueError("Staging integrity failure")
        print("STAGED", filename, path.stat().st_size, flush=True)
    request = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File",
                                   "ContainerEntrypoint": ["bash", "-lc"],
                                   "ContainerArguments": ["python /opt/ml/input/data/input/aws_sealion_train.py"]},
        "RoleArn": ROLE,
        "InputDataConfig": [{"ChannelName": "input", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
            "S3DataDistributionType": "FullyReplicated"}}, "InputMode": "File"}],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/"},
        "ResourceConfig": {"InstanceType": "ml.g4dn.2xlarge", "InstanceCount": 1, "VolumeSizeInGB": 60},
        "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        "Environment": {"SEALION_OUTPUT_BUCKET": BUCKET, "SEALION_OUTPUT_PREFIX": f"{prefix}/results",
                        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
                        "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1"},
        "Tags": [{"Key": "Project", "Value": "pycon-davao-model2vec-demo"},
                 {"Key": "Purpose", "Value": "sealion-two-tokenlearn-students-no-pca"}],
        "EnableNetworkIsolation": False,
    }
    write_new(directory / "create-training-job-request.json", request)
    response = sm.create_training_job(**request)
    write_new(directory / "launch-response.json", response)
    state = sm.describe_training_job(TrainingJobName=name)
    write_new(directory / "initial-job-state.json", state)
    record = {"job_name": name, "profile": "admin-general", "region": REGION,
              "s3_results": f"s3://{BUCKET}/{prefix}/results/", "run_directory": str(directory),
              "status": state["TrainingJobStatus"], "instance_type": "ml.g4dn.2xlarge",
              "training_rate_usd_per_hour": 1.315, "credits_verified": False,
              "max_training_runtime_hours": 24, "max_training_runtime_compute_usd": 31.56,
              "billing_note": "Provisioning/billable time and storage may add charges; this is not a spend limit.",
              "automatic_release": "Managed training compute/volume released on job end; no warm pool or endpoint.",
              "input_files": records}
    write_new(directory / "resource-record.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
