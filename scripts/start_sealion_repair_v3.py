"""Launch one authorized dependency repair, using unchanged staged corpora."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import boto3

ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-southeast-1"
BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
OLD_JOB = "pycon-sealion-20261004-073934-2eccba"
OLD_PREFIX = "pycon-davao-model2vec/" + OLD_JOB


def checksum(path):
    value = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, indent=2, default=str)
        target.write("\n")


def main():
    session = boto3.Session(profile_name="admin-general", region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != "<aws-account-id>":
        raise ValueError("Wrong AWS account")
    sm, s3 = session.client("sagemaker"), session.client("s3")
    old = sm.describe_training_job(TrainingJobName=OLD_JOB)
    if old["TrainingJobStatus"] != "Failed":
        raise RuntimeError("Original job is not confirmed terminal Failed")
    for page in sm.get_paginator("list_training_jobs").paginate(
            NameContains="pycon-sealion", StatusEquals="InProgress"):
        if page["TrainingJobSummaries"]:
            raise RuntimeError("Active SEA-LION job exists; refusing duplicate")
    previous_job = "pycon-sealion-repair-20261004-082757-c1f372"
    if sm.describe_training_job(TrainingJobName=previous_job)["TrainingJobStatus"] != "Failed":
        raise RuntimeError("Previous repair is not terminal Failed; refusing duplicate")
    preflight_path = ROOT / "runs/sealion-local-preflight-20261004-v3/receipt.json"
    preflight = json.loads(preflight_path.read_text())
    if not preflight["all_training_text_preflighted"] or not preflight["original_text_unchanged"]:
        raise ValueError("Full corpus preflight has not passed")
    if preflight["arms"]["external_wikipedia"]["source_rows"] != 175658:
        raise ValueError("Unexpected corpus preflight count")
    original = ROOT / "runs" / OLD_JOB
    old_record = json.loads((original / "resource-record.json").read_text())
    pinned = {
        "settings.json": "07adb22c9fd66c57dfc75424b75ca00bf8316ee0c36e5bc6ea10aec4d8a84208",
        "aws_sealion_train.py": "435b2d6839183afaead8b0c5ab7021d296a6857153d89dad8d2eb3b677103623",
        "sealion_training.py": "0dbbb50b2c88ed093c38ec206fffd0d9ed7f279c75a153f14d5212b98812155f",
    }
    for filename, spec in old_record["input_files"].items():
        if checksum(spec["local_path"]) != spec["sha256"]:
            raise ValueError("Original local staged artifact changed: " + filename)
        if filename in pinned and spec["sha256"] != pinned[filename]:
            raise ValueError("Frozen source/config hash mismatch")
        head = s3.head_object(Bucket=BUCKET, Key=f"{OLD_PREFIX}/input/{filename}")
        if head["ContentLength"] != spec["bytes"] or head["Metadata"].get("sha256") != spec["sha256"]:
            raise ValueError("Original S3 input metadata changed: " + filename)
    config = json.loads(Path(old_record["input_files"]["settings.json"]["local_path"]).read_text())
    if (preflight["teacher"] != config["teacher"]
            or preflight["arms"]["approved_articles"]["safe_windows"] != 6546
            or preflight["arms"]["external_wikipedia"]["safe_windows"] != 176178):
        raise ValueError("Full-corpus preflight teacher/window counts differ")
    if (config["teacher"]["dimension"] != 1024 or config["initialization"]["pca_dims"] is not None
            or config["tokenlearn"]["target_pca_dims"] is not None):
        raise ValueError("Unapproved PCA/dimension change")
    recovery = {"source_job": OLD_JOB, "bucket": BUCKET, "files": {},
                "teacher": config["teacher"], "initialization_recipe": config["initialization"],
                "verification": "Every file streamed and SHA256 checked inside NEW GPU job before reuse"}
    expected_files = {"README.md", "config.json", "model.safetensors", "modules.json", "tokenizer.json"}
    found = s3.list_objects_v2(Bucket=BUCKET, Prefix=OLD_PREFIX + "/results/models/initialization/")
    for obj in found.get("Contents", []):
        filename = obj["Key"].rsplit("/", 1)[-1]
        if filename not in expected_files:
            raise ValueError("Unexpected initialization artifact: " + filename)
        head = s3.head_object(Bucket=BUCKET, Key=obj["Key"])
        digest = head["Metadata"].get("sha256")
        if not digest or len(digest) != 64:
            raise ValueError("Missing initialization checksum")
        recovery["files"][filename] = {"key": obj["Key"], "bytes": head["ContentLength"], "sha256": digest}
    if set(recovery["files"]) != expected_files:
        raise ValueError("Incomplete preserved initialization")
    if recovery["files"]["model.safetensors"]["sha256"] != "a58a9d3b0cd6d9682348520797fe9da23bd450e02d1bd28a214e8eb8a6f9bcef":
        raise ValueError("Unexpected initial weights")
    feature_prefix = "pycon-davao-model2vec/" + previous_job + "/results/"
    feature_spec = {"source_job": previous_job,
                    "layout_sha256": "9f224fd18244e91763bb5ba7ea4f7ae48c043c3badbb7a91328213f9271bdd32",
                    "files": {}}
    for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=feature_prefix + "features/approved_articles/").get("Contents", []):
        filename = obj["Key"].rsplit("/", 1)[-1]
        head = s3.head_object(Bucket=BUCKET, Key=obj["Key"])
        feature_spec["files"][filename] = {"key": obj["Key"], "bytes": head["ContentLength"],
                                           "sha256": head["Metadata"]["sha256"]}
    expected_features = {f"features-{i:08d}.{ext}" for i in range(0,6546,1024) for ext in ("json","npy")}
    if set(feature_spec["files"]) != expected_features:
        raise ValueError("Approved feature pairs incomplete")
    recovery["approved_features"] = feature_spec
    name = "pycon-sealion-repair-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    prefix = "pycon-davao-model2vec/" + name
    directory = ROOT / "runs" / name
    directory.mkdir(exist_ok=False)
    write_new(directory / "initialization-recovery.json", recovery)
    files = {
        "aws_sealion_train_repair_v3.py": ROOT / "scripts/aws_sealion_train_repair_v3.py",
        "sealion_runtime.py": ROOT / "src/pycon_model2vec_demo/sealion_runtime_v3.py",
        "initialization-recovery.json": directory / "initialization-recovery.json",
        "sealion_source_integrity.py": ROOT / "src/pycon_model2vec_demo/sealion_source_integrity.py",
        "full-corpus-preflight.json": preflight_path,
    }
    records = {filename: {"local_path": str(path), "bytes": path.stat().st_size,
                          "sha256": checksum(path)} for filename, path in files.items()}
    write_new(directory / "staging-plan.json", {"job_name": name, "settings": config,
              "new_files": records, "reused_inputs": old_record["input_files"],
              "initialization_recovery": recovery, "authorization":
              "User explicitly requested fixing and proceeding without stopping until results, October 4 2026."})
    for filename, path in files.items():
        with path.open("rb") as source:
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/repair-input/{filename}", Body=source,
                          ContentLength=path.stat().st_size, IfNoneMatch="*",
                          ServerSideEncryption="AES256", Metadata={"sha256": records[filename]["sha256"]})
        head = s3.head_object(Bucket=BUCKET, Key=f"{prefix}/repair-input/{filename}")
        if head["Metadata"].get("sha256") != records[filename]["sha256"] or head["ContentLength"] != records[filename]["bytes"]:
            raise ValueError("Repair staging metadata mismatch")
        print("STAGED", filename, flush=True)
    request = json.loads((original / "create-training-job-request.json").read_text())
    request["TrainingJobName"] = name
    request["AlgorithmSpecification"]["ContainerArguments"] = [
        "python /opt/ml/input/data/repair/aws_sealion_train_repair_v3.py"]
    request["InputDataConfig"].append({"ChannelName": "repair", "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/{prefix}/repair-input/",
        "S3DataDistributionType": "FullyReplicated"}}, "InputMode": "File"})
    request["OutputDataConfig"]["S3OutputPath"] = f"s3://{BUCKET}/{prefix}/native-output/"
    request["Environment"]["SEALION_OUTPUT_PREFIX"] = f"{prefix}/results"
    request["Environment"]["SEALION_RECOVERY_MANIFEST"] = "/opt/ml/input/data/repair/initialization-recovery.json"
    write_new(directory / "create-training-job-request.json", request)
    response = sm.create_training_job(**request)
    write_new(directory / "launch-response.json", response)
    state = sm.describe_training_job(TrainingJobName=name)
    write_new(directory / "initial-job-state.json", state)
    record = {**{k: v for k, v in old_record.items() if k not in ("job_name", "run_directory", "s3_results", "status", "input_files")},
              "job_name": name, "run_directory": str(directory), "s3_results": f"s3://{BUCKET}/{prefix}/results/",
              "status": state["TrainingJobStatus"], "input_files": old_record["input_files"],
              "repair_files": records, "recovery": recovery, "failed_predecessors": [OLD_JOB, previous_job], "local_preflight_sha256": checksum(preflight_path)}
    write_new(directory / "resource-record.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
