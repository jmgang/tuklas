#!/usr/bin/env python3
"""Launch the EmbeddingGemma-300m quality + runtime job (GPU, ml.g4dn.xlarge).

Transformer on the T4 GPU only; Tuklas pruned runs as the harness anchor.
RUNTIME_DRY_RUN=1 prints the plan and stages nothing.  HF_TOKEN, if set in
the caller's shell, is passed to the job as an environment variable (option B:
no Secrets Manager; the token is then visible in the job's settings).  Without
HF_TOKEN, the AmazonSageMaker-hf-token secret must exist.  The launcher never
prints the token.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import boto3

ROOT = Path(__file__).resolve().parents[1]
REGION, PROFILE, ACCOUNT = "ap-southeast-1", "admin-general", "<aws-account-id>"
BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
ROLE = "arn:aws:iam::<aws-account-id>:role/service-role/AmazonSageMaker-ExecutionRole-20220924T130717"
IMAGE = "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
INSTANCE = "ml.g4dn.xlarge"
TRAINING_USD_PER_HOUR = 1.03            # AWS Price List API, APS1-Train
HOSTING = {"cpu": ("ml.m5.xlarge", 0.288), "gpu": ("ml.g4dn.xlarge", 1.0304)}
MAX_RUNTIME_SECONDS = 3_600
HF_TOKEN_SECRET = "AmazonSageMaker-hf-token"   # readable by the SageMaker role
GPU_PREFIXES = ("ml.g4dn.", "ml.g5.", "ml.g6.", "ml.p")
SHOOTOUT_INPUT = "pycon-davao-model2vec/pycon-shootout-filipino-20261008-053701-be01db/input/"
LARGE_INPUTS = {
    "shared-layout.json": "740b0b97e406e13be18af14c440116f34e17cd3b1c85bcf540d0ed2424b81141",
    "annotations.all.jsonl": "5e4cb1d6550aa74ad5f3d9a1269b079a5077094311b1a5c752dd45f8037aef7d",
}
SMALL_SOURCES = {
    "aws_embeddinggemma_eval.py": ROOT / "scripts/aws_embeddinggemma_eval.py",
    "aws_runtime_comparison_v2.py": ROOT / "scripts/aws_runtime_comparison_v2.py",
    "quality_metrics_complete.py": ROOT / "scripts/quality_metrics_complete.py",
    "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
    "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
    "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
}
SYSTEMS = {
    "tuklas-pruned": {
        "runtime": "static-cpu", "device": "cpu", "full_index": True, "query_prefix_from_config": True,
        "s3_prefix": "pycon-davao-model2vec/pycon-score-prunebest-20261007-182144-87966f/results/pruned-model/",
        "weights_etag": "37a2a90c873cdacbdc76327082c5f909-5", "etag_part_bytes": 8_388_608,
        "timeout_seconds": 600},
    "embeddinggemma-gpu": {
        "runtime": "transformer-gpu", "device": "cuda", "full_index": True,
        # Prompts are read from the model's own config_sentence_transformers.json in the job.
        "model": {"id": "google/embeddinggemma-300m",
                  "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
                  "batch_size": 64, "max_seq_length": 512},
        "timeout_seconds": 1500},
}


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    dry_run = os.environ.get("RUNTIME_DRY_RUN") == "1"
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    s3, sagemaker = session.client("s3"), session.client("sagemaker")

    try:
        session.client("secretsmanager").describe_secret(SecretId=HF_TOKEN_SECRET)
        secret_ready = True
    except session.client("secretsmanager").exceptions.ResourceNotFoundException:
        secret_ready = False
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if not secret_ready and not env_token and not dry_run:
        raise RuntimeError(f"set HF_TOKEN or create the {HF_TOKEN_SECRET} secret first (gated model)")

    running = []
    for page in sagemaker.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        for row in page["TrainingJobSummaries"]:
            kind = sagemaker.describe_training_job(TrainingJobName=row["TrainingJobName"])["ResourceConfig"]["InstanceType"]
            running.append((row["TrainingJobName"], kind))
    gpu = [j for j in running if j[1].startswith(GPU_PREFIXES)]
    if any(k == INSTANCE for _, k in gpu) or len(gpu) >= 3:
        raise RuntimeError(f"GPU slot unavailable: {gpu}")

    large = {}
    for name, expected in LARGE_INPUTS.items():
        head = s3.head_object(Bucket=BUCKET, Key=SHOOTOUT_INPUT + name)
        if head.get("Metadata", {}).get("sha256") != expected:
            raise ValueError(name + ": sha256 metadata changed")
        large[name] = {"key": SHOOTOUT_INPUT + name, "sha256": expected}
    pruned = SYSTEMS["tuklas-pruned"]
    etag = s3.get_object_attributes(Bucket=BUCKET, Key=pruned["s3_prefix"] + "model.safetensors",
                                    ObjectAttributes=["ETag"])["ETag"].strip('"')
    if etag != pruned["weights_etag"]:
        raise ValueError("pruned package ETag changed")

    job = f"pycon-gemma-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    prefix = f"pycon-davao-model2vec/{job}"
    plan = {
        "bucket": BUCKET, "region": REGION, "instance_type": INSTANCE,
        "output_prefix": f"{prefix}/results/", "hf_token_secret": HF_TOKEN_SECRET,
        "order": ["tuklas-pruned", "embeddinggemma-gpu"], "systems": SYSTEMS,
        "protocol": {"cpu_threads": 4, "window": 192, "stride": 128, "chunks": 40_152,
                     "development": 197, "heldout": 3_984, "heldout_use": "verification only",
                     "timing": "same child and protocol as pycon-runtime2-20261013-134452-39cb6c",
                     "quality": "same evaluator as pycon-quality-repair-20261013-171154-974bc7"},
        "usd_per_hour": {"cpu": HOSTING["cpu"][1], "gpu": HOSTING["gpu"][1]},
        "instances": {"cpu": HOSTING["cpu"][0], "gpu": HOSTING["gpu"][0]},
        "limitations": [
            "Transformer timed on the T4 GPU only, in its checkpoint dtype (EmbeddingGemma does not support fp16)",
            "Default 768-dimension output; Matryoshka truncation to 512/256/128 not measured",
            "Per-query cost assumes one sequential stream kept fully busy",
            "Catalog prices; credit coverage unverified",
        ],
        "large_inputs": large,
        "staged_sha256": {**{n: digest(p) for n, p in SMALL_SOURCES.items()},
                          **{n: s["sha256"] for n, s in large.items()}},
    }
    request = {
        "TrainingJobName": job, "RoleArn": ROLE,
        "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File",
            "ContainerArguments": ["python", "/opt/ml/input/data/input/aws_embeddinggemma_eval.py"]},
        "InputDataConfig": [{"ChannelName": "input", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/"},
        "ResourceConfig": {"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 50},
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": "embeddinggemma-eval"}],
    }
    if env_token:
        request["Environment"] = {"HF_TOKEN": env_token}
    summary = {"job": job, "usd_per_hour": TRAINING_USD_PER_HOUR,
               "maximum_compute_usd": round(TRAINING_USD_PER_HOUR * MAX_RUNTIME_SECONDS / 3600, 4),
               "credit_coverage": "unverified", "secret_ready": secret_ready,
               "token_source": "HF_TOKEN environment variable" if env_token else "Secrets Manager",
               "running_jobs": running,
               "results": f"s3://{BUCKET}/{prefix}/results/"}
    if dry_run:
        shown = {**request, "Environment": {"HF_TOKEN": "<redacted>"}} if env_token else request
        print(json.dumps({"dry_run": True, "plan": plan, "request": shown, **summary}, indent=2))
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
        s3.copy_object(Bucket=BUCKET, Key=f"{prefix}/input/{name}", CopySource={"Bucket": BUCKET, "Key": spec["key"]},
                       MetadataDirective="COPY", ServerSideEncryption="AES256")
    redacted = {**request, "Environment": {"HF_TOKEN": "<redacted>"}} if env_token else request
    (run / "create-training-job-request.json").write_text(json.dumps(redacted, indent=2, sort_keys=True) + "\n")
    response = sagemaker.create_training_job(**request)
    (run / "create-training-job-response.json").write_text(json.dumps(response, indent=2, default=str) + "\n")
    print(json.dumps({**summary, "run": str(run)}, indent=2))


if __name__ == "__main__":
    main()
