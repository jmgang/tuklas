#!/usr/bin/env python3
"""Launch speed/memory/cost timing for every non-lexical comparator.

GPU job on ml.g4dn.xlarge, the same host type as pycon-runtime-20261013-112910-a01fd7,
so its numbers line up with Tuklas / teacher / Cohere.  Tuklas pruned runs again
as an anchor to show the two jobs' CPU timings agree.

RUNTIME_DRY_RUN=1 prints the plan and request and stages nothing.  No other
environment variable is read.
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
INSTANCE = "ml.g4dn.xlarge"
TRAINING_USD_PER_HOUR = 1.03            # AWS Price List API, APS1-Train, 2026-10-13
HOSTING = {"cpu": ("ml.m5.xlarge", 0.288), "gpu": ("ml.g4dn.xlarge", 1.0304)}
MAX_RUNTIME_SECONDS = 14_400
GPU_PREFIXES = ("ml.g4dn.", "ml.g5.", "ml.g6.", "ml.p")
SHOOTOUT_INPUT = "pycon-davao-model2vec/pycon-shootout-filipino-20261008-053701-be01db/input/"
CLASSICAL_INPUT = "pycon-davao-model2vec/pycon-classical-classic2-20261008-052644-a7ec32/input/"
LARGE_INPUTS = {
    "shared-layout.json": (SHOOTOUT_INPUT, "740b0b97e406e13be18af14c440116f34e17cd3b1c85bcf540d0ed2424b81141"),
    "annotations.all.jsonl": (SHOOTOUT_INPUT, "5e4cb1d6550aa74ad5f3d9a1269b079a5077094311b1a5c752dd45f8037aef7d"),
    "training.tl.jsonl.gz": (CLASSICAL_INPUT, "0857cbdc061877369334660bfdd7bb1eacb011710debb489387ae8e71481b478"),
}
SMALL_SOURCES = {
    "aws_runtime_comparison_v2.py": ROOT / "scripts/aws_runtime_comparison_v2.py",
    "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
    "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
    "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
}

# Development Recall@10 recorded by the job that scored each system (S3
# results/shootout.json or classical.json of the jobs named in each entry).
STATIC_HF = [
    # (system, repo, revision, expected dev, source job)
    ("potion-multilingual-128M", "minishlab/potion-multilingual-128M", "73908c3438cf03b6a01bcb9611d62b23d0726f08",
     0.7258883248730964, "pycon-shootout-multi-20261008-042806-56f6da"),
    ("amgix-static-retrieval-69m", "amgix/static-retrieval-multilingual-69m-v1", "f777a93cd958b280b6c7fad5aaac3cd59d676822",
     0.6903553299492385, "pycon-shootout-static-20261008-045008-17f996"),
    ("nexus666-e5-large-m2v", "nexus666/multilingual-e5-large-m2v", "3bc4fda5d8c351ef8568661d3820ded8f83a013c",
     0.6294416243654822, "pycon-shootout-static-20261008-045008-17f996"),
    ("m2v-multilingual-output", "minishlab/M2V_multilingual_output", "62144420694cbf4ebe8f3b0fe3c1c49d8a0724b5",
     0.583756345177665, "pycon-shootout-static-20261008-045008-17f996"),
    ("tss-m2v-bge-m3-1024d", "tss-deposium/m2v-bge-m3-1024d", "bc15712c51130dd4737db102a349adafc72cfe26",
     0.4467005076142132, "pycon-shootout-static-20261008-045008-17f996"),
]
STATIC_ST = ("static-similarity-mrl", {"id": "sentence-transformers/static-similarity-mrl-multilingual-v1",
             "revision": "b68f4122911bcffcd6e1f695f2d99cd6788972d8", "batch_size": 256},
             0.5888324873096447, "pycon-shootout-stmrl2-20261008-051019-1ceb59")
GENSIM = [
    ("word2vec-cbow-512", {"kind": "word2vec", "sg": 0}, 0.3299492385786802),
    ("word2vec-skipgram-512", {"kind": "word2vec", "sg": 1}, 0.5583756345177665),
    ("fasttext-512", {"kind": "fasttext", "sg": 1}, 0.649746192893401),
]
TRANSFORMERS = [
    # (system stem, model spec, expected dev, source job, gpu timeout, cpu timeout)
    ("e5-large", {"id": "intfloat/multilingual-e5-large", "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
                  "batch_size": 32, "query_prefix": "query: ", "document_prefix": "passage: "},
     0.9949238578680203, "pycon-shootout-filipino-20261008-053701-be01db", 1500, 2100),
    ("bge-m3", {"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181",
                "batch_size": 32, "max_seq_length": 8192},
     0.9949238578680203, "pycon-shootout-multi-20261008-042806-56f6da", 1500, 2100),
    ("e5-base", {"id": "intfloat/multilingual-e5-base", "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
                 "batch_size": 64, "query_prefix": "query: ", "document_prefix": "passage: "},
     0.9746192893401016, "pycon-shootout-multi-20261008-042806-56f6da", 900, 1500),
    ("e5-small", {"id": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                  "batch_size": 128, "query_prefix": "query: ", "document_prefix": "passage: "},
     0.9644670050761421, "pycon-shootout-multi-20261008-042806-56f6da", 600, 900),
    ("granite-97m", {"id": "ibm-granite/granite-embedding-97m-multilingual-r2",
                     "revision": "835ad14087e140460703cf0fae09f97d469d65c2", "batch_size": 128},
     0.9086294416243654, "pycon-shootout-filipino-20261008-053701-be01db", 600, 900),
    ("labse", {"id": "sentence-transformers/LaBSE", "revision": "836121a0533e5664b21c7aacc5d22951f2b8b25b",
               "batch_size": 64, "max_seq_length": 256},
     0.883248730964467, "pycon-shootout-multi-20261008-042806-56f6da", 900, 1500),
    ("meedan-filipino-mpnet", {"id": "meedan/paraphrase-filipino-mpnet-base-v2",
                               "revision": "20d8b8d26840bc2f9d47e06ad9f576fa9ef86af3", "batch_size": 64},
     0.7715736040609137, "pycon-shootout-filipino-20261008-053701-be01db", 900, 1500),
]


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def build_systems() -> tuple[dict, list[str]]:
    systems, order = {}, []
    systems["tuklas-pruned"] = {
        "runtime": "static-cpu", "device": "cpu", "full_index": True, "stage_key": "tuklas-pruned",
        "s3_prefix": "pycon-davao-model2vec/pycon-score-prunebest-20261007-182144-87966f/results/pruned-model/",
        "weights_etag": "37a2a90c873cdacbdc76327082c5f909-5", "etag_part_bytes": 8_388_608,
        "query_prefix_from_config": True,
        "expected_development": 0.9441624365482234, "gate_mode": "exact",
        "source_job": "pycon-runtime-20261013-112910-a01fd7", "timeout_seconds": 600}
    order.append("tuklas-pruned")
    for name, repo, revision, expected, source in STATIC_HF:
        systems[name] = {"runtime": "static-cpu", "device": "cpu", "full_index": True, "stage_key": name,
                         "model": {"id": repo, "revision": revision},
                         "expected_development": expected, "gate_mode": "exact",
                         "source_job": source, "timeout_seconds": 600}
        order.append(name)
    name, spec, expected, source = STATIC_ST
    # Scored on CPU (stmrl2 plan device: cpu), so an exact match is expected.
    systems[name] = {"runtime": "st-static-cpu", "device": "cpu", "full_index": True, "stage_key": name,
                     "model": spec, "expected_development": expected, "gate_mode": "exact",
                     "source_job": source, "timeout_seconds": 600}
    order.append(name)
    for name, recipe, expected in GENSIM:
        systems[name] = {"runtime": "gensim-cpu", "device": "cpu", "full_index": True, "stage_key": name,
                         "recipe": {**recipe, "dimension": 512, "window": 5, "min_count": 3,
                                    "workers": 8, "epochs": 5, "seed": 42,
                                    "expected_source_rows": 75_658},
                         # Retrained here; multi-threaded gensim is not bit-reproducible.
                         "expected_development": expected, "gate_mode": "band", "gate_band": 0.05,
                         "source_job": "pycon-classical-classic2-20261008-052644-a7ec32",
                         "timeout_seconds": 900}
        order.append(name)
    for stem, spec, expected, source, gpu_timeout, cpu_timeout in TRANSFORMERS:
        systems[stem + "-gpu"] = {"runtime": "transformer-gpu", "device": "cuda", "full_index": True,
                                  "stage_key": stem, "model": spec, "expected_development": expected,
                                  "gate_mode": "one-question", "source_job": source,
                                  "timeout_seconds": gpu_timeout}
        systems[stem + "-cpu"] = {"runtime": "transformer-cpu", "device": "cpu", "full_index": False,
                                  "stage_key": stem, "model": spec, "borrow_from": stem + "-gpu",
                                  "expected_development": expected, "gate_mode": "one-question",
                                  "source_job": source, "timeout_seconds": cpu_timeout}
        order += [stem + "-gpu", stem + "-cpu"]
    return systems, order


def main() -> None:
    dry_run = os.environ.get("RUNTIME_DRY_RUN") == "1"
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    s3 = session.client("s3")
    sagemaker = session.client("sagemaker")

    running = []
    for page in sagemaker.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        for row in page["TrainingJobSummaries"]:
            detail = sagemaker.describe_training_job(TrainingJobName=row["TrainingJobName"])
            running.append((row["TrainingJobName"], detail["ResourceConfig"]["InstanceType"]))
    gpu = [job for job in running if job[1].startswith(GPU_PREFIXES)]
    if any(kind == INSTANCE for _, kind in gpu):
        raise RuntimeError(f"{INSTANCE} slot is busy: {gpu}")
    if len(gpu) >= 3:
        raise RuntimeError(f"three GPU jobs already running: {gpu}")

    large = {}
    for name, (source, expected) in LARGE_INPUTS.items():
        head = s3.head_object(Bucket=BUCKET, Key=source + name)
        recorded = head.get("Metadata", {}).get("sha256", "")
        if recorded != expected:
            raise ValueError(f"{name}: recorded sha256 {recorded} != {expected}")
        large[name] = {"key": source + name, "sha256": recorded, "bytes": head["ContentLength"]}

    systems, order = build_systems()
    pruned = systems["tuklas-pruned"]
    attributes = s3.get_object_attributes(Bucket=BUCKET, Key=pruned["s3_prefix"] + "model.safetensors",
                                          ObjectAttributes=["ETag"])
    if attributes["ETag"].strip('"') != pruned["weights_etag"]:
        raise ValueError("pruned package ETag changed")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = f"pycon-runtime2-{timestamp}-{uuid4().hex[:6]}"
    prefix = f"pycon-davao-model2vec/{job}"
    plan = {
        "bucket": BUCKET, "region": REGION, "instance_type": INSTANCE,
        "output_prefix": f"{prefix}/results/",
        "order": order, "systems": systems,
        "protocol": {"cpu_threads": 4, "window": 192, "stride": 128, "chunks": 40_152,
                     "timing_queries": "197 development questions", "heldout_read": False,
                     "local_repetitions": 3, "sample_step": 20,
                     "companion_job": "pycon-runtime-20261013-112910-a01fd7"},
        "usd_per_hour": {"cpu": HOSTING["cpu"][1], "gpu": HOSTING["gpu"][1]},
        "instances": {"cpu": HOSTING["cpu"][0], "gpu": HOSTING["gpu"][0]},
        "limitations": [
            "One host: CPU numbers come from the g4dn.xlarge's 4 vCPUs; the cost model prices them at ml.m5.xlarge",
            "Transformers run in their checkpoint dtype (see torch_dtype); fp16 on the T4 would be faster",
            "Transformer-CPU index throughput is measured on a 2,008-chunk sample; static and GPU use the full index",
            "Word2Vec/fastText were retrained here with the scored recipe; timing depends on vocabulary and width, not on the exact weights",
            "Word2Vec/fastText encode with the per-word Python loop that produced their scores",
            "fastText load time and memory include gensim rebuilding its vectors from 2M x 512 n-gram buckets on load",
            "Transformer-CPU memory includes the borrowed GPU-twin search matrix (up to 164 MB)",
            "tss-deposium/m2v-bge-m3-1024d ships per-token weights that model2vec applies in encode; the timed library path uses them, but the scored 0.4467 used an unweighted mean (see library_vs_harness_min_cosine)",
            "Per-query cost assumes one sequential stream kept fully busy; batching or concurrency changes it",
            "Exact search, no ANN index; RSS includes the Python runtime and imported libraries",
            "Catalog prices; credit coverage unverified",
        ],
        "large_inputs": large,
        "staged_sha256": {**{name: digest(path) for name, path in SMALL_SOURCES.items()},
                          **{name: spec["sha256"] for name, spec in large.items()}},
    }
    request = {
        "TrainingJobName": job, "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE, "TrainingInputMode": "File",
            "ContainerArguments": ["python", "/opt/ml/input/data/input/aws_runtime_comparison_v2.py"],
        },
        "InputDataConfig": [{"ChannelName": "input", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/"},
        "ResourceConfig": {"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 100},
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        "Tags": [{"Key": "pycon-davao-model2vec", "Value": "runtime-comparison-v2"}],
    }
    summary = {"job": job, "instance": INSTANCE, "usd_per_hour": TRAINING_USD_PER_HOUR,
               "maximum_compute_usd": round(TRAINING_USD_PER_HOUR * MAX_RUNTIME_SECONDS / 3600, 4),
               "systems": len(order), "child_timeout_total_seconds": sum(s["timeout_seconds"] for s in systems.values()),
               "credit_coverage": "unverified", "running_jobs": running,
               "results": f"s3://{BUCKET}/{prefix}/results/"}
    if dry_run:
        print(json.dumps({"dry_run": True, "plan": plan, "request": request, **summary}, indent=2))
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
        s3.copy_object(Bucket=BUCKET, Key=f"{prefix}/input/{name}",
                       CopySource={"Bucket": BUCKET, "Key": spec["key"]},
                       MetadataDirective="COPY", ServerSideEncryption="AES256")
    (run / "create-training-job-request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    response = sagemaker.create_training_job(**request)
    (run / "create-training-job-response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({**summary, "run": str(run)}, indent=2))


if __name__ == "__main__":
    main()
