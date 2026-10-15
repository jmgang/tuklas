#!/usr/bin/env python3
"""Stage and launch one immutable direct SEA-LION managed training job."""

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


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        stream.write("\n")


def main() -> None:
    instance_type = os.environ.get("DIRECT_SEALION_INSTANCE_TYPE", "ml.g4dn.2xlarge")
    # Each g4dn size carries its own quota of 1, so distinct sizes are distinct
    # concurrency slots.  All of these are a single T4: the larger sizes buy host
    # RAM and vCPU, not GPU throughput.
    if instance_type not in {
        "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g4dn.4xlarge", "ml.g4dn.8xlarge",
    }:
        raise ValueError("unapproved direct SEA-LION training instance type")
    maximum_runtime_seconds = int(
        os.environ.get("DIRECT_SEALION_MAX_RUNTIME_SECONDS", "10800")
    )
    if maximum_runtime_seconds not in {7_200, 10_800, 14_400, 21_600}:
        raise ValueError("direct SEA-LION runtime cap must be 2, 3, 4 or 6 hours")
    hourly_rates = {
        "ml.g4dn.xlarge": 0.736, "ml.g4dn.2xlarge": 1.315,
        "ml.g4dn.4xlarge": 2.106, "ml.g4dn.8xlarge": 2.72,
    }
    recipe_relative = os.environ.get(
        "DIRECT_SEALION_RECIPE", "configs/multiwikiqa-direct-parent-v1.json"
    )
    recipe_path = ROOT / recipe_relative
    recipe = json.loads(recipe_path.read_text())
    tagalog_only = recipe.get("language") == "tl"
    capacity_frontier = str(recipe.get("protocol", "")).startswith(
        "multiwikiqa-sealion-tagalog-capacity-frontier-v"
    )
    candidate_id = os.environ.get("DIRECT_SEALION_CANDIDATE", "") if capacity_frontier else ""
    candidate = None
    if capacity_frontier:
        from pycon_model2vec_demo.tagalog_capacity_frontier import validate_candidate_recipe

        validate_candidate_recipe(recipe)
        candidates = {row["id"]: row for row in recipe["candidates"]}
        if candidate_id not in candidates:
            raise ValueError("DIRECT_SEALION_CANDIDATE must select a frozen frontier candidate")
        candidate = candidates[candidate_id]
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("wrong AWS account")
    sagemaker = session.client("sagemaker")
    s3 = session.client("s3")
    active = []
    for status in ("InProgress", "Stopping"):
        active.extend(
            sagemaker.list_training_jobs(StatusEquals=status, MaxResults=100)["TrainingJobSummaries"]
        )
    # A dry run creates nothing and uploads nothing, so concurrency is
    # irrelevant to it -- and inspecting a planned launch is most useful
    # precisely while other jobs are still running.
    if active and not os.environ.get("DIRECT_SEALION_DRY_RUN"):
        allowed = {
            name.strip()
            for name in os.environ.get("DIRECT_SEALION_ALLOW_PARALLEL_WITH", "").split(",")
            if name.strip()
        }
        active_names = {row["TrainingJobName"] for row in active}
        if not allowed or not active_names <= allowed:
            raise RuntimeError(
                "unexpected active SageMaker training job(s): "
                f"{sorted(active_names - allowed)}; refusing duplicate paid compute"
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_kind = (
        f"pycon-sealion-frontier-{int(candidate['dimension'])}" if capacity_frontier
        else "pycon-direct-sealion-tl" if tagalog_only
        else "pycon-direct-sealion"
    )
    job = f"{job_kind}-{timestamp}-{uuid4().hex[:6]}"
    run = ROOT / "runs" / job
    run.mkdir(parents=True, exist_ok=False)
    prefix = f"pycon-davao-model2vec/{job}"

    vocabulary = None
    vocabulary_path = None
    if capacity_frontier and candidate["vocabulary_mode"] == "full-parent":
        pass
    else:
        full_vocabulary_path = ROOT / (
            candidate["vocabulary_source"] if capacity_frontier
            else recipe["model2vec"]["vocabulary_source"]
        )
        vocabulary = json.loads(full_vocabulary_path.read_text())
        expected_vocabulary = int(candidate.get("vocabulary_size", 160_000)) if capacity_frontier else 80_000
        if len(vocabulary) != expected_vocabulary or len(set(vocabulary)) != expected_vocabulary:
            raise ValueError(f"frozen workflow vocabulary must contain {expected_vocabulary} entries")
        vocabulary_path = run / ("vocabulary-selected.json" if capacity_frontier else "vocabulary-80000.json")
        write_new(vocabulary_path, vocabulary)

    verified = ROOT / "runs/multiwikiqa-sonnet45-20261005-041012-bb7ee5/verified-20261005T042822Z"
    synthetic_path = Path(os.environ.get(
        "DIRECT_SEALION_SYNTHETIC_ADMITTED",
        str(verified / "admitted.jsonl"),
    )).resolve()
    synthetic_report_path = Path(os.environ.get(
        "DIRECT_SEALION_SYNTHETIC_REPORT",
        str(verified / "verification-report.json"),
    )).resolve()
    synthetic_records = [
        json.loads(line)
        for line in synthetic_path.open(encoding="utf-8")
        if line.strip()
    ]
    # Mirror the worker's own filter: it compares len(tagalog subset) against
    # this number, so counting every row overstates it for a bilingual file.
    synthetic_rows = (
        sum(1 for row in synthetic_records if row.get("language") == "tl")
        if tagalog_only
        else len(synthetic_records)
    )
    if synthetic_rows != len(synthetic_records):
        print(
            f"NOTE synthetic file holds {len(synthetic_records)} rows, "
            f"{synthetic_rows} Tagalog; staging the Tagalog count",
            flush=True,
        )
    declared_admitted = json.loads(synthetic_report_path.read_text()).get("admitted_pairs")
    if declared_admitted is not None and int(declared_admitted) != synthetic_rows:
        raise ValueError(
            f"synthetic receipt declares {int(declared_admitted)} admitted pairs "
            f"but {synthetic_path.name} yields {synthetic_rows} for this cohort; "
            "set DIRECT_SEALION_SYNTHETIC_ADMITTED/_REPORT to a matching pair "
            "(the capacity wrapper points at "
            "runs/multiwikiqa-tagalog-ranking-expanded-15000-20261006T080600Z)"
        )
    recipe_minimum_mined = int(
        recipe["training"]["ranking_data_expansion"]["minimum_safely_mined_pairs"]
    )
    minimum_mined = int(os.environ.get(
        "DIRECT_SEALION_MINIMUM_MINED",
        str(recipe_minimum_mined) if capacity_frontier else "4000" if tagalog_only else "8000",
    ))
    allowed_minimums = (
        {recipe_minimum_mined}
        if recipe["protocol"].endswith(("-v4", "-v5-512"))
        else {4_000, recipe_minimum_mined}
    )
    if capacity_frontier and minimum_mined not in allowed_minimums:
        raise ValueError(f"capacity-frontier minimum mined count must be one of {sorted(allowed_minimums)}")
    worker_relative = os.environ.get(
        "DIRECT_SEALION_WORKER", "scripts/aws_multiwikiqa_direct_sealion.py"
    )
    # The default worker has no stage-1 block.  Launching a "-stage1-" protocol
    # against it completes, costs full price, and stamps the report with a
    # protocol the model never followed.
    if "-stage1-" in str(recipe.get("protocol", "")) and "stage1" not in worker_relative:
        raise ValueError(
            f"protocol {recipe['protocol']} requires the stage-1 worker, "
            f"got {worker_relative}"
        )
    if recipe["training"].get("unsupervised_pretraining") and "stage1" not in worker_relative:
        raise ValueError("unsupervised_pretraining requires the stage-1 worker")
    sources = {
        "aws_multiwikiqa_direct_sealion.py": ROOT / worker_relative,
        "aws_sealion_train_repair_v3.py": ROOT / "scripts/aws_sealion_train_repair_v3.py",
        "direct_parent_protocol.py": ROOT / "src/pycon_model2vec_demo/direct_parent_protocol.py",
        "tagalog_capacity_frontier.py": ROOT / "src/pycon_model2vec_demo/tagalog_capacity_frontier.py",
        "full_rank_pca.py": ROOT / "src/pycon_model2vec_demo/full_rank_pca.py",
        "multiwikiqa_compact_retrieval.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_compact_retrieval.py",
        "multiwikiqa_evaluation.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_evaluation.py",
        "multiwikiqa_ranking_v5.py": ROOT / "src/pycon_model2vec_demo/multiwikiqa_ranking_v5.py",
        "tagalog_ranking_v8.py": ROOT / "src/pycon_model2vec_demo/tagalog_ranking_v8.py",
        "sealion_runtime.py": ROOT / "src/pycon_model2vec_demo/sealion_runtime_v3.py",
        "sealion_training.py": ROOT / "src/pycon_model2vec_demo/sealion_training.py",
        "sealion_source_integrity.py": ROOT / "src/pycon_model2vec_demo/sealion_source_integrity.py",
        "settings.json": ROOT / "runs/pycon-multiwikiqa-mixed-20261004-143440-0eb179/settings.json",
        "mixed-plan.json": ROOT / "runs/pycon-multiwikiqa-mixed-20261004-143440-0eb179/mixed-plan.json",
        "direct-parent.json": recipe_path,
        "synthetic-admitted.jsonl": synthetic_path,
        "synthetic-verification-report.json": synthetic_report_path,
    }
    if vocabulary_path is not None:
        sources[vocabulary_path.name] = vocabulary_path
    wants_pairs = bool(recipe["training"].get("unsupervised_pretraining"))
    unsupervised = os.environ.get("DIRECT_SEALION_UNSUPERVISED_PAIRS", "")
    if unsupervised:
        sources["unsupervised-admitted.jsonl"] = Path(unsupervised)
    # A large input already staged by an earlier job can be referenced by key and
    # copied server-side.  Re-uploading an 869MB file that is already in the
    # bucket wastes bandwidth, and keeping a local copy only to re-upload it
    # wastes disk; neither is necessary for an immutable input.
    s3_sources: dict[str, dict[str, str]] = {}
    reuse_key = os.environ.get("DIRECT_SEALION_UNSUPERVISED_S3_KEY", "")
    if reuse_key:
        if unsupervised:
            raise ValueError("pass either a local pair file or an S3 key, not both")
        reuse_sha = os.environ["DIRECT_SEALION_UNSUPERVISED_SHA256"]
        declared = recipe["training"].get("unsupervised_pretraining") or {}
        recipe_sha = declared.get("admitted_sha256")
        if recipe_sha and recipe_sha != reuse_sha:
            raise ValueError(
                "DIRECT_SEALION_UNSUPERVISED_SHA256 does not match the recipe's "
                f"admitted_sha256 ({reuse_sha[:16]}... vs {recipe_sha[:16]}...); "
                "the S3 key names a different pair set than this protocol "
                "describes"
            )
        s3_sources["unsupervised-admitted.jsonl"] = {"key": reuse_key, "sha256": reuse_sha}
    if wants_pairs and "unsupervised-admitted.jsonl" not in sources and not s3_sources:
        # Otherwise the worker only discovers the absence after distillation,
        # TokenLearn, teacher encoding and mining have all been paid for.
        raise ValueError(
            "this recipe declares unsupervised_pretraining but no pair file was "
            "supplied; set DIRECT_SEALION_UNSUPERVISED_PAIRS or "
            "DIRECT_SEALION_UNSUPERVISED_S3_KEY"
        )
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing staged source {name}: {path}")
    records = {
        name: {"local_path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        for name, path in sources.items()
    }
    for name, spec in s3_sources.items():
        head = s3.head_object(Bucket=BUCKET, Key=spec["key"])
        if name == "unsupervised-admitted.jsonl":
            declared_bytes = (
                recipe["training"].get("unsupervised_pretraining") or {}
            ).get("admitted_bytes")
            if declared_bytes and int(head["ContentLength"]) != int(declared_bytes):
                raise ValueError(
                    f"S3 pair object is {int(head['ContentLength'])} bytes but the "
                    f"recipe declares {int(declared_bytes)}"
                )
        records[name] = {
            "s3_source_key": spec["key"],
            "bytes": int(head["ContentLength"]),
            "sha256": spec["sha256"],
        }
    manifest = {
        "protocol": recipe["protocol"],
        "job": job,
        "files": records,
        "teacher": {
            "model": "aisingapore/SEA-LION-E5-Embedding-600M",
            "revision": "6a97cedd22714f3d4eb7c6fd1a452bf310899683",
            "dimension": 1024,
        },
        "model2vec": {
            "entrypoint": "model2vec.distill.distill",
            "candidate": candidate_id if capacity_frontier else None,
            "vocabulary": len(vocabulary) if vocabulary is not None else "full-parent",
            "pca_dims": int(candidate["dimension"]) if capacity_frontier else "auto",
        },
        "tokenlearn": {
            "same_parent_targets": 182_724,
            "dimension": int(candidate["dimension"]) if capacity_frontier else 1024,
            "best_validation_checkpoint_required": capacity_frontier,
        },
        "language": recipe.get("language", "bilingual"),
        "retrieval": {
            "hard_negatives": 4,
            "synthetic_queries": synthetic_rows,
            "minimum_safely_mined": minimum_mined,
        },
        "export": {
            "dimension": int(candidate["dimension"]) if capacity_frontier else 1024,
            "dtype": "int8",
            "maximum_bytes": (
                int(candidate["hard_maximum_complete_artifact_bytes"])
                if capacity_frontier else 100_000_000
            ),
        },
        "heldout_policy": (
            "3984 Tagalog evaluation questions unavailable to training/checkpoint selection"
            if tagalog_only
            else "7922 evaluation questions unavailable to training/checkpoint selection"
        ),
    }
    manifest_path = run / "input-manifest.json"
    write_new(manifest_path, manifest)

    # Mechanical guard against launching an arm whose inputs silently differ
    # from the run it will be compared against.  A reviewer reading a prose
    # launch plan cannot see an omitted environment variable; this can.  The
    # failure it exists to stop: calling this launcher directly instead of
    # through the capacity wrapper fell back to a hardcoded synthetic directory
    # holding 5,382 Tagalog questions instead of 15,431, which the worker only
    # rejected 13 minutes in, after TokenLearn had been paid for.
    reference_job = os.environ.get("DIRECT_SEALION_REFERENCE_JOB", "")
    if reference_job:
        reference_path = ROOT / "runs" / reference_job / "input-manifest.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"no manifest for reference job: {reference_path}")
        reference = json.loads(reference_path.read_text())
        allowed = {
            name.strip()
            for name in os.environ.get(
                "DIRECT_SEALION_REFERENCE_ALLOW_DIFF", ""
            ).split(",")
            if name.strip()
        } | {"input-manifest.json", "direct-parent.json"}
        drifted: list[str] = []
        for name, spec in records.items():
            if name in allowed:
                continue
            other = reference.get("files", {}).get(name)
            if other is None:
                drifted.append(f"{name}: absent from reference")
            elif other.get("sha256") != spec.get("sha256"):
                drifted.append(
                    f"{name}: sha256 {str(spec.get('sha256'))[:12]} vs reference "
                    f"{str(other.get('sha256'))[:12]} "
                    f"({spec.get('bytes')} vs {other.get('bytes')} bytes)"
                )
        for key in ("retrieval",):
            if manifest.get(key) != reference.get(key):
                drifted.append(f"{key}: {manifest.get(key)} vs reference {reference.get(key)}")
        if drifted:
            raise ValueError(
                "inputs differ from reference job "
                + reference_job
                + " in ways not declared in DIRECT_SEALION_REFERENCE_ALLOW_DIFF:\n  "
                + "\n  ".join(drifted)
            )
        print(f"PREFLIGHT inputs match reference {reference_job} "
              f"(allowing {sorted(allowed)})", flush=True)

    if os.environ.get("DIRECT_SEALION_DRY_RUN"):
        # Everything above is local; nothing has been uploaded and no job
        # exists.  Print what *would* be staged so a launch can be inspected at
        # zero cost before any money is committed.
        print(json.dumps({
            "dry_run": True,
            "protocol": recipe["protocol"],
            "would_stage": {
                name: {"bytes": spec.get("bytes"), "sha256": str(spec.get("sha256"))[:16],
                       "s3_source_key": spec.get("s3_source_key"),
                       "local_path": spec.get("local_path")}
                for name, spec in sorted(records.items())
            },
            "retrieval": manifest.get("retrieval"),
            "instance": instance_type,
            "maximum_runtime_seconds": maximum_runtime_seconds,
        }, indent=2, sort_keys=True))
        return
    records["input-manifest.json"] = {
        "local_path": str(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "sha256": digest(manifest_path),
    }

    for name, spec in records.items():
        key = f"{prefix}/input/{name}"
        if "s3_source_key" in spec:
            s3.copy_object(
                Bucket=BUCKET,
                Key=key,
                CopySource={"Bucket": BUCKET, "Key": spec["s3_source_key"]},
                MetadataDirective="REPLACE",
                ServerSideEncryption="AES256",
                Metadata={"sha256": spec["sha256"]},
            )
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] != spec["bytes"] or head["Metadata"].get("sha256") != spec["sha256"]:
                raise ValueError("S3 server-side copy verification failed: " + name)
            print("COPIED", name, f'from {spec["s3_source_key"]}', flush=True)
            continue
        path = Path(spec["local_path"])
        with path.open("rb") as stream:
            s3.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=stream,
                ContentLength=path.stat().st_size,
                IfNoneMatch="*",
                ServerSideEncryption="AES256",
                Metadata={"sha256": spec["sha256"]},
            )
        head = s3.head_object(Bucket=BUCKET, Key=key)
        if head["ContentLength"] != spec["bytes"] or head["Metadata"].get("sha256") != spec["sha256"]:
            raise ValueError("S3 staging verification failed: " + name)
        print("STAGED", name, flush=True)

    request = {
        "TrainingJobName": job,
        "RoleArn": ROLE,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE,
            "TrainingInputMode": "File",
            "ContainerArguments": ["python", "/opt/ml/input/data/input/aws_multiwikiqa_direct_sealion.py"],
        },
        "InputDataConfig": [{
            "ChannelName": "input",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/{prefix}/input/",
                "S3DataDistributionType": "FullyReplicated",
            }},
            "InputMode": "File",
        }],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{prefix}/native-output/"},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": maximum_runtime_seconds},
        "EnableNetworkIsolation": False,
        "EnableInterContainerTrafficEncryption": False,
        "EnableManagedSpotTraining": False,
        "Environment": {
            "SEALION_OUTPUT_BUCKET": BUCKET,
            "SEALION_OUTPUT_PREFIX": f"{prefix}/results",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "SEALION_CAPACITY_CANDIDATE": candidate_id,
        },
        "Tags": [
            {"Key": "Project", "Value": "pycon-davao-model2vec-demo"},
            {"Key": "Protocol", "Value": recipe["protocol"]},
        ],
    }
    write_new(run / "create-training-job-request.json", request)
    write_new(run / "staging-record.json", {
        "job": job,
        "profile": PROFILE,
        "region": REGION,
        "bucket": BUCKET,
        "prefix": prefix,
        "files": records,
        "managed_training": True,
        "no_endpoint": True,
        "temporary_training_volume": True,
        "runtime_ceiling_seconds": maximum_runtime_seconds,
        "instance_type": instance_type,
        "hourly_rate_usd": hourly_rates[instance_type],
        "maximum_compute_usd": hourly_rates[instance_type] * maximum_runtime_seconds / 3600,
        "credits_verified": False,
    })
    response = sagemaker.create_training_job(**request)
    write_new(run / "create-training-job-response.json", response)
    state = sagemaker.describe_training_job(TrainingJobName=job)
    write_new(run / "initial-state.json", state)
    print(json.dumps({
        "job": job,
        "status": state["TrainingJobStatus"],
        "instance": instance_type,
        "maximum_runtime_seconds": maximum_runtime_seconds,
        "maximum_compute_usd": hourly_rates[instance_type] * maximum_runtime_seconds / 3600,
        "results": f"s3://{BUCKET}/{prefix}/results/",
        "run": str(run),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
