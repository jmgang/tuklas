"""Import a completed SageMaker teacher cache without touching CPU partial data."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.model_registry import local_model_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_name")
    args = parser.parse_args()
    launch_path = ROOT / "runs/aws-input" / f"{args.job_name}-launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch["job_name"] != args.job_name or launch["region"] != "ap-southeast-1":
        raise ValueError("Launch record does not match job name and region")
    teacher = local_model_artifacts(ROOT, require_all=True)["teacher"]
    if teacher.fingerprint != launch["input_manifest"]["model_fingerprint"]:
        raise ValueError("Teacher artifact differs from the uploaded model")

    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    job = session.client("sagemaker").describe_training_job(TrainingJobName=args.job_name)
    if job["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"Training job is not complete: {job['TrainingJobStatus']}")
    uri = launch["s3_gpu_output"].removeprefix("s3://")
    bucket, prefix = uri.split("/", 1)
    result_key = prefix.rstrip("/") + "/result.json"
    result_bytes = session.client("s3").get_object(Bucket=bucket, Key=result_key)["Body"].read()
    result = json.loads(result_bytes)
    expected = launch["input_manifest"]
    if (
        result["status"] != "complete"
        or result["corpus_sha256"] != expected["corpus_sha256"]
        or result["model_fingerprint"] != teacher.fingerprint
        or result["teacher_archive_sha256"] != expected["teacher_archive_sha256"]
        or result["corpus_passage_count"] != expected["expected_passage_count"]
        or result["dimension"] != expected["dimension"]
        or result["dtype"] != "float32"
    ):
        raise ValueError("SageMaker result provenance differs from frozen inputs")

    cache = ROOT / "runs/embeddings" / f"teacher-{teacher.fingerprint[:12]}.npy"
    cache_manifest_path = cache.with_suffix(".json")
    imported_partial = cache.with_name(cache.name + f".aws-import-{args.job_name}.partial")
    result_path = ROOT / "runs/aws-input" / f"{args.job_name}-result.json"
    parity_path = ROOT / "runs/aws-input" / f"{args.job_name}-parity.json"
    for path in (cache, cache_manifest_path, imported_partial, result_path, parity_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    s3 = session.client("s3")
    vectors = np.lib.format.open_memmap(
        imported_partial,
        mode="w+",
        dtype=np.float32,
        shape=(result["corpus_passage_count"], result["dimension"]),
    )
    next_start = 0
    try:
        for index, chunk in enumerate(result["chunks"]):
            if chunk["index"] != index or chunk["start"] != next_start:
                raise ValueError("SageMaker chunk list is not contiguous")
            body = s3.get_object(Bucket=bucket, Key=chunk["s3_key"])["Body"].read()
            if sha256(body).hexdigest() != chunk["sha256"]:
                raise ValueError(f"Chunk {index} SHA256 mismatch")
            data = np.load(io.BytesIO(body), allow_pickle=False)
            if data.shape != (chunk["stop"] - chunk["start"], result["dimension"]) or data.dtype != np.float32:
                raise ValueError(f"Chunk {index} has unexpected shape or dtype")
            if not np.isfinite(data).all():
                raise ValueError(f"Chunk {index} has non-finite values")
            vectors[chunk["start"]:chunk["stop"]] = data
            next_start = chunk["stop"]
            if index % 10 == 0 or index + 1 == len(result["chunks"]):
                print(f"imported={next_start}/{result['corpus_passage_count']}", flush=True)
        if next_start != result["corpus_passage_count"]:
            raise ValueError("SageMaker chunks do not cover the full corpus")

        cpu_partial = cache.with_name(cache.name + ".partial")
        local = np.load(cpu_partial, mmap_mode="r")
        if local.shape != vectors.shape:
            raise ValueError("Original CPU partial shape differs from GPU output")
        samples = [index for index in (0, 1, 127, 1024, 4096, 16384, 43000) if index < len(local)]
        checks = []
        for index in samples:
            cpu = np.asarray(local[index], dtype=np.float32)
            gpu = np.asarray(vectors[index], dtype=np.float32)
            if not np.any(cpu):
                continue
            cosine = float(np.dot(cpu, gpu) / (np.linalg.norm(cpu) * np.linalg.norm(gpu)))
            checks.append({"row": index, "cosine": cosine, "max_abs_difference": float(np.max(np.abs(cpu - gpu)))})
        if len(checks) < 3 or min(row["cosine"] for row in checks) < 0.9999:
            raise ValueError(f"GPU/CPU teacher parity failed: {checks}")
        vectors.flush()
        del vectors
        vectors = None

        with result_path.open("xb") as output:
            output.write(result_bytes)
        with parity_path.open("x", encoding="utf-8") as output:
            json.dump({"status": "passed", "samples": checks}, output, sort_keys=True, indent=2)
            output.write("\n")
        os.link(imported_partial, cache)
        manifest = {
            "corpus_sha256": result["corpus_sha256"],
            "corpus_passage_count": result["corpus_passage_count"],
            "model_fingerprint": result["model_fingerprint"],
            "dimension": result["dimension"],
            "dtype": result["dtype"],
            "zero_vector_passages": result["zero_vector_passages"],
            "truncated_passages": result["truncated_passages"],
            "source_ids_first_last": result["source_ids_first_last"],
            "external_build_seconds": result["full_corpus_embedding_build_seconds"],
            "external_build_environment": result["environment"],
            "external_build_job": args.job_name,
            "gpu_cpu_parity": checks,
        }
        with cache_manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, sort_keys=True, indent=2)
            output.write("\n")
        print(f"published={cache}", flush=True)
    finally:
        if vectors is not None:
            vectors.flush()
            del vectors
        # Preserve imported partial and every input/output even on failure.


if __name__ == "__main__":
    main()
