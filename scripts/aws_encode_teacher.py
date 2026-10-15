"""One-off SageMaker GPU corpus encoding; writes immutable S3 chunks.

This job creates no model endpoint and never reads or modifies the local CPU
partial cache. Each chunk has a unique S3 key and is uploaded conditionally.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


PACKAGE_VERSIONS = {
    "sentence-transformers": "5.7.0",
    "transformers": "5.3.0",
    "huggingface-hub": "1.33.0",
}


def install_encoder_dependencies() -> None:
    missing = []
    for name, version in PACKAGE_VERSIONS.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing.append(f"{name}=={version}")
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing],
            check=True,
        )


def file_sha256(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = sha256()
    opener = gzip.open if decompress_gzip else Path.open
    with opener(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def put_new_json(client: object, bucket: str, key: str, value: dict) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, sort_keys=True, indent=2).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
        IfNoneMatch="*",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("SM_CHANNEL_INPUT", "/opt/ml/input/data/input")))
    parser.add_argument("--output-bucket", default=os.environ.get("PYCON_OUTPUT_BUCKET"))
    parser.add_argument("--output-prefix", default=os.environ.get("PYCON_OUTPUT_PREFIX"))
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if not args.output_bucket or not args.output_prefix or args.chunk_size <= 0:
        parser.error("output bucket, unique output prefix, and positive chunk size are required")

    import boto3
    import numpy as np

    input_dir = args.input_dir
    input_manifest = json.loads((input_dir / "input-manifest.json").read_text(encoding="utf-8"))
    corpus_path = input_dir / "corpus.jsonl.gz"
    archive_path = input_dir / "teacher.tar.gz"
    if file_sha256(corpus_path, decompress_gzip=True) != input_manifest["corpus_sha256"]:
        raise ValueError("Input corpus SHA256 mismatch")
    if file_sha256(archive_path) != input_manifest["teacher_archive_sha256"]:
        raise ValueError("Teacher archive SHA256 mismatch")

    install_encoder_dependencies()
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("SageMaker GPU job has no CUDA device")
    model_root = Path("/tmp/pycon-teacher-snapshot")
    model_root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(model_root, filter="data")
    model_dir = model_root / input_manifest["teacher_directory"]
    model = SentenceTransformer(str(model_dir), device="cuda", local_files_only=True, trust_remote_code=False)
    model.max_seq_length = input_manifest["max_length"]
    if model.get_embedding_dimension() != input_manifest["dimension"]:
        raise ValueError("Teacher dimension differs from frozen configuration")

    s3 = boto3.client("s3", region_name="ap-southeast-1")
    prefix = args.output_prefix.rstrip("/")
    count = 0
    zero_rows = 0
    truncated_rows = 0
    first_id = None
    last_id = None
    chunks = []
    batch_texts = []
    started = time.monotonic()

    def upload_chunk(texts: list[str], index: int, offset: int) -> dict:
        nonlocal zero_rows, truncated_rows
        prefixed = ["passage: " + text for text in texts]
        lengths = model.tokenizer(prefixed, add_special_tokens=True, truncation=False, return_length=True, verbose=False)["length"]
        truncated_rows += sum(length > input_manifest["max_length"] for length in lengths)
        vectors = np.asarray(
            model.encode(
                prefixed,
                batch_size=input_manifest["batch_size"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.shape != (len(texts), input_manifest["dimension"]) or not np.isfinite(vectors).all():
            raise ValueError("Invalid teacher vector chunk")
        zero_rows += int(np.count_nonzero(np.linalg.norm(vectors, axis=1) == 0))
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        body = buffer.getvalue()
        key = f"{prefix}/chunks/{index:05d}.npy"
        s3.put_object(
            Bucket=args.output_bucket,
            Key=key,
            Body=body,
            ContentType="application/octet-stream",
            ServerSideEncryption="AES256",
            IfNoneMatch="*",
        )
        result = {"index": index, "start": offset, "stop": offset + len(texts), "s3_key": key, "sha256": sha256(body).hexdigest()}
        print(f"encoded={result['stop']}/{input_manifest['expected_passage_count']} chunk={index}", flush=True)
        return result

    with gzip.open(corpus_path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            corpus_id = row["corpus_id"]
            if first_id is None:
                first_id = corpus_id
            last_id = corpus_id
            batch_texts.append(row["text"])
            count += 1
            if len(batch_texts) == args.chunk_size:
                chunks.append(upload_chunk(batch_texts, len(chunks), count - len(batch_texts)))
                batch_texts = []
    if batch_texts:
        chunks.append(upload_chunk(batch_texts, len(chunks), count - len(batch_texts)))
    if count != input_manifest["expected_passage_count"]:
        raise ValueError(f"Expected {input_manifest['expected_passage_count']} passages, read {count}")
    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": input_manifest["corpus_sha256"],
        "model_fingerprint": input_manifest["model_fingerprint"],
        "teacher_archive_sha256": input_manifest["teacher_archive_sha256"],
        "corpus_passage_count": count,
        "dimension": input_manifest["dimension"],
        "dtype": "float32",
        "source_ids_first_last": [first_id, last_id],
        "zero_vector_passages": zero_rows,
        "truncated_passages": truncated_rows,
        "chunks": chunks,
        "full_corpus_embedding_build_seconds": round(time.monotonic() - started, 3),
        "environment": {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
    }
    put_new_json(s3, args.output_bucket, f"{prefix}/result.json", result)
    Path("/opt/ml/model").mkdir(parents=True, exist_ok=True)
    Path("/opt/ml/model/result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"completed={count} seconds={result['full_corpus_embedding_build_seconds']}", flush=True)


if __name__ == "__main__":
    main()
