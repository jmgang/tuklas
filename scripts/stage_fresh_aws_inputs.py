"""Stage raw DisastIR inputs and verified frozen weights, never old outputs."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pycon_model2vec_demo.model_registry import local_model_artifacts
from pycon_model2vec_demo.distillation import file_hashes


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    resources_path = Path(sys.argv[1]).resolve()
    resources = json.loads(resources_path.read_text())
    directory = resources_path.parent
    artifacts = local_model_artifacts(ROOT, require_all=True)
    clean = {}
    files = []
    for name, artifact in artifacts.items():
        clean[name] = {"path": artifact.path.relative_to(ROOT).as_posix(),
            "settings": artifact.settings, "files_sha256": artifact.files_sha256,
            "fingerprint": artifact.fingerprint}
        files.extend(artifact.path / relative for relative in artifact.files_sha256)
    tokenizer = ROOT / "models/cohere-v3-tokenizer-a73b09960122"
    clean["cohere_v3_tokenizer"] = {"path": tokenizer.relative_to(ROOT).as_posix(),
        "files_sha256": file_hashes(tokenizer)}
    files.extend(tokenizer / relative for relative in clean["cohere_v3_tokenizer"]["files_sha256"])
    files.extend(sorted((ROOT / "data/raw").glob("*")))
    files.extend(sorted((ROOT / "src").rglob("*.py")))
    files.extend(ROOT / "configs" / filename for filename in [
        "disastir-qa-all-categories.json", "models-v1.json", "grade3-v1.json",
        "measurement-v1.json", "retrieval-v1.json"])
    manifest = {"frozen_models": clean, "inputs": {
        path.relative_to(ROOT).as_posix(): digest(path) for path in files if path.is_file()},
        "reuse_policy": "raw dataset and verified frozen weights only; no old embeddings, catalogs, API batches, retrieval results, measurements or reports",
        "fresh_environment": resources["name"]}
    manifest_path = directory / "fresh-input-manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    archive = directory / "fresh-inputs.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for path in sorted(set(files)):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        tar.add(manifest_path, arcname="fresh-input-manifest.json", recursive=False)
    archive_hash = digest(archive)
    print("ARCHIVE", archive, "BYTES", archive.stat().st_size, "SHA256", archive_hash, flush=True)
    bucket, prefix = resources["s3_prefix"].removeprefix("s3://").split("/", 1)
    key = prefix + "inputs/fresh-inputs.tar.gz"
    s3 = boto3.Session(profile_name="admin-general", region_name=resources["region"]).client("s3")
    with archive.open("rb") as stream:
        s3.put_object(Bucket=bucket, Key=key, Body=stream, ContentLength=archive.stat().st_size,
            ServerSideEncryption="AES256", IfNoneMatch="*", Metadata={"sha256": archive_hash})
    with (directory / "staged-inputs.json").open("x") as stream:
        json.dump({"bucket": bucket, "key": key, "sha256": archive_hash,
                   "bytes": archive.stat().st_size}, stream, indent=2)
    print("STAGED", f"s3://{bucket}/{key}", flush=True)


if __name__ == "__main__":
    main()
