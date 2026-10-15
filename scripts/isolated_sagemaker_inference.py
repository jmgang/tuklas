"""SageMaker PyTorch inference hooks for the dedicated E5 GPU endpoint."""
import json
import os
from pathlib import Path
import sys


def model_fn(model_dir):
    root = Path(model_dir)
    sys.path.insert(0, str(root / "src"))
    # Only frozen teacher inputs are extracted from the checksummed raw-input
    # archive. No old catalogs, vectors or reports exist in this archive.
    import boto3
    from hashlib import sha256
    import tarfile
    settings = json.loads((root / "teacher-source.json").read_text())
    target = Path("/opt/ml/isolated-teacher")
    target.mkdir(exist_ok=False)
    archive = target / "raw-inputs.tar.gz"
    reply = boto3.client("s3", region_name="ap-southeast-1").get_object(
        Bucket=settings["bucket"], Key=settings["key"])
    hasher = sha256()
    with archive.open("xb") as stream:
        while chunk := reply["Body"].read(1024 * 1024):
            hasher.update(chunk); stream.write(chunk)
    reply["Body"].close()
    if hasher.hexdigest() != settings["sha256"]:
        raise ValueError("Raw teacher input checksum differs")
    manifest = json.loads((root / "fresh-input-manifest.json").read_text())
    row = manifest["frozen_models"]["teacher"]
    expected = {row["path"] + "/" + relative for relative in row["files_sha256"]}
    with tarfile.open(archive) as package:
        members = [m for m in package.getmembers() if m.name in expected]
        if {m.name for m in members} != expected or any(not m.isfile() for m in members):
            raise ValueError("Unexpected teacher archive members")
        package.extractall(target, members=members, filter="data")
    with (target / "fresh-input-manifest.json").open("x") as stream:
        json.dump(manifest, stream)
    from pycon_model2vec_demo.isolated_serving import ModelServer
    return ModelServer(target, "teacher", os.environ["PYCON_RESOURCE_ID"], device="cuda")


def input_fn(body, content_type):
    if content_type != "application/json":
        raise ValueError("JSON required")
    return json.loads(body)


def predict_fn(request, model):
    return model.handle(request)


def output_fn(response, accept):
    if accept != "application/json":
        raise ValueError("JSON required")
    # Hugging Face's MMS handler expects a serialized body, not PyTorch's
    # (body, content_type) tuple; it sets the response content type itself.
    return json.dumps(response)
