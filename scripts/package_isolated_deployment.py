"""Stage tiny replacement code packages. Does not launch paid compute."""
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from uuid import uuid4
import boto3

ROOT = Path(__file__).resolve().parents[1]


def add_bytes(package, name, value):
    info = tarfile.TarInfo(name); info.size = len(value)
    package.addfile(info, io.BytesIO(value))


def main():
    s = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    if s.client("sts").get_caller_identity()["Account"] != "<aws-account-id>":
        raise RuntimeError("Wrong account")
    source = json.loads((ROOT / "runs/pycon-fresh-20261002-162759-c570dd/staged-inputs.json").read_text())
    manifest = json.loads((ROOT / "runs/pycon-fresh-20261002-162759-c570dd/fresh-input-manifest.json").read_text())
    name = f"pycon-isolated-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    directory = ROOT / "runs" / name
    directory.mkdir(exist_ok=False)
    prefix = "pycon-davao-model2vec/" + name
    scripts = ["isolated_model_http.py", "isolated_sagemaker_inference.py", "run_fresh_aws_benchmark.py",
               "run_isolated_aws.py", "control_isolated_aws.py"]
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / p for p in scripts]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
    for kind in ("teacher", "host"):
        archive = directory / f"{kind}-code.tar.gz"
        with tarfile.open(archive, "x:gz") as package:
            for path in files:
                package.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
            add_bytes(package, "fresh-input-manifest.json", json.dumps(manifest).encode())
            add_bytes(package, "teacher-source.json", json.dumps(source).encode())
            if kind == "teacher":
                package.add(ROOT / "scripts/isolated_sagemaker_inference.py", arcname="code/inference.py", recursive=False)
                add_bytes(package, "code/requirements.txt", b"sentence-transformers==5.7.0\ntransformers==5.3.0\nhuggingface-hub==1.33.0\nnumpy==2.5.3\npsutil==7.2.2\nboto3\n")
        value = archive.read_bytes()
        s.client("s3").put_object(Bucket=source["bucket"], Key=f"{prefix}/inputs/{archive.name}",
            Body=value, IfNoneMatch="*", ServerSideEncryption="AES256",
            Metadata={"sha256": sha256(value).hexdigest()})
    record = {"name": name, "region": "ap-southeast-1", "account": "<aws-account-id>",
        "bucket": source["bucket"], "prefix": prefix, "raw_inputs": source,
        "frozen_models": manifest["frozen_models"], "code_sha256": hashes,
        "old_outputs_reused": False, "compute_launched": False, "runtime_cap_seconds": 10800,
        "gpu_serving_image": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference@sha256:52ca14fa145ef92f9bc63f02cdded090e3a591bec2aafb188638c7ec5dca9233",
        "gpu_framework": "PyTorch 2.6.0 CUDA 12.4 FP32; serving libraries recorded at readiness",
        "measurement_protocol": {"concurrency": 1, "batch_size": 1, "warmup_rounds": 2,
            "measured_rounds": 5, "development_queries": 32, "request_timeout_seconds": 60,
            "application_retries": 0, "connection_reuse": True, "serving_threads": 4,
            "failures": "record failed attempt; no latency percentile excludes failures without error count",
            "timed_phases_overlap_bulk_jobs": False}}
    with (directory / "deployment-inputs.json").open("x") as out:
        json.dump(record, out, indent=2)
    print(json.dumps({"staged": str(directory), "compute_launched": False}), flush=True)


if __name__ == "__main__":
    main()
