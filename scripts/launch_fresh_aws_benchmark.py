"""Launch exactly one fresh AWS runner after verified host preparation."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys
from uuid import uuid4
import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = Path(sys.argv[1]).resolve()
    resources = json.loads((directory / "resources.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name=resources["region"])
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0}, request_checksum_calculation="when_required")
    ssm = session.client("ssm", config=cfg)
    prep_records = list(directory.glob("host-preparation-*.json"))
    if len(prep_records) != 1:
        raise RuntimeError("Expected exactly one input preparation command")
    prep = json.loads(prep_records[0].read_text())
    status = ssm.get_command_invocation(CommandId=prep["command_id"], InstanceId=resources["instance_id"])
    if status["Status"] != "Success":
        raise RuntimeError("AWS input preparation is not successful")
    if list(directory.glob("benchmark-launch-*.json")):
        raise FileExistsError("Benchmark launch already issued; inspect it, do not duplicate")
    source = ROOT / "scripts/run_fresh_aws_benchmark.py"
    body = source.read_bytes()
    source_hash = sha256(body).hexdigest()
    bucket, prefix = resources["s3_prefix"].removeprefix("s3://").split("/", 1)
    key = prefix + "source/run_fresh_aws_benchmark-" + source_hash + ".py"
    session.client("s3", config=cfg).put_object(Bucket=bucket, Key=key, Body=body,
        Metadata={"sha256": source_hash}, IfNoneMatch="*", ServerSideEncryption="AES256")
    stop_at = datetime.fromisoformat(resources["automatic_stop"]["utc"])
    remaining = int((stop_at-datetime.now(timezone.utc)).total_seconds())
    if remaining < 300:
        raise RuntimeError("Three-hour compute stop is imminent; do not silently extend it")
    install = f"""import boto3, hashlib, pathlib
root = pathlib.Path('/opt/pycon-fresh/experiment')
assert (root / 'aws-input-verification.json').is_file(), 'Inputs unverified'
assert not (root / 'aws-benchmark').exists(), 'Existing benchmark must not be overwritten or resumed'
target = pathlib.Path('/opt/pycon-fresh/fresh-runner.py')
payload = boto3.client('s3', region_name='ap-southeast-1').get_object(Bucket={bucket!r}, Key={key!r})['Body'].read()
assert hashlib.sha256(payload).hexdigest() == {source_hash!r}
compile(payload, str(target), 'exec')
with target.open('xb') as output: output.write(payload)
print('Fresh runner source verified')
"""
    commands = ["set -eu", "/opt/pycon-fresh/venv/bin/python -c " + shlex.quote(install),
        "test ! -e /opt/pycon-fresh/benchmark-service.log",
        "systemd-run --unit=pycon-fresh-benchmark --service-type=exec "
        "--property=Restart=no --property=KillMode=control-group "
        f"--property=RuntimeMaxSec={remaining} "
        "--property=StandardOutput=append:/opt/pycon-fresh/benchmark-service.log "
        "--property=StandardError=append:/opt/pycon-fresh/benchmark-service.log "
        "--setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYCON_FRESH_ROOT=/opt/pycon-fresh/experiment "
        "--setenv=HF_HUB_OFFLINE=1 --setenv=TOKENIZERS_PARALLELISM=false "
        "--setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=8 --setenv=MKL_NUM_THREADS=8 "
        f"--setenv=PYCON_RESULT_BUCKET={bucket} --setenv=PYCON_RESULT_PREFIX={prefix}benchmark "
        "/opt/pycon-fresh/venv/bin/python -u /opt/pycon-fresh/fresh-runner.py "
        "--directory /opt/pycon-fresh/experiment/aws-benchmark",
        "systemctl is-active pycon-fresh-benchmark",
    ]
    result = ssm.send_command(InstanceIds=[resources["instance_id"]], DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands, "executionTimeout": ["300"]}, TimeoutSeconds=120,
        OutputS3BucketName=bucket, OutputS3KeyPrefix=prefix+"launch-logs/",
        Comment="Launch one fresh AWS-only DisastIR benchmark, no old outputs or resume")
    record = {"command_id": result["Command"]["CommandId"], "source_sha256": source_hash,
              "instance_id": resources["instance_id"], "systemd_unit": "pycon-fresh-benchmark",
              "remaining_compute_seconds": remaining, "benchmark_start_not_yet_verified": True}
    with (directory / f"benchmark-launch-{uuid4().hex}.json").open("x") as out:
        json.dump(record, out, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
