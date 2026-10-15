"""Load exclusively fresh inputs on the new host and verify private DNS.

This prepares the environment, not a benchmark result. It never launches any
old runner or imports its result cache. SSM command IDs are recorded exclusively.
"""
import json
from pathlib import Path
import sys
from uuid import uuid4

import boto3


def main():
    directory = Path(sys.argv[1]).resolve()
    resources = json.loads((directory / "resources.json").read_text())
    staged = json.loads((directory / "staged-inputs.json").read_text())
    ec2_session = boto3.Session(profile_name="admin-general", region_name=resources["region"])
    endpoint = ec2_session.client("ec2").describe_vpc_endpoints(VpcEndpointIds=[resources["bedrock_endpoint_id"]])["VpcEndpoints"][0]
    interfaces = ec2_session.client("ec2").describe_network_interfaces(NetworkInterfaceIds=endpoint["NetworkInterfaceIds"])["NetworkInterfaces"]
    expected_ips = [interface["PrivateIpAddress"] for interface in interfaces]
    payload = json.dumps({**staged, "private_endpoint_ips": expected_ips})
    # The new host has no previous experiment directories; all outputs are new.
    setup = f"""import hashlib, json, pathlib, socket, tarfile
import boto3
settings = json.loads({payload!r})
resolved = sorted({{row[4][0] for row in socket.getaddrinfo('bedrock-runtime.ap-southeast-1.amazonaws.com', 443, family=socket.AF_INET)}})
assert resolved and set(resolved) <= set(settings['private_endpoint_ips']), (resolved, settings['private_endpoint_ips'])
root = pathlib.Path('/opt/pycon-fresh/experiment')
root.mkdir(exist_ok=False)
archive = root / 'fresh-inputs.tar.gz'
s3 = boto3.client('s3', region_name='ap-southeast-1')
response = s3.get_object(Bucket=settings['bucket'], Key=settings['key'])
checksum = hashlib.sha256()
with archive.open('xb') as output:
    while True:
        chunk = response['Body'].read(1024 * 1024)
        if not chunk: break
        checksum.update(chunk)
        output.write(chunk)
assert checksum.hexdigest() == settings['sha256'], 'Input archive checksum mismatch'
with tarfile.open(archive, 'r:gz') as package:
    members = package.getmembers()
    for member in members:
        assert member.isfile() and not member.name.startswith('/') and '..' not in pathlib.PurePosixPath(member.name).parts
    package.extractall(root, members=members, filter='data')
manifest = json.loads((root / 'fresh-input-manifest.json').read_text())
for relative, expected in manifest['inputs'].items():
    hasher = hashlib.sha256()
    with (root / relative).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): hasher.update(chunk)
    assert hasher.hexdigest() == expected, relative
with (root / 'aws-input-verification.json').open('x') as output:
    json.dump({{'private_bedrock_dns_ips': resolved, 'input_archive_sha256': checksum.hexdigest(),
               'verified_file_count': len(manifest['inputs']), 'old_outputs_imported': False}}, output, indent=2)
print('AWS inputs and private Bedrock DNS verified', flush=True)
"""
    commands = [
        "set -eu",
        "while [ ! -f /opt/pycon-fresh/bootstrap-ready.txt ]; do sleep 5; done",
        "export PYTHONDONTWRITEBYTECODE=1",
        "/opt/pycon-fresh/venv/bin/python -u -c " + __import__("shlex").quote(setup),
        "/opt/pycon-fresh/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0'",
        "/opt/pycon-fresh/venv/bin/pip install 'numpy==2.5.3' 'sentence-transformers==5.7.0' 'transformers==5.3.0' 'huggingface-hub==1.33.0' 'model2vec==0.9.0' psutil",
        "/opt/pycon-fresh/venv/bin/python -c 'import torch, model2vec, sentence_transformers; print(\"Benchmark dependencies imported\"); print(torch.__version__)'",
    ]
    command = ec2_session.client("ssm").send_command(
        InstanceIds=[resources["instance_id"]], DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands, "executionTimeout": ["5400"]},
        TimeoutSeconds=120, OutputS3BucketName=staged["bucket"],
        OutputS3KeyPrefix=staged["key"].rsplit("/inputs/", 1)[0] + "/setup-logs/",
        Comment="Fresh AWS-only benchmark environment preparation; no old results reused")
    command_id = command["Command"]["CommandId"]
    with (directory / f"host-preparation-{uuid4().hex}.json").open("x") as out:
        json.dump({"command_id": command_id, "instance_id": resources["instance_id"],
                   "expected_private_endpoint_ips": expected_ips, "benchmark_started": False}, out, indent=2)
    print("PREPARATION_COMMAND", command_id, flush=True)


if __name__ == "__main__":
    main()
