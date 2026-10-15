"""Stop exact benchmark compute only after verification; retain all data/disks."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys
import time

import boto3
from botocore.config import Config

EXPECTED = {"i-0b284f6d0109a7040", "i-00e90619978d9e17b", "i-0f441ffa3b38f4047",
            "i-06f07be22c5245b25", "i-0970e7d36b364d59a", "i-0edb972cc663d77ee"}


def main(folder, verified):
    folder, verified = Path(folder), Path(verified)
    settings = json.loads((folder / "resources.json").read_text())
    proof = json.loads((verified / "verification.json").read_text())
    assert proof["held_out_queries"] == 1127 and proof["measured_samples_per_model"] == 160
    assert len(proof["verified_models"]) == 6 and proof["cutoff"] == 10
    ids = {r["instance_id"] for r in settings["restored_volumes"]}
    assert ids == EXPECTED and settings["region"] == "ap-southeast-1"
    session = boto3.Session(profile_name="admin-general", region_name=settings["region"])
    config = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    ec, s3, sm, ssm = [session.client(n, config=config) for n in ("ec2", "s3", "sagemaker", "ssm")]
    assert session.client("sts").get_caller_identity()["Account"] == settings["account"]
    for relative, expected in proof["local_content_sha256"].items():
        assert sha256((verified / relative).read_bytes()).hexdigest() == expected, relative
        header = s3.head_object(Bucket=settings["bucket"], Key=settings["prefix"] + "/benchmark/" + relative)
        assert header["Metadata"]["sha256"] == expected, relative
    marker = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    record = folder / ("verified-compute-stop-" + marker)
    record.mkdir(exist_ok=False)
    def save(label, value):
        body = json.dumps(value, indent=2, default=str).encode()
        with (record / (label + ".json")).open("xb") as out:
            out.write(body)
        s3.put_object(Bucket=settings["bucket"], Key=settings["prefix"] + "/completion-preservation/" + marker + "/" + label + ".json",
                      Body=body, Metadata={"sha256": sha256(body).hexdigest()}, IfNoneMatch="*", ServerSideEncryption="AES256")
    inventory = ec.describe_instances(InstanceIds=sorted(ids))
    hosts = [i for g in inventory["Reservations"] for i in g["Instances"]]
    assert {i["InstanceId"] for i in hosts} == ids
    assert all(any(t["Key"] == "Project" and t["Value"] == "pycon-davao-model2vec" for t in i["Tags"]) for i in hosts)
    save("instance-inventory-before", inventory)
    endpoint = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])
    endpoint_config = sm.describe_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])
    models = [sm.describe_model(ModelName=v["ModelName"]) for v in endpoint_config["ProductionVariants"]]
    assert endpoint["EndpointStatus"] == "InService"
    assert endpoint_config["ProductionVariants"][0]["InstanceType"] == "ml.g4dn.xlarge"
    save("endpoint-model-config-preserved", {"endpoint": endpoint, "config": endpoint_config, "models": models})
    save("verified-results", proof)
    code = f'''import socket,json,psutil,pathlib,subprocess
active=[]
for p in psutil.process_iter(['cmdline']):
 args=p.info['cmdline'] or []
 if any(a.startswith('/opt/pycon-isolated/') and a.endswith(('-worker.py','-controller.py','/run_isolated_aws.py','/isolated_model_http.py','/control_dedicated_models.py')) for a in args):active.append(p.pid)
assert not active,active
routes={{}}
for domain,expected in [('bedrock-runtime.ap-southeast-1.amazonaws.com',{settings['bedrock_private_ips']!r}),('runtime.sagemaker.ap-southeast-1.amazonaws.com',{settings['sagemaker_runtime_private_ips']!r})]:
 ips={{r[4][0] for r in socket.getaddrinfo(domain,443,family=socket.AF_INET)}}
 assert ips and ips<=set(expected),(domain,ips)
 routes[domain]=sorted(ips)
subprocess.run(['sync'],check=True)
print(json.dumps({{'workers_idle':True,'private_routes':routes,'sync_completed':True,'files_deleted':False}}))
'''
    requests = {}
    for host in hosts:
        assert host["State"]["Name"] == "running", host["State"]
        response = ssm.send_command(InstanceIds=[host["InstanceId"]], DocumentName="AWS-RunShellScript", Parameters={
            "commands": ["PYTHONDONTWRITEBYTECODE=1 " + settings["python"] + " -u -c " + shlex.quote(code)], "executionTimeout": ["120"]},
            TimeoutSeconds=120, OutputS3BucketName=settings["bucket"], OutputS3KeyPrefix=settings["prefix"] + "/completion-preservation/" + marker + "/ssm")
        requests[host["InstanceId"]] = response["Command"]["CommandId"]
    save("idle-sync-commands", requests)
    pending = dict(requests)
    deadline = time.monotonic() + 180
    while pending and time.monotonic() < deadline:
        for host, command in list(pending.items()):
            try:
                reply = ssm.get_command_invocation(CommandId=command, InstanceId=host)
            except ssm.exceptions.InvocationDoesNotExist:
                continue
            if reply["Status"] in ("Pending", "InProgress", "Delayed"):
                continue
            save("idle-sync-" + host, reply)
            assert reply["Status"] == "Success", reply
            pending.pop(host)
        if pending:
            time.sleep(2)
    assert not pending, pending
    save("stop-request", ec.stop_instances(InstanceIds=sorted(ids)))
    sm.delete_endpoint(EndpointName=settings["teacher_endpoint"])
    save("endpoint-serving-stop-request", {"name": settings["teacher_endpoint"], "endpoint_only_removed": True,
        "model_and_configuration_retained": True, "data_disks_snapshots_and_s3_retained": True})
    print(json.dumps({"record": str(record), "instances_stopping": sorted(ids), "endpoint_deleting": settings["teacher_endpoint"]}), flush=True)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        state = ec.describe_instances(InstanceIds=sorted(ids))
        states = {i["InstanceId"]: i["State"]["Name"] for g in state["Reservations"] for i in g["Instances"]}
        try:
            ep = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])["EndpointStatus"]
        except Exception as error:
            if "Could not find endpoint" not in str(error):
                raise
            ep = "absent"
        if set(states.values()) == {"stopped"} and ep == "absent":
            result = {"instances": states, "endpoint": ep, "retained_disks_snapshots_data": True,
                      "retained_storage_and_private_network_endpoints_still_charge": True}
            save("final-state", result)
            print(json.dumps(result, indent=2), flush=True)
            return
        time.sleep(5)
    raise TimeoutError("Stop requests submitted; final state not yet confirmed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
