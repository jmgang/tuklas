"""Launch the repaired Cohere worker and continuation controller on AWS once."""
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys
import time
import boto3
from botocore.config import Config


def main(folder, evidence_file=None):
    folder = Path(folder)
    settings = json.loads((folder / "resources.json").read_text())
    root = Path(__file__).resolve().parents[1]
    evidence_path = Path(evidence_file) if evidence_file else next(folder.glob("parity-evidence-*.json"))
    evidence = json.loads(evidence_path.read_text())
    repeat = evidence["identical_fresh_requests"]
    assert not repeat["strict_99999_passes"] and repeat["minimum_cosine"] >= .999
    assert repeat["all_rows_match_own_index"]
    session = boto3.Session(profile_name="admin-general", region_name=settings["region"])
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    s3, ssm, iam = [session.client(x, config=cfg) for x in ("s3", "ssm", "iam")]
    assert session.client("sts", config=cfg).get_caller_identity()["Account"] == settings["account"]
    with (folder / "continuation-launch-once.lock").open("x"):
        pass
    sources = {}
    for label, path in (("worker", root / "scripts/resume_experiment_worker.py"),
                        ("controller", root / "scripts/control_recovered_experiment.py"),
                        ("parity-evidence", evidence_path)):
        body = path.read_bytes()
        sources[label] = {"key": settings["prefix"] + "/inputs/" + label + path.suffix,
                          "sha256": sha256(body).hexdigest(),
                          "target": "/opt/pycon-isolated/" + settings["name"] + "-" + label + path.suffix}
        s3.put_object(Bucket=settings["bucket"], Key=sources[label]["key"], Body=body,
                      Metadata={"sha256": sources[label]["sha256"]}, IfNoneMatch="*", ServerSideEncryption="AES256")
    # The controller restores exactly the retained E5 endpoint/config after the
    # bulk phase; existing stop guards already cover that endpoint and six hosts.
    endpoint_arn = f"arn:aws:sagemaker:{settings['region']}:{settings['account']}:endpoint/{settings['teacher_endpoint']}"
    endpoint_config_arn = f"arn:aws:sagemaker:{settings['region']}:{settings['account']}:endpoint-config/{settings['teacher_endpoint']}"
    iam.put_role_policy(RoleName=settings["role_arn"].rsplit("/", 1)[1], PolicyName=settings["name"] + "-endpoint-restore",
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
            "Action": "sagemaker:CreateEndpoint", "Resource": [endpoint_arn, endpoint_config_arn]},
            {"Effect": "Allow", "Action": "sagemaker:AddTags", "Resource": endpoint_arn}]}))
    current = {**settings, "restore_teacher_endpoint_after_builds": True,
        "provider_replay_policy": {"minimum_cosine": .999, "maximum_abs_difference": .01,
            "reason": "Repeated identical provider requests vary numerically"},
        "provider_replay_evidence_path": sources["parity-evidence"]["target"],
        "provider_replay_evidence_sha256": sources["parity-evidence"]["sha256"],
        "continuation_sources": sources}
    row = current["dedicated_models"]["cohere_v3"]
    cohere = {**current, "client_instance": row["instance_id"], "client_private_ip": row["private_ip"],
        "server_instance": row["instance_id"], "server_private_ip": row["private_ip"],
        "local_resource_id": row["instance_id"],
        "cohere_snapshot_id": next(r["snapshot_id"] for r in current["restored_volumes"] if r["instance_id"] == row["instance_id"])}
    launches = []
    def launch(host, config, label):
        sources_to_stage = {label: sources[label]}
        if label == "worker": sources_to_stage["parity-evidence"] = sources["parity-evidence"]
        config_target = "/opt/pycon-isolated/" + current["name"] + ("-cohere_v3-settings.json" if label == "worker" else "-controller-settings.json")
        script = sources[label]["target"]
        unit = current["name"] + ("-cohere_v3" if label == "worker" else "-controller")
        log = "/opt/pycon-isolated/" + unit + ".log"
        argv = [current["python"], "-u", script]
        argv += ["--settings", config_target, "--phase", "cohere", "--model", "cohere_v3"] if label == "worker" else [config_target]
        code = f'''import boto3,hashlib,pathlib,json,subprocess,psutil,socket
assert pathlib.Path('/opt/pycon-isolated/ready.json').exists()
active=[p.pid for p in psutil.process_iter(['cmdline']) if any(a.endswith(('/run_isolated_aws.py','/isolated_model_http.py')) or ('pycon-' in a and a.endswith(('-worker.py','-controller.py'))) for a in (p.info['cmdline'] or []))]
assert not active,active
for key,expected in {current['code_sha256']!r}.items():
 b=(pathlib.Path('/opt/pycon-isolated/code')/key).read_bytes();assert hashlib.sha256(b).hexdigest()==expected,key
routes={{}}
for domain,expected in [('bedrock-runtime.ap-southeast-1.amazonaws.com',{current['bedrock_private_ips']!r}),('runtime.sagemaker.ap-southeast-1.amazonaws.com',{current['sagemaker_runtime_private_ips']!r})]:
 ips={{r[4][0] for r in socket.getaddrinfo(domain,443,family=socket.AF_INET)}};assert ips and ips<=set(expected),(domain,ips);routes[domain]=sorted(ips)
client=boto3.client('s3',region_name={current['region']!r})
for row in {sources_to_stage!r}.values():
 b=client.get_object(Bucket={current['bucket']!r},Key=row['key'])['Body'].read();assert hashlib.sha256(b).hexdigest()==row['sha256']
 with pathlib.Path(row['target']).open('xb') as out:out.write(b)
with pathlib.Path({config_target!r}).open('x') as out:json.dump(json.loads({json.dumps(config)!r}),out,indent=2)
if {bool(settings.get('fixed_time_cap_removed'))!r}:
 subprocess.run(['shutdown','-c'],check=True)
 assert not pathlib.Path('/run/systemd/shutdown/scheduled').exists()
else:
 subprocess.run(['shutdown','-h','+180'],check=True)
subprocess.run(['systemd-run','--unit='+{unit!r},'--property=Restart=no','--property=StandardOutput=append:'+{log!r},'--property=StandardError=inherit','--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OPENBLAS_NUM_THREADS=8',*{argv!r}],check=True)
print(json.dumps({{'unit':{unit!r},'private_routes':routes,'settings':{config_target!r},'sha256':{sources[label]['sha256']!r}}}),flush=True)
'''
        response = ssm.send_command(InstanceIds=[host], DocumentName="AWS-RunShellScript", Parameters={
            "commands": ["PYTHONDONTWRITEBYTECODE=1 " + current["python"] + " -u -c " + shlex.quote(code)],
            "executionTimeout": ["180"]}, TimeoutSeconds=120,
            OutputS3BucketName=current["bucket"], OutputS3KeyPrefix=current["prefix"] + "/recovery/launch-logs")
        record = {"instance_id": host, "command_id": response["Command"]["CommandId"], "unit": unit,
                  "settings": config_target, "script": script, "log": log, "sha256": sources[label]["sha256"]}
        with (folder / (label + "-launch.json")).open("x") as out: json.dump(record, out, indent=2)
        launches.append(record)
        print("CONTINUATION_LAUNCH_REQUEST", json.dumps(record), flush=True)
    with (folder / "continuation-settings.json").open("x") as out: json.dump(current, out, indent=2)
    with (folder / "cohere-settings.json").open("x") as out: json.dump(cohere, out, indent=2)
    launch(row["instance_id"], cohere, "worker")
    launch(current["client_instance"], current, "controller")
    pending = list(launches); until = time.monotonic() + 240
    while pending and time.monotonic() < until:
        for row in pending.copy():
            try: result = ssm.get_command_invocation(CommandId=row["command_id"], InstanceId=row["instance_id"])
            except ssm.exceptions.InvocationDoesNotExist: continue
            if result["Status"] in ("Pending", "InProgress", "Delayed"): continue
            with (folder / (row["unit"] + "-launch-result.json")).open("x") as out: json.dump(result, out, indent=2, default=str)
            assert result["Status"] == "Success", result
            print("CONTINUATION_LAUNCHED", result["StandardOutputContent"], flush=True)
            pending.remove(row)
        if pending: time.sleep(3)
    assert not pending, pending


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
