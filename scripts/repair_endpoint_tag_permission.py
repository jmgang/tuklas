"""Add the missing endpoint-scoped permission and resume only the controller.

Preserves the failed controller directory and every completed AWS result. The
unchanged hash-pinned controller gets a new local output directory, but publishes
to the same experiment prefix with its existing identical-or-new checks.
"""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import boto3
from botocore.config import Config


def main(folder):
    folder = Path(folder)
    settings = json.loads((folder / "continuation-settings.json").read_text())
    launch = json.loads((folder / "controller-launch.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name=settings["region"])
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    iam, s3, ssm, sm = [session.client(n, config=cfg) for n in ("iam", "s3", "ssm", "sagemaker")]
    assert session.client("sts").get_caller_identity()["Account"] == settings["account"]
    for name in (*settings["dedicated_models"], "teacher"):
        s3.head_object(Bucket=settings["bucket"], Key=settings["prefix"] + f"/benchmark/{name}/quality.json")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=settings["bucket"], Prefix=settings["prefix"] + "/benchmark/"):
        assert not any(o["Key"].endswith("/measurement.json") for o in page.get("Contents", [])), "Timing already exists; do not restart"
    endpoint = settings["teacher_endpoint"]
    endpoint_arn = f"arn:aws:sagemaker:{settings['region']}:{settings['account']}:endpoint/{endpoint}"
    endpoint_config_arn = f"arn:aws:sagemaker:{settings['region']}:{settings['account']}:endpoint-config/{endpoint}"
    sm.describe_endpoint_config(EndpointConfigName=endpoint)
    try:
        sm.describe_endpoint(EndpointName=endpoint)
    except Exception as error:
        assert "Could not find endpoint" in str(error), error
    else:
        raise RuntimeError("Endpoint already exists; investigate before repair")
    stamp = str(time.time_ns())
    policy_name = settings["name"] + "-endpoint-tags-" + stamp
    record = folder / ("endpoint-permission-repair-" + stamp)
    record.mkdir(exist_ok=False)
    document = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "sagemaker:AddTags", "Resource": endpoint_arn},
        {"Effect": "Allow", "Action": "sagemaker:CreateEndpoint", "Resource": [endpoint_arn, endpoint_config_arn]}]}
    with (record / "policy.json").open("x") as out:
        json.dump(document, out, indent=2)
    assert policy_name not in iam.list_role_policies(RoleName=settings["role_arn"].rsplit("/", 1)[1])["PolicyNames"]
    iam.put_role_policy(RoleName=settings["role_arn"].rsplit("/", 1)[1], PolicyName=policy_name, PolicyDocument=json.dumps(document))
    result = iam.simulate_principal_policy(PolicySourceArn=settings["role_arn"], ActionNames=["sagemaker:CreateEndpoint"], ResourceArns=[endpoint_arn, endpoint_config_arn])
    result["EvaluationResults"].extend(iam.simulate_principal_policy(PolicySourceArn=settings["role_arn"], ActionNames=["sagemaker:AddTags"], ResourceArns=[endpoint_arn])["EvaluationResults"])
    with (record / "permission-verification.json").open("x") as out:
        json.dump(result, out, indent=2, default=str)
    assert all(r["EvalDecision"] == "allowed" for r in result["EvaluationResults"]), result
    new_settings = {**settings, "output_directory": settings["output_directory"] + "-latency-" + stamp}
    config_path = "/opt/pycon-isolated/" + settings["name"] + "-controller-latency-" + stamp + "-settings.json"
    unit = settings["name"] + "-controller-latency-" + stamp
    log = "/opt/pycon-isolated/" + unit + ".log"
    argv = [settings["python"], "-u", launch["script"], config_path]
    code = f'''import hashlib,json,pathlib,subprocess,psutil
assert hashlib.sha256(pathlib.Path({launch['script']!r}).read_bytes()).hexdigest()=={launch['sha256']!r}
state=subprocess.check_output(['systemctl','show',{launch['unit']!r},'-p','ActiveState','--value'],text=True).strip()
assert state in ('failed','inactive'),state
active=[p.pid for p in psutil.process_iter(['cmdline']) if any('pycon-' in a and a.endswith(('-controller.py','-worker.py')) for a in (p.info['cmdline'] or []))]
assert not active,active
assert not pathlib.Path({new_settings['output_directory']!r}).exists()
assert not pathlib.Path('/run/systemd/shutdown/scheduled').exists()
with pathlib.Path({config_path!r}).open('x') as out:json.dump(json.loads({json.dumps(new_settings)!r}),out,indent=2)
subprocess.run(['systemd-run','--unit='+{unit!r},'--property=Restart=no','--property=StandardOutput=append:'+{log!r},'--property=StandardError=inherit','--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OPENBLAS_NUM_THREADS=8',*{argv!r}],check=True)
print(json.dumps({{'unit':{unit!r},'settings':{config_path!r},'log':{log!r},'controller_sha256':{launch['sha256']!r},'output_directory':{new_settings['output_directory']!r}}}))
'''
    response = ssm.send_command(InstanceIds=[settings["client_instance"]], DocumentName="AWS-RunShellScript", Parameters={"commands": ["PYTHONDONTWRITEBYTECODE=1 " + settings["python"] + " -u -c " + shlex.quote(code)], "executionTimeout": ["120"]}, TimeoutSeconds=120, OutputS3BucketName=settings["bucket"], OutputS3KeyPrefix=settings["prefix"] + "/recovery/endpoint-permission-repair")
    command = response["Command"]["CommandId"]
    with (record / "launch.json").open("x") as out:
        json.dump({"command_id": command, "instance_id": settings["client_instance"], "unit": unit, "log": log, "settings": config_path, "output_directory": new_settings["output_directory"], "policy_name": policy_name, "controller_sha256": launch["sha256"]}, out, indent=2)
    print(json.dumps({"repair_record": str(record), "command_id": command, "unit": unit}), flush=True)
    for _ in range(30):
        try:
            result = ssm.get_command_invocation(CommandId=command, InstanceId=settings["client_instance"])
        except ssm.exceptions.InvocationDoesNotExist:
            time.sleep(2)
            continue
        if result["Status"] in ("Pending", "InProgress", "Delayed"):
            time.sleep(2)
            continue
        with (record / "launch-result.json").open("x") as out:
            json.dump(result, out, indent=2, default=str)
        assert result["Status"] == "Success", result
        print(result["StandardOutputContent"], flush=True)
        return
    raise TimeoutError(command)


if __name__ == "__main__":
    main(sys.argv[1])
