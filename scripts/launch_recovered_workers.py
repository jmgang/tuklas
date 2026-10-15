"""Stage a hash-pinned recovery wrapper and launch disjoint AWS workers once."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

import boto3
from botocore.config import Config


def main(folder):
    folder = Path(folder)
    r = json.loads((folder / "resources.json").read_text())
    source = Path(__file__).with_name("resume_experiment_worker.py")
    body = source.read_bytes(); digest = hashlib.sha256(body).hexdigest()
    session = boto3.Session(profile_name="admin-general", region_name=r["region"])
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0})
    s3, ssm = [session.client(name, config=cfg) for name in ("s3", "ssm")]
    with (folder / "workers-launch-once.lock").open("x"):
        pass
    key = r["prefix"] + "/inputs/resume-experiment-worker.py"
    s3.put_object(Bucket=r["bucket"], Key=key, Body=body, Metadata={"sha256": digest},
                  IfNoneMatch="*", ServerSideEncryption="AES256")
    for name in ("cohere_v3", "potion_retrieval", "plain_student", "custom_student", "tokenlearn_custom_student"):
        row = r["dedicated_models"][name]
        settings = {**r, "server_instance": row["instance_id"], "server_private_ip": row["private_ip"],
                    "local_resource_id": row["instance_id"], "recovery_worker_sha256": digest}
        if name == "cohere_v3":
            settings["client_instance"] = row["instance_id"]
            settings["cohere_snapshot_id"] = next(v["snapshot_id"] for v in r["restored_volumes"] if v["instance_id"] == row["instance_id"])
        phase = "cohere" if name == "cohere_v3" else "quality"
        target_script = "/opt/pycon-isolated/" + r["name"] + "-worker.py"
        target_settings = "/opt/pycon-isolated/" + r["name"] + "-" + name + "-settings.json"
        code = f'''import boto3,hashlib,pathlib,json,subprocess,psutil
base=pathlib.Path('/opt/pycon-isolated')
assert (base/'bootstrap.json').exists() and (base/'ready.json').exists()
active=[p.pid for p in psutil.process_iter(['cmdline']) if any(a.endswith(('/run_isolated_aws.py','/control_dedicated_models.py','/isolated_model_http.py')) for a in (p.info['cmdline'] or []))]
assert not active,active
reply=boto3.client('s3',region_name='ap-southeast-1').get_object(Bucket={r['bucket']!r},Key={key!r});b=reply['Body'].read()
assert hashlib.sha256(b).hexdigest()=={digest!r}
with pathlib.Path({target_script!r}).open('xb') as out:out.write(b)
with pathlib.Path({target_settings!r}).open('x') as out:json.dump(json.loads({json.dumps(settings)!r}),out,indent=2)
subprocess.run(['systemd-run','--unit='+{(r['name']+'-'+name)!r},'--property=Restart=no','--property=StandardOutput=append:'+{('/opt/pycon-isolated/'+r['name']+'-'+name+'.log')!r},'--property=StandardError=inherit','--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OPENBLAS_NUM_THREADS=8',
 {r['python']!r},'-u',{target_script!r},'--settings',{target_settings!r},'--phase',{phase!r},'--model',{name!r}],check=True)
print('RECOVERY_WORKER_LAUNCHED',{name!r},{row['instance_id']!r})
'''
        response = ssm.send_command(InstanceIds=[row["instance_id"]], DocumentName="AWS-RunShellScript",
            Parameters={"commands": ["set -eu", "PYTHONDONTWRITEBYTECODE=1 " + r["python"] + " -u -c " + shlex.quote(code)],
                        "executionTimeout": ["180"]}, TimeoutSeconds=120,
            OutputS3BucketName=r["bucket"], OutputS3KeyPrefix=r["prefix"] + "/recovery/setup-logs")
        record = {"name": name, "instance_id": row["instance_id"], "command_id": response["Command"]["CommandId"],
                  "settings_path": target_settings, "worker_path": target_script, "unit": r["name"] + "-" + name,
                  "phase": phase, "sha256": digest}
        with (folder / ("worker-" + name + ".json")).open("x") as out:
            json.dump(record, out, indent=2)
        print("RECOVERY_LAUNCH", json.dumps(record), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
