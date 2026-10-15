"""User-requested pause: quiesce recovered workers, stop compute, snapshot disks."""
from datetime import datetime, timezone
import json, shlex, time
from pathlib import Path
from uuid import uuid4
import boto3
from botocore.config import Config


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/pycon-resume-20261003-034045-7df262"
    settings = json.loads((run / "resources.json").read_text())
    restore = [json.loads(p.read_text()) for p in sorted(run.glob("restore-*.json"))]
    assert len(restore) == 6
    instances = sorted({row["instance_id"] for row in restore})
    volumes = sorted({row["volume_id"] for row in restore})
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0})
    ec, ssm, sm, s3 = [session.client(x, config=cfg) for x in ("ec2", "ssm", "sagemaker", "s3")]
    assert session.client("sts", config=cfg).get_caller_identity()["Account"] == settings["account"]
    inv = {i["InstanceId"]: i for g in ec.describe_instances(InstanceIds=instances)["Reservations"] for i in g["Instances"]}
    assert set(inv) == set(instances)
    assert all(i["State"]["Name"] in ("running", "pending", "stopping", "stopped") for i in inv.values())
    marker = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    out = run / ("pause-snapshot-" + marker); out.mkdir(exist_ok=False)
    prefix = settings["prefix"] + "/pause-snapshots/" + marker

    def save(label, value):
        data = json.dumps(value, indent=2, default=str).encode()
        with (out / (label + ".json")).open("xb") as f: f.write(data)
        s3.put_object(Bucket=settings["bucket"], Key=prefix + "/" + label + ".json", Body=data,
                      IfNoneMatch="*", ServerSideEncryption="AES256")

    commands = {}
    body = '''import subprocess,psutil,json
units=[u.split()[0] for u in subprocess.check_output(['systemctl','list-units','--all','--type=service','--no-legend','--plain']).decode().splitlines() if u.split() and u.split()[0].startswith('pycon-')]
before={u:subprocess.run(['systemctl','show',u,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],capture_output=True,text=True).stdout for u in units}
units.sort(key=lambda u:0 if 'controller' in u else 1)
stops={u:subprocess.run(['systemctl','stop',u],capture_output=True,text=True,timeout=120).returncode for u in units}
subprocess.run(['sync'],check=True)
remaining=[p.pid for p in psutil.process_iter(['cmdline']) if any(a.endswith(('/run_isolated_aws.py','/resume_experiment_worker.py','/control_recovered_experiment.py','/isolated_model_http.py')) for a in (p.info['cmdline'] or []))]
assert not remaining,remaining
print(json.dumps({'before':before,'stops':stops,'remaining':remaining,'sync':True}),flush=True)
'''
    for instance in instances:
        if inv[instance]["State"]["Name"] not in ("running", "pending"): continue
        reply = ssm.send_command(InstanceIds=[instance], DocumentName="AWS-RunShellScript",
            Parameters={"commands": [settings["python"] + " -u -c " + shlex.quote(body)], "executionTimeout": ["240"]},
            TimeoutSeconds=120)
        commands[instance] = reply["Command"]["CommandId"]
    save("quiesce-commands", commands)
    pending = dict(commands); deadline = time.monotonic() + 360; outputs = {}
    while pending and time.monotonic() < deadline:
        for instance, command in list(pending.items()):
            try: reply = ssm.get_command_invocation(CommandId=command, InstanceId=instance)
            except ssm.exceptions.InvocationDoesNotExist: continue
            if reply["Status"] == "Success": outputs[instance] = reply; pending.pop(instance)
            elif reply["Status"] not in ("Pending", "InProgress", "Delayed"):
                save("quiesce-failure-" + instance, reply); raise RuntimeError("quiesce failed")
        if pending: time.sleep(3)
    if pending: raise RuntimeError("quiesce incomplete: " + repr(pending))
    for instance, reply in outputs.items(): save("quiesced-" + instance, reply)
    active = [i for i in instances if inv[i]["State"]["Name"] in ("running", "pending")]
    stop_reply = ec.stop_instances(InstanceIds=active) if active else {"Instances": []}
    save("ec2-stop-request", stop_reply)
    try:
        endpoint = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])
        if endpoint["EndpointStatus"] not in ("Deleting", "Deleted"): sm.delete_endpoint(EndpointName=settings["teacher_endpoint"])
        save("gpu-endpoint-delete-request", {"endpoint": settings["teacher_endpoint"], "status_before": endpoint["EndpointStatus"]})
    except sm.exceptions.ResourceNotFound: save("gpu-endpoint-delete-request", {"endpoint": settings["teacher_endpoint"], "already_absent": True})
    snapshots=[]
    for row in restore:
        snap=ec.create_snapshot(VolumeId=row["volume_id"],Description="Pause recovered PyCon benchmark; retain until user releases "+marker,
            TagSpecifications=[{"ResourceType":"snapshot","Tags":[{"Key":"Project","Value":"pycon-davao-model2vec"},{"Key":"RecoveryRun","Value":settings["name"]},{"Key":"PauseSnapshot","Value":marker},{"Key":"SourceInstance","Value":row["instance_id"]}]}])
        record={"instance_id":row["instance_id"],"volume_id":row["volume_id"],"snapshot_id":snap["SnapshotId"],"state":snap["State"]};snapshots.append(record);save("snapshot-"+row["instance_id"],record)
    save("pause-manifest",{"created_at_utc":datetime.now(timezone.utc).isoformat(),"instances":instances,"volumes":volumes,"snapshots":snapshots,"endpoint":settings["teacher_endpoint"],"retained_until_user_instruction":True,"compute_restart_authorized":False,"files_deleted":False,"s3_backup":"s3://"+settings["bucket"]+"/"+prefix+"/"})
    print("PAUSE_REQUESTED", json.dumps({"instances":instances,"snapshots":snapshots,"s3_prefix":"s3://"+settings["bucket"]+"/"+prefix+"/"}), flush=True)


if __name__ == "__main__": main()
