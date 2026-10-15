"""User-authorized removal of shutdowns and restart of the six retained hosts.

Does not launch benchmark workers, replace disks, or modify saved experiment data.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time
from uuid import uuid4
import boto3
from botocore.config import Config


def main():
    root = Path(__file__).resolve().parents[1]
    old = root / "runs/pycon-continue-20261003-053452-4bc896"
    settings = json.loads((old / "continuation-settings.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    config = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    ec, scheduler, ssm, s3 = [session.client(x, config=config) for x in ("ec2", "scheduler", "ssm", "s3")]
    assert session.client("sts", config=config).get_caller_identity()["Account"] == settings["account"]
    ids = {r["instance_id"] for r in settings["restored_volumes"]}
    assert len(ids) == 6
    inventory = {i["InstanceId"]: i for group in ec.describe_instances(InstanceIds=sorted(ids))["Reservations"] for i in group["Instances"]}
    assert all(i["State"]["Name"] in ("stopped", "running") for i in inventory.values())
    name = "pycon-repair-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    folder = root / "runs" / name
    folder.mkdir(exist_ok=False)
    prefix = settings["original_prefix"] + "/" + name

    def save(label, record):
        body = json.dumps(record, indent=2, default=str).encode()
        with (folder / (label + ".json")).open("xb") as output:
            output.write(body)
        s3.put_object(Bucket=settings["bucket"], Key=prefix + "/recovery/" + label + ".json", Body=body,
                      ServerSideEncryption="AES256", IfNoneMatch="*")

    save("authorization", {"user_request": "fix it and remove that shutdown; don't stop till everything is fixed",
                           "previous_run": settings["name"], "fixed_time_cap_removed": True,
                           "new_instances_or_volumes": False, "stop_after_verified_completion": True})
    changes = []
    for page in scheduler.get_paginator("list_schedules").paginate():
        for entry in page.get("Schedules", []):
            if entry["State"] != "ENABLED" or not entry["Name"].startswith("pycon-"):
                continue
            before = scheduler.get_schedule(Name=entry["Name"], GroupName=entry["GroupName"])
            target = before["Target"]
            payload = json.loads(target.get("Input", "{}"))
            matching = (target["Arn"].endswith("ec2:stopInstances") and bool(ids & set(payload.get("InstanceIds", [])))) or (
                target["Arn"].endswith("sagemaker:deleteEndpoint") and payload.get("EndpointName") == settings["teacher_endpoint"])
            if not matching:
                continue
            assert target["RoleArn"] == settings["guard_role"]
            if "InstanceIds" in payload:
                assert set(payload["InstanceIds"]) <= ids, "Refuse unrelated target"
            args = {k: before[k] for k in ("Name", "GroupName", "ScheduleExpression", "ScheduleExpressionTimezone",
                "FlexibleTimeWindow", "Target", "Description", "KmsKeyArn", "StartDate", "EndDate", "ActionAfterCompletion") if k in before}
            args["State"] = "DISABLED"
            scheduler.update_schedule(**args)
            after = scheduler.get_schedule(Name=before["Name"], GroupName=before["GroupName"])
            assert after["State"] == "DISABLED" and after["Target"] == target
            changes.append({"name": before["Name"], "expression": before["ScheduleExpression"], "state": after["State"]})
    save("shutdown-schedules-disabled", changes)
    current = {**settings, "name": name, "prefix": prefix,
        "output_directory": "/opt/pycon-isolated/" + name,
        "previous_recovery_prefix": settings["prefix"],
        "resume_chunk_prefix": settings["prefix"],
        "recovery_deadline_utc": None, "new_stop_guards": [], "fixed_time_cap_removed": True,
        "shutdown_policy": "Stop paid compute after verified completion and output preservation",
        "restore_teacher_endpoint_after_builds": True}
    save("resources", current)
    stopped = [i for i in sorted(ids) if inventory[i]["State"]["Name"] == "stopped"]
    if stopped:
        save("start-request", ec.start_instances(InstanceIds=stopped))
    print("REPAIR_FOLDER", str(folder), flush=True)
    print("RESTARTED_EXISTING_HOSTS", json.dumps(stopped), flush=True)
    until = time.monotonic() + 360
    while time.monotonic() < until:
        online = {x["InstanceId"] for x in ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": sorted(ids)}])["InstanceInformationList"]
                  if x["PingStatus"] == "Online"}
        if online == ids:
            break
        time.sleep(5)
    else:
        raise TimeoutError("Existing hosts did not all become SSM Online")
    code = """import pathlib,subprocess,json,psutil
r=subprocess.run(['shutdown','-c'],capture_output=True,text=True)
scheduled=pathlib.Path('/run/systemd/shutdown/scheduled')
assert not scheduled.exists(), 'Host shutdown remains scheduled'
active=[{'pid':p.pid,'argv':p.info['cmdline']} for p in psutil.process_iter(['cmdline']) if any(a.startswith('/opt/pycon-isolated/') and a.endswith(('-worker.py','-controller.py','/run_isolated_aws.py','/isolated_model_http.py')) for a in (p.info['cmdline'] or []))]
print(json.dumps({'shutdown_cancelled':r.returncode,'scheduled_file_exists':scheduled.exists(),'active_workers':active}))
"""
    result = ssm.send_command(InstanceIds=sorted(ids), DocumentName="AWS-RunShellScript", Parameters={
        "commands": ["PYTHONDONTWRITEBYTECODE=1 " + settings["python"] + " -c " + shlex.quote(code)],
        "executionTimeout": ["120"]}, TimeoutSeconds=120,
        OutputS3BucketName=settings["bucket"], OutputS3KeyPrefix=prefix + "/recovery/host-inspection")
    save("inspection-request", {"command_id": result["Command"]["CommandId"], "instances": sorted(ids)})
    print("INSPECTION_COMMAND", result["Command"]["CommandId"], flush=True)


if __name__ == "__main__":
    main()
