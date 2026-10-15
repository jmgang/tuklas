"""Restore the user's last six pause snapshots to their stopped AWS hosts.

Retains detached source volumes, snapshots and all existing experiment outputs.
Creates immutable local/S3 journals and exact three-hour compute stop guards.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import boto3
from botocore.config import Config


def main():
    root = Path(__file__).resolve().parents[1]
    previous = root / "runs/pycon-resume-20261003-034045-7df262"
    settings = json.loads((previous / "resources.json").read_text())
    pause = json.loads((previous / "pause-snapshot-20261003-035840-e54ac580/pause-manifest.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name=settings["region"])
    config = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    ec, s3, iam, scheduler = [session.client(x, config=config) for x in ("ec2", "s3", "iam", "scheduler")]
    assert session.client("sts", config=config).get_caller_identity()["Account"] == settings["account"]
    ids = pause["instances"]
    inventory = {i["InstanceId"]: i for group in ec.describe_instances(InstanceIds=ids)["Reservations"] for i in group["Instances"]}
    assert len(inventory) == 6 and all(i["State"]["Name"] == "stopped" for i in inventory.values())
    snaps = {s["SnapshotId"]: s for s in ec.describe_snapshots(SnapshotIds=[r["snapshot_id"] for r in pause["snapshots"]])["Snapshots"]}
    assert all(s["State"] == "completed" and s["OwnerId"] == settings["account"] for s in snaps.values())
    marker = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    name = "pycon-continue-" + marker
    folder = root / "runs" / name
    folder.mkdir(exist_ok=False)
    prefix = settings["original_prefix"] + "/" + name

    def save(label, value):
        body = json.dumps(value, indent=2, default=str).encode()
        with (folder / (label + ".json")).open("xb") as out:
            out.write(body)
        s3.put_object(Bucket=settings["bucket"], Key=prefix + "/recovery/" + label + ".json", Body=body,
                      IfNoneMatch="*", ServerSideEncryption="AES256")

    save("authorization", {"user_request": "restore the snapshot and then continue the experiments", "source_pause": pause,
                           "old_disks_retained": True, "three_hour_cap_seconds": 10800})
    print("RECOVERY_FOLDER", str(folder), flush=True)

    def restore(row):
        host = inventory[row["instance_id"]]
        attached = host["BlockDeviceMappings"]
        assert len(attached) == 1 and attached[0]["Ebs"]["VolumeId"] == row["volume_id"]
        device = attached[0]["DeviceName"]
        volume = ec.create_volume(AvailabilityZone=host["Placement"]["AvailabilityZone"],
            SnapshotId=row["snapshot_id"], VolumeType="gp3", Iops=3000, Throughput=125, Encrypted=True,
            ClientToken=name + "-" + host["InstanceId"],
            TagSpecifications=[{"ResourceType": "volume", "Tags": [{"Key": "Project", "Value": "pycon-davao-model2vec"},
                {"Key": "RecoveryRun", "Value": name}, {"Key": "SourceSnapshot", "Value": row["snapshot_id"]},
                {"Key": "RestoreInstance", "Value": host["InstanceId"]}]}])
        record = {"instance_id": host["InstanceId"], "snapshot_id": row["snapshot_id"], "volume_id": volume["VolumeId"],
                  "retained_source_volume": row["volume_id"], "device": device, "private_ip": host["PrivateIpAddress"]}
        save("volume-created-" + host["InstanceId"], record)
        ec.get_waiter("volume_available").wait(VolumeIds=[volume["VolumeId"]], WaiterConfig={"Delay": 3, "MaxAttempts": 100})
        ec.detach_volume(VolumeId=row["volume_id"], InstanceId=host["InstanceId"], Device=device)
        ec.get_waiter("volume_available").wait(VolumeIds=[row["volume_id"]], WaiterConfig={"Delay": 3, "MaxAttempts": 100})
        ec.attach_volume(VolumeId=volume["VolumeId"], InstanceId=host["InstanceId"], Device=device)
        ec.get_waiter("volume_in_use").wait(VolumeIds=[volume["VolumeId"]], WaiterConfig={"Delay": 3, "MaxAttempts": 100})
        ec.modify_instance_attribute(InstanceId=host["InstanceId"], BlockDeviceMappings=[{
            "DeviceName": device, "Ebs": {"DeleteOnTermination": False}}])
        save("restored-" + host["InstanceId"], record)
        print("SNAPSHOT_RESTORED", json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=6) as pool:
        restored = list(pool.map(restore, pause["snapshots"]))
    guard_role = settings["guard_role"]
    guard_name = guard_role.rsplit("/", 1)[1]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=10800)
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "ec2:StopInstances", "Resource": [
            f"arn:aws:ec2:{settings['region']}:{settings['account']}:instance/{i}" for i in ids]},
        {"Effect": "Allow", "Action": "sagemaker:DeleteEndpoint", "Resource":
            f"arn:aws:sagemaker:{settings['region']}:{settings['account']}:endpoint/{settings['teacher_endpoint']}"}]}
    iam.put_role_policy(RoleName=guard_name, PolicyName=name, PolicyDocument=json.dumps(policy))
    schedules = []
    for suffix, action, payload in (("cpu-stop", "ec2:stopInstances", {"InstanceIds": ids}),
                                     ("gpu-stop", "sagemaker:deleteEndpoint", {"EndpointName": settings["teacher_endpoint"]})):
        schedule = name + "-" + suffix
        response = scheduler.create_schedule(Name=schedule, ScheduleExpression=f"at({deadline:%Y-%m-%dT%H:%M:%S})",
            ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"}, State="ENABLED",
            Target={"Arn": "arn:aws:scheduler:::aws-sdk:" + action, "RoleArn": guard_role, "Input": json.dumps(payload),
                    "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2}})
        verified = scheduler.get_schedule(Name=schedule)
        assert verified["State"] == "ENABLED" and json.loads(verified["Target"]["Input"]) == payload
        schedules.append({"name": schedule, "arn": response["ScheduleArn"], "deadline_utc": deadline.isoformat()})
    save("stop-guards", schedules)
    new = {**settings, "name": name, "prefix": prefix, "output_directory": "/opt/pycon-isolated/" + name,
           "previous_recovery_prefix": settings["prefix"], "previous_recovery_directory": settings["output_directory"],
           "restored_volumes": restored, "recovery_deadline_utc": deadline.isoformat(), "resume_authorized": True,
           "source_pause_snapshots": [r["snapshot_id"] for r in restored], "new_stop_guards": schedules}
    save("resources", new)
    time.sleep(5)
    response = ec.start_instances(InstanceIds=ids)
    save("start-request", response)
    print("RECOVERY_STARTED", json.dumps({"folder": str(folder), "instances": ids, "deadline_utc": deadline.isoformat()}), flush=True)


if __name__ == "__main__":
    main()
