"""Explicitly authorized recovery, same hosts/models, fresh three-hour guard.

No existing disks, snapshots, files, outputs or AWS schedules are overwritten.
Restore six required disks. The obsolete shared server stays stopped/diskless.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import boto3
from botocore.config import Config


def main():
    root = Path(__file__).resolve().parents[1]
    previous = root / "runs/pycon-isolated-20261002-192911-eb3adc"
    r = json.loads((previous / "resources.json").read_text())
    dedicated = json.loads((previous / "dedicated-resources.json").read_text())["models"]
    backup = previous / "preserved-stop-20261002-210911-5011ace5"
    old = json.loads((backup / "halt-manifest.json").read_text())
    mapping = {x["instance_id"]: x for x in old["snapshots"]}
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0})
    ec, sm, sched, s3 = [session.client(name, config=cfg) for name in ("ec2", "sagemaker", "scheduler", "s3")]
    assert session.client("sts", config=cfg).get_caller_identity()["Account"] == r["account"]
    targets = [r["client_instance"], dedicated["cohere_v3"]["instance_id"],
               *[v["instance_id"] for k, v in dedicated.items() if k != "cohere_v3"]]
    assert len(targets) == len(set(targets)) == 6
    assert r["server_instance"] not in targets
    name = "pycon-resume-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    folder = root / "runs" / name
    folder.mkdir(exist_ok=False)
    prefix = r["prefix"] + "/" + name

    def save(label, value):
        body = json.dumps(value, indent=2, default=str).encode()
        with (folder / (label + ".json")).open("xb") as out:
            out.write(body)
        s3.put_object(Bucket=r["bucket"], Key=prefix + "/recovery/" + label + ".json", Body=body,
                      IfNoneMatch="*", ServerSideEncryption="AES256")

    instances = {i["InstanceId"]: i for group in ec.describe_instances(InstanceIds=targets)["Reservations"] for i in group["Instances"]}
    assert all(i["State"]["Name"] == "stopped" and not i["BlockDeviceMappings"] for i in instances.values())
    snaps = {s["SnapshotId"]: s for s in ec.describe_snapshots(SnapshotIds=[mapping[i]["snapshot_id"] for i in targets])["Snapshots"]}
    for target in targets:
        snap = snaps[mapping[target]["snapshot_id"]]
        assert snap["State"] == "completed" and snap["OwnerId"] == r["account"]
        assert snap["VolumeId"] == mapping[target]["volume_id"]
    deadline = datetime.now(timezone.utc) + timedelta(hours=3)
    # New guards represent the user's new recovery authorization; never alter
    # the old experiment's expired shutdown deadline or frozen source records.
    for label, target_arn, inputs in [
        ("ec2-stop", "arn:aws:scheduler:::aws-sdk:ec2:stopInstances", {"InstanceIds": targets}),
        ("gpu-stop", "arn:aws:scheduler:::aws-sdk:sagemaker:deleteEndpoint", {"EndpointName": r["teacher_endpoint"]}),
    ]:
        schedule_name = name + "-" + label
        sched.create_schedule(Name=schedule_name, ScheduleExpression=f"at({deadline:%Y-%m-%dT%H:%M:%S})",
            ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"}, State="ENABLED",
            Target={"Arn": target_arn, "RoleArn": r["guard_role"], "Input": json.dumps(inputs),
                    "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2}})
        guard = sched.get_schedule(Name=schedule_name)
        assert guard["State"] == "ENABLED" and json.loads(guard["Target"]["Input"]) == inputs
        save(label + "-guard", guard)
    restored = []
    for target in targets:
        host = instances[target]
        reply = ec.create_volume(SnapshotId=mapping[target]["snapshot_id"], AvailabilityZone=host["Placement"]["AvailabilityZone"],
            VolumeType="gp3", Encrypted=True, ClientToken=name + "-" + target,
            TagSpecifications=[{"ResourceType": "volume", "Tags": [
                {"Key": "Project", "Value": "pycon-davao-model2vec"}, {"Key": "RecoveryRun", "Value": name},
                {"Key": "SourceInstance", "Value": target}, {"Key": "Name", "Value": name + "-" + target}]}])
        row = {"instance_id": target, "volume_id": reply["VolumeId"], "snapshot_id": mapping[target]["snapshot_id"],
               "device": host["RootDeviceName"], "private_ip": host["PrivateIpAddress"]}
        save("restore-" + target, row)
        restored.append(row)
        print("RESTORE_VOLUME_REQUESTED", json.dumps(row), flush=True)
    for row in restored:
        ec.get_waiter("volume_available").wait(VolumeIds=[row["volume_id"]], WaiterConfig={"Delay": 3, "MaxAttempts": 80})
        ec.attach_volume(VolumeId=row["volume_id"], InstanceId=row["instance_id"], Device=row["device"])
    for row in restored:
        ec.get_waiter("volume_in_use").wait(VolumeIds=[row["volume_id"]], WaiterConfig={"Delay": 3, "MaxAttempts": 80})
        ec.modify_instance_attribute(InstanceId=row["instance_id"], BlockDeviceMappings=[
            {"DeviceName": row["device"], "Ebs": {"DeleteOnTermination": False, "VolumeId": row["volume_id"]}}])
    settings = {**r, "name": name, "prefix": prefix,
        "original_dedicated_prefix": r["prefix"] + "/dedicated", "original_prefix": r["prefix"],
        "dedicated_models": dedicated, "output_directory": "/opt/pycon-isolated/" + name,
        "recovery_deadline_utc": deadline.isoformat(), "restored_volumes": restored,
        "resume_authorized": True, "preserved_completed_vectors_reused": True,
        "dataset_changed": False, "model_weights_changed": False, "three_hour_cap": True}
    save("resources", settings)
    with (folder / "settings.json").open("x") as out:
        json.dump(settings, out, indent=2)
    save("ec2-start-request", ec.start_instances(InstanceIds=targets))
    restore_record = json.loads((backup / "gpu-endpoint-restore-record.json").read_text())
    # Existing retained definitions are unchanged; endpoint compute is recreated.
    save("gpu-create-request", sm.create_endpoint(EndpointName=r["teacher_endpoint"],
        EndpointConfigName=restore_record["config"]["EndpointConfigName"],
        Tags=[{"Key": "Project", "Value": "pycon-davao-model2vec"}, {"Key": "RecoveryRun", "Value": name}]))
    print("RESTORED_HOSTS_STARTING", targets, flush=True)
    print("GPU_ENDPOINT_CREATING", r["teacher_endpoint"], flush=True)
    print("RECOVERY_DIRECTORY", str(folder), flush=True)
    print("NEW_HARD_STOP_UTC", deadline.isoformat(), flush=True)


if __name__ == "__main__":
    main()
