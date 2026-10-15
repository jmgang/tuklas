"""User-authorized removal of the exact rejected benchmark instance and disk.

Never searches for broad deletion targets, deletes local files, or modifies S3.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4
import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = "i-00aa8bfc3bf2b424e"
VOLUME = "vol-0147e6b3e6a602e55"
RUN = "pycon-fresh-20261002-162759-c570dd"


def main():
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    cfg = Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 0})
    if session.client("sts", config=cfg).get_caller_identity()["Account"] != "<aws-account-id>":
        raise RuntimeError("Unexpected AWS account")
    ec2 = session.client("ec2", config=cfg)
    instance = ec2.describe_instances(InstanceIds=[INSTANCE])["Reservations"][0]["Instances"][0]
    if {t["Key"]: t["Value"] for t in instance["Tags"]}.get("Name") != RUN:
        raise RuntimeError("Target instance tag differs")
    disks = instance["BlockDeviceMappings"]
    if len(disks) != 1 or disks[0]["Ebs"]["VolumeId"] != VOLUME:
        raise RuntimeError("Unexpected attached disks")
    volume = ec2.describe_volumes(VolumeIds=[VOLUME])["Volumes"][0]
    if len(volume["Attachments"]) != 1 or volume["Attachments"][0]["InstanceId"] != INSTANCE:
        raise RuntimeError("Disk ownership mismatch")
    if {t["Key"]: t["Value"] for t in volume["Tags"]}.get("Name") != RUN:
        raise RuntimeError("Disk tag mismatch")
    record = {"authorized_action": "user explicitly requested deleting rejected EC2 and its data",
              "time": datetime.now(timezone.utc).isoformat(), "instance": INSTANCE, "volume": VOLUME,
              "before": {"instance_state": instance["State"]["Name"], "volume_state": volume["State"]},
              "local_project_and_frozen_models": "preserved", "s3_evidence": "preserved"}
    directory = ROOT / "runs" / RUN
    with (directory / f"removal-authorized-{uuid4().hex}.json").open("x") as out:
        json.dump(record, out, indent=2)
    # This exact disk is the one the user requested erasing. Enable automatic
    # deletion on termination, and verify the flag before terminating compute.
    ec2.modify_instance_attribute(InstanceId=INSTANCE, BlockDeviceMappings=[{
        "DeviceName": disks[0]["DeviceName"], "Ebs": {"VolumeId": VOLUME, "DeleteOnTermination": True}}])
    verified = ec2.describe_instances(InstanceIds=[INSTANCE])["Reservations"][0]["Instances"][0]
    if not verified["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"]:
        raise RuntimeError("Disk deletion flag not verified")
    print("TERMINATE", json.dumps(ec2.terminate_instances(InstanceIds=[INSTANCE])["TerminatingInstances"]), flush=True)
    scheduler = session.client("scheduler", config=cfg)
    stop = scheduler.get_schedule(Name=RUN+"-stop")
    scheduler.update_schedule(Name=stop["Name"], GroupName=stop["GroupName"],
        ScheduleExpression=stop["ScheduleExpression"], ScheduleExpressionTimezone=stop["ScheduleExpressionTimezone"],
        FlexibleTimeWindow=stop["FlexibleTimeWindow"], Target=stop["Target"],
        ActionAfterCompletion=stop["ActionAfterCompletion"], State="DISABLED")
    print("STOP_SCHEDULE_DISABLED", flush=True)
    for _ in range(18):
        state = ec2.describe_instances(InstanceIds=[INSTANCE])["Reservations"][0]["Instances"][0]["State"]["Name"]
        print("INSTANCE_STATE", state, flush=True)
        if state == "terminated":
            break
        time.sleep(5)
    from botocore.exceptions import ClientError
    try:
        disk_state = ec2.describe_volumes(VolumeIds=[VOLUME])["Volumes"][0]["State"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "InvalidVolume.NotFound":
            raise
        disk_state = "deleted"
    result = {**record, "after": {"instance_state": state, "volume_state": disk_state}}
    with (directory / f"removal-result-{uuid4().hex}.json").open("x") as out:
        json.dump(result, out, indent=2)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
