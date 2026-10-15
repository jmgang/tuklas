"""Delete only the seven explicitly authorized disks, after completed backups.

Keeps every snapshot, S3 object, local artifact and stopped instance. No compute
is started. Full recovery mappings are persisted locally and in S3 before
detaching any root disk. Never force-detach or delete an unbacked disk.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import boto3
from botocore.config import Config


EXPECTED_VOLUMES = {
    "vol-015e58938a0be79c0", "vol-06f5714a5bfdb65a8",
    "vol-0f16726a53d6d0d2b", "vol-0023d2c80a5105d88",
    "vol-0c832de5ac3108509", "vol-0cde15e54e432bc4c",
    "vol-08500e5caf9d45e42",
}


def main():
    root = Path(__file__).resolve().parents[1]
    preserved = root / "runs/pycon-isolated-20261002-192911-eb3adc/preserved-stop-20261002-210911-5011ace5"
    manifest = json.loads((preserved / "halt-manifest.json").read_text())
    rows = manifest["snapshots"]
    assert {r["volume_id"] for r in rows} == EXPECTED_VOLUMES
    assert len(rows) == 7
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    config = Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 0})
    ec = session.client("ec2", config=config)
    s3 = session.client("s3", config=config)
    assert session.client("sts", config=config).get_caller_identity()["Account"] == "<aws-account-id>"
    marker = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output = preserved / ("disk-removal-" + marker)
    output.mkdir(exist_ok=False)
    bucket = "sagemaker-ap-southeast-1-<aws-account-id>"
    prefix = "pycon-davao-model2vec/pycon-isolated-20261002-192911-eb3adc/preservation/20261002-210911-5011ace5/disk-removal-" + marker

    def save(name, value):
        body = json.dumps(value, indent=2, default=str).encode()
        with (output / (name + ".json")).open("xb") as stream:
            stream.write(body)
        s3.put_object(Bucket=bucket, Key=prefix + "/" + name + ".json", Body=body,
                      IfNoneMatch="*", ServerSideEncryption="AES256")

    snapshots = []
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        snapshots = ec.describe_snapshots(SnapshotIds=[r["snapshot_id"] for r in rows])["Snapshots"]
        by_id = {s["SnapshotId"]: s for s in snapshots}
        for row in rows:
            snap = by_id[row["snapshot_id"]]
            assert snap["VolumeId"] == row["volume_id"]
            assert snap["OwnerId"] == "<aws-account-id>"
            assert snap["Encrypted"]
            tags = {x["Key"]: x["Value"] for x in snap.get("Tags", [])}
            assert tags["Project"] == "pycon-davao-model2vec"
            assert tags["SourceInstance"] == row["instance_id"]
            if snap["State"] == "error":
                save("snapshot-error", snapshots)
                raise RuntimeError("Snapshot failed; no disks will be deleted")
        print("SNAPSHOT_PROGRESS", json.dumps({s["SnapshotId"]: [s["State"], s["Progress"]] for s in snapshots}), flush=True)
        if all(s["State"] == "completed" for s in snapshots):
            break
        time.sleep(20)
    else:
        raise RuntimeError("Snapshots not complete; source disks retained")

    instances = [i for group in ec.describe_instances(InstanceIds=manifest["instance_ids"])["Reservations"] for i in group["Instances"]]
    assert all(i["State"]["Name"] == "stopped" for i in instances)
    volumes = ec.describe_volumes(VolumeIds=sorted(EXPECTED_VOLUMES))["Volumes"]
    by_volume = {v["VolumeId"]: v for v in volumes}
    for row in rows:
        volume = by_volume[row["volume_id"]]
        assert volume["Size"] == 80
        assert {t["Key"]: t["Value"] for t in volume["Tags"]}["Project"] == "pycon-davao-model2vec"
        assert all(a["InstanceId"] == row["instance_id"] for a in volume["Attachments"])
    save("completed-backups-and-recovery-map", {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": rows, "completed_snapshots": snapshots,
        "source_volumes": volumes, "stopped_instances": instances,
        "backup_storage": "AWS-managed S3 via EBS Standard snapshots; not ordinary user-bucket objects",
        "user_authorized_original_disk_deletion": True,
        "snapshot_deletion_authorized": False,
        "compute_restart_authorized": False,
    })
    print("ALL_SEVEN_BACKUPS_COMPLETE_AND_RECORDED", flush=True)

    for row in rows:
        for attachment in by_volume[row["volume_id"]]["Attachments"]:
            reply = ec.detach_volume(VolumeId=row["volume_id"], InstanceId=attachment["InstanceId"],
                                     Device=attachment["Device"], Force=False)
            save("detach-" + row["volume_id"], reply)
            print("DISK_DETACH_REQUESTED", row["volume_id"], flush=True)

    pending = dict((r["volume_id"], r) for r in rows)
    deadline = time.monotonic() + 300
    while pending and time.monotonic() < deadline:
        for volume_id in list(pending):
            volume = ec.describe_volumes(VolumeIds=[volume_id])["Volumes"][0]
            if volume["State"] != "available" or volume["Attachments"]:
                continue
            # Recheck the completed backup immediately before irreversible deletion.
            snap = ec.describe_snapshots(SnapshotIds=[pending[volume_id]["snapshot_id"]])["Snapshots"][0]
            assert snap["State"] == "completed" and snap["VolumeId"] == volume_id
            reply = ec.delete_volume(VolumeId=volume_id)
            save("delete-" + volume_id, {"requested_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "recovery_snapshot": snap["SnapshotId"], "response": reply})
            pending.pop(volume_id)
            print("DISK_DELETE_REQUESTED", volume_id, "RECOVERABLE_FROM", snap["SnapshotId"], flush=True)
        if pending:
            time.sleep(3)
    assert not pending, "Some disks are not safely detached; inspect preserved records"

    absent = []
    for row in rows:
        ec.get_waiter("volume_deleted").wait(VolumeIds=[row["volume_id"]], WaiterConfig={"Delay": 3, "MaxAttempts": 40})
        absent.append(row["volume_id"])
    final_snapshots = ec.describe_snapshots(SnapshotIds=[r["snapshot_id"] for r in rows])["Snapshots"]
    assert len(final_snapshots) == 7 and all(s["State"] == "completed" for s in final_snapshots)
    save("deletion-verification", {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "deleted_volumes": absent, "provisioned_gib_removed": 560,
                                  "retained_complete_snapshots": rows,
                                  "snapshot_count": 7, "source_disk_monthly_charge_removed_usd": 53.76,
                                  "instance_shells_retained_stopped": True,
                                  "snapshot_s3_and_interface_endpoint_charges_remain": True})
    print("ALL_SEVEN_DISKS_DELETED_BACKUPS_RETAINED", str(output), flush=True)


if __name__ == "__main__":
    main()
