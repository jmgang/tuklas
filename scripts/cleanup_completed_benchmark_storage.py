"""Explicitly authorized cleanup: twelve exact disks and two private endpoints.

Preserve all snapshots, S3/local data, verified results and stopped EC2 shells.
Back up latest disks before removal; older detached disks already have completed
pause snapshots. Never force-detach, delete a snapshot, or start compute.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import boto3
from botocore.config import Config

VOLUMES = {
    "vol-09c438a58d7a55aa1", "vol-0231bd3d6040dd6f9", "vol-09598adb1c344426a",
    "vol-0d961745047418528", "vol-002d2ef31ddf76959", "vol-061cc5e22ad13ef43",
    "vol-0abfa03381a188e38", "vol-0f1d1d58a8375951b", "vol-04978de37686e60a8",
    "vol-03e26d2e105e1d7e2", "vol-057f1450450797f06", "vol-0e19df04a24e6e885",
}
ENDPOINTS = {"vpce-058d30102ee2844d0", "vpce-07c6df2501a422efa"}


def main(folder, verified):
    folder, verified = Path(folder), Path(verified)
    settings = json.loads((folder / "resources.json").read_text())
    proof = json.loads((verified / "verification.json").read_text())
    assert len(proof["verified_models"]) == 6
    rows = settings["restored_volumes"]
    assert {r["volume_id"] for r in rows} | {r["retained_source_volume"] for r in rows} == VOLUMES
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    cfg = Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2})
    ec, s3 = [session.client(n, config=cfg) for n in ("ec2", "s3")]
    assert session.client("sts").get_caller_identity()["Account"] == "<aws-account-id>"
    inventory = ec.describe_volumes(VolumeIds=sorted(VOLUMES))["Volumes"]
    assert len(inventory) == 12
    by_volume = {v["VolumeId"]: v for v in inventory}
    hosts = [i for g in ec.describe_instances(InstanceIds=[r["instance_id"] for r in rows])["Reservations"] for i in g["Instances"]]
    assert len(hosts) == 6 and all(i["State"]["Name"] == "stopped" for i in hosts)
    vpcs = {h["VpcId"] for h in hosts}
    assert len(vpcs) == 1
    endpoints = ec.describe_vpc_endpoints(VpcEndpointIds=sorted(ENDPOINTS))["VpcEndpoints"]
    assert len(endpoints) == 2
    assert all(v["VpcId"] in vpcs and v["VpcEndpointType"] == "Interface" for v in endpoints)
    assert {e["ServiceName"] for e in endpoints} == {"com.amazonaws.ap-southeast-1.bedrock-runtime", "com.amazonaws.ap-southeast-1.sagemaker.runtime"}
    for relative, expected in proof["local_content_sha256"].items():
        assert sha256((verified / relative).read_bytes()).hexdigest() == expected
        assert s3.head_object(Bucket=settings["bucket"], Key=settings["prefix"] + "/benchmark/" + relative)["Metadata"]["sha256"] == expected
    marker = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = folder / ("authorized-storage-cleanup-" + marker)
    output.mkdir(exist_ok=False)
    prefix = settings["prefix"] + "/authorized-storage-cleanup/" + marker
    def save(label, value):
        body = json.dumps(value, indent=2, default=str).encode()
        with (output / (label + ".json")).open("xb") as out:
            out.write(body)
        s3.put_object(Bucket=settings["bucket"], Key=prefix + "/" + label + ".json", Body=body,
                      Metadata={"sha256": sha256(body).hexdigest()}, IfNoneMatch="*", ServerSideEncryption="AES256")
    save("before", {"volumes": inventory, "endpoints": endpoints, "hosts": hosts, "verified_results": proof,
        "authorized_scope": "Delete twelve exact provisioned volumes and two interface endpoints after backups; preserve snapshots, S3, local artifacts, stopped instances"})
    backups = []
    for row in rows:
        for vol_id in (row["retained_source_volume"], row["volume_id"]):
            volume = by_volume[vol_id]
            assert volume["Size"] == 80 and volume["Encrypted"]
            assert {t["Key"]: t["Value"] for t in volume["Tags"]}["Project"] == "pycon-davao-model2vec"
            if vol_id == row["retained_source_volume"]:
                assert volume["State"] == "available" and not volume["Attachments"]
                snap = ec.describe_snapshots(SnapshotIds=[row["snapshot_id"]])["Snapshots"][0]
                assert snap["State"] == "completed" and snap["VolumeId"] == vol_id and snap["Encrypted"]
                backup = {"volume_id": vol_id, "instance_id": row["instance_id"], "snapshot_id": snap["SnapshotId"], "existing_completed_backup": True}
            else:
                assert len(volume["Attachments"]) == 1 and volume["Attachments"][0]["InstanceId"] == row["instance_id"]
                snap = ec.create_snapshot(VolumeId=vol_id, Description="Final verified PyCon benchmark disk preservation " + marker,
                    TagSpecifications=[{"ResourceType": "snapshot", "Tags": [{"Key": "Project", "Value": "pycon-davao-model2vec"},
                        {"Key": "Cleanup", "Value": marker}, {"Key": "SourceInstance", "Value": row["instance_id"]}]}])
                backup = {"volume_id": vol_id, "instance_id": row["instance_id"], "snapshot_id": snap["SnapshotId"], "existing_completed_backup": False}
                print("FINAL_DISK_SNAPSHOT_REQUESTED", json.dumps(backup), flush=True)
            backups.append(backup)
            save("backup-" + vol_id, backup)
    save("recovery-map", backups)
    # Network endpoints contain no model or disk data; remove their idle hourly
    # charges while the final disk snapshots complete. Keep the S3 gateway.
    result = ec.delete_vpc_endpoints(VpcEndpointIds=sorted(ENDPOINTS))
    save("interface-endpoint-deletion-request", result)
    assert not result["Unsuccessful"], result
    print("INTERFACE_ENDPOINT_DELETION_REQUESTED", sorted(ENDPOINTS), flush=True)
    pending = {b["volume_id"]: b for b in backups}
    deadline = time.monotonic() + 1800
    while pending and time.monotonic() < deadline:
        snapshots = {s["SnapshotId"]: s for s in ec.describe_snapshots(SnapshotIds=[b["snapshot_id"] for b in pending.values()])["Snapshots"]}
        for vol_id, backup in list(pending.items()):
            snap = snapshots[backup["snapshot_id"]]
            assert snap["VolumeId"] == vol_id and snap["OwnerId"] == "<aws-account-id>" and snap["Encrypted"]
            if snap["State"] == "error":
                save("snapshot-failure-" + vol_id, snap)
                raise RuntimeError("Backup failed; preserve remaining disks")
            if snap["State"] != "completed":
                continue
            volume = ec.describe_volumes(VolumeIds=[vol_id])["Volumes"][0]
            if volume["Attachments"]:
                attachment = volume["Attachments"][0]
                if attachment["State"] == "attached":
                    response = ec.detach_volume(VolumeId=vol_id, InstanceId=backup["instance_id"], Device=attachment["Device"], Force=False)
                    save("detach-" + vol_id, response)
                    print("BACKED_UP_DISK_DETACH_REQUESTED", vol_id, flush=True)
                continue
            if volume["State"] != "available":
                continue
            save("verified-complete-snapshot-" + vol_id, snap)
            response = ec.delete_volume(VolumeId=vol_id)
            save("delete-" + vol_id, {"recovery": backup, "response": response})
            print("BACKED_UP_DISK_DELETE_REQUESTED", vol_id, flush=True)
            pending.pop(vol_id)
        if pending:
            print("WAITING_FOR_BACKUP_OR_DETACH", sorted(pending), flush=True)
            time.sleep(10)
    assert not pending, "Remaining disks preserved; inspect pending backups"
    absent = []
    for vol_id in sorted(VOLUMES):
        ec.get_waiter("volume_deleted").wait(VolumeIds=[vol_id], WaiterConfig={"Delay": 3, "MaxAttempts": 30})
        absent.append(vol_id)
    for endpoint_id in sorted(ENDPOINTS):
        for _ in range(60):
            try:
                rows_ep = ec.describe_vpc_endpoints(VpcEndpointIds=[endpoint_id])["VpcEndpoints"]
            except Exception as error:
                if getattr(error, "response", {}).get("Error", {}).get("Code") != "InvalidVpcEndpointId.NotFound":
                    raise
                break
            if not rows_ep or rows_ep[0]["State"] == "deleted":
                break
            time.sleep(5)
        else:
            raise TimeoutError("Endpoint deletion not confirmed: " + endpoint_id)
    snapshots = ec.describe_snapshots(SnapshotIds=[b["snapshot_id"] for b in backups])["Snapshots"]
    assert len(snapshots) == 12 and all(s["State"] == "completed" for s in snapshots)
    remaining = ec.describe_volumes(Filters=[{"Name": "tag:Project", "Values": ["pycon-davao-model2vec"]}])["Volumes"]
    assert not remaining, "Unexpected benchmark disks remain"
    result = {"deleted_volumes": absent, "deleted_interface_endpoints": sorted(ENDPOINTS), "removed_provisioned_GiB": 960,
        "removed_monthly_disk_charge_usd": 92.16, "removed_30day_endpoint_charge_usd": 18.72,
        "retained_complete_recovery_map": backups, "preserved_verified_results": str(verified),
        "snapshots_s3_models_results_and_local_files_preserved": True, "compute_not_started": True,
        "backup_storage": "EBS snapshots in AWS-managed S3; not ordinary exported disk files",
        "recurring_snapshot_and_s3_storage_costs_remain": True}
    save("deletion-verification", result)
    print("CLEANUP_VERIFIED", json.dumps({"records": str(output), "disks_removed": 12, "interface_endpoints_removed": 2, "monthly_base_charge_removed_usd": 110.88}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
