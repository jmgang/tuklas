"""Read-only progress and preserved-output checks for the current continuation."""
import argparse
import json
from pathlib import Path
import boto3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    settings = json.loads((args.folder / "resources.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name=settings["region"])
    s3, ec, sm = [session.client(n) for n in ("s3", "ec2", "sagemaker")]
    prefix = settings["prefix"] + "/benchmark/"
    objects = [obj for page in s3.get_paginator("list_objects_v2").paginate(Bucket=settings["bucket"], Prefix=prefix)
               for obj in page.get("Contents", [])]
    keys = {obj["Key"]: obj for obj in objects}
    events = []
    for key in sorted(k for k in keys if k.startswith(prefix + "events/"))[-8:]:
        reply = s3.get_object(Bucket=settings["bucket"], Key=key)
        events.append(json.loads(reply["Body"].read()))
        reply["Body"].close()
    chunk_keys = [k for k in keys if k.startswith(prefix + "cohere_v3/chunks/") and k.endswith(".json")]
    def ranges(selected):
        result = {}
        for key in selected:
            start, stop = map(int, key.rsplit("/", 1)[1].split(".")[0].split("-"))
            assert 55680 <= start < stop <= 240765 and (start - 55680) % 96 == 0
            assert stop == min(start + 96, 240765)
            result[start] = stop
        return result
    copied_or_new = ranges(chunk_keys)
    source = settings.get("resume_chunk_prefix")
    saved_source = {}
    if source:
        source_path = source + "/benchmark/cohere_v3/chunks/"
        saved_source = ranges([obj["Key"] for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=settings["bucket"], Prefix=source_path) for obj in page.get("Contents", [])
            if obj["Key"].endswith(".json")])
    union = {**saved_source, **copied_or_new}
    completed = 55680 + sum(stop - start for start, stop in union.items())
    imported = sum(stop - start for start, stop in copied_or_new.items() if start in saved_source)
    newly_encoded = sum(stop - start for start, stop in copied_or_new.items() if start not in saved_source)
    results = {name: {kind: prefix + name + "/" + kind + ".json" in keys for kind in ("build", "quality", "measurement")}
               for name in ("potion_retrieval", "plain_student", "custom_student", "tokenlearn_custom_student", "teacher", "cohere_v3")}
    ids = [r["instance_id"] for r in settings["restored_volumes"]]
    hosts = {i["InstanceId"]: i["State"]["Name"] for group in ec.describe_instances(InstanceIds=ids)["Reservations"] for i in group["Instances"]}
    try: endpoint = sm.describe_endpoint(EndpointName=settings["teacher_endpoint"])["EndpointStatus"]
    except Exception as error:
        if "Could not find endpoint" not in str(error): raise
        endpoint = "absent; restored by controller after bulk work"
    print(json.dumps({"run": settings["name"], "hosts": hosts, "teacher_endpoint": endpoint,
        "cohere_saved_segments": completed, "total_segments": 240765, "percent": completed / 240765 * 100,
        "current_prefix_chunks": len(chunk_keys), "copied_saved_segment_rows": imported,
        "new_encoded_segment_rows": newly_encoded,
        "saved_source_chunk_rows": sum(stop-start for start,stop in saved_source.items()),
        "restored_chunk_provenance_complete": prefix + "cohere_v3/restored-chunks-provenance.json" in keys if source else None,
        "results": results, "latest_events": events,
        "combined_report": prefix + "reports/combined-report.json" in keys,
        "stop_deadline_utc": settings["recovery_deadline_utc"]}, indent=2))


if __name__ == "__main__":
    main()
