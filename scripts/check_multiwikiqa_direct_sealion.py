#!/usr/bin/env python3
"""Read-only status checker for a direct SEA-LION managed training run."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import boto3


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    record = json.loads((run / "staging-record.json").read_text())
    session = boto3.Session(profile_name=record["profile"], region_name=record["region"])
    state = session.client("sagemaker").describe_training_job(TrainingJobName=record["job"])
    s3 = session.client("s3")
    prefix = record["prefix"] + "/results/"
    objects = []
    token = None
    while True:
        kwargs = {"Bucket": record["bucket"], "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        objects.extend({"key": row["Key"], "bytes": row["Size"], "utc": row["LastModified"]} for row in page.get("Contents", []))
        token = page.get("NextContinuationToken")
        if not token:
            break
    events = sorted((row for row in objects if "/events/" in row["key"]), key=lambda row: row["key"])
    latest = None
    if events:
        response = s3.get_object(Bucket=record["bucket"], Key=events[-1]["key"])
        latest = json.loads(response["Body"].read())
        response["Body"].close()
    print(json.dumps({
        "job": record["job"],
        "status": state["TrainingJobStatus"],
        "secondary_status": state.get("SecondaryStatus"),
        "latest_event": latest,
        "result_objects": len(objects),
        "billable_seconds": state.get("BillableTimeInSeconds"),
        "failure_reason": state.get("FailureReason"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
