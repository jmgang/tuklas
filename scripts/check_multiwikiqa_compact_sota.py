#!/usr/bin/env python3
"""Read-only status check for one compact SOTA SageMaker run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    record = json.loads((args.run / "staging-record.json").read_text())
    session = boto3.Session(profile_name=record["profile"], region_name=record["region"])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    state = sm.describe_training_job(TrainingJobName=record["job"])
    prefix = record["prefix"] + "/results/"
    objects = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=record["bucket"], Prefix=prefix):
        objects.extend(page.get("Contents", []))
    events = sorted(
        (item for item in objects if "/events/" in item["Key"]),
        key=lambda item: item["Key"],
    )
    reports = sorted(
        (item for item in objects if "/reports/" in item["Key"]),
        key=lambda item: item["Key"],
    )
    latest = None
    if events:
        latest = json.loads(s3.get_object(Bucket=record["bucket"], Key=events[-1]["Key"])["Body"].read())
    result = {
        "job": record["job"],
        "status": state["TrainingJobStatus"],
        "secondary_status": state.get("SecondaryStatus"),
        "failure_reason": state.get("FailureReason"),
        "latest_event": latest,
        "event_count": len(events),
        "report_count": len(reports),
        "billable_seconds": state.get("BillableTimeInSeconds"),
        "training_seconds": state.get("TrainingTimeInSeconds"),
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
