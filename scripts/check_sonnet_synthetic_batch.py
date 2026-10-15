#!/usr/bin/env python3
"""Read-only status and output-count check for a Sonnet batch run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    resource = json.loads((args.run / "resource-record.json").read_text(encoding="utf-8"))
    launch = json.loads((args.run / "create-job-response.json").read_text(encoding="utf-8"))
    session = boto3.Session(profile_name=resource["profile"], region_name=resource["region"])
    bedrock, s3 = session.client("bedrock"), session.client("s3")
    status = bedrock.get_model_invocation_job(jobIdentifier=launch["jobArn"])
    listed = s3.list_objects_v2(Bucket=resource["bucket"], Prefix=f"{resource['prefix']}/output/")
    outputs = [{"key": item["Key"], "bytes": item["Size"], "updated": item["LastModified"].isoformat()} for item in listed.get("Contents", [])]
    print(json.dumps({
        "job_arn": status["jobArn"],
        "status": status["status"],
        "message": status.get("message"),
        "submit_time": status.get("submitTime"),
        "last_modified_time": status.get("lastModifiedTime"),
        "end_time": status.get("endTime"),
        "outputs": outputs,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
