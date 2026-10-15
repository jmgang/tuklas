"""Read-only live monitor; transient local network failures never restart jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_directory",type=Path)
    args=parser.parse_args()
    record=json.loads((args.run_directory/"resource-record.json").read_text())
    session=boto3.Session(profile_name=record["profile"],region_name=record["region"])
    config=Config(connect_timeout=10,read_timeout=30,retries={"mode":"standard","max_attempts":2})
    sm,s3=session.client("sagemaker",config=config),session.client("s3",config=config)
    uri=urlparse(record["s3_results"])
    previous=None
    while True:
        try:
            job=sm.describe_training_job(TrainingJobName=record["job_name"])
            events=[]
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=uri.netloc,Prefix=uri.path.strip("/")+"/events/"):
                events.extend(page.get("Contents",[]))
            latest=(json.loads(s3.get_object(Bucket=uri.netloc,
                    Key=max(events,key=lambda x:x["Key"])["Key"])["Body"].read()) if events else None)
            value={"status":job["TrainingJobStatus"],"secondary":job.get("SecondaryStatus"),
                   "latest":latest,"failure":job.get("FailureReason")}
            signature=json.dumps(value,sort_keys=True)
            if signature!=previous:
                print(signature,flush=True)
                previous=signature
            if job["TrainingJobStatus"] in ("Completed","Failed","Stopped"):
                print(json.dumps({"terminal":True,"billable_seconds":job.get("BillableTimeInSeconds")}),flush=True)
                return
        except (BotoCoreError,ClientError,OSError) as exc:
            print(json.dumps({"monitoring_error":str(exc)[:500],
                  "checked_at_utc":datetime.now(timezone.utc).isoformat(),
                  "cloud_job_state_not_inferred":True,"retry_seconds":45}),flush=True)
        time.sleep(45)


if __name__=="__main__":
    main()
