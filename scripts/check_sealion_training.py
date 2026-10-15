"""Read-only progress check; optional full streaming verification on completion.

Never restarts, deletes, overwrites, or creates paid resources.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import struct
from urllib.parse import urlparse

import boto3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--verify-complete", action="store_true")
    args = parser.parse_args()
    record = json.loads((args.run_directory / "resource-record.json").read_text())
    session = boto3.Session(profile_name=record["profile"], region_name=record["region"])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    job = sm.describe_training_job(TrainingJobName=record["job_name"])
    uri = urlparse(record["s3_results"])
    bucket, prefix = uri.netloc, uri.path.strip("/")
    objects = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        objects.extend(page.get("Contents", []))
    events = sorted((o for o in objects if "/events/" in o["Key"]), key=lambda o: o["Key"])
    latest = json.loads(s3.get_object(Bucket=bucket, Key=events[-1]["Key"])["Body"].read()) if events else None
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "job_name": record["job_name"], "status": job["TrainingJobStatus"],
              "secondary_status": job.get("SecondaryStatus"), "failure_reason": job.get("FailureReason"),
              "latest_event": latest, "saved_objects": len(objects),
              "training_seconds": job.get("TrainingTimeInSeconds"),
              "billable_seconds": job.get("BillableTimeInSeconds"),
              "training_compute_charge_estimate_usd": (job["BillableTimeInSeconds"] / 3600 * 1.315
                                                      if job.get("BillableTimeInSeconds") is not None else None),
              "credits_verified": False,
              "saved_reports": [o["Key"] for o in objects if "/reports/" in o["Key"]]}
    if args.verify_complete:
        if job["TrainingJobStatus"] != "Completed":
            raise RuntimeError("Job has not completed; refusing completion verification")
        complete = json.loads(s3.get_object(Bucket=bucket, Key=prefix + "/reports/completion.json")["Body"].read())
        protocol = complete["protocol"]
        if (complete["status"] != "complete" or protocol["teacher"]["dimension"] != 1024
                or protocol["initialization"]["pca_dims"] is not None
                or protocol["tokenlearn"]["target_pca_dims"] is not None):
            raise ValueError("Dimension/PCA/completion protocol mismatch")
        reports = complete["training_reports"]
        if {r["arm"] for r in reports} != {"external_wikipedia", "approved_articles"}:
            raise ValueError("Missing corpus arm")
        if reports[0]["initial_hashes"] != reports[1]["initial_hashes"]:
            raise ValueError("Different initial models")
        for report in reports:
            if (report["dimension"] != 1024 or not report["reload_parity"]
                    or report["changed_token_rows"] <= 0 or not report["validation_history"]):
                raise ValueError("Student training/reload evidence failed")
        checked = 0
        weight_hashes = []
        # Stream bytes: independently verify ALL preserved outputs without local
        # downloads, disk growth, or trusting custom checksum metadata alone.
        for relative, spec in complete["artifacts"].items():
            response = s3.get_object(Bucket=bucket, Key=prefix + "/" + relative)
            if response["ContentLength"] != spec["bytes"]:
                raise ValueError("Object size mismatch: " + relative)
            digest = sha256()
            body = response["Body"]
            try:
                for block in iter(lambda: body.read(1024 * 1024), b""):
                    digest.update(block)
            finally:
                body.close()
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError("Independent S3 byte checksum mismatch: " + relative)
            checked += 1
            if relative.startswith("models/") and relative.endswith("model.safetensors"):
                header_length = s3.get_object(Bucket=bucket, Key=prefix + "/" + relative, Range="bytes=0-7")["Body"].read()
                length = struct.unpack("<Q", header_length)[0]
                header = json.loads(s3.get_object(Bucket=bucket, Key=prefix + "/" + relative,
                                                 Range=f"bytes=8-{7 + length}")["Body"].read())
                tensors = [v for k, v in header.items() if k != "__metadata__" and len(v["shape"]) == 2]
                if not tensors or not any(v["shape"][1] == 1024 and v["dtype"] == "F32" for v in tensors):
                    raise ValueError("Static matrix dimension/precision mismatch")
                if "/initialization/" not in relative:
                    weight_hashes.append(spec["sha256"])
        if len(weight_hashes) != 2 or len(set(weight_hashes)) != 2:
            raise ValueError("Missing/distinct student weights")
        result["independent_verification"] = {"objects_stream_hashed": checked, "students": 2,
                                              "dimension": 1024, "pca": False, "shared_initialization": True,
                                              "managed_compute_released": True,
                                              "evaluation_not_performed": True}
    print(json.dumps(result, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
