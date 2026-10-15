"""Publish the already verified clean input archive using observable multipart IO.

Old transfers, files and partial uploads are preserved. This creates a distinct
immutable S3 key, not a second experiment or compute instance.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
import boto3
from botocore.config import Config


def main():
    directory = Path(sys.argv[1]).resolve()
    resources = json.loads((directory / "resources.json").read_text())
    if (directory / "staged-inputs.json").exists():
        raise FileExistsError("Inputs already published; do not duplicate")
    archive = directory / "fresh-inputs.tar.gz"
    hasher = sha256()
    with archive.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    expected = "1acb3e46d80165aba30225693e1b1792cc6cb2d886242aa10e6702805d4bfb11"
    if hasher.hexdigest() != expected:
        raise ValueError("Clean archive changed")
    bucket, prefix = resources["s3_prefix"].removeprefix("s3://").split("/", 1)
    key = prefix + "inputs/fresh-multipart-" + uuid4().hex + ".tar.gz"
    client = boto3.Session(profile_name="admin-general", region_name=resources["region"]).client(
        "s3", config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 0},
                            request_checksum_calculation="when_required"))
    events = directory / "transfer-events"
    events.mkdir(exist_ok=False)

    def event(value):
        value = {"time": datetime.now(timezone.utc).isoformat(), **value}
        with (events / f"{time.time_ns()}-{uuid4().hex}.json").open("x") as out:
            json.dump(value, out)
        print(json.dumps(value), flush=True)

    upload = client.create_multipart_upload(Bucket=bucket, Key=key,
        ServerSideEncryption="AES256", Metadata={"sha256": expected})["UploadId"]
    event({"stage": "multipart-created", "key": key, "upload_id": upload,
           "bytes": archive.stat().st_size})
    size = 8 * 1024 * 1024

    def part(number, offset):
        with archive.open("rb") as source:
            source.seek(offset)
            body = source.read(size)
        started = time.perf_counter()
        result = client.upload_part(Bucket=bucket, Key=key, UploadId=upload,
            PartNumber=number, Body=body, ContentLength=len(body))
        return {"PartNumber": number, "ETag": result["ETag"]}, len(body), time.perf_counter() - started

    parts, completed = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(part, number, offset) for number, offset in enumerate(
            range(0, archive.stat().st_size, size), 1)]
        for job in as_completed(jobs):
            value, count, elapsed = job.result()
            parts.append(value)
            completed += count
            event({"stage": "part-complete", "part": value["PartNumber"],
                   "part_seconds": elapsed, "completed_bytes": completed,
                   "total_bytes": archive.stat().st_size})
    client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload,
        MultipartUpload={"Parts": sorted(parts, key=lambda item: item["PartNumber"])}, IfNoneMatch="*")
    head = client.head_object(Bucket=bucket, Key=key)
    assert head["ContentLength"] == archive.stat().st_size
    assert head["Metadata"]["sha256"] == expected
    with (directory / "staged-inputs.json").open("x") as out:
        json.dump({"bucket": bucket, "key": key, "sha256": expected,
                   "bytes": archive.stat().st_size}, out, indent=2)
    event({"stage": "transfer-complete", "key": key, "completed_bytes": completed})


if __name__ == "__main__":
    main()
