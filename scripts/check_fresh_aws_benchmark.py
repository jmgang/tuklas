"""Read-only, red/green check for the missing AWS benchmark launch."""
import json
from pathlib import Path
import sys
import boto3
from botocore.config import Config


def main():
    directory = Path(sys.argv[1])
    resources = json.loads((directory / "resources.json").read_text())
    bucket, prefix = resources["s3_prefix"].removeprefix("s3://").split("/", 1)
    session = boto3.Session(profile_name="admin-general", region_name=resources["region"])
    client = session.client("s3", config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 0}))
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix+"benchmark/events/", MaxKeys=1000)
    events = [json.loads(client.get_object(Bucket=bucket, Key=row["Key"])["Body"].read())
              for row in response.get("Contents", [])]
    started = [item for item in events if item.get("stage") == "benchmark-started"]
    assert len(started) == 1, f"FAIL: expected exactly one fresh AWS benchmark launch; found {len(started)}"
    failures = [item for item in events if item.get("stage") == "failed"]
    assert not failures, f"FAIL: benchmark failed: {failures}"
    print("PASS: exactly one fresh AWS benchmark launch", started[0])
    print("LATEST:", max(events, key=lambda item: item["time"]))


if __name__ == "__main__":
    main()
