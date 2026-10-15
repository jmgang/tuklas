#!/usr/bin/env python3
"""Remove rejected MultiWikiQA custom-student packages and keep an audit receipt.

Only packaged custom model directories are in scope.  Datasets, teacher and
Potion baselines, feature targets, source bundles, reports, metrics and event
logs are deliberately excluded.  S3 versioning is checked before deletion and
recorded because deleting from an unversioned bucket is permanent.  Local
packages are moved to the macOS Trash so they remain recoverable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import boto3


PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"
BUCKET = "sagemaker-ap-southeast-1-<aws-account-id>"
PREFIX = "pycon-davao-model2vec/"
PROFILE = "admin-general"
REGION = "ap-southeast-1"
RUN_PREFIXES = (
    "pycon-sealion-",
    "pycon-multiwikiqa-",
    "pycon-mqa-",
    "pycon-compact-sota-",
)


def is_rejected_student_key(key: str) -> bool:
    if not key.startswith(PREFIX) or "/results/models/" not in key:
        return False
    run_name = key[len(PREFIX) :].split("/", 1)[0]
    return run_name.startswith(RUN_PREFIXES)


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_dir = RUNS / f"rejected-bilingual-students-{timestamp}"
    receipt_dir.mkdir(parents=False, exist_ok=False)

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    s3 = session.client("s3")
    versioning = s3.get_bucket_versioning(Bucket=BUCKET).get("Status", "Disabled")
    objects: list[dict[str, object]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if is_rejected_student_key(key):
                objects.append(
                    {
                        "key": key,
                        "size": int(item["Size"]),
                        "etag": str(item["ETag"]).strip('"'),
                        "last_modified": item["LastModified"].isoformat(),
                    }
                )
    if not objects:
        raise RuntimeError("no rejected bilingual student objects matched the exact scope")

    local_models: list[dict[str, object]] = []
    for weight in sorted(RUNS.glob("pycon-compact-sota-*/verification-*/model/model.safetensors")):
        model_dir = weight.parent
        local_models.append(
            {
                "path": str(model_dir),
                "bytes": tree_bytes(model_dir),
                "model_sha256": file_sha256(weight),
            }
        )

    inventory = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Rejected: failed bilingual teacher-distance quality gate",
        "quality_gate": {
            "english_recall_at_10_floor": 0.90,
            "maximum_teacher_gap_per_language": 0.10,
            "teacher_english_recall_at_10": 0.9835967193438687,
            "teacher_tagalog_recall_at_10": 0.9698060387922416,
        },
        "preserved": [
            "datasets and annotations",
            "SEA-LION teacher and Potion baseline artifacts",
            "feature targets and vector evidence",
            "source code and immutable input bundles",
            "reports, metrics, manifests and event logs",
            "all DisastIR artifacts (not comparable to the bilingual gate)",
        ],
        "s3": {
            "bucket": BUCKET,
            "versioning": versioning,
            "object_count": len(objects),
            "bytes": sum(int(row["size"]) for row in objects),
            "objects": objects,
        },
        "local": {"model_directory_count": len(local_models), "models": local_models},
    }
    inventory_path = receipt_dir / "pre-deletion-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    deleted: list[dict[str, str]] = []
    for start in range(0, len(objects), 1000):
        batch = objects[start : start + 1000]
        response = s3.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": str(row["key"])} for row in batch], "Quiet": False},
        )
        errors = response.get("Errors", [])
        if errors:
            raise RuntimeError(f"S3 deletion returned errors: {errors}")
        deleted.extend({"key": row["Key"]} for row in response.get("Deleted", []))

    trash_root = Path.home() / ".Trash" / f"pycon-rejected-bilingual-students-{timestamp}"
    trash_root.mkdir(parents=False, exist_ok=False)
    moved_local: list[dict[str, str]] = []
    for index, row in enumerate(local_models, start=1):
        source = Path(str(row["path"]))
        if not source.exists():
            raise RuntimeError(f"local model disappeared before rejection: {source}")
        destination = trash_root / f"{index:02d}-{source.parent.parent.name}-{source.parent.name}-model"
        shutil.move(str(source), str(destination))
        moved_local.append({"source": str(source), "trash": str(destination)})

    remaining_s3 = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for item in page.get("Contents", []):
            if is_rejected_student_key(str(item["Key"])):
                remaining_s3.append(str(item["Key"]))
    remaining_local = [row["path"] for row in local_models if Path(str(row["path"])).exists()]
    if remaining_s3 or remaining_local or len(deleted) != len(objects):
        raise RuntimeError(
            f"post-deletion verification failed: s3={len(remaining_s3)} "
            f"local={len(remaining_local)} deleted={len(deleted)}/{len(objects)}"
        )

    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(inventory_path),
        "s3_versioning": versioning,
        "s3_deleted_objects": len(deleted),
        "s3_deleted_bytes": inventory["s3"]["bytes"],
        "s3_remaining_matching_objects": 0,
        "local_moved_to_trash": moved_local,
        "local_remaining_matching_directories": 0,
        "trash_root": str(trash_root),
    }
    (receipt_dir / "deletion-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
