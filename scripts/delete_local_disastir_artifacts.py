#!/usr/bin/env python3
"""Permanently delete the exact obsolete local DisastIR artifact set.

The target list is deliberately explicit.  It excludes shared baseline models,
MultiWikiQA/SEA-LION data, source code, S3 objects and unrelated runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "runs"
TARGETS = (
    "disastir-ranking-repair-20261003-v1",
    "disastir-ranking-repair-20261003-v2",
    "disastir-512-rebuild-20261003-v2",
    "disastir-512-baseline-20261003-v1",
    "disastir-512-improved-20261003-v1",
    "parent-segment-grade3-d94014294ee7",
    "embeddings",
    "measurement-caches",
)


def measure(root: Path) -> tuple[int, int, int]:
    logical = allocated = files = 0
    for item in root.rglob("*"):
        if item.is_file() and not item.is_symlink():
            stat = item.stat()
            logical += stat.st_size
            allocated += stat.st_blocks * 512
            files += 1
    return logical, allocated, files


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_dir = RUNS / f"deleted-local-disastir-artifacts-{timestamp}"
    receipt_dir.mkdir(parents=False, exist_ok=False)

    inventory = []
    resolved_runs = RUNS.resolve()
    for name in TARGETS:
        path = RUNS / name
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"expected ordinary target directory is missing: {path}")
        resolved = path.resolve()
        if resolved.parent != resolved_runs:
            raise RuntimeError(f"target escaped the runs directory: {resolved}")
        logical, allocated, files = measure(path)
        inventory.append(
            {
                "path": str(path),
                "logical_bytes": logical,
                "allocated_bytes": allocated,
                "file_count": files,
            }
        )

    pre = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "obsolete local DisastIR model runs, segment vectors and measurement caches",
        "recoverable": False,
        "targets": inventory,
        "total_logical_bytes": sum(row["logical_bytes"] for row in inventory),
        "total_allocated_bytes": sum(row["allocated_bytes"] for row in inventory),
        "preserved": [
            "shared models/ directory including Potion baseline",
            "MultiWikiQA and SEA-LION runs/data/features",
            "source code, tests, specifications and documentation",
            "all S3 objects",
            "other local DisastIR reports/runs not named in the exact target list",
        ],
    }
    (receipt_dir / "pre-deletion-inventory.json").write_text(
        json.dumps(pre, indent=2, sort_keys=True) + "\n"
    )

    for row in inventory:
        shutil.rmtree(row["path"])

    remaining = [row["path"] for row in inventory if Path(row["path"]).exists()]
    if remaining:
        raise RuntimeError(f"post-deletion verification failed: {remaining}")
    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deleted_directories": len(inventory),
        "deleted_files": sum(row["file_count"] for row in inventory),
        "deleted_logical_bytes": pre["total_logical_bytes"],
        "freed_allocated_bytes": pre["total_allocated_bytes"],
        "verified_absent": True,
        "recoverable": False,
    }
    (receipt_dir / "deletion-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
