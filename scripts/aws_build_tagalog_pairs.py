#!/usr/bin/env python3
"""Build Tagalog unsupervised pair sets inside AWS and write them to S3.

The pair payloads are 0.7-1.6 GB each and the builder is 25 seconds of pure
stdlib CPU, so generating them on a workstation and uploading is the wrong way
round: it fills local disk and then pays for a slow upload of bytes AWS could
have produced in place.  This job uploads only the small inputs (the corpus is
already in the bucket as a 35MB gzip and is copied server-side) and writes each
admitted.jsonl straight to S3.

``scripts/build_tagalog_unsupervised_pairs.py`` runs UNCHANGED as a subprocess,
so every guard it carries -- the benchmark-parent exclusion, the Tagalog-only
assertion, the duplicate-query rejection -- applies exactly as reviewed.  It
resolves its inputs relative to its own parents[1], so the three files are
staged into that layout rather than the script being edited.

Determinism is the acceptance test: the builder is seeded, so a build performed
here must reproduce the SHA256 recorded by the same recipe locally.  The job
fails if it does not, which makes a silently different pair file impossible.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

INPUT = Path("/opt/ml/input/data/input")
BUILD = Path("/opt/ml/build")


def main() -> None:
    plan = json.loads((INPUT / "build-plan.json").read_text())
    bucket = plan["bucket"]
    import boto3

    s3 = boto3.client("s3")

    # Recreate the layout build_tagalog_unsupervised_pairs.py expects, so the
    # script itself needs no modification.
    scripts = BUILD / "scripts"
    corpus = BUILD / "runs/multiwikiqa-training-text-20261004-v5"
    preparation = BUILD / "runs/multiwikiqa-preparation-20261004-v1"
    for directory in (scripts, corpus, preparation):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(INPUT / "build_tagalog_unsupervised_pairs.py", scripts)
    shutil.copy(INPUT / "audit.json", corpus / "audit.json")
    shutil.copy(INPUT / "parents.jsonl", preparation / "parents.jsonl")

    # The corpus arrives gzipped (35MB instead of 92MB); the builder reads plain
    # JSONL, so decompress once here.
    source = corpus / "training.tl.jsonl"
    with gzip.open(INPUT / "training.tl.jsonl.gz", "rb") as raw, source.open("wb") as out:
        shutil.copyfileobj(raw, out)
    rows = sum(1 for _ in source.open(encoding="utf-8"))
    if rows != plan["expected_source_rows"]:
        raise ValueError(f"corpus changed: {rows} rows, expected {plan['expected_source_rows']}")
    print(f"corpus ready: {rows} Tagalog passages", flush=True)

    for arm in plan["arms"]:
        name = arm["name"]
        out_dir = BUILD / "runs" / f"pairs-{name}"
        command = [
            sys.executable, str(scripts / "build_tagalog_unsupervised_pairs.py"),
            "--target", str(arm["target"]),
            "--crops-per-passage", str(arm["crops_per_passage"]),
            "--inverse-cloze-per-passage", str(arm["inverse_cloze_per_passage"]),
            "--cross-passage-target", str(arm["cross_passage_target"]),
            "--seed", str(arm["seed"]),
            "--out", str(out_dir),
        ]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

        report = json.loads((out_dir / "verification-report.json").read_text())
        admitted = out_dir / "admitted.jsonl"
        # Acceptance test: the seeded builder must reproduce the digest this arm
        # was planned against.  A mismatch means the corpus, the builder or the
        # arguments drifted, and the recipe's declared admitted_sha256 would no
        # longer describe the file the training job receives.
        expected_digest = arm.get("expected_sha256")
        if expected_digest and report["admitted_sha256"] != expected_digest:
            raise ValueError(
                f"{name}: built digest {report['admitted_sha256']} != planned "
                f"{expected_digest}; refusing to publish a pair set the "
                "recipe does not describe"
            )
        expected_pairs = arm.get("expected_pairs")
        if expected_pairs and report["pairs"] != expected_pairs:
            raise ValueError(f"{name}: {report['pairs']} pairs, expected {expected_pairs}")
        if expected_digest:
            # This arm was built once on a workstation, so a match here proves
            # the seeded builder is reproducible in the container and that the
            # recipe's declared digest still describes the published bytes.
            print(f"{name}: {report['pairs']} pairs, digest REPRODUCED", flush=True)
        else:
            # A new arm has no prior digest; record it so the recipe can be
            # written against the bytes that were actually published.
            print(f"{name}: {report['pairs']} pairs, digest "
                  f"{report['admitted_sha256']} (new arm)", flush=True)

        prefix = f"{plan['pair_sets_prefix']}{name}/"
        s3.upload_file(str(admitted), bucket, prefix + "admitted.jsonl")
        s3.upload_file(str(out_dir / "verification-report.json"), bucket,
                       prefix + "verification-report.json")
        print("UPLOADED", prefix, report["admitted_bytes"], "bytes", flush=True)
        # Free the container's disk before the next, larger arm.
        admitted.unlink()

    print("ALL ARMS BUILT", flush=True)


if __name__ == "__main__":
    main()
