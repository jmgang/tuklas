"""Recovery workers: immutable copies, checked Cohere prefix, unchanged scoring.

Run on restored AWS hosts only. Never writes into the preserved output trees.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def validated_prefix(values, batch_size=96):
    """Return only complete normalized batches; verify the entire zero tail."""
    import numpy as np
    if values.ndim != 2 or values.shape[1] != 1024 or values.dtype != np.float32:
        raise ValueError("Unexpected preserved Cohere vector format")
    boundary = len(values)
    found = False
    for start in range(0, len(values), 4096):
        block = np.asarray(values[start:start + 4096])
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite preserved vectors")
        norms = np.linalg.norm(block, axis=1)
        zeros = np.flatnonzero(norms == 0)
        if not found and len(zeros):
            boundary = start + int(zeros[0]); found = True
        absolute = np.arange(start, start + len(block))
        complete = absolute < boundary
        if not np.allclose(norms[complete], 1, atol=1e-4):
            raise ValueError("Preserved prefix is not normalized")
        if found and np.any(norms[absolute >= boundary] != 0):
            raise ValueError("Noncontiguous preserved output; do not infer a prefix")
    return boundary if boundary == len(values) else boundary // batch_size * batch_size


def cohere_replay_check(replay, preserved, policy=None):
    """Check provider replay with a documented numerical tolerance and row order."""
    import numpy as np
    if replay.shape != preserved.shape or replay.ndim != 2:
        raise ValueError("Provider replay shape changed")
    if not np.isfinite(replay).all() or not np.isfinite(preserved).all():
        raise ValueError("Nonfinite provider replay")
    cosine = np.sum(replay * preserved, axis=1)
    if policy is None:
        if float(cosine.min()) <= .99999:
            raise ValueError("Provider/preserved-vector parity changed")
        return {"minimum_cosine": float(cosine.min()), "policy": "legacy-strict"}
    minimum = float(policy["minimum_cosine"])
    max_difference = float(policy["maximum_abs_difference"])
    assert minimum >= .999 and max_difference <= .01
    assert policy["reason"] == "Repeated identical provider requests vary numerically"
    difference = float(np.max(np.abs(replay - preserved)))
    matrix = replay @ preserved.T
    aligned = bool(np.all(matrix.max(axis=1) <= cosine + 1e-5))
    if float(cosine.min()) < minimum or difference > max_difference or not aligned:
        raise ValueError("Provider replay exceeds tolerance or row alignment changed")
    return {"minimum_cosine": float(cosine.min()), "mean_cosine": float(cosine.mean()),
            "maximum_abs_difference": difference, "row_alignment_verified": aligned,
            "policy": policy}


def validate_saved_chunk_range(record, begin, total):
    """Validate an immutable batch journal; never infer completeness from counts."""
    start, stop = record.get("start"), record.get("stop")
    checksum = record.get("sha256", "")
    if (type(start) is not int or type(stop) is not int or
            not begin <= start < total or (start - begin) % 96 or
            stop != min(start + 96, total) or not isinstance(checksum, str) or
            len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)):
        raise ValueError("Invalid saved chunk range or checksum")
    return start, stop


def remaining_chunk_starts(begin, total, saved_starts):
    allowed = set(range(begin, total, 96))
    if not set(saved_starts) <= allowed:
        raise ValueError("Saved chunks are outside this frozen layout")
    return [start for start in range(begin, total, 96) if start not in saved_starts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--phase", choices=["quality", "cohere", "measure"], required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    code = Path("/opt/pycon-isolated/code")
    os.environ["PYCON_FRESH_ROOT"] = "/opt/pycon-isolated/input"
    os.environ["PYCON_RESULT_BUCKET"] = settings["bucket"]
    os.environ["PYCON_RESULT_PREFIX"] = settings["prefix"] + "/benchmark"
    sys.path.insert(0, str(code / "scripts"))
    sys.path.insert(0, str(code / "src"))
    import numpy as np
    import boto3
    import run_isolated_aws as worker
    import run_fresh_aws_benchmark as shared

    directory = Path(settings["output_directory"])
    directory.mkdir(exist_ok=True)
    s3 = boto3.client("s3", region_name="ap-southeast-1")
    original_publish = worker.publish
    def publish_identical_or_new(output, path, config):
        key = config["prefix"] + "/benchmark/" + path.relative_to(output).as_posix()
        expected = shared.file_digest(path)
        try:
            response = s3.head_object(Bucket=config["bucket"], Key=key)
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
            try:
                return original_publish(output, path, config)
            except Exception as conflict:
                if getattr(conflict, "response", {}).get("Error", {}).get("Code") != "PreconditionFailed":
                    raise
                response = s3.head_object(Bucket=config["bucket"], Key=key)
        assert response["Metadata"]["sha256"] == expected and response["ContentLength"] == path.stat().st_size, key
    worker.publish = publish_identical_or_new
    for relative, expected in settings["code_sha256"].items():
        assert shared.file_digest(code / relative) == expected, relative

    def copy_from(prefix, relative, destination=None):
        response = s3.get_object(Bucket=settings["bucket"], Key=prefix + "/benchmark/" + relative)
        expected = response["Metadata"]["sha256"]
        path = directory / (destination or relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert shared.file_digest(path) == expected, relative
            response["Body"].close()
            return path
        with path.open("xb") as out:
            for block in iter(lambda: response["Body"].read(1024 * 1024), b""):
                out.write(block)
        response["Body"].close()
        assert shared.file_digest(path) == expected, relative
        worker.publish(directory, path, settings)
        return path

    source = settings["original_dedicated_prefix"]
    for file in ("shared-catalog.json", "shared-catalog.sha256.json", "timing-queries.json"):
        copy_from(source, file)
    catalog = shared.read(directory / "shared-catalog.json")
    assert catalog["passage_count"] == 239704 and catalog["segment_count"] == 240765
    if args.phase == "quality":
        for file in ("build.json", "segment-vectors.npy", "queries-vectors.npy", "development-vectors.npy"):
            copy_from(source, args.model + "/" + file)
        worker.quality(directory, args.model, settings)
        return
    if args.phase == "measure":
        copy_from(settings["prefix"], "all-builds-complete.json")
        if args.model != "cohere_v3":
            for file in ("build.json", "segment-vectors.npy", "queries-vectors.npy", "development-vectors.npy"):
                copy_from(source, args.model + "/" + file)
        worker.measure(directory, args.model, settings)
        return

    assert args.model == "cohere_v3"
    actual_private_ips = {x[4][0] for x in socket.getaddrinfo("bedrock-runtime.ap-southeast-1.amazonaws.com", 443, family=socket.AF_INET)}
    assert actual_private_ips and actual_private_ips <= set(settings["bedrock_private_ips"])
    corpus, queries, dev, _, _ = shared.prepare()
    assert catalog["corpus_texts_sha256"] == shared.digest(corpus.texts)
    assert catalog["query_texts_sha256"] == shared.digest([q.query for q in queries])
    preserved = Path("/opt/pycon-isolated/dedicated-benchmark/cohere_v3/segment-vectors.npy")
    original = np.load(preserved, mmap_mode="r")
    assert original.shape == (240765, 1024)
    begin = validated_prefix(original)
    model_dir = directory / "cohere_v3"
    model_dir.mkdir(exist_ok=False)
    chunks = model_dir / "chunks"; chunks.mkdir(exist_ok=False)
    _, bedrock = worker.session_clients()
    enc = worker.CohereEncoder(bedrock)
    # Three fixed batches verify restored vectors still match this provider and
    # the unchanged catalog. These are not selected by relevance or performance.
    checks = []
    replay_policy = settings.get("provider_replay_policy")
    if replay_policy:
        evidence = shared.read(Path(settings["provider_replay_evidence_path"]))
        assert shared.file_digest(Path(settings["provider_replay_evidence_path"])) == settings["provider_replay_evidence_sha256"]
        repeated = evidence["identical_fresh_requests"]
        assert repeated["all_rows_match_own_index"] and repeated["minimum_cosine"] >= .999
        assert not repeated["strict_99999_passes"], "Evidence must reproduce the strict-check failure on identical requests"
        assert evidence["catalog_sha256"] == shared.file_digest(directory / "shared-catalog.json")
    if begin >= 96:
        for start in sorted({0, (begin // 192) * 96, begin - 96}):
            texts = [corpus.texts[o][a:b] for o, a, b in catalog["segments"][start:start + 96]]
            replay, metadata = enc.encode(texts, "corpus")
            comparison = cohere_replay_check(replay, original[start:start + 96], replay_policy)
            checks.append({"start": start, **comparison, "request": metadata})
    seed = model_dir / "preserved-prefix.npy"
    with seed.open("xb") as out:
        np.save(out, original[:begin], allow_pickle=False)
    shared.save(model_dir / "resume-provenance.json", {
        "preserved_source": str(preserved), "source_snapshot_id": settings["cohere_snapshot_id"],
        "source_file_sha256": shared.file_digest(preserved), "validated_preserved_rows": begin,
        "validated_prefix_sha256": shared.file_digest(seed), "catalog_sha256": shared.file_digest(directory / "shared-catalog.json"),
        "parity_checks": checks, "provider_private_ips": sorted(actual_private_ips),
        "dataset_changed": False, "original_file_modified": False, "truncate": "NONE",
        "new_batch_concurrency": 4, "token_budget_per_minute": 285000,
        "provider_replay_evidence_sha256": settings.get("provider_replay_evidence_sha256"),
    })
    worker.publish(directory, model_dir / "resume-provenance.json", settings)
    worker.publish(directory, seed, settings)
    restored_starts = set(); restored_rows = 0; restored_requests = []; restored_checksums = {}
    chunk_source = settings.get("resume_chunk_prefix")
    if chunk_source:
        # Recovery requires a NEW output directory and prefix. Failed output is
        # read-only, including out-of-order batches; no batch count implies a prefix.
        assert chunk_source != settings["prefix"]
        proof_path = copy_from(chunk_source, "cohere_v3/resume-provenance.json",
                               "cohere_v3/restored-source-provenance.json")
        proof = shared.read(proof_path)
        assert proof["catalog_sha256"] == shared.file_digest(directory / "shared-catalog.json")
        assert proof["validated_preserved_rows"] == begin
        assert proof["validated_prefix_sha256"] == shared.file_digest(seed)
        assert proof["provider_replay_evidence_sha256"] == settings.get("provider_replay_evidence_sha256")
        source_path = chunk_source + "/benchmark/cohere_v3/chunks/"
        keys = [obj["Key"] for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=settings["bucket"], Prefix=source_path) for obj in page.get("Contents", [])
            if obj["Key"].endswith(".json")]
        for key in sorted(keys):
            relative = "cohere_v3/chunks/" + key.rsplit("/", 1)[1]
            journal = copy_from(chunk_source, relative)
            record = shared.read(journal)
            start, stop = validate_saved_chunk_range(record, begin, len(original))
            assert journal.name == f"{start:09d}-{stop:09d}.json"
            assert start not in restored_starts
            part = copy_from(chunk_source, relative[:-5] + ".npy")
            assert shared.file_digest(part) == record["sha256"]
            saved = np.load(part, allow_pickle=False)
            assert saved.shape == (stop - start, 1024) and saved.dtype == np.float32
            assert np.isfinite(saved).all() and np.allclose(np.linalg.norm(saved, axis=1), 1, atol=1e-4)
            restored_starts.add(start); restored_rows += stop - start
            restored_requests.append(record["request"])
            restored_checksums[part.name] = record["sha256"]
        restored_proof = model_dir / "restored-chunks-provenance.json"
        shared.save(restored_proof, {"source_prefix": chunk_source,
            "source_provenance_sha256": shared.file_digest(proof_path),
            "catalog_sha256": proof["catalog_sha256"], "restored_rows": restored_rows,
            "restored_chunk_sha256": restored_checksums, "source_objects_modified": False})
        worker.publish(directory, restored_proof, settings)
    shared.event(directory, stage="cohere-resume-verified", model="cohere_v3",
                 completed_segments=begin + restored_rows, total_segments=240765)
    started = time.perf_counter(); requests = []; completed = begin + restored_rows; token_count = 0

    def encode_batch(start):
        stop = min(start + 96, len(original))
        texts = [corpus.texts[o][a:b] for o, a, b in catalog["segments"][start:stop]]
        encoded, metadata = enc.encode(texts, "corpus")
        path = chunks / f"{start:09d}-{stop:09d}.npy"
        with path.open("xb") as out:
            np.save(out, encoded, allow_pickle=False)
        checksum = shared.file_digest(path)
        journal = path.with_suffix(".json")
        shared.save(journal, {"start": start, "stop": stop, "sha256": checksum, "request": metadata})
        worker.publish(directory, path, settings)
        worker.publish(directory, journal, settings)
        return start, stop, metadata

    with ThreadPoolExecutor(max_workers=4) as pool:
        # Bounded outstanding tasks, shared token reservation, maximum 96 texts.
        from concurrent.futures import wait, FIRST_COMPLETED
        starts = iter(remaining_chunk_starts(begin, len(original), restored_starts)); pending = {}
        def submit_one():
            start = next(starts, None)
            if start is not None:
                pending[pool.submit(encode_batch, start)] = start
        for _ in range(4): submit_one()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                start, stop, metadata = future.result()
                completed += stop - start; requests.append(metadata)
                token_count += metadata["input_tokens"] or int(metadata["characters"] / 3)
                if token_count / 1e6 * .1 > 15:
                    raise RuntimeError("Cohere recovery exceeded the $15 API estimate guard")
                if len(requests) % 20 == 0 or completed == len(original):
                    shared.event(directory, stage="cohere-resume-build", model="cohere_v3", completed_segments=completed,
                        total_segments=len(original), remaining_segments=len(original)-completed,
                        newly_billed_tokens=token_count, elapsed_seconds=time.perf_counter()-started)
                submit_one()
    path = model_dir / "segment-vectors.npy"
    assert not path.exists()
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=original.shape)
    values[:begin] = original[:begin]
    for start in range(begin, len(original), 96):
        stop = min(start + 96, len(original)); part = chunks / f"{start:09d}-{stop:09d}.npy"
        record = shared.read(part.with_suffix(".json"))
        assert shared.file_digest(part) == record["sha256"]
        values[start:stop] = np.load(part, allow_pickle=False)
    values.flush()
    assert validated_prefix(values) == len(values)
    for label, group in (("queries", queries), ("development", dev)):
        encoded = [enc.encode([q.query for q in group[a:a + 96]], label)[0] for a in range(0, len(group), 96)]
        with (model_dir / f"{label}-vectors.npy").open("xb") as out:
            np.save(out, np.concatenate(encoded), allow_pickle=False)
    files = {p.name: shared.file_digest(p) for p in model_dir.glob("*-vectors.npy")}
    shared.save(model_dir / "build.json", {"model": "cohere_v3", "identity": enc.identity,
        "catalog_sha256": shared.file_digest(directory / "shared-catalog.json"), "files_sha256": files,
        "vector_bytes": path.stat().st_size, "fresh_rows": len(original)-begin-restored_rows,
        "preserved_same_experiment_rows": begin, "restored_same_experiment_chunk_rows": restored_rows,
        "restored_bulk_requests": restored_requests, "newly_billed_tokens": token_count,
        "build_seconds": time.perf_counter()-started, "bulk_requests": requests,
        "original_same_experiment_vectors_reused_after_parity_verification": True})
    for name in (*files, "build.json"):
        worker.publish(directory, model_dir / name, settings)
    shared.event(directory, stage="cohere-resume-build-complete", model="cohere_v3", completed_segments=len(original))
    # Quality search finishes before the controller permits latency measurement.
    worker.quality(directory, "cohere_v3", settings)


if __name__ == "__main__":
    main()
