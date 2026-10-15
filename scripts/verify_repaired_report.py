"""Independent read-only AWS result checks and immutable local report copies."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

NAMES = ("potion_retrieval", "plain_student", "custom_student",
         "tokenlearn_custom_student", "teacher", "cohere_v3")


def validate_report(report, catalog_hash, frozen, dedicated, client_id):
    if set(report["models"]) != set(NAMES) or report["catalog_sha256"] != catalog_hash:
        raise ValueError("Wrong model set or shared catalog")
    reference = None
    for name in NAMES:
        row = report["models"][name]; quality = row["quality"]; timing = row["measurement"]
        if quality["catalog_sha256"] != catalog_hash or quality["summary"]["query_count"] != 1127:
            raise ValueError("Quality provenance/cohort mismatch: " + name)
        queries = quality["per_query"]
        ids = [q["query_id"] for q in queries]
        if len(ids) != 1127 or len(set(ids)) != 1127:
            raise ValueError("Incomplete or duplicate queries: " + name)
        if reference is None:
            reference = ids
        if ids != reference:
            raise ValueError("Models do not share the identical ordered query cohort")
        for query in queries:
            parents = [p["corpus_id"] for p in query["top10"]]
            if len(parents) != 10 or len(set(parents)) != 10:
                raise ValueError("Ranking is not ten unique parents: " + name)
            scores = query["metrics"]
            recall = [scores[k] for k in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10")]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()) or recall != sorted(recall):
                raise ValueError("Invalid retrieval metric: " + name)
        if (timing["model"] != name or timing["expected_measured_requests"] != 160 or
                timing["valid_successful_measurements"] != 160 or timing["failures"] or
                len(timing["samples"]) != 160):
            raise ValueError("Incomplete/failed measurement: " + name)
        identity = timing["server"]
        if name != "cohere_v3":
            if (identity["fingerprint"] != frozen[name]["fingerprint"] or
                    identity["active_models"] != [name] or identity["resource_id"] == client_id):
                raise ValueError("Wrong weights or nonisolated model: " + name)
            if name == "teacher":
                if identity["device"] != "cuda" or not identity.get("gpu_name"):
                    raise ValueError("Teacher not served on GPU")
            elif identity["resource_id"] != dedicated[name]["instance_id"]:
                raise ValueError("Static model was not on its assigned host")
        elif identity["resource_id"] != "bedrock:cohere.embed-english-v3":
            raise ValueError("Wrong managed Cohere model")
        for sample in timing["samples"]:
            elapsed = sample["client_round_trip_seconds"]
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid timing sample")
    if report["cohere_v4"]["status"] != "blocked":
        raise ValueError("Unverified Cohere v4 result")


def main(folder):
    import boto3
    folder = Path(folder); settings = json.loads((folder / "resources.json").read_text())
    session = boto3.Session(profile_name="admin-general", region_name="ap-southeast-1")
    assert session.client("sts").get_caller_identity()["Account"] == settings["account"]
    s3 = session.client("s3"); prefix = settings["prefix"] + "/benchmark/"
    fetched = {}
    def fetch(relative):
        response = s3.get_object(Bucket=settings["bucket"], Key=prefix + relative)
        body = response["Body"].read(); response["Body"].close()
        expected = response["Metadata"].get("sha256")
        if expected != sha256(body).hexdigest():
            raise ValueError("Content checksum mismatch: " + relative)
        fetched[relative] = body
        return body
    report = json.loads(fetch("reports/combined-report.json"))
    catalog_body = fetch("shared-catalog.json"); catalog = json.loads(catalog_body)
    assert catalog["passage_count"] == 239704 and catalog["segment_count"] == 240765
    validate_report(report, sha256(catalog_body).hexdigest(), settings["frozen_models"],
                    settings["dedicated_models"], settings["client_instance"])
    fetch("reports/combined-report.md")
    gate = json.loads(fetch("all-builds-complete.json"))
    assert gate["bulk_and_quality_workers_idle_verified"] is True
    assert set(gate["models"]) == set(NAMES)
    for name in NAMES:
        for artifact in ("quality.json", "measurement.json", "build.json", "isolation-proof.json"):
            raw = fetch(name + "/" + artifact)
            value = json.loads(raw)
            if artifact in ("quality.json", "measurement.json"):
                assert value == report["models"][name][artifact[:-5]], (name, artifact)
            if artifact == "build.json":
                assert sha256(raw).hexdigest() == gate["builds"][name]
                assert value["catalog_sha256"] == report["catalog_sha256"]
                vector_prefix = prefix if name == "cohere_v3" else settings["original_dedicated_prefix"] + "/benchmark/"
                for filename, expected in value["files_sha256"].items():
                    header = s3.head_object(Bucket=settings["bucket"], Key=vector_prefix + name + "/" + filename)
                    assert header["Metadata"]["sha256"] == expected, (name, filename)
            if artifact == "isolation-proof.json":
                assert value["all_bulk_and_quality_workers_idle"] and value["model_hosts_distinct"]
                assert value["build_gate_sha256"] == sha256(fetched["all-builds-complete.json"]).hexdigest()
                assert value["server_identity"] == report["models"][name]["measurement"]["server"]
    source = json.loads(fetch("cohere_v3/restored-chunks-provenance.json"))
    assert source["restored_rows"] == 81216 and len(source["restored_chunk_sha256"]) == 846
    # All checks above precede creating the new local verification directory.
    destination = folder / ("verified-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    destination.mkdir(exist_ok=False)
    for relative, body in fetched.items():
        path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output: output.write(body)
    summary = {"report": str(destination / "reports/combined-report.md"),
        "verified_models": list(NAMES), "held_out_queries": 1127, "parent_count": 239704,
        "segment_count": 240765, "cutoff": 10, "measured_samples_per_model": 160,
        "local_content_sha256": {key: sha256(body).hexdigest() for key, body in fetched.items()},
        "vector_check_note": "S3 recorded checksums cross-checked; AWS quality workers checked vector bytes before scoring"}
    with (destination / "verification.json").open("x") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k != "local_content_sha256"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
