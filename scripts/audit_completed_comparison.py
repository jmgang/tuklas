"""Recompute diagnostic evidence from preserved reports; never run models/AWS.

Output must be a NEW file. Bootstrap intervals are conditional on this fixed
benchmark, with queries sampled independently, not disaster/topic blocks.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


MODELS = ("potion_retrieval", "plain_student", "custom_student",
          "tokenlearn_custom_student", "teacher", "cohere_v3")


def audit(root):
    qualities, measurements, hashes = {}, {}, {}
    for name in MODELS:
        for kind, records in (("quality", qualities), ("measurement", measurements)):
            path = root / name / f"{kind}.json"
            raw = path.read_bytes()
            hashes[str(path)] = sha256(raw).hexdigest()
            records[name] = json.loads(raw)
    query_ids = [row["query_id"] for row in qualities[MODELS[0]]["per_query"]]
    assert len(query_ids) == 1127 and len(set(query_ids)) == 1127
    values, categories = {}, {}
    for name, report in qualities.items():
        by_id = {row["query_id"]: row for row in report["per_query"]}
        assert set(by_id) == set(query_ids)
        assert all(len(row["top10"]) == 10 and
                   len({item["corpus_id"] for item in row["top10"]}) == 10
                   for row in by_id.values())
        values[name] = np.array([by_id[q]["metrics"]["recall_at_10"] for q in query_ids])
        categories[name] = {}
        for category in sorted({row["category"] for row in by_id.values()}):
            selected = [by_id[q]["metrics"]["recall_at_10"] for q in query_ids
                        if by_id[q]["category"] == category]
            categories[name][category] = {"queries": len(selected),
                                         "recall_at_10_percent": float(np.mean(selected) * 100)}
    summaries = {}
    for name, report in measurements.items():
        assert report["valid_successful_measurements"] == 160
        assert not report["failures"]
        timed = [sample for sample in report["samples"] if "encode_seconds" in sample]
        residuals = [(s["client_round_trip_seconds"] - s["encode_seconds"]) * 1000
                     for s in timed]
        summaries[name] = {
            "recall_at_10_percent": float(values[name].mean() * 100),
            "request_p95_ms": report["round_trip"]["p95_ms"],
            "encode_p95_ms": (report.get("server_encode") or {}).get("p95_ms"),
            "non_encode_residual_median_ms": float(np.median(residuals)) if residuals else None,
            "ram_mib": report["server_max_rss_bytes"] / 2**20
                       if report.get("server_max_rss_bytes") is not None else None,
            "samples": report["valid_successful_measurements"],
        }
    comparisons = {}
    for other in ("custom_student", "potion_retrieval", "teacher"):
        difference = values["tokenlearn_custom_student"] - values[other]
        generator = np.random.default_rng(20261003)
        samples = []
        for _ in range(100):
            indices = generator.integers(0, len(difference), size=(100, len(difference)))
            samples.extend(difference[indices].mean(axis=1) * 100)
        comparisons[other] = {
            "tokenlearn_minus_other_percentage_points": float(difference.mean() * 100),
            "paired_bootstrap_95_percent_interval": np.quantile(samples, [.025, .975]).tolist(),
            "wins": int((difference > 0).sum()), "ties": int((difference == 0).sum()),
            "losses": int((difference < 0).sum()),
        }
    example = "0291b8ce4f12580d907808480c1d1f64c3efb47d2a9fa9daec64ad3e85a04865"
    return {
        "source_directory": str(root.resolve()), "source_sha256": hashes,
        "queries": len(query_ids), "summaries": summaries,
        "paired_comparisons": comparisons, "categories": categories,
        "oyster_example": {
            name: next(row for row in qualities[name]["per_query"] if row["query_id"] == example)
            for name in MODELS
        },
        "limitations": [
            "No new inference, training, encoding, retrieval scoring or AWS timing is performed.",
            "Residual is not network time alone: it includes all work outside the encode timer.",
            "Bootstrap treats queries as independent; correlated topics and pooled judgments limit generalization.",
            "Historical latency remains unchanged; Linux/AWS confirmation of the serving patch is pending.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing diagnostic outputs; choose a new path")
    result = audit(args.verified)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summaries": result["summaries"],
                      "paired_comparisons": result["paired_comparisons"]}, indent=2))
