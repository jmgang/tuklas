"""Download and prepare a new immutable MultiWikiQA bilingual data audit.

No models, training, paid APIs, AWS resources, deletion or overwritten artifacts.
"""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq
import requests
from pycon_model2vec_demo.multiwikiqa import prepare


def json_file(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def json_lines(path, values):
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or ("multiwikiqa-preparation-" +
                             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run ID must be one new directory name")
    config_path = ROOT / "configs/multiwikiqa-data-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    directory = ROOT / "runs" / run_id
    directory.mkdir(exist_ok=False)
    (directory / "raw").mkdir()
    json_file(directory / "settings.json", config)
    json_file(directory / "source-provenance.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "dataset": config["dataset"],
        "revision": config["revision"], "config_sha256": file_hash(config_path),
        "runner_sha256": file_hash(Path(__file__)),
        "module_sha256": file_hash(ROOT / "src/pycon_model2vec_demo/multiwikiqa.py"),
        "no_model_calls": True, "no_training": True, "no_aws_resources": True,
        "existing_experiments_modified": False,
    })
    rows = {}
    for language in config["languages"]:
        spec = config["files"][language]
        url = f'https://huggingface.co/datasets/{config["dataset"]}/resolve/{config["revision"]}/{spec["path"]}'
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        data = response.content
        if len(data) != spec["bytes"] or sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"Pinned source checksum/size mismatch: {language}")
        with (directory / "raw" / f"{language}.parquet").open("xb") as stream:
            stream.write(data)
        rows[language] = pq.read_table(BytesIO(data)).to_pylist()
        print(json.dumps({"stage": "verified-raw-file", "language": language,
                          "rows": len(rows[language]), "bytes": len(data)}), flush=True)
    card_url = f'https://huggingface.co/datasets/{config["dataset"]}/resolve/{config["revision"]}/README.md'
    card = requests.get(card_url, timeout=(15, 90))
    card.raise_for_status()
    with (directory / "raw" / "README.source.md").open("xb") as stream:
        stream.write(card.content)
    result = prepare(rows, config)
    json_lines(directory / "parents.jsonl", result["parents"])
    json_lines(directory / "chunks.candidate.jsonl", result["chunks"])
    json_lines(directory / "annotations.all.jsonl", result["annotations"])
    json_lines(directory / "structural-exclusions.jsonl",
               [query for query in result["annotations"] if query["exclusion_reasons"]])
    for partition in ("development", "evaluation"):
        json_lines(directory / f"queries.{partition}.candidate.jsonl",
                   [query for query in result["candidate_queries"] if query["partition"] == partition])
    json_lines(directory / "semantic-review.pending.jsonl", result["review_queue"])
    json_file(directory / "audit.json", result["audit"])
    counts = result["audit"]["counts"]
    report = ["# MultiWikiQA bilingual preparation — not model results", "",
              f'Revision: `{config["revision"]}`. Licence: {config["license"]}.', "",
              "| Language | Raw QA | Structurally eligible | Development | Evaluation | Source articles |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for language in config["languages"]:
        report.append(f'| {language} | {counts.get("raw_" + language, 0)} | '
                      f'{counts.get("eligible_" + language, 0)} | '
                      f'{counts.get("eligible_development_" + language, 0)} | '
                      f'{counts.get("eligible_evaluation_" + language, 0)} | '
                      f'{counts.get("parents_" + language, 0)} |')
    report += ["", f'Candidate chunks: {result["audit"]["chunks"]}.', "",
               "Every raw row is preserved. Exclusions are recorded, not deleted.",
               "All eligible QA rows remain pending semantic review. Character chunks",
               "require tokenizer/prefix budget validation before any model scores.",
               "No teacher, training corpus, model result, generation score or latency claim exists for this run."]
    with (directory / "preparation-report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    hashes = {str(path.relative_to(directory)): file_hash(path)
              for path in sorted(directory.rglob("*")) if path.is_file()}
    json_file(directory / "manifest.json", {
        "complete": True, "kind": "structural-data-preparation-only",
        "dataset": config["dataset"], "revision": config["revision"],
        "artifacts_sha256": hashes, "audit": result["audit"],
    })
    print(json.dumps({"stage": "prepared", "directory": str(directory),
                      "audit": result["audit"]}), flush=True)


if __name__ == "__main__":
    main()
