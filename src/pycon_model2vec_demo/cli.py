"""Small CLI for reproducible dataset preparation."""

import argparse
import json
from pathlib import Path

from .data import load_disastir_slice, load_qa_collection, source_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a pinned DisastIR judged-candidate slice")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if config.get("dataset") != "DMIR01/DisastIR":
        parser.error("Config must name DMIR01/DisastIR")
    if "categories" in config:
        prepared = load_qa_collection(
            data_dir=args.data_dir,
            revision=config["revision"],
            category_checksums=config["categories"],
            seed=config["split_seed"],
            evaluation_fraction=config["evaluation_fraction"],
        )
    else:
        task = config["task"]
        category = config["category"]
        prepared = load_disastir_slice(
            query_path=source_path(args.data_dir, f"{task}_{category}.json"),
            qrels_path=source_path(args.data_dir, f"{task}_{category}_qrels.json"),
            revision=config["revision"],
            task=task,
            category=category,
            seed=config["split_seed"],
            evaluation_fraction=config["evaluation_fraction"],
            expected_query_sha256=config["query_sha256"],
            expected_qrels_sha256=config["qrels_sha256"],
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as destination:
        json.dump(prepared.manifest, destination, indent=2)
        destination.write("\n")
    print(
        f"Prepared {prepared.manifest['query_count']} queries and "
        f"{prepared.manifest['judged_pair_count']} judged pairs: "
        f"{len(prepared.development)} development / {len(prepared.evaluation)} evaluation."
    )
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
