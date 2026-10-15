"""Audit all published QA judgments against stable corpus IDs, without scoring models."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection


def main() -> None:
    config = json.loads((ROOT / "configs/disastir-qa-all-categories.json").read_text())
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=config["corpus_sha256"])
    prepared = load_qa_collection(
        data_dir=ROOT / "data/raw", revision=config["revision"],
        category_checksums=config["categories"], seed=config["split_seed"],
        evaluation_fraction=config["evaluation_fraction"],
    )
    _, report = map_query_groups(prepared.development + prepared.evaluation, corpus)
    content = {
        "status": "coverage-only; no model retrieval or held-out scoring",
        "dataset_revision": config["revision"],
        "corpus_sha256": config["corpus_sha256"],
        "corpus_passages": len(corpus.ids),
        "corpus_duplicate_texts": corpus.duplicate_text_count,
        "raw_judgments": prepared.manifest["raw_judgment_count"],
        "orphan_judgments_excluded": prepared.manifest["excluded_orphan_judgment_count"],
        "mapping": report,
    }
    target = ROOT / "runs/corpus-mapping-coverage.json"
    with target.open("x", encoding="utf-8") as output:
        json.dump(content, output, indent=2, sort_keys=True)
        output.write("\n")
    print(target)


if __name__ == "__main__":
    main()
