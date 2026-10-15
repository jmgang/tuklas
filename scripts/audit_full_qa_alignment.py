"""Read-only audit of all QA judgments against the pinned corpus."""

from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"


def digest(text: str) -> bytes:
    return sha256(text.encode("utf-8")).digest()


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    exact: set[bytes] = set()
    whitespace: set[bytes] = set()
    corpus_rows = 0
    with gzip.open(DATA / "corpus.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            text = json.loads(line)["text"]
            corpus_rows += 1
            exact.add(digest(text))
            whitespace.add(digest(normalize(text)))

    categories: dict[str, dict] = {}
    overall = Counter()
    for qrels_path in sorted(DATA.glob("QA_*_qrels.json.gz")):
        category = qrels_path.name.removeprefix("QA_").removesuffix("_qrels.json.gz")
        with (DATA / f"QA_{category}.json").open(encoding="utf-8") as source:
            published_queries = {row["user_query"] for row in json.load(source)}
        with gzip.open(qrels_path, "rt", encoding="utf-8") as source:
            qrels = json.load(source)

        counts = Counter()
        positives: dict[str, list[int]] = {}
        for row in qrels:
            query = row["user_query"]
            if query not in published_queries:
                counts["orphan_judgments"] += 1
                continue
            counts["judged_pairs"] += 1
            if row["score"] > 0:
                counts["positive_pairs"] += 1
                positives.setdefault(query, [0, 0])[0] += 1
            passage = row["passage"]
            if digest(passage) in exact:
                match_kind = "exact_matches"
            elif digest(normalize(passage)) in whitespace:
                match_kind = "whitespace_matches"
            else:
                match_kind = "unmatched"
            counts[match_kind] += 1
            if row["score"] > 0:
                counts[f"positive_{match_kind}"] += 1
                if match_kind != "unmatched":
                    positives[query][1] += 1
        counts["queries_with_unmatched_positives"] = sum(
            mapped < total for total, mapped in positives.values()
        )
        counts["queries_with_no_mapped_positives"] = sum(
            mapped == 0 for _, mapped in positives.values()
        )
        categories[category] = {"queries": len(published_queries), **counts}
        overall.update(counts)

    print(
        json.dumps(
            {
                "corpus_rows": corpus_rows,
                "unique_exact_corpus_texts": len(exact),
                "categories": categories,
                "total": dict(overall),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
