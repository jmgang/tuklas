"""Read-only check: can released qrels passages be mapped to corpus IDs?"""

from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/raw/corpus.jsonl.gz"
QRELS = ROOT / "data/raw/QA_meteorological_hydrological_qrels.json.gz"


def normalized(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    exact: dict[bytes, str] = {}
    space_normalized: dict[bytes, str] = {}
    duplicate_corpus_texts = 0
    corpus_rows = 0
    with gzip.open(CORPUS, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            corpus_rows += 1
            text = row["text"]
            exact_key = sha256(text.encode("utf-8")).digest()
            normalized_key = sha256(normalized(text).encode("utf-8")).digest()
            if exact_key in exact:
                duplicate_corpus_texts += 1
            exact.setdefault(exact_key, row["corpus_id"])
            space_normalized.setdefault(normalized_key, row["corpus_id"])

    with gzip.open(QRELS, "rt", encoding="utf-8") as source:
        qrels = json.load(source)
    result = Counter()
    positives_by_query: dict[str, list[int]] = {}
    unmatched_examples: list[dict[str, object]] = []
    for row in qrels:
        result["qrels_rows"] += 1
        if row["score"] > 0:
            result["positive_rows"] += 1
            positives_by_query.setdefault(row["user_query"], [0, 0])[0] += 1
        exact_key = sha256(row["passage"].encode("utf-8")).digest()
        normalized_key = sha256(normalized(row["passage"]).encode("utf-8")).digest()
        if exact_key in exact:
            result["exact_matches"] += 1
            if row["score"] > 0:
                result["positive_exact_matches"] += 1
                positives_by_query[row["user_query"]][1] += 1
        elif normalized_key in space_normalized:
            result["whitespace_only_matches"] += 1
            if row["score"] > 0:
                result["positive_whitespace_only_matches"] += 1
                positives_by_query[row["user_query"]][1] += 1
        elif len(unmatched_examples) < 3:
            unmatched_examples.append(
                {"score": row["score"], "query": row["user_query"][:100], "passage": row["passage"][:150]}
            )
    print(
        json.dumps(
            {
                "corpus_rows": corpus_rows,
                "unique_exact_corpus_texts": len(exact),
                "duplicate_corpus_texts": duplicate_corpus_texts,
                **result,
                "queries_with_unmatched_positives": sum(
                    mapped < total for total, mapped in positives_by_query.values()
                ),
                "queries_with_no_mapped_positives": sum(
                    mapped == 0 for _, mapped in positives_by_query.values()
                ),
                "unmatched_examples": unmatched_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
