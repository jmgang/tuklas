"""Build the frozen custom vocabulary from released corpus text only."""

from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.data import _file_sha256
from pycon_model2vec_demo.vocabulary import iter_released_passages, load_stopwords, select_vocabulary


def main() -> None:
    dataset_config = json.loads((ROOT / "configs/disastir-qa-all-categories.json").read_text())
    model_config = json.loads((ROOT / "configs/models-v1.json").read_text())
    rule = model_config["students"]["custom_vocabulary"]
    corpus_path = ROOT / "data/raw/corpus.jsonl.gz"
    corpus_hash = _file_sha256(corpus_path)
    if corpus_hash != dataset_config["corpus_sha256"]:
        raise ValueError("Corpus differs from pinned DisastIR release")
    stopword_path = ROOT / "configs/stopwords-en-v1.txt"
    stopwords, stopword_sha256 = load_stopwords(stopword_path)
    selected, counts = select_vocabulary(
        iter_released_passages(corpus_path),
        stopwords=stopwords,
        ngram_lengths=tuple(rule["ngram_lengths"]),
        min_passage_frequency=rule["min_passage_frequency"],
        candidate_cap=rule["candidate_cap"],
    )
    if counts["passage_count"] != 239704 or len(selected) != rule["candidate_cap"]:
        raise RuntimeError("Corpus size or selected vocabulary count is unexpected")
    content = {
        "status": "frozen-before-held-out-evaluation",
        "source": "released-unlabeled-passage-text-only",
        "dataset_revision": dataset_config["revision"],
        "corpus_sha256": corpus_hash,
        "stopword_list": stopword_path.relative_to(ROOT).as_posix(),
        "stopword_list_sha256": stopword_sha256,
        "normalization": "NFKC + casefold; alphabetic tokens >=2 characters",
        "term_counts": counts,
        "rule": rule,
        "requested_terms": [{"term": term, "passage_frequency": count} for term, count in selected],
    }
    vocabulary_bytes = json.dumps(content["requested_terms"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    content["requested_terms_sha256"] = sha256(vocabulary_bytes).hexdigest()
    target = ROOT / "runs/custom-vocabulary-v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump(content, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(target)


if __name__ == "__main__":
    main()
