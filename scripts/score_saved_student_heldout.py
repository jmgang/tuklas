#!/usr/bin/env python3
"""Score a preserved int8 student package on dev and held-out, locally on CPU.

A candidate that passes both promotion gates reaches the final report, where a
pre-existing scope bug (``mining_manifest`` was local to the mining function)
raised before the held-out numbers were persisted.  The package itself is in S3,
so the measurement is recoverable without any GPU or retraining.

Query handling mirrors the worker exactly: the ``[QRY]`` role token is prepended
to every question, padding and unknown ids are stripped, token vectors are mean
pooled and L2 normalised.  Ranking and scoring reuse
``multiwikiqa_evaluation.rank_parents`` and ``score`` so the numbers are
comparable with every other figure in this project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.multiwikiqa_compact_retrieval import strip_padding_and_unknown  # noqa: E402
from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score  # noqa: E402
from pycon_model2vec_demo.multiwikiqa_ranking_v5 import prepare  # noqa: E402

# Frozen teacher reference for the 3,984-question Tagalog held-out cohort, from
# docs/full-heldout-results-20261005.md.
TEACHER_HELDOUT_RECALL_AT_10 = 0.9694
TEACHER_DEVELOPMENT_RECALL_AT_10 = 0.9746192893401015
METRICS = (
    "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
    "AnswerSpanCoverage@1", "AnswerSpanCoverage@3",
    "AnswerSpanCoverage@5", "AnswerSpanCoverage@10",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="local int8 package directory")
    parser.add_argument("--label", default="student")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    from model2vec import StaticModel
    from model2vec.train.utils import get_probable_pad_token_id

    student = StaticModel.from_pretrained(str(arguments.model))
    weights = np.asarray(student.embedding, dtype=np.float32)
    tokenizer = student.tokenizer
    config = json.loads((arguments.model / "config.json").read_text())
    query_prefix = config.get("query_prefix", "")
    pad_id = get_probable_pad_token_id(tokenizer)
    unknown_id = student.unk_token_id

    layout_path = ROOT / "runs/pycon-compact-sota-20261005-055331-d48482/verification-20261005T090654Z/evaluation/shared-layout.json"
    annotations_path = layout_path.parent / "annotations.all.jsonl"
    layout = json.loads(layout_path.read_text())
    annotations = read_jsonl(annotations_path)
    chunks = layout["chunks"]
    parents = layout["parents"]
    positions = {parent: index for index, parent in enumerate(parents)}
    owners = [positions[chunk["parent_id"]] for chunk in chunks]

    tagalog = [row for row in annotations if row["language"] == "tl"]
    _, development, _ = prepare(tagalog, chunks)
    heldout = [row for row in tagalog if row["partition"] == "evaluation"]
    if len(development) != 197 or len(heldout) != 3984:
        raise ValueError(f"cohorts changed: dev={len(development)} heldout={len(heldout)}")

    def encode(texts: list[str], *, query: bool) -> np.ndarray:
        values = [(query_prefix + text) if query else text for text in texts]
        out = np.zeros((len(values), weights.shape[1]), dtype=np.float32)
        for start in range(0, len(values), 1024):
            batch = values[start : start + 1024]
            encodings = tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            for offset, encoding in enumerate(encodings):
                ids = strip_padding_and_unknown(
                    encoding.ids, pad_id=pad_id, unknown_id=unknown_id
                )
                if not ids:
                    raise ValueError("empty token sequence")
                out[start + offset] = weights[ids].mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    print(f"encoding {len(chunks)} documents ...", flush=True)
    document_vectors = encode([row["text"] for row in chunks], query=False)

    def evaluate(cohort: list[dict], name: str) -> dict:
        print(f"encoding {len(cohort)} {name} questions ...", flush=True)
        query_vectors = encode([row["question"] for row in cohort], query=True)
        rows = []
        for start in range(0, len(cohort), 256):
            rankings = rank_parents(
                query_vectors[start : start + 256], document_vectors, owners, parents
            )
            for query, ranking in zip(cohort[start : start + 256], rankings, strict=True):
                rows.append(score(query, ranking, chunks))
        result = {"questions": len(rows)}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        return result

    development_metrics = evaluate(development, "development")
    heldout_metrics = evaluate(heldout, "held-out")

    report = {
        "label": arguments.label,
        "model_directory": str(arguments.model),
        "dimension": int(weights.shape[1]),
        "vocabulary_rows": int(weights.shape[0]),
        "package_bytes": sum(p.stat().st_size for p in arguments.model.rglob("*") if p.is_file()),
        "development": development_metrics,
        "heldout": heldout_metrics,
        "teacher_development_recall_at_10": TEACHER_DEVELOPMENT_RECALL_AT_10,
        "teacher_heldout_recall_at_10": TEACHER_HELDOUT_RECALL_AT_10,
        "development_gap": TEACHER_DEVELOPMENT_RECALL_AT_10 - development_metrics["Recall@10"],
        "heldout_gap": TEACHER_HELDOUT_RECALL_AT_10 - heldout_metrics["Recall@10"],
        "development_minus_heldout": development_metrics["Recall@10"] - heldout_metrics["Recall@10"],
        "gate_maximum_gap": 0.10,
        "development_gate_passed": (
            TEACHER_DEVELOPMENT_RECALL_AT_10 - development_metrics["Recall@10"] <= 0.10
        ),
        "heldout_gate_passed": (
            TEACHER_HELDOUT_RECALL_AT_10 - heldout_metrics["Recall@10"] <= 0.10
        ),
    }
    out = arguments.out or (arguments.model.parent / f"{arguments.label}-dev-and-heldout.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
