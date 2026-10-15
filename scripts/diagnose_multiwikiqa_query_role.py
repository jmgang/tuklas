"""Ablate query text prefixes on the frozen approved-article student.

This is a quality diagnosis, not a latency benchmark and not a new training
run.  It uses the exact development-validation questions, shared chunks and
parent-ranking protocol from the frozen teacher-margin experiment.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from model2vec import StaticModel

from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score
from pycon_model2vec_demo.multiwikiqa_ranking_v5 import prepare


EXPECTED_MODEL_SHA256 = "ca6538b51f0f6affd6a67be4703595ef32cf798e598558f873dc350cabb625a7"
TEACHER_QUERY_PROMPT = (
    "Instruct: Given a passage that is guaranteed to contain the answer, "
    "retrieve relevant passages that answer the query.\nQuery: "
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source]


def summarize(rows: list[dict]) -> dict:
    output = {}
    for language in ("en", "tl"):
        group = [row for row in rows if row["language"] == language]
        output[language] = {
            "questions": len(group),
            **{
                metric: float(np.mean([row["metrics"][metric] for row in group]))
                for metric in ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")
            },
        }
    output["combined"] = {
        "questions": len(rows),
        **{
            metric: float(np.mean([row["metrics"][metric] for row in rows]))
            for metric in ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.model / "model.safetensors") != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen approved-student weights changed")

    layout = json.loads(args.layout.read_text())
    chunks = layout["chunks"]
    parent_ids = layout["parents"]
    annotations = read_jsonl(args.annotations)
    _, validation, _ = prepare(annotations, chunks)
    if len(validation) != 338 or {row["partition"] for row in validation} != {"development"}:
        raise ValueError("development-validation cohort changed")

    parent_index = {parent_id: index for index, parent_id in enumerate(parent_ids)}
    owners = [parent_index[chunk["parent_id"]] for chunk in chunks]
    if owners != sorted(owners):
        raise ValueError("shared chunk ownership order changed")

    model = StaticModel.from_pretrained(args.model)
    documents = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=512,
        max_length=None,
        use_multiprocessing=False,
    )
    variants = {
        "unprefixed_frozen_baseline": "",
        "short_query_label": "query: ",
        "teacher_retrieval_instruction": TEACHER_QUERY_PROMPT,
    }
    reports = {}
    for name, prefix in variants.items():
        vectors = model.encode(
            [prefix + row["question"] for row in validation],
            batch_size=512,
            max_length=None,
            use_multiprocessing=False,
        )
        rankings = rank_parents(vectors, documents, owners, parent_ids)
        rows = []
        for query, ranking in zip(validation, rankings):
            rows.append(
                {
                    "query_id": query["query_id"],
                    "language": query["language"],
                    "gold_parent_id": query["gold_parent_id"],
                    "metrics": score(query, ranking, chunks),
                    "top10_parent_ids": [result[0] for result in ranking],
                }
            )
        reports[name] = {"prefix": prefix, "summary": summarize(rows), "per_query": rows}

    payload = {
        "purpose": "query-role prefix ablation on one frozen model; no training and no latency measurement",
        "model_weights_sha256": EXPECTED_MODEL_SHA256,
        "shared_layout_sha256": digest(args.layout),
        "validation_questions": len(validation),
        "documents": len(parent_ids),
        "chunks": len(chunks),
        "reports": reports,
        "interpretation_guard": (
            "An untrained textual prefix is not equivalent to the proposed atomic trainable [QRY] token. "
            "This ablation only tests whether the frozen model already supports textual query instructions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(payload, destination, ensure_ascii=False, sort_keys=True, indent=2)
        destination.write("\n")
    print(json.dumps({"output": str(args.output), "summaries": {k: v["summary"] for k, v in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
