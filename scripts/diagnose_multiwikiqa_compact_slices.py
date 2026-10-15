"""Measure untrained prefix/quantization behavior of the frozen student.

The result is diagnostic only: the frozen model was not trained with a
Matryoshka objective, so these values must not be presented as the final compact
student.  No latency is measured and no model artifact is written.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from model2vec import StaticModel, quantize_model

from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score
from pycon_model2vec_demo.multiwikiqa_ranking_v5 import prepare


EXPECTED_MODEL_SHA256 = "ca6538b51f0f6affd6a67be4703595ef32cf798e598558f873dc350cabb625a7"
DIMENSIONS = (1024, 768, 512, 480, 448, 384, 256)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source]


def metrics(queries, query_vectors, document_vectors, owners, parent_ids, chunks):
    rankings = rank_parents(query_vectors, document_vectors, owners, parent_ids)
    rows = [
        {"language": query["language"], "metrics": score(query, ranking, chunks)}
        for query, ranking in zip(queries, rankings)
    ]
    output = {}
    for language in ("en", "tl", "combined"):
        group = rows if language == "combined" else [row for row in rows if row["language"] == language]
        output[language] = {
            "questions": len(group),
            **{
                name: float(np.mean([row["metrics"][name] for row in group]))
                for name in ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")
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
    annotations = read_jsonl(args.annotations)
    _, validation, _ = prepare(annotations, layout["chunks"])
    if len(validation) != 338:
        raise ValueError("development-validation cohort changed")
    parent_index = {parent_id: index for index, parent_id in enumerate(layout["parents"])}
    owners = [parent_index[chunk["parent_id"]] for chunk in layout["chunks"]]
    texts = [chunk["text"] for chunk in layout["chunks"]]
    questions = [query["question"] for query in validation]

    base = StaticModel.from_pretrained(args.model)
    full_documents = np.asarray(base.encode(texts, batch_size=512, max_length=None, use_multiprocessing=False))
    full_queries = np.asarray(base.encode(questions, batch_size=512, max_length=None, use_multiprocessing=False))
    reports = {}
    for dimension in DIMENSIONS:
        float_documents = full_documents[:, :dimension]
        float_queries = full_queries[:, :dimension]
        reports[f"float32_prefix_{dimension}"] = {
            "dimension": dimension,
            "dtype": "float32",
            "trained_for_prefix": False,
            "summary": metrics(
                validation, float_queries, float_documents, owners, layout["parents"], layout["chunks"]
            ),
        }
        quantized = quantize_model(base, quantize_to="int8", dimensionality=dimension)
        int8_documents = np.asarray(
            quantized.encode(texts, batch_size=512, max_length=None, use_multiprocessing=False)
        )
        int8_queries = np.asarray(
            quantized.encode(questions, batch_size=512, max_length=None, use_multiprocessing=False)
        )
        reports[f"int8_prefix_{dimension}"] = {
            "dimension": dimension,
            "dtype": "int8",
            "trained_for_prefix": False,
            "matrix_bytes": int(quantized.embedding.nbytes),
            "summary": metrics(
                validation, int8_queries, int8_documents, owners, layout["parents"], layout["chunks"]
            ),
        }
        del quantized, int8_documents, int8_queries
        gc.collect()

    payload = {
        "purpose": "untrained prefix and int8 ablation; no training and no latency measurement",
        "model_weights_sha256": EXPECTED_MODEL_SHA256,
        "shared_layout_sha256": digest(args.layout),
        "validation_questions": len(validation),
        "reports": reports,
        "interpretation_guard": (
            "The frozen model was trained only at 1024 dimensions. Results below 1024 are a failure probe, "
            "not evidence for or against a future Matryoshka-trained compact export."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(payload, destination, ensure_ascii=False, sort_keys=True, indent=2)
        destination.write("\n")
    compact = {
        name: {
            "en_R10": row["summary"]["en"]["Recall@10"],
            "tl_R10": row["summary"]["tl"]["Recall@10"],
            "combined_R10": row["summary"]["combined"]["Recall@10"],
        }
        for name, row in reports.items()
    }
    print(json.dumps({"output": str(args.output), "recall10": compact}, indent=2))


if __name__ == "__main__":
    main()
