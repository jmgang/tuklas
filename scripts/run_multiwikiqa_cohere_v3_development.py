#!/usr/bin/env python3
"""Score Cohere multilingual v3 on the 197 Tagalog development questions.

The frozen Cohere scaffold targets the 3,984-question held-out partition.  This
runs the same provider, tokenizer snapshot, segmentation policy and metric code
against the *development* cohort instead, so the number is directly comparable
with the student's 173/197 and held-out stays closed.

Comparability is structural, not approximate: the ranking and scoring come from
``multiwikiqa_evaluation.rank_parents`` and ``score`` -- the identical functions
the student is measured with -- and the question cohort is selected by the same
``multiwikiqa_ranking_v5.prepare`` predicate that produced the student's
197-question report cohort.

Default mode is preflight: it counts tokens with the frozen Cohere tokenizer and
prints the cost estimate without calling Bedrock.  ``--execute`` embeds, caching
each batch to an immutable ``.npy`` so an interrupted run resumes without
re-paying.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.multiwikiqa_cohere_v3 import (  # noqa: E402
    _load_tokenizer,
    _segment_layout,
    file_sha256,
)
from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents, score  # noqa: E402
from pycon_model2vec_demo.multiwikiqa_ranking_v5 import prepare  # noqa: E402

CONFIG = ROOT / "configs/multiwikiqa-cohere-multilingual-v3-eval-v1.json"
# Bedrock Cohere Embed input-token price; the receipt records it as an estimate.
USD_PER_1K_INPUT_TOKENS = 0.0001

# Per-cohort baselines.  These must be selected by cohort: emitting the
# development student and teacher alongside a held-out Cohere score compared a
# held-out number against development references and overstated Cohere's lead
# by 5.08 points on development and its deficit to the teacher by 0.87 points
# on held-out.  The student figure was additionally a superseded 173/197
# checkpoint; the current best is 183/197 on development.
# Baselines at the re-chunked window-192 index.  Mixing these with the frozen
# 1024-char figures is the mismatched-comparison bug this file already shipped
# once: a report that pairs Cohere at window 192 with a student measured at
# window 1024 is wrong by several points.
BASELINES_RECHUNKED_192 = {
    "development": {
        "student_recall_at_10": 0.9441624365482234,
        "teacher_recall_at_10": 1.0,          # saturated at this window
    },
    "evaluation": {
        "student_recall_at_10": 0.9359939759036144,
        "teacher_recall_at_10": 0.985943775100402,
    },
}

BASELINES = {
    "development": {
        # 183/197, pycon-sealion-frontier-512-20261006-210615-260853
        "student_recall_at_10": 183 / 197,
        "teacher_recall_at_10": 192 / 197,
    },
    "evaluation": {
        "student_recall_at_10": 0.886546,
        # 3848/3984; NOT the 0.9694 bilingual figure in
        # docs/full-heldout-results-20261005.md, which is a different cohort.
        "teacher_recall_at_10": 3848 / 3984,
    },
}

METRIC_NAMES = (
    "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
    "AnswerSpanCoverage@1", "AnswerSpanCoverage@3",
    "AnswerSpanCoverage@5", "AnswerSpanCoverage@10",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    # Re-chunk the frozen layout before segmentation so Cohere is measured on
    # the SAME index as the student.  The published 0.952309 was taken at the
    # frozen 1024-char layout and is not comparable with a window-192 student
    # score.  A side benefit: 192-character chunks sit far below Cohere's token
    # limit, so _segment_layout becomes a no-op and the segments-vs-chunks
    # mismatch that made AnswerSpanCoverage incomparable disappears.
    parser.add_argument("--rechunk-window", type=int, default=0)
    parser.add_argument("--rechunk-stride", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--cohort", choices=("development", "evaluation"), default="development",
        help="development = the 197 report questions; evaluation = the 3,984 held-out",
    )
    parser.add_argument(
        "--reuse-documents", type=Path, default=None,
        help="an earlier run directory whose cached document vectors to reuse",
    )
    arguments = parser.parse_args()
    if arguments.rechunk_window:
        if (arguments.rechunk_window, arguments.rechunk_stride) != (192, 128):
            raise ValueError(
                "baselines are only recorded for window 192 / stride 128; "
                "add them before comparing at another index"
            )
        baselines = BASELINES_RECHUNKED_192[arguments.cohort]
        baseline_index = "window 192 / stride 128, 40,152 chunks"
    else:
        baselines = BASELINES[arguments.cohort]
        baseline_index = "frozen layout, 6,546 chunks / 6,842 Cohere segments"

    config = json.loads(CONFIG.read_text())
    dataset = config["dataset"]
    layout = json.loads((ROOT / dataset["layout"]).read_text())
    annotations = read_jsonl(ROOT / dataset["annotations"])
    if file_sha256(ROOT / dataset["layout"]) != dataset["layout_sha256"]:
        raise ValueError("frozen evaluation layout changed")
    if file_sha256(ROOT / dataset["annotations"]) != dataset["annotations_sha256"]:
        raise ValueError("frozen annotations changed")

    tagalog = [row for row in annotations if row["language"] == "tl"]
    if arguments.cohort == "development":
        _, report_cohort, _ = prepare(tagalog, layout["chunks"])
        if len(report_cohort) != 197:
            raise ValueError(f"expected the 197-question report cohort, got {len(report_cohort)}")
        if any(row["partition"] != "development" for row in report_cohort):
            raise ValueError("held-out question entered the Cohere development cohort")
    else:
        # The held-out partition is the project's reporting surface for third
        # party baselines; Cohere has no training relationship to it.
        report_cohort = [row for row in tagalog if row["partition"] == "evaluation"]
        if len(report_cohort) != int(dataset["questions"]):
            raise ValueError(
                f"expected {dataset['questions']} held-out questions, got {len(report_cohort)}"
            )

    if arguments.rechunk_window:
        window = int(arguments.rechunk_window)
        stride = int(arguments.rechunk_stride or window)
        from collections import defaultdict

        grouped = defaultdict(list)
        for chunk in layout["chunks"]:
            grouped[chunk["parent_id"]].append(chunk)
        rebuilt = {}
        for parent_id, group in grouped.items():
            group = sorted(group, key=lambda c: c["start"])
            buffer = [None] * max(c["end"] for c in group)
            for chunk in group:
                if len(chunk["text"]) != chunk["end"] - chunk["start"]:
                    raise ValueError(f"{parent_id}: chunk text disagrees with its span")
                for offset, character in enumerate(chunk["text"]):
                    position = chunk["start"] + offset
                    if buffer[position] is not None and buffer[position] != character:
                        raise ValueError(f"{parent_id}: overlapping windows disagree")
                    buffer[position] = character
            if any(c is None for c in buffer):
                raise ValueError(f"{parent_id}: reconstruction hole")
            rebuilt[parent_id] = "".join(buffer)
        if set(rebuilt) != set(layout["parents"]):
            raise ValueError("reconstructed parent set differs from the frozen layout")
        fresh = []
        for parent_id in layout["parents"]:
            text = rebuilt[parent_id]
            starts = list(range(0, max(1, len(text) - window + 1), stride))
            if starts[-1] + window < len(text):
                starts.append(len(text) - window)
            seen = set()
            for begin in starts:
                begin = max(0, begin)
                end = min(len(text), begin + window)
                if end <= begin or (begin, end) in seen:
                    continue
                seen.add((begin, end))
                fresh.append({
                    "chunk_id": f"{parent_id}:{begin}:{end}",
                    "parent_id": parent_id, "text": text[begin:end],
                    "start": begin, "end": end,
                })
        print(f"re-chunked {len(layout['chunks'])} -> {len(fresh)} chunks "
              f"at window={window} stride={stride}", flush=True)
        layout = {**layout, "chunks": fresh}

    # Re-chunking deliberately departs from the frozen 6,842-segment
    # configuration, so its count assertion cannot apply; the per-segment
    # token ceiling still does.
    segmented = _segment_layout(
        ROOT, config, layout, frozen_counts=not arguments.rechunk_window
    )
    segments = segmented["chunks"]
    parents = segmented["parents"]
    positions = {parent: index for index, parent in enumerate(parents)}
    owners = [positions[segment["parent_id"]] for segment in segments]

    tokenizer = _load_tokenizer(ROOT, config)
    document_texts = [segment["text"] for segment in segments]
    query_texts = [row["question"] for row in report_cohort]
    document_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=True).ids) for text in document_texts
    )
    query_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=True).ids) for text in query_texts
    )
    total_tokens = document_tokens + query_tokens
    estimate = total_tokens / 1000.0 * USD_PER_1K_INPUT_TOKENS

    preflight = {
        "model_id": config["model"]["model_id"],
        "dimension": config["model"]["dimension"],
        "cohort": f"tagalog-{arguments.cohort}-{len(report_cohort)}",
        "questions": len(report_cohort),
        "parents": len(parents),
        "source_chunks": len(layout["chunks"]),
        "searchable_segments": len(segments),
        "document_tokens": document_tokens,
        "query_tokens": query_tokens,
        "total_input_tokens": total_tokens,
        "usd_per_1k_input_tokens": USD_PER_1K_INPUT_TOKENS,
        "estimated_usd": round(estimate, 4),
        "heldout_questions_read": True,
        "heldout_read_purpose": "scored when --cohort evaluation; the development path reads held-out text only for prepare()'s leak-exclusion guard",
        "rechunk_window": arguments.rechunk_window or None,
        "rechunk_stride": arguments.rechunk_stride or None,
        "heldout_used_for_student_training_or_selection": False,
        "student_comparison_recall_at_10": baselines["student_recall_at_10"],
        "baseline_cohort": arguments.cohort,
    }
    print(json.dumps(preflight, indent=2))
    if not arguments.execute:
        print("\npreflight only; pass --execute to call Bedrock")
        return

    import boto3
    from pycon_model2vec_demo.bedrock_embeddings import BedrockCohere

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = arguments.out or ROOT / f"runs/multiwikiqa-cohere-v3-{arguments.cohort}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    session = boto3.Session(
        profile_name=config["aws"]["profile"], region_name=config["aws"]["region"]
    )
    encoder = BedrockCohere(session.client("bedrock-runtime"), config["model"])
    batch = int(config["model"]["maximum_texts_per_request"])

    def encode_all(texts: list[str], role: str, directory: Path) -> tuple[np.ndarray, list[dict]]:
        directory.mkdir(parents=True, exist_ok=True)
        parts, receipts = [], []
        for start in range(0, len(texts), batch):
            piece = texts[start : start + batch]
            stem = f"{start:06d}-{start + len(piece):06d}"
            vector_path = directory / f"{stem}.npy"
            receipt_path = directory / f"{stem}.json"
            if vector_path.exists() and receipt_path.exists():
                parts.append(np.load(vector_path))
                receipts.append(json.loads(receipt_path.read_text()))
                continue
            vectors, receipt = encoder.encode(piece, role)
            with vector_path.open("xb") as stream:
                np.save(stream, vectors.astype(np.float32, copy=False), allow_pickle=False)
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True)
                stream.write("\n")
            parts.append(vectors.astype(np.float32, copy=False))
            receipts.append(receipt)
            done = start + len(piece)
            print(f"  {role}: {done}/{len(texts)}", flush=True)
        return np.concatenate(parts), receipts

    print("\nembedding documents...", flush=True)
    document_directory = (
        arguments.reuse_documents / "documents"
        if arguments.reuse_documents else out / "documents"
    )
    document_vectors, document_receipts = encode_all(
        document_texts, config["model"]["document_input_type"], document_directory
    )
    print("embedding queries...", flush=True)
    query_vectors, query_receipts = encode_all(
        query_texts, config["model"]["query_input_type"], out / "queries"
    )

    rows = []
    for start in range(0, len(report_cohort), 64):
        rankings = rank_parents(
            query_vectors[start : start + 64], document_vectors, owners, parents
        )
        for query, ranking in zip(report_cohort[start : start + 64], rankings, strict=True):
            rows.append({"query_id": query["query_id"], "metrics": score(query, ranking, segments)})

    metrics = {"questions": len(rows)}
    for name in METRIC_NAMES:
        values = [r["metrics"][name] for r in rows if r["metrics"][name] is not None]
        metrics[name] = float(np.mean(values)) if values else None

    billed = sum(r["input_tokens"] or 0 for r in document_receipts + query_receipts)
    report = {
        **preflight,
        "executed_utc": stamp,
        "metrics": metrics,
        "billed_input_tokens": billed,
        "billed_usd_estimate": round(billed / 1000.0 * USD_PER_1K_INPUT_TOKENS, 4),
        "requests": len(document_receipts) + len(query_receipts),
        "student_recall_at_10": baselines["student_recall_at_10"],
        "teacher_recall_at_10": baselines["teacher_recall_at_10"],
        "baseline_index": baseline_index,
        "delta_vs_student": (metrics["Recall@10"] or 0.0) - baselines["student_recall_at_10"],
        "delta_vs_teacher": (metrics["Recall@10"] or 0.0) - baselines["teacher_recall_at_10"],
        # AnswerSpanCoverage is NOT comparable with the student: Cohere is
        # scored over 6,842 segments and the student over 6,546 chunks, and
        # coverage requires one selected unit to contain the whole span, so the
        # 296 split chunks penalise Cohere systematically.
        "answer_span_coverage_comparable_with_student": False,
    }
    (out / f"{arguments.cohort}-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print("\n" + json.dumps({k: report[k] for k in (
        "metrics", "billed_input_tokens", "billed_usd_estimate",
        "delta_vs_student", "delta_vs_teacher")}, indent=2))
    print(f"\nwrote {out / (arguments.cohort + '-report.json')}")


if __name__ == "__main__":
    main()
