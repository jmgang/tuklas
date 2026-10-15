#!/usr/bin/env python3
"""Sweep retrieval chunk size for a saved student, on CPU, inside AWS.

Why this can matter more than any training change tried so far: the frozen
layout indexes 1024-character windows (median 159 words). A static model has
one fixed vector per token and mean-pools them, so a query matching eight
relevant tokens in a 159-word window is averaged against ~151 irrelevant ones.
The teacher has attention and can concentrate on the relevant span; a lookup
table physically cannot. Shrinking the window should therefore help the student
disproportionately, and it needs **no retraining at all** -- only re-indexing.

The parent texts are reconstructed from the frozen layout's overlapping windows
by offset (verified lossless: 1,294 parents, zero conflicts or holes), so this
changes only how the same corpus is segmented, never its content.

Held-out isolation: the window is chosen on the 197-question development
cohort. Held-out is scored once at the selected window, and the full held-out
curve is emitted only as a post-hoc observation, explicitly not used to select.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/rechunk-work")
sys.path.insert(0, str(INPUT))

TEACHER_DEVELOPMENT_RECALL_AT_10 = 192 / 197
TEACHER_HELDOUT_RECALL_AT_10 = 3848 / 3984
METRICS = ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    plan = json.loads((INPUT / "rechunk-plan.json").read_text())
    bucket = plan["bucket"]
    model_prefix = plan["model_prefix"].rstrip("/") + "/"
    output_prefix = plan["output_prefix"].rstrip("/") + "/"

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         f"model2vec=={plan['model2vec_version']}"],
        check=True,
    )

    import boto3
    import numpy as np
    from model2vec import StaticModel
    from model2vec.train.utils import get_probable_pad_token_id

    from multiwikiqa_compact_retrieval import strip_padding_and_unknown
    from multiwikiqa_evaluation import rank_parents, score
    from multiwikiqa_ranking_v5 import prepare

    s3 = boto3.client("s3")
    model_dir = WORK / "package"
    model_dir.mkdir(parents=True, exist_ok=True)
    package_bytes = 0
    for row in s3.list_objects_v2(Bucket=bucket, Prefix=model_prefix).get("Contents", []):
        name = row["Key"][len(model_prefix):]
        if not name:
            continue
        destination = model_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, row["Key"], str(destination))
        package_bytes += int(row["Size"])
    if not package_bytes:
        raise ValueError("no package objects under " + model_prefix)

    student = StaticModel.from_pretrained(str(model_dir))
    weights = np.asarray(student.embedding, dtype=np.float32)
    tokenizer = student.tokenizer
    config = json.loads((model_dir / "config.json").read_text())
    query_prefix = config.get("query_prefix", "")
    pad_id = get_probable_pad_token_id(tokenizer)
    unknown_id = student.unk_token_id

    layout = json.loads((INPUT / "shared-layout.json").read_text())
    annotations = read_jsonl(INPUT / "annotations.all.jsonl")
    frozen_chunks = layout["chunks"]
    parents = layout["parents"]

    tagalog = [row for row in annotations if row["language"] == "tl"]
    _, development, _ = prepare(tagalog, frozen_chunks)
    heldout = [row for row in tagalog if row["partition"] == "evaluation"]
    if len(development) != 197 or len(heldout) != 3_984:
        raise ValueError(f"cohorts changed: dev={len(development)} heldout={len(heldout)}")

    # Reconstruct each parent's text from the frozen overlapping windows.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in frozen_chunks:
        grouped[chunk["parent_id"]].append(chunk)
    parent_text: dict[str, str] = {}
    for parent_id, group in grouped.items():
        group = sorted(group, key=lambda c: c["start"])
        size = max(c["end"] for c in group)
        buffer: list[str | None] = [None] * size
        for chunk in group:
            text = chunk["text"]
            if len(text) != chunk["end"] - chunk["start"]:
                raise ValueError(f"{parent_id}: chunk text length disagrees with its span")
            for offset, character in enumerate(text):
                position = chunk["start"] + offset
                if buffer[position] is not None and buffer[position] != character:
                    raise ValueError(f"{parent_id}: overlapping windows disagree at {position}")
                buffer[position] = character
        if any(character is None for character in buffer):
            raise ValueError(f"{parent_id}: reconstruction left a hole")
        parent_text[parent_id] = "".join(buffer)  # type: ignore[arg-type]
    if set(parent_text) != set(parents):
        raise ValueError("reconstructed parent set differs from the frozen layout")
    print(f"reconstructed {len(parent_text)} parent texts losslessly", flush=True)

    def encode(texts: list[str], *, query: bool) -> np.ndarray:
        values = [(query_prefix + text) if query else text for text in texts]
        out = np.zeros((len(values), weights.shape[1]), dtype=np.float32)
        for start in range(0, len(values), 1024):
            batch = values[start : start + 1024]
            for offset, encoding in enumerate(
                tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            ):
                ids = strip_padding_and_unknown(
                    encoding.ids, pad_id=pad_id, unknown_id=unknown_id
                )
                if ids:
                    out[start + offset] = weights[ids].mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def build(window: int, stride: int) -> tuple[list[dict], list[int]]:
        """Re-window every parent, preserving parent order and char offsets."""
        chunks: list[dict] = []
        owners: list[int] = []
        for index, parent_id in enumerate(parents):
            text = parent_text[parent_id]
            starts = list(range(0, max(1, len(text) - window + 1), stride))
            if starts[-1] + window < len(text):
                starts.append(len(text) - window)
            seen: set[tuple[int, int]] = set()
            for begin in starts:
                begin = max(0, begin)
                end = min(len(text), begin + window)
                if (begin, end) in seen or end <= begin:
                    continue
                seen.add((begin, end))
                chunks.append({
                    "parent_id": parent_id, "text": text[begin:end],
                    "start": begin, "end": end,
                })
                owners.append(index)
        return chunks, owners

    # Queries are window-independent, so encode them once.
    development_vectors = encode([row["question"] for row in development], query=True)
    heldout_vectors = encode([row["question"] for row in heldout], query=True)

    def evaluate(cohort, query_vectors, chunks, owners, document_vectors) -> dict:
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

    results = []
    for candidate in plan["windows"]:
        window, stride = int(candidate["window"]), int(candidate["stride"])
        chunks, owners = build(window, stride)
        document_vectors = encode([c["text"] for c in chunks], query=False)
        dev = evaluate(development, development_vectors, chunks, owners, document_vectors)
        held = evaluate(heldout, heldout_vectors, chunks, owners, document_vectors)
        row = {
            "window": window, "stride": stride, "chunks": len(chunks),
            "development": dev, "heldout": held,
            "development_recall_at_10": dev["Recall@10"],
            "heldout_recall_at_10": held["Recall@10"],
        }
        results.append(row)
        print(f"window={window} stride={stride} chunks={len(chunks)} "
              f"dev={dev['Recall@10']:.6f} heldout={held['Recall@10']:.6f}", flush=True)
        # Preserve after every setting, so a later failure cannot lose the sweep.
        path = WORK / "rechunk-sweep.json"
        path.write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + "rechunk-sweep.json")

    # Selection is on development only.
    best = max(results, key=lambda r: r["development_recall_at_10"])
    frozen = next(
        (r for r in results if r["window"] == 1024 and r["stride"] == 896), None
    )
    report = {
        "label": plan["label"],
        "model_prefix": model_prefix,
        "source_job": plan.get("source_job"),
        "package_bytes": package_bytes,
        "retrained": False,
        "scored_on_cpu": True,
        "parents": len(parents),
        "frozen_layout": {"window": 1024, "stride": 896, "chunks": len(frozen_chunks)},
        "results": results,
        "selected_on": "development",
        "selected": {k: best[k] for k in ("window", "stride", "chunks",
                                          "development_recall_at_10",
                                          "heldout_recall_at_10")},
        "baseline_in_sweep": (
            {k: frozen[k] for k in ("window", "stride", "chunks",
                                    "development_recall_at_10",
                                    "heldout_recall_at_10")}
            if frozen else None
        ),
        "teacher_development_recall_at_10": TEACHER_DEVELOPMENT_RECALL_AT_10,
        "teacher_heldout_recall_at_10": TEACHER_HELDOUT_RECALL_AT_10,
        "heldout_use": (
            "verification-only; the window is selected on development and the "
            "held-out curve is reported post hoc, never used to select"
        ),
        "heldout_used_for_training_or_checkpoint_selection": False,
    }
    path = WORK / "rechunk-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    s3.upload_file(str(path), bucket, output_prefix + "rechunk-report.json")
    print(json.dumps({k: report[k] for k in
                      ("selected", "baseline_in_sweep", "selected_on")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
