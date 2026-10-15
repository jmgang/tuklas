#!/usr/bin/env python3
"""Score off-the-shelf multilingual embedding models on the frozen benchmark.

All systems share ONE index: the 1,294 benchmark articles re-chunked at
window 192 / stride 128 (40,152 chunks), the same setting the custom student is
served at.  Mixing chunkings is the single easiest way to produce a meaningless
comparison, so it is a module constant here, not a per-model choice.

Each model's own prompt convention is applied, because getting it wrong
silently understates a model: multilingual-e5 REQUIRES "query: " / "passage: "
prefixes and loses several points without them; BGE-M3 and LaBSE take no
prefix; potion-multilingual is a static model2vec table.

Gate: the custom student is re-scored in the same harness and must reproduce
its known 0.935994 held-out / 0.944162 development before any other number is
trusted.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/shootout-work")
sys.path.insert(0, str(INPUT))

WINDOW, STRIDE = 192, 128
STUDENT_DEVELOPMENT = 0.9441624365482234
STUDENT_HELDOUT = 0.9359939759036144


def windows(text: str) -> list[tuple[int, int]]:
    starts = list(range(0, max(1, len(text) - WINDOW + 1), STRIDE))
    if starts and starts[-1] + WINDOW < len(text):
        starts.append(len(text) - WINDOW)
    out, seen = [], set()
    for begin in starts:
        begin = max(0, begin)
        end = min(len(text), begin + WINDOW)
        if end <= begin or (begin, end) in seen:
            continue
        seen.add((begin, end))
        out.append((begin, end))
    return out


def encode_weighted_static(model, values):
    """Apply Model2Vec's shipped token weights and mapping before scoring."""
    import numpy as np
    out = np.asarray(model.encode(values, max_length=None,
        batch_size=1024, use_multiprocessing=False), dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    plan = json.loads((INPUT / "shootout-plan.json").read_text())
    bucket = plan["bucket"]
    output_prefix = plan["output_prefix"].rstrip("/") + "/"

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
         "model2vec==0.9.0", "sentence-transformers==5.7.0", "transformers==5.3.0",
         "huggingface-hub==1.33.0", "numpy==2.5.3", "boto3==1.43.108"],
        check=True,
    )

    import boto3
    import numpy as np
    import torch
    from model2vec import StaticModel
    from model2vec.train.utils import get_probable_pad_token_id
    from sentence_transformers import SentenceTransformer

    from multiwikiqa_compact_retrieval import strip_padding_and_unknown
    from multiwikiqa_evaluation import rank_parents, score
    from multiwikiqa_ranking_v5 import prepare

    import importlib.metadata as metadata

    versions = {}
    for package in ("model2vec", "sentence-transformers", "transformers",
                    "tokenizers", "numpy", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    print("versions", json.dumps(versions, sort_keys=True), flush=True)

    s3 = boto3.client("s3")
    layout = json.loads((INPUT / "shared-layout.json").read_text())
    annotations = [
        json.loads(line)
        for line in (INPUT / "annotations.all.jsonl").read_text().splitlines()
        if line.strip()
    ]
    frozen = layout["chunks"]
    parents = layout["parents"]
    tagalog = [row for row in annotations if row["language"] == "tl"]
    _, development, _ = prepare(tagalog, frozen)
    heldout = [row for row in tagalog if row["partition"] == "evaluation"]
    if len(development) != 197 or len(heldout) != 3_984:
        raise ValueError(f"cohorts changed: {len(development)}/{len(heldout)}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in frozen:
        grouped[chunk["parent_id"]].append(chunk)
    texts: dict[str, str] = {}
    for parent_id, group in grouped.items():
        group = sorted(group, key=lambda c: c["start"])
        buffer: list[str | None] = [None] * max(c["end"] for c in group)
        for chunk in group:
            if len(chunk["text"]) != chunk["end"] - chunk["start"]:
                raise ValueError(f"{parent_id}: chunk text length disagrees with its span")
            for offset, character in enumerate(chunk["text"]):
                position = chunk["start"] + offset
                if buffer[position] is not None and buffer[position] != character:
                    raise ValueError(f"{parent_id}: overlapping windows disagree at {position}")
                buffer[position] = character
        if any(c is None for c in buffer):
            raise ValueError(f"{parent_id}: reconstruction hole")
        texts[parent_id] = "".join(buffer)  # type: ignore[arg-type]
    if set(texts) != set(parents):
        raise ValueError("reconstructed parent set differs from the frozen layout")

    chunk_records: list[dict] = []
    owners: list[int] = []
    for index, parent_id in enumerate(parents):
        text = texts[parent_id]
        spans = windows(text) or [(0, len(text))]
        for begin, end in spans:
            chunk_records.append({
                "parent_id": parent_id, "text": text[begin:end],
                "start": begin, "end": end,
            })
            owners.append(index)
    if len(chunk_records) != 40_152 or len(parents) != 1_294:
        raise ValueError(
            f"index changed: {len(chunk_records)} chunks over {len(parents)} articles, "
            "expected 40,152 over 1,294"
        )
    print(f"shared index: {len(chunk_records):,} chunks over {len(parents):,} articles",
          flush=True)
    documents = [c["text"] for c in chunk_records]

    def evaluate(document_vectors, query_vectors, cohort) -> dict:
        rows = []
        for start in range(0, len(cohort), 64):
            rankings = rank_parents(
                query_vectors[start : start + 64], document_vectors, owners, parents
            )
            for query, ranking in zip(cohort[start : start + 64], rankings, strict=True):
                rows.append(score(query, ranking, chunk_records))
        return {
            "Recall@10": float(np.mean([r["Recall@10"] for r in rows])),
            "Recall@1": float(np.mean([r["Recall@1"] for r in rows])),
            "MRR@10": float(np.mean([r["MRR@10"] for r in rows])),
            "questions": len(rows),
        }

    results: dict[str, dict] = {}

    def record(name: str, D, Qdev, Qheld, extra: dict) -> None:
        dev = evaluate(D, Qdev, development)
        held = evaluate(D, Qheld, heldout)
        results[name] = {"development": dev, "heldout": held, **extra}
        print(f"  {name}: dev {dev['Recall@10']:.6f}  heldout {held['Recall@10']:.6f}",
              flush=True)
        path = WORK / "shootout.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + "shootout.json")

    # ---- the custom student, as the harness gate -------------------------
    student_dir = WORK / "student"
    student_dir.mkdir(parents=True, exist_ok=True)
    prefix = plan["student_prefix"].rstrip("/") + "/"
    student_bytes = 0
    for row in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        name = row["Key"][len(prefix):]
        if not name:
            continue
        target = student_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, row["Key"], str(target))
        student_bytes += int(row["Size"])
    if not student_bytes:
        raise ValueError("student package is empty under " + prefix)
    student = StaticModel.from_pretrained(str(student_dir))
    weights = np.asarray(student.embedding, dtype=np.float32)
    tokenizer = student.tokenizer
    query_prefix = json.loads((student_dir / "config.json").read_text()).get("query_prefix", "")
    pad_id = get_probable_pad_token_id(tokenizer)
    unknown_id = student.unk_token_id

    def static_encode(values: list[str], *, query: bool) -> np.ndarray:
        items = [(query_prefix + v) if query else v for v in values]
        out = np.zeros((len(items), weights.shape[1]), dtype=np.float32)
        for start in range(0, len(items), 2048):
            for offset, enc in enumerate(
                tokenizer.encode_batch_fast(items[start : start + 2048], add_special_tokens=False)
            ):
                ids = strip_padding_and_unknown(enc.ids, pad_id=pad_id, unknown_id=unknown_id)
                if ids:
                    out[start + offset] = weights[ids].mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    D = static_encode(documents, query=False)
    record(
        "custom-student-512d",
        D,
        static_encode([q["question"] for q in development], query=True),
        static_encode([q["question"] for q in heldout], query=True),
        {"bytes": student_bytes, "kind": "static", "dimension": int(weights.shape[1])},
    )
    gate = results["custom-student-512d"]
    if (abs(gate["development"]["Recall@10"] - STUDENT_DEVELOPMENT) > 1e-6
            or abs(gate["heldout"]["Recall@10"] - STUDENT_HELDOUT) > 1e-6):
        raise ValueError(
            "harness does not reproduce the student's known scores "
            f"({gate['development']['Recall@10']} / {gate['heldout']['Recall@10']}); "
            "every other number here would be untrustworthy"
        )
    print("GATE ok: harness reproduces the student exactly", flush=True)
    del D

    # ---- static multilingual competitor ---------------------------------
    for spec in plan["static_models"]:
        try:
            try:
                model = StaticModel.from_pretrained(spec["id"], revision=spec["revision"])
                pinned = spec["revision"]
            except TypeError:
                # model2vec 0.9.0 may not expose revision; record that it is unpinned
                # rather than silently implying it was.
                model = StaticModel.from_pretrained(spec["id"])
                pinned = None
            w = np.asarray(model.embedding, dtype=np.float32)
            tk = model.tokenizer
            pid = get_probable_pad_token_id(tk)
            uid = model.unk_token_id

            def enc(values: list[str]) -> np.ndarray:
                if model.weights is not None:
                    # The old manual mean omitted shipped per-token weights.
                    return encode_weighted_static(model, values)
                out = np.zeros((len(values), w.shape[1]), dtype=np.float32)
                for start in range(0, len(values), 2048):
                    for offset, e in enumerate(
                        tk.encode_batch_fast(values[start : start + 2048], add_special_tokens=False)
                    ):
                        ids = strip_padding_and_unknown(e.ids, pad_id=pid, unknown_id=uid)
                        if ids:
                            out[start + offset] = w[ids].mean(axis=0)
                n = np.linalg.norm(out, axis=1, keepdims=True)
                return out / np.maximum(n, 1e-12)

            Dm = enc(documents)
            record(spec["id"], Dm,
                   enc([q["question"] for q in development]),
                   enc([q["question"] for q in heldout]),
                   {"kind": "static", "dimension": int(w.shape[1]),
                    "revision_requested": spec["revision"],
                    "revision_pinned": pinned})
            del Dm
        except Exception as error:  # a missing model must not kill the run
            results[spec["id"]] = {"error": repr(error)[:300], "kind": "static"}
            print(f"  {spec['id']}: FAILED {error!r}"[:220], flush=True)

    # ---- transformer competitors ----------------------------------------
    for spec in plan["transformer_models"]:
        try:
            model = SentenceTransformer(
                spec["id"], device=plan.get("device", "cuda"),
                trust_remote_code=False, revision=spec["revision"],
            )
            # A StaticEmbedding module has no sequence limit, so the property
            # is read-only there.  Setting it is an optimisation for real
            # transformers, never a requirement.
            try:
                model.max_seq_length = int(spec.get("max_seq_length", 512))
            except AttributeError:
                pass

            def enc(values: list[str], prompt: str) -> np.ndarray:
                kwargs = {"prompt": prompt} if prompt else {}
                return model.encode(
                    values, batch_size=int(spec.get("batch_size", 64)),
                    normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False, **kwargs,
                ).astype(np.float32, copy=False)

            Dm = enc(documents, spec.get("document_prefix", ""))
            record(spec["id"], Dm,
                   enc([q["question"] for q in development], spec.get("query_prefix", "")),
                   enc([q["question"] for q in heldout], spec.get("query_prefix", "")),
                   {"kind": "transformer",
                    "dimension": model.get_sentence_embedding_dimension(),
                    "revision": spec["revision"],
                    "query_prefix": spec.get("query_prefix", ""),
                    "document_prefix": spec.get("document_prefix", ""),
                    # A StaticEmbedding reports inf (no sequence limit), which
                    # int() cannot represent.
                    "max_seq_length": (
                        int(model.max_seq_length)
                        if math.isfinite(float(model.max_seq_length))
                        else "unbounded"
                    ),
                    **({"head": spec["head"]} if spec.get("head") else {})})
        except Exception as error:
            results[spec["id"]] = {"error": repr(error)[:300], "kind": "transformer"}
            print(f"  {spec['id']}: FAILED {error!r}"[:220], flush=True)
        finally:
            # Must run even when a query encode raises: otherwise the previous
            # model stays resident on the GPU while the next one is built, and
            # empty_cache() cannot reclaim live tensors.
            Dm = None
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "results": results,
        "window": WINDOW, "stride": STRIDE,
        "chunks": len(chunk_records), "articles": len(parents),
        "shared_index": True,
        "library_versions": versions,
        "scored_on_cpu": False,
        "device": plan.get("device", "cuda"),
        "harness_reproduces_student": True,
        "heldout_use": "verification-only; no selection performed here",
    }
    out = WORK / "shootout-report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    s3.upload_file(str(out), bucket, output_prefix + "shootout-report.json")
    print(json.dumps(
        {k: (v.get("heldout", {}).get("Recall@10") or v.get("error"))
         for k, v in results.items()}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
