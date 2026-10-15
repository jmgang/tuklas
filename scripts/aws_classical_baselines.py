#!/usr/bin/env python3
"""Score CLASSICAL retrieval baselines on the frozen benchmark.

TF-IDF is BM25's direct ancestor: BM25 is TF-IDF plus term-frequency
saturation (k1) and document-length normalisation (b), so the pair measures
what those two refinements are worth.

Word2Vec is a different lineage entirely -- dense static vectors learned
from co-occurrence rather than distilled from a transformer.  It is trained
HERE on the same approved Tagalog corpus, at the same 512 dimensions, and
pooled and indexed identically to the student, so the only thing that
differs is how the vectors were learned.  That isolates what distillation
from SEA-LION actually bought over classical co-occurrence training.

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
import gzip
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path
import subprocess
import sys

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/classical-work")
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


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    plan = json.loads((INPUT / "classical-plan.json").read_text())
    bucket = plan["bucket"]
    output_prefix = plan["output_prefix"].rstrip("/") + "/"

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
         "model2vec==0.9.0", "gensim==4.4.0", "numpy==2.5.3", "boto3==1.43.108"],
        check=True,
    )

    import boto3
    import numpy as np
    from model2vec import StaticModel
    from model2vec.train.utils import get_probable_pad_token_id

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
        path = WORK / "classical.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + "classical.json")

    # ---- gate: the custom student, in this harness --------------------------
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
    record("custom-student-512d", D,
           static_encode([q["question"] for q in development], query=True),
           static_encode([q["question"] for q in heldout], query=True),
           {"bytes": student_bytes, "kind": "dense-static", "dimension": int(weights.shape[1])})
    gate = results["custom-student-512d"]
    if (abs(gate["development"]["Recall@10"] - STUDENT_DEVELOPMENT) > 1e-6
            or abs(gate["heldout"]["Recall@10"] - STUDENT_HELDOUT) > 1e-6):
        raise ValueError("harness does not reproduce the student's known scores")
    print("GATE ok: harness reproduces the student exactly", flush=True)
    del D

    # ---- sparse lexical: TF-IDF and BM25 -----------------------------------
    WORD = re.compile(r"\w+", re.UNICODE)

    def words(text: str) -> list[str]:
        return WORD.findall(unicodedata.normalize("NFKC", text).casefold())

    doc_words = [words(t) for t in documents]
    lengths = np.array([len(d) for d in doc_words], dtype=np.float32)
    average = float(lengths.mean())
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for j, doc in enumerate(doc_words):
        counts: dict[str, int] = defaultdict(int)
        for word in doc:
            counts[word] += 1
        for word, freq in counts.items():
            postings[word].append((j, freq))
    total = len(doc_words)
    # BM25's probabilistic idf and TF-IDF's classic smoothed idf differ; use each
    # scheme's own definition rather than sharing one.
    idf_bm25 = {w: np.log(1 + (total - len(p) + 0.5) / (len(p) + 0.5)) for w, p in postings.items()}
    idf_tfidf = {w: np.log(total / (1 + len(p))) + 1.0 for w, p in postings.items()}
    owners_array = np.asarray(owners)
    names_array = np.asarray(parents)
    starts = np.r_[0, np.flatnonzero(owners_array[1:] != owners_array[:-1]) + 1].astype(np.int64)

    def lexical_hits(cohort, scheme: str, k1=1.5, b=0.75) -> list[float]:
        hits = []
        for query in cohort:
            scores = np.zeros(total, dtype=np.float32)
            terms = words(query["question"])
            qcounts: dict[str, int] = defaultdict(int)
            for w in terms:
                qcounts[w] += 1
            for word, qf in qcounts.items():
                p = postings.get(word)
                if not p:
                    continue
                j = np.fromiter((x[0] for x in p), np.int64, len(p))
                f = np.fromiter((x[1] for x in p), np.float32, len(p))
                if scheme == "bm25":
                    scores[j] += idf_bm25[word] * (f * (k1 + 1)) / (
                        f + k1 * (1 - b + b * lengths[j] / average)
                    )
                else:
                    # classic tf-idf: raw tf x idf, cosine-normalised by length
                    scores[j] += qf * idf_tfidf[word] * f / np.maximum(lengths[j], 1.0)
            best = np.maximum.reduceat(scores, starts)
            top = np.lexsort((names_array, -best))[:10]
            hits.append(float(query["gold_parent_id"] in names_array[top]))
        return hits

    for scheme, label in (("bm25", "BM25 (k1=1.5 b=0.75)"), ("tfidf", "TF-IDF (tf x idf, length-normalised)")):
        dev = float(np.mean(lexical_hits(development, scheme)))
        held = float(np.mean(lexical_hits(heldout, scheme)))
        results[label] = {
            "development": {"Recall@10": dev}, "heldout": {"Recall@10": held},
            "kind": "sparse-lexical", "dimension": len(postings),
        }
        print(f"  {label}: dev {dev:.6f}  heldout {held:.6f}", flush=True)
        path = WORK / "classical.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + "classical.json")

    # ---- dense static from co-occurrence: Word2Vec CBOW / skip-gram, fastText
    from gensim.models import FastText, Word2Vec

    corpus = WORK / "training.tl.jsonl"
    with gzip.open(INPUT / "training.tl.jsonl.gz", "rb") as raw, corpus.open("wb") as out:
        shutil.copyfileobj(raw, out)
    sentences = []
    with corpus.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                sentences.append(words(json.loads(line)["text"]))
    if len(sentences) != plan["expected_source_rows"]:
        raise ValueError(f"corpus changed: {len(sentences)} rows")
    print(f"training corpus: {len(sentences):,} passages, "
          f"{sum(len(s) for s in sentences):,} tokens", flush=True)

    dimension = int(plan["classical_dimension"])
    for label, builder in (
        (f"Word2Vec CBOW {dimension}d (trained here)",
         lambda: Word2Vec(sentences, vector_size=dimension, sg=0, window=5,
                          min_count=3, workers=8, epochs=5, seed=42)),
        (f"Word2Vec skip-gram {dimension}d (trained here)",
         lambda: Word2Vec(sentences, vector_size=dimension, sg=1, window=5,
                          min_count=3, workers=8, epochs=5, seed=42)),
        (f"fastText {dimension}d subword (trained here)",
         lambda: FastText(sentences, vector_size=dimension, sg=1, window=5,
                          min_count=3, workers=8, epochs=5, seed=42)),
    ):
        try:
            model = builder()
            kv = model.wv

            def embed(values: list[str]) -> np.ndarray:
                out = np.zeros((len(values), dimension), dtype=np.float32)
                for i, text in enumerate(values):
                    vectors = [kv[w] for w in words(text) if w in kv]
                    if vectors:
                        out[i] = np.mean(vectors, axis=0)
                n = np.linalg.norm(out, axis=1, keepdims=True)
                return out / np.maximum(n, 1e-12)

            Dm = embed(documents)
            record(label, Dm,
                   embed([q["question"] for q in development]),
                   embed([q["question"] for q in heldout]),
                   {"kind": "dense-static", "dimension": dimension,
                    "vocabulary": len(kv), "trained_on": "approved Tagalog corpus",
                    "epochs": 5, "min_count": 3, "window": 5})
            del Dm, model, kv
        except Exception as error:
            results[label] = {"error": repr(error)[:300], "kind": "dense-static"}
            print(f"  {label}: FAILED {error!r}"[:220], flush=True)

    report = {
        "results": results,
        "window": WINDOW, "stride": STRIDE,
        "chunks": len(chunk_records), "articles": len(parents),
        "shared_index": True,
        "classical_dimension": dimension,
        "library_versions": versions,
        "scored_on_cpu": True,
        "harness_reproduces_student": True,
        "heldout_use": "verification-only; no selection performed here",
    }
    out_path = WORK / "classical-report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    s3.upload_file(str(out_path), bucket, output_prefix + "classical-report.json")
    print(json.dumps(
        {k: (v.get("heldout", {}).get("Recall@10") or v.get("error"))
         for k, v in results.items()}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
