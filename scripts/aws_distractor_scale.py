#!/usr/bin/env python3
"""Scale the retrieval haystack from 1,294 articles to ~41,366 and re-measure.

The benchmark retrieves one article out of 1,294, and its questions are
generated from those articles, so every system scores in the 90s and the task
looks trivial.  The approved unsupervised corpus holds 40,072 further Tagalog
Wikipedia articles which **exclude all 1,294 benchmark parents** (asserted
below), so they are pure distractors: same questions, same gold answers, a 32x
larger haystack.

Nothing is fabricated -- the system simply stops being handed a tiny candidate
pool.  A lexical matcher has 32x more chances to be fooled by a coincidental
rare-term hit, and a 512-dimensional static space has 32x more entities to keep
distinguishable, so both should fall.  Which falls further is the measurement.

Correctness gate: ``rank_parents`` loops over parents in Python, which is
unusable at 41,366 parents x 3,984 queries.  This uses ``np.maximum.reduceat``
instead, and **asserts the vectorised path reproduces rank_parents exactly on
the 1,294-article index** before running the scaled one.  A ranking shortcut
that silently disagreed would invalidate every number here.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/scale-work")
sys.path.insert(0, str(INPUT))

WINDOW, STRIDE = 192, 128
WORD = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


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
    plan = json.loads((INPUT / "scale-plan.json").read_text())
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
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, row["Key"], str(target))
        package_bytes += int(row["Size"])
    if not package_bytes:
        raise ValueError("no package objects under " + model_prefix)
    print(f"model package {package_bytes:,} bytes", flush=True)

    student = StaticModel.from_pretrained(str(model_dir))
    weights = np.asarray(student.embedding, dtype=np.float32)
    tokenizer = student.tokenizer
    config = json.loads((model_dir / "config.json").read_text())
    query_prefix = config.get("query_prefix", "")
    pad_id = get_probable_pad_token_id(tokenizer)
    unknown_id = student.unk_token_id
    print(f"embedding {weights.shape}", flush=True)

    layout = json.loads((INPUT / "shared-layout.json").read_text())
    annotations = [
        json.loads(line)
        for line in (INPUT / "annotations.all.jsonl").read_text().splitlines()
        if line.strip()
    ]
    frozen = layout["chunks"]
    benchmark_parents = layout["parents"]
    tagalog = [row for row in annotations if row["language"] == "tl"]
    _, development, _ = prepare(tagalog, frozen)
    heldout = [row for row in tagalog if row["partition"] == "evaluation"]
    if len(development) != 197 or len(heldout) != 3_984:
        raise ValueError(f"cohorts changed: {len(development)}/{len(heldout)}")

    # Benchmark articles, reconstructed by offset from the frozen windows.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in frozen:
        grouped[chunk["parent_id"]].append(chunk)
    texts: dict[str, str] = {}
    for parent_id, group in grouped.items():
        group = sorted(group, key=lambda c: c["start"])
        buffer: list[str | None] = [None] * max(c["end"] for c in group)
        for chunk in group:
            for offset, character in enumerate(chunk["text"]):
                buffer[chunk["start"] + offset] = character
        if any(c is None for c in buffer):
            raise ValueError(f"{parent_id}: reconstruction hole")
        texts[parent_id] = "".join(buffer)  # type: ignore[arg-type]

    # Distractors: the approved external corpus, grouped into articles.
    corpus = WORK / "training.tl.jsonl"
    with gzip.open(INPUT / "training.tl.jsonl.gz", "rb") as raw, corpus.open("wb") as out:
        shutil.copyfileobj(raw, out)
    external: dict[str, list[str]] = defaultdict(list)
    rows = 0
    with corpus.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("language") != "tl":
                raise ValueError("non-Tagalog row in the approved corpus")
            external["external:" + str(row.get("source_article_id"))].append(row["text"])
            rows += 1
    if rows != plan["expected_source_rows"]:
        raise ValueError(f"corpus changed: {rows} rows")
    # Decisive: no benchmark parent may enter the distractor set.
    overlap = set(external) & set(benchmark_parents)
    if overlap:
        raise ValueError(f"benchmark parent leaked into distractors: {sorted(overlap)[:3]}")
    print(f"{len(external):,} distractor articles from {rows:,} passages", flush=True)

    def build(parent_texts: dict[str, str]):
        names = sorted(parent_texts)
        chunk_texts: list[str] = []
        owners: list[int] = []
        spans_out: list[tuple[int, int]] = []
        for index, name in enumerate(names):
            text = parent_texts[name]
            spans = windows(text)
            if not spans:
                spans = [(0, len(text))]
            for begin, end in spans:
                chunk_texts.append(text[begin:end])
                owners.append(index)
                spans_out.append((begin, end))
        return chunk_texts, np.asarray(owners), names, spans_out

    def encode(values: list[str], *, query: bool) -> np.ndarray:
        items = [(query_prefix + v) if query else v for v in values]
        out = np.zeros((len(items), weights.shape[1]), dtype=np.float32)
        for start in range(0, len(items), 2048):
            batch = items[start : start + 2048]
            for offset, enc in enumerate(
                tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            ):
                ids = strip_padding_and_unknown(enc.ids, pad_id=pad_id, unknown_id=unknown_id)
                if ids:
                    out[start + offset] = weights[ids].mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def segment_starts(owners: np.ndarray) -> np.ndarray:
        return np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1].astype(np.int64)

    def dense_hits(cohort, D, owners, names, starts, batch=64) -> list[float]:
        """Vectorised: per-parent max via reduceat, then top-10 with name ties."""
        Q = encode([q["question"] for q in cohort], query=True)
        order_names = np.asarray(names)
        hits = []
        for begin in range(0, len(cohort), batch):
            sims = Q[begin : begin + batch] @ D.T
            for row, query in zip(sims, cohort[begin : begin + batch]):
                best = np.maximum.reduceat(row, starts)
                top = np.lexsort((order_names, -best))[:10]
                hits.append(float(query["gold_parent_id"] in order_names[top]))
        return hits

    def bm25_hits(cohort, chunk_texts, owners, names, starts, k1=1.5, b=0.75) -> list[float]:
        docs = [normalise(t) for t in chunk_texts]
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        average = float(lengths.mean())
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for j, doc in enumerate(docs):
            counts: dict[str, int] = defaultdict(int)
            for word in doc:
                counts[word] += 1
            for word, freq in counts.items():
                postings[word].append((j, freq))
        total = len(docs)
        idf = {w: np.log(1 + (total - len(p) + 0.5) / (len(p) + 0.5)) for w, p in postings.items()}
        order_names = np.asarray(names)
        hits = []
        for query in cohort:
            scores = np.zeros(total, dtype=np.float32)
            for word in set(normalise(query["question"])):
                p = postings.get(word)
                if not p:
                    continue
                j = np.fromiter((x[0] for x in p), np.int64, len(p))
                f = np.fromiter((x[1] for x in p), np.float32, len(p))
                scores[j] += idf[word] * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * lengths[j] / average)
                )
            best = np.maximum.reduceat(scores, starts)
            top = np.lexsort((order_names, -best))[:10]
            hits.append(float(query["gold_parent_id"] in order_names[top]))
        return hits

    results = {}

    # ---- Gate: reproduce the known baseline on the 1,294-article index, and
    # prove the vectorised ranking agrees with rank_parents exactly.
    base_texts = {name: texts[name] for name in benchmark_parents}
    chunk_texts, owners, names, spans = build(base_texts)
    starts = segment_starts(owners)
    D = encode(chunk_texts, query=False)
    print(f"baseline index: {len(chunk_texts):,} chunks over {len(names):,} articles", flush=True)

    reference = []
    chunk_records = [
        {"parent_id": names[o], "text": t, "start": b, "end": e}
        for t, o, (b, e) in zip(chunk_texts, owners, spans)
    ]
    Qdev = encode([q["question"] for q in development], query=True)
    for begin in range(0, len(development), 64):
        rankings = rank_parents(Qdev[begin : begin + 64], D, owners.tolist(), names)
        for query, ranking in zip(development[begin : begin + 64], rankings, strict=True):
            reference.append(float(score(query, ranking, chunk_records)["Recall@10"]))
    vectorised = dense_hits(development, D, owners, names, starts)
    if reference != vectorised:
        raise ValueError(
            "vectorised ranking disagrees with rank_parents on development: "
            f"{sum(reference)} vs {sum(vectorised)} hits"
        )
    print(f"GATE ok: vectorised == rank_parents on development "
          f"({sum(reference):.0f}/{len(reference)})", flush=True)

    for label, cohort in (("development", development), ("heldout", heldout)):
        results[f"baseline_1294/student/{label}"] = float(
            np.mean(dense_hits(cohort, D, owners, names, starts))
        )
        results[f"baseline_1294/bm25/{label}"] = float(
            np.mean(bm25_hits(cohort, chunk_texts, owners, names, starts))
        )
        print(f"  1,294 articles {label}: student "
              f"{results[f'baseline_1294/student/{label}']:.6f}  bm25 "
              f"{results[f'baseline_1294/bm25/{label}']:.6f}", flush=True)

    path = WORK / "distractor-scale.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    s3.upload_file(str(path), bucket, output_prefix + "distractor-scale.json")

    # ---- Scaled index: benchmark articles + external distractors.
    scaled_texts = dict(base_texts)
    for name, passages in external.items():
        scaled_texts[name] = "\n".join(passages)
    chunk_texts, owners, names, spans = build(scaled_texts)
    starts = segment_starts(owners)
    print(f"scaled index: {len(chunk_texts):,} chunks over {len(names):,} articles", flush=True)
    D = encode(chunk_texts, query=False)
    print("  encoded", flush=True)

    for label, cohort in (("development", development), ("heldout", heldout)):
        results[f"scaled_{len(names)}/student/{label}"] = float(
            np.mean(dense_hits(cohort, D, owners, names, starts))
        )
        print(f"  {len(names):,} articles {label}: student "
              f"{results[f'scaled_{len(names)}/student/{label}']:.6f}", flush=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + "distractor-scale.json")
    del D

    for label, cohort in (("development", development), ("heldout", heldout)):
        results[f"scaled_{len(names)}/bm25/{label}"] = float(
            np.mean(bm25_hits(cohort, chunk_texts, owners, names, starts))
        )
        print(f"  {len(names):,} articles {label}: bm25 "
              f"{results[f'scaled_{len(names)}/bm25/{label}']:.6f}", flush=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + "distractor-scale.json")

    report = {
        "results": results,
        "window": WINDOW, "stride": STRIDE,
        "benchmark_articles": len(benchmark_parents),
        "scaled_articles": len(names),
        "scaled_chunks": len(chunk_texts),
        "distractor_articles": len(external),
        "model_prefix": model_prefix,
        "package_bytes": package_bytes,
        "pruned": False,
        "benchmark_parents_in_distractors": 0,
        "questions_unchanged": True,
        "gold_answers_unchanged": True,
        "vectorised_ranking_matches_rank_parents": True,
        "heldout_use": "verification-only; no selection performed here",
    }
    out = WORK / "distractor-scale-report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    s3.upload_file(str(out), bucket, output_prefix + "distractor-scale-report.json")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
