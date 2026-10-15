#!/usr/bin/env python3
"""Ask the custom Tagalog student a question and see which articles it ranks.

Everything here is the same arithmetic the benchmark uses: look up a vector per
token, mean-pool, L2 normalise, cosine against every chunk, keep each article's
best chunk, return the top 10.  No vector database and no ANN index -- 40,152
chunks is one NumPy matmul, so exact search is faster than approximating it.

First run encodes the corpus (~30s) and caches it to demo/index-192.npz;
later runs load the cache and start instantly.

    python demo/ask.py                      # interactive
    python demo/ask.py "Kailan naitatag ang University of Vermont?"
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from urllib.parse import unquote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "demo/model"
CACHE = ROOT / "demo/index-192.npz"
EVAL = ROOT / "runs/pycon-compact-sota-20261005-055331-d48482/verification-20261005T090654Z/evaluation"
WINDOW, STRIDE = 192, 128          # the setting selected on development


def parent_texts(chunks: list[dict]) -> dict[str, str]:
    """Rebuild each article from the frozen overlapping windows, by offset."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["parent_id"]].append(chunk)
    texts = {}
    for parent_id, group in grouped.items():
        group = sorted(group, key=lambda c: c["start"])
        buffer: list[str | None] = [None] * max(c["end"] for c in group)
        for chunk in group:
            for offset, character in enumerate(chunk["text"]):
                buffer[chunk["start"] + offset] = character
        if any(c is None for c in buffer):
            raise ValueError(f"{parent_id}: reconstruction left a hole")
        texts[parent_id] = "".join(buffer)  # type: ignore[arg-type]
    return texts


def rechunk(parents: list[str], texts: dict[str, str]):
    chunks, owners = [], []
    for index, parent_id in enumerate(parents):
        text = texts[parent_id]
        starts = list(range(0, max(1, len(text) - WINDOW + 1), STRIDE))
        if starts[-1] + WINDOW < len(text):
            starts.append(len(text) - WINDOW)
        seen = set()
        for begin in starts:
            begin = max(0, begin)
            end = min(len(text), begin + WINDOW)
            if (begin, end) in seen or end <= begin:
                continue
            seen.add((begin, end))
            chunks.append({"parent_id": parent_id, "text": text[begin:end]})
            owners.append(index)
    return chunks, np.asarray(owners)


def main() -> None:
    from model2vec import StaticModel
    from model2vec.train.utils import get_probable_pad_token_id
    sys.path.insert(0, str(ROOT / "src"))
    from pycon_model2vec_demo.multiwikiqa_compact_retrieval import strip_padding_and_unknown

    student = StaticModel.from_pretrained(str(MODEL))
    weights = np.asarray(student.embedding, dtype=np.float32)
    tokenizer = student.tokenizer
    config = json.loads((MODEL / "config.json").read_text())
    query_prefix = config.get("query_prefix", "")
    pad_id = get_probable_pad_token_id(tokenizer)
    unknown_id = student.unk_token_id
    print(f"model: {weights.shape[0]:,} rows x {weights.shape[1]}d  "
          f"({sum(p.stat().st_size for p in MODEL.rglob('*') if p.is_file()):,} bytes)")

    def encode(texts: list[str], *, query: bool) -> np.ndarray:
        values = [(query_prefix + t) if query else t for t in texts]
        out = np.zeros((len(values), weights.shape[1]), dtype=np.float32)
        for start in range(0, len(values), 1024):
            batch = values[start : start + 1024]
            for offset, enc in enumerate(tokenizer.encode_batch_fast(batch, add_special_tokens=False)):
                ids = strip_padding_and_unknown(enc.ids, pad_id=pad_id, unknown_id=unknown_id)
                if ids:
                    out[start + offset] = weights[ids].mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    layout = json.loads((EVAL / "shared-layout.json").read_text())
    parents = layout["parents"]
    chunks, owners = rechunk(parents, parent_texts(layout["chunks"]))
    titles = {}
    for row in (json.loads(l) for l in (EVAL / "annotations.all.jsonl").read_text().splitlines() if l.strip()):
        url_title = unquote(row.get("source_url", "").rsplit("/", 1)[-1]).replace("_", " ")
        titles.setdefault(row["gold_parent_id"], row.get("source_title") or url_title or row["gold_parent_id"])

    if CACHE.exists():
        document_vectors = np.load(CACHE)["d"]
        if document_vectors.shape[0] != len(chunks):
            raise ValueError("cache is stale; delete demo/index-192.npz")
        print(f"index: {len(chunks):,} chunks over {len(parents):,} articles (cached)")
    else:
        print(f"encoding {len(chunks):,} chunks over {len(parents):,} articles ...", flush=True)
        began = time.time()
        document_vectors = encode([c["text"] for c in chunks], query=False)
        np.savez_compressed(CACHE, d=document_vectors)
        print(f"  done in {time.time()-began:.1f}s, cached to {CACHE.name}")

    names = np.asarray(parents)
    boundaries = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
    spans = list(zip(boundaries, np.r_[boundaries[1:], len(owners)]))

    def ask(question: str, top: int = 10) -> None:
        began = time.time()
        q = encode([question], query=True)[0]
        similarities = document_vectors @ q
        best = np.array([a + int(np.argmax(similarities[a:b])) for a, b in spans])
        scores = similarities[best]
        order = np.lexsort((names, -scores))[:top]
        elapsed = (time.time() - began) * 1000
        print(f'\n"{question}"   [{elapsed:.0f} ms]')
        for rank, i in enumerate(order, 1):
            snippet = " ".join(chunks[best[i]]["text"].split())[:96]
            print(f"  {rank:>2}. {scores[i]:.4f}  {titles.get(parents[i], parents[i])[:44]:<44} | {snippet}")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return
    print("\nType a Tagalog question (blank line or Ctrl-D to quit).")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            break
        ask(line)


if __name__ == "__main__":
    main()
