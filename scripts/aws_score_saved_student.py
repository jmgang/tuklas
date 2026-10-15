#!/usr/bin/env python3
"""Score a preserved int8 student package on development and held-out, in AWS.

Runs on CPU: a static model is a lookup table, so scoring needs no GPU.  The
package is read from its immutable S3 prefix and every artifact is written back
to S3, so nothing touches a local workstation disk.

Why this exists: a candidate that passes both promotion gates reaches the final
report, where ``mining_manifest`` was referenced outside the scope it was
assigned in.  That raised before the held-out numbers were persisted, so a
passing candidate's held-out measurement was computed and then lost.  The
package itself survives, so the measurement is recoverable without retraining.

Query handling mirrors the training worker exactly: the ``[QRY]`` role token
prefixes every question, padding and unknown ids are stripped, token vectors are
mean pooled then L2 normalised.  Ranking and scoring reuse
``multiwikiqa_evaluation.rank_parents`` and ``score`` so the numbers are
directly comparable with every other figure in the project.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/score-work")
sys.path.insert(0, str(INPUT))

TEACHER_DEVELOPMENT_RECALL_AT_10 = 0.9746192893401015
# Teacher reference for the 3,984-question Tagalog held-out cohort, measured by
# the training workers themselves.  docs/full-heldout-results-20261005.md quotes
# 0.9694, but that is the earlier *bilingual* evaluation over a different cohort
# (7,922 questions) and overstates this cohort's teacher by 0.354 points.  Two
# independent runs agree on the real value through their own gate arithmetic:
#   544D: student 0.816265 + reported gap 0.149598 = 0.965863
#   173 : student 0.817771 + reported gap 0.148092 = 0.965863
# which is exactly 3848/3984.
TEACHER_HELDOUT_RECALL_AT_10 = 3848 / 3984
METRICS = (
    "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
    "AnswerSpanCoverage@1", "AnswerSpanCoverage@3",
    "AnswerSpanCoverage@5", "AnswerSpanCoverage@10",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    plan = json.loads((INPUT / "score-plan.json").read_text())
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
    listing = s3.list_objects_v2(Bucket=bucket, Prefix=model_prefix)
    package_bytes = 0
    for row in listing.get("Contents", []):
        name = row["Key"][len(model_prefix):]
        if not name:
            continue
        destination = model_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, row["Key"], str(destination))
        package_bytes += int(row["Size"])
        print("FETCHED", name, row["Size"], flush=True)
    if not package_bytes:
        raise ValueError("no package objects under " + model_prefix)

    # Optional vocabulary pruning.  SEA-LION's 250k rows cover eleven South-East
    # Asian languages; only ~70k are ever emitted for Tagalog plus the
    # single-character insurance set.  The tokenizer is Unigram, so ids are list
    # indices and pruning is a subset selection: Viterbi picks the
    # max-probability segmentation, and deleting a token that never appears in a
    # chosen segmentation leaves that segmentation available and still optimal.
    keep_path = INPUT / "keep-token-ids.json"
    pruned_dir = WORK / "pruned"
    if keep_path.exists():
        from tokenizers import Tokenizer
        import shutil

        raw = json.loads((model_dir / "tokenizer.json").read_text())
        if raw["model"]["type"] != "Unigram":
            raise ValueError("pruning assumes a Unigram tokenizer")
        vocab = raw["model"]["vocab"]
        requested = set(json.loads(keep_path.read_text()))
        # [QRY] is an *added* token appended past model.vocab by
        # add_neutral_query_role, so the embedding matrix has len(vocab)+1 rows.
        # Filtering ids by len(vocab) silently dropped the query-role row, which
        # cost every question its role token.  Keep vocab rows and appended rows
        # separately, and order the matrix slice to match the rebuilt tokenizer.
        vocab_ids = sorted(i for i in requested if i < len(vocab))
        appended_ids = sorted(i for i in requested if i >= len(vocab))
        keep = vocab_ids + appended_ids
        remap = {old: new for new, old in enumerate(keep)}
        pruned = json.loads(json.dumps(raw))
        pruned["model"]["vocab"] = [vocab[i] for i in vocab_ids]
        pruned["model"]["unk_id"] = remap[raw["model"]["unk_id"]]
        pruned["added_tokens"] = [
            {**t, "id": remap[t["id"]]}
            for t in (raw.get("added_tokens") or []) if t["id"] in remap
        ]
        for token in raw.get("added_tokens") or []:
            if token["id"] not in remap:
                raise ValueError(f"added token {token['content']} would be pruned away")
        if raw["model"]["unk_id"] not in remap:
            raise ValueError("the unknown-token row must be kept")
        # get_probable_pad_token_id falls back to 0 with only a log warning, which
        # would make strip_padding_and_unknown delete real token id 0 everywhere.
        pad_names = {"<pad>", "[PAD]", "<padding>"}
        pad_rows = {i for i, (t, _) in enumerate(vocab) if t in pad_names}
        if pad_rows and not (pad_rows & set(remap)):
            raise ValueError("the padding-token row must be kept")
        original_tokenizer = Tokenizer.from_str((model_dir / "tokenizer.json").read_text())
        pruned_dir.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "modules.json", "README.md"):
            if (model_dir / name).exists():
                shutil.copy2(model_dir / name, pruned_dir / name)
        (pruned_dir / "tokenizer.json").write_text(json.dumps(pruned))

        from safetensors.numpy import load_file, save_file
        tensors = load_file(str(model_dir / "model.safetensors"))
        key = next(iter(tensors))
        full = tensors[key]
        if full.shape[0] not in (len(vocab), len(vocab) + 1):
            raise ValueError(f"matrix rows {full.shape[0]} do not match vocabulary {len(vocab)}")
        if keep[-1] >= full.shape[0]:
            raise ValueError(f"keep-set id {keep[-1]} exceeds matrix rows {full.shape[0]}")
        save_file({key: full[keep]}, str(pruned_dir / "model.safetensors"))

        pruned_tokenizer = Tokenizer.from_str((pruned_dir / "tokenizer.json").read_text())
        print(f"PRUNED {full.shape[0]:,} -> {len(keep):,} rows "
              f"({len(vocab_ids):,} vocab + {len(appended_ids):,} appended)", flush=True)
        prune_audit = {
            "source_rows": int(full.shape[0]),
            "kept_rows": len(keep),
            "kept_vocab_rows": len(vocab_ids),
            "kept_appended_rows": len(appended_ids),
            "pruned_bytes": sum(p.stat().st_size for p in pruned_dir.rglob("*") if p.is_file()),
            "original_bytes": package_bytes,
        }
    else:
        original_tokenizer = pruned_tokenizer = None
        prune_audit = None

    student = StaticModel.from_pretrained(str(pruned_dir if prune_audit else model_dir))
    weights = np.asarray(student.embedding, dtype=np.float32)
    tokenizer = student.tokenizer
    config = json.loads((model_dir / "config.json").read_text())
    query_prefix = config.get("query_prefix", "")
    pad_id = get_probable_pad_token_id(tokenizer)
    unknown_id = student.unk_token_id

    layout = json.loads((INPUT / "shared-layout.json").read_text())
    annotations = read_jsonl(INPUT / "annotations.all.jsonl")
    chunks = layout["chunks"]
    parents = layout["parents"]
    positions = {parent: index for index, parent in enumerate(parents)}
    owners = [positions[chunk["parent_id"]] for chunk in chunks]

    tagalog = [row for row in annotations if row["language"] == "tl"]
    _, development, _ = prepare(tagalog, chunks)
    heldout = [row for row in tagalog if row["partition"] == "evaluation"]
    if len(development) != 197 or len(heldout) != 3_984:
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

    print(f"encoding {len(chunks)} documents", flush=True)
    document_vectors = encode([row["text"] for row in chunks], query=False)

    def evaluate(cohort: list[dict], name: str, slug: str) -> dict:
        print(f"encoding {len(cohort)} {name} questions", flush=True)
        query_vectors = encode([row["question"] for row in cohort], query=True)
        rows = []
        # Per-question rows are emitted so arms can be compared with
        # ``paired_cluster_ci``: the aggregate alone cannot distinguish a real
        # difference from sampling noise, and the 3,984 questions sit on only
        # 1,294 source articles, so they are not independent observations.  The
        # row shape (query_id, gold_parent_id, metrics) is exactly what that
        # bootstrap consumes, and the order follows the cohort, which is
        # deterministic across jobs.
        detail = []
        for start in range(0, len(cohort), 256):
            rankings = rank_parents(
                query_vectors[start : start + 256], document_vectors, owners, parents
            )
            for query, ranking in zip(cohort[start : start + 256], rankings, strict=True):
                metrics = score(query, ranking, chunks)
                rows.append(metrics)
                detail.append({
                    "query_id": query["query_id"],
                    "gold_parent_id": query["gold_parent_id"],
                    "metrics": metrics,
                })
        path = WORK / f"per-question-{slug}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in detail:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        s3.upload_file(str(path), bucket, output_prefix + path.name)
        print("UPLOADED", output_prefix + path.name, flush=True)
        # Four arms emit identically-named files whose query_id order is
        # identical by construction, so pairing the wrong file with the wrong
        # arm would not raise anywhere downstream.  Bind each file to its arm.
        manifest = WORK / f"per-question-{slug}-manifest.json"
        manifest.write_text(json.dumps({
            "label": plan["label"],
            "model_prefix": model_prefix,
            "source_job": plan.get("source_job"),
            "cohort": name,
            "slug": slug,
            "questions": len(detail),
            "rows": path.name,
            "dimension": int(weights.shape[1]),
            "vocabulary_rows": int(weights.shape[0]),
            "duplicate_query_identities": len(detail) - len({r["query_id"] for r in detail}),
        }, indent=2, sort_keys=True) + "\n")
        s3.upload_file(str(manifest), bucket, output_prefix + manifest.name)
        print("UPLOADED", output_prefix + manifest.name, flush=True)
        # Judged only after both artifacts are preserved.
        if len({row["query_id"] for row in detail}) != len(detail):
            raise ValueError(f"duplicate query identities in {name} rows")
        result = {"questions": len(rows)}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        return result

    development_metrics = evaluate(development, "development", "development")
    heldout_metrics = evaluate(heldout, "held-out", "heldout")
    development_gap = TEACHER_DEVELOPMENT_RECALL_AT_10 - development_metrics["Recall@10"]
    heldout_gap = TEACHER_HELDOUT_RECALL_AT_10 - heldout_metrics["Recall@10"]

    verification = None
    if prune_audit is not None:
        # The decisive check: pruning must not change how the indexed corpus or
        # the development questions tokenise.
        changed = 0
        # Compare remapped ids, not sequence lengths: a re-segmentation that
        # happens to keep the token count would otherwise pass as unchanged.
        # Questions carry the query prefix so the [QRY] role token's atomicity
        # is verified too -- the same invariant tokenize() asserts in training.
        probe = (
            [row["text"] for row in chunks]
            + [query_prefix + row["question"] for row in development]
        )
        for start in range(0, len(probe), 1000):
            batch = probe[start : start + 1000]
            before = original_tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            after = pruned_tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            for a, b in zip(before, after):
                if [remap[i] for i in a.ids if i in remap] != list(b.ids):
                    changed += 1
                elif any(i not in remap for i in a.ids):
                    changed += 1
        if changed:
            raise ValueError(
                f"pruning changed tokenisation on {changed} of {len(probe)} texts"
            )
        verification = {
            **prune_audit,
            "indexed_and_development_texts_checked": len(probe),
            "tokenization_length_changed": changed,
        }
        print("VERIFICATION", json.dumps(verification), flush=True)
        for path in sorted(pruned_dir.rglob("*")):
            if path.is_file():
                s3.upload_file(
                    str(path), bucket,
                    output_prefix + "pruned-model/" + path.relative_to(pruned_dir).as_posix(),
                )
        print("UPLOADED pruned-model/", flush=True)

    report = {
        "pruning": verification,
        "label": plan["label"],
        "model_prefix": model_prefix,
        "source_job": plan.get("source_job"),
        "dimension": int(weights.shape[1]),
        "vocabulary_rows": int(weights.shape[0]),
        "package_bytes": prune_audit["pruned_bytes"] if prune_audit else package_bytes,
        "original_package_bytes": package_bytes,
        "development": development_metrics,
        "heldout": heldout_metrics,
        "teacher_development_recall_at_10": TEACHER_DEVELOPMENT_RECALL_AT_10,
        "teacher_heldout_recall_at_10": TEACHER_HELDOUT_RECALL_AT_10,
        "teacher_heldout_source": "measured by the training workers (3848/3984); not the 0.9694 bilingual figure in docs/full-heldout-results-20261005.md",
        "development_gap": development_gap,
        "heldout_gap": heldout_gap,
        "development_minus_heldout": development_metrics["Recall@10"] - heldout_metrics["Recall@10"],
        "gate_maximum_gap": 0.10,
        "development_gate_passed": bool(development_gap <= 0.10),
        "heldout_gate_passed": bool(heldout_gap <= 0.10),
        "scored_on_cpu": True,
        "retrained": False,
        "heldout_use": (
            "verification-only; arm selection is decided on the development "
            "paired interval and held-out only confirms the 10-point gate"
        ),
        "heldout_used_for_training_or_checkpoint_selection": False,
    }
    path = WORK / "dev-and-heldout-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    s3.upload_file(str(path), bucket, output_prefix + "dev-and-heldout-report.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("UPLOADED", output_prefix + "dev-and-heldout-report.json", flush=True)


if __name__ == "__main__":
    main()
