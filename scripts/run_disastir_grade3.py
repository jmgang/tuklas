"""Evaluate the frozen DisastIR models with grade-3 judgments only.

Two separate views are produced: all queries with mapped grade-3 passages,
and the strict subset with exactly one raw and mapped grade-3 passage.
Existing model vectors and the original held-out report are read only.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycon_model2vec_demo.corpus import load_corpus, map_query_groups
from pycon_model2vec_demo.data import load_qa_collection
from pycon_model2vec_demo.grade3 import paired_with_teacher, score_query, select_cohorts, summarize
from pycon_model2vec_demo.model_registry import MODEL_NAMES, local_model_artifacts
from pycon_model2vec_demo.retrieval import exact_top_k, open_corpus_embeddings


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def verify_or_write(path: Path, value: object) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing artifact has different provenance: {path}")
    else:
        write_new(path, value)


def model_result(name: str, artifact: object, queries: tuple, corpus: object, corpus_hash: str, output_dir: Path, cohort_hash: str, block_size: int) -> dict:
    result_path = output_dir / f"{name}-query-results.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["cohort_sha256"] != cohort_hash or result["model_fingerprint"] != artifact.fingerprint or result["corpus_sha256"] != corpus_hash:
            raise ValueError(f"Saved {name} result has different provenance")
        if [row["query_id"] for row in result["per_query"]] != [query.query_id for query in queries]:
            raise ValueError(f"Saved {name} result query order differs")
        print(f"Verified existing result: {name}", flush=True)
        return result

    cache = ROOT / "runs/embeddings" / f"{name}-{artifact.fingerprint[:12]}.npy"
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise FileNotFoundError(f"Missing existing full DisastIR corpus embeddings for {name}: {cache}")
    vectors = open_corpus_embeddings(cache, corpus=corpus, corpus_sha256=corpus_hash, model_fingerprint=artifact.fingerprint)
    encoder = artifact.load_encoder()
    rows: list[dict] = []
    zero_vectors = 0
    for start in range(0, len(queries), 8):
        batch = queries[start:start + 8]
        query_vectors = encoder.encode_queries([query.query for query in batch])
        zero_vectors += int(np.count_nonzero(np.linalg.norm(query_vectors, axis=1) == 0))
        rankings = exact_top_k(query_vectors, vectors, corpus.ids, k=100, corpus_block_size=block_size)
        for query, ranking in zip(batch, rankings):
            rows.append({"query_id": query.query_id, "category": query.category, "raw_grade3_count": query.raw_grade3_count, "mapped_grade3_count": len(query.gold_ids), "metrics": score_query(query, ranking)})
        if (start // 8 + 1) % 20 == 0:
            print(f"{name}: scored {start + len(batch)}/{len(queries)} queries", flush=True)
    result = {
        "model": name,
        "model_fingerprint": artifact.fingerprint,
        "model_files_sha256": artifact.files_sha256,
        "corpus_sha256": corpus_hash,
        "cohort_sha256": cohort_hash,
        "zero_query_vectors": zero_vectors,
        "per_query": rows,
    }
    write_new(result_path, result)
    print(f"Finished {name}", flush=True)
    return result


def render_markdown(report: dict) -> str:
    lines = [
        f"# DisastIR grade-3 retrieval: {report['cohort']}", "",
        f"{report['query_count']:,} held-out questions; {report['corpus_passage_count']:,} passages; exact full-corpus search.", "",
        "| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Unjudged@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_100", "mrr_at_10", "ndcg_at_10", "unjudged_at_10")
    for name in MODEL_NAMES:
        overall = report["models"][name]["overall"]
        lines.append("| " + name + " | " + " | ".join(f"{overall[key]:.4f}" for key in labels) + " |")
    audit = report["selection_audit"]
    lines.extend([
        "", "## Selection audit", "",
        f"- Original held-out questions: {audit['heldout_queries']:,}.",
        f"- Strict single-gold questions: {audit['strict_single_queries']:,}.",
        f"- Questions with at least one mapped grade-3 passage: {audit['all_mapped_grade3_queries']:,}.",
        f"- Raw grade-3 judgments not mapped or collapsed: {audit['unmapped_or_collapsed_grade3']:,}.",
        f"- Strict selection counts: `{json.dumps(audit['strict_selection_counts'], sort_keys=True)}`.",
        "", "## Interpretation", "",
        "- Only released grade-3 judgments count as relevant. Grade-1/2 judgments are not labeled negative.",
        "- Recall divides by mapped grade-3 passages. In the single-gold view, each query has exactly one such passage.",
        "- The query subset was defined after the first held-out evaluation; these are exploratory results, not a fresh untouched test set.",
        "- DisastIR queries are generated and judgments are pooled and LLM-rated. Unjudged retrieved passages have unknown relevance.",
        "- This is passage retrieval, not generated-answer or production-user performance.",
        "", "## Provenance", "",
        f"- Corpus SHA-256: `{report['provenance']['corpus_sha256']}`.",
        f"- Cohort SHA-256: `{report['provenance']['cohort_sha256']}`.",
        f"- Protocol SHA-256: `{report['provenance']['protocol_sha256']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    data_path = ROOT / "configs/disastir-qa-all-categories.json"
    protocol_path = ROOT / "configs/grade3-v1.json"
    retrieval_path = ROOT / "configs/retrieval-v1.json"
    data = read_json(data_path)
    protocol = read_json(protocol_path)
    retrieval = read_json(retrieval_path)
    if protocol["dataset"] != data["dataset"] or protocol["corpus_sha256"] != data["corpus_sha256"] or protocol["grade"] != 3 or tuple(protocol["models"]) != MODEL_NAMES or protocol["top_k"] != 100:
        raise ValueError("Grade-3 protocol differs from frozen DisastIR inputs")
    if retrieval["secondary_top_k"] != 100 or retrieval["corpus_block_size"] <= 0:
        raise ValueError("Existing retrieval settings differ from grade-3 protocol")
    corpus = load_corpus(ROOT / "data/raw/corpus.jsonl.gz", expected_sha256=data["corpus_sha256"])
    prepared = load_qa_collection(data_dir=ROOT / "data/raw", revision=data["revision"], category_checksums=data["categories"], seed=data["split_seed"], evaluation_fraction=data["evaluation_fraction"])
    groups = prepared.evaluation
    if [group.query_id for group in groups] != prepared.manifest["evaluation_query_ids"]:
        raise ValueError("Held-out query order changed")
    mapped, mapping_audit = map_query_groups(groups, corpus)
    strict, broad, selection_audit = select_cohorts(groups, mapped, corpus)
    if len(groups) != 1265 or len(strict) != 337 or len(broad) != 1127:
        raise ValueError("Grade-3 query counts differ from the audited pinned release")
    artifacts = local_model_artifacts(ROOT)
    protocol_sha = sha256(protocol_path.read_bytes()).hexdigest()
    cohort = {
        "dataset": data["dataset"], "revision": data["revision"], "corpus_sha256": data["corpus_sha256"],
        "protocol_sha256": protocol_sha,
        "source_sha256": {"config": sha256(data_path.read_bytes()).hexdigest(), "categories": data["categories"]},
        "model_fingerprints": {name: artifacts[name].fingerprint for name in MODEL_NAMES},
        "heldout_query_ids": [group.query_id for group in groups],
        "all_grade3_query_ids": [query.query_id for query in broad],
        "strict_single_query_ids": [query.query_id for query in strict],
        "selection_audit": selection_audit,
        "mapping_audit": mapping_audit,
    }
    cohort_sha = sha256(json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_dir = ROOT / "runs" / f"disastir-grade3-{cohort_sha[:12]}"
    verify_or_write(output_dir / "cohort.json", cohort)
    print(f"Frozen DisastIR cohorts: {len(strict)} single-gold, {len(broad)} all-grade3; run {cohort_sha[:12]}", flush=True)
    results = {name: model_result(name, artifacts[name], broad, corpus, data["corpus_sha256"], output_dir, cohort_sha, retrieval["corpus_block_size"]) for name in MODEL_NAMES}
    strict_ids = {query.query_id for query in strict}
    for cohort_name in ("single", "all"):
        selected_rows = {name: [row for row in results[name]["per_query"] if cohort_name == "all" or row["query_id"] in strict_ids] for name in MODEL_NAMES}
        expected_ids = [query.query_id for query in (strict if cohort_name == "single" else broad)]
        if any([row["query_id"] for row in selected_rows[name]] != expected_ids for name in MODEL_NAMES):
            raise ValueError("Scored query order differs from selected cohort")
        report = {
            "status": "posthoc-grade3-evaluation-repair",
            "cohort": cohort_name,
            "query_count": len(expected_ids),
            "corpus_passage_count": len(corpus.ids),
            "selection_audit": selection_audit,
            "mapping_audit": mapping_audit,
            "provenance": {"corpus_sha256": data["corpus_sha256"], "cohort_sha256": cohort_sha, "protocol_sha256": protocol_sha, "model_fingerprints": cohort["model_fingerprints"], "query_ids": expected_ids, "source_sha256": cohort["source_sha256"]},
            "models": {name: {**summarize(selected_rows[name]), "zero_query_vectors_in_all_grade3": results[name]["zero_query_vectors"]} for name in MODEL_NAMES},
            "paired_with_teacher": {name: paired_with_teacher(selected_rows["teacher"], selected_rows[name]) for name in MODEL_NAMES if name != "teacher"},
            "limitations": ["The subset was defined after opening the original held-out results, so it is exploratory.", "Relevance judgments are pooled and LLM-rated, and unjudged passages have unknown relevance.", "Some grade-3 judgments do not map to the released corpus; coverage is reported.", "The released QA questions are generated rather than production-user questions.", "The original operational measurements use different hardware for the teacher corpus build and student builds."],
        }
        report_path = output_dir / f"{cohort_name}-report.json"
        verify_or_write(report_path, report)
        markdown_path = output_dir / f"{cohort_name}-report.md"
        markdown = render_markdown(report)
        if markdown_path.exists():
            if markdown_path.read_text(encoding="utf-8") != markdown:
                raise ValueError(f"Existing Markdown differs: {markdown_path}")
        else:
            with markdown_path.open("x", encoding="utf-8") as stream:
                stream.write(markdown)
        print(f"{cohort_name} report: {report_path}", flush=True)
        for name in MODEL_NAMES:
            print(f"{cohort_name} {name}: {report['models'][name]['overall']}", flush=True)


if __name__ == "__main__":
    main()
