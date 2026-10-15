#!/usr/bin/env python3
"""Direct SEA-LION -> Model2Vec -> TokenLearn retrieval student.

This managed SageMaker worker uses one pinned neural parent throughout.  The
held-out evaluation questions are not read by checkpoint selection; they are
scored once only after the development and complete-artifact gates pass.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import gc
import importlib
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import sys

import numpy as np


INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/direct-sealion-work")
sys.path.insert(0, str(INPUT))
os.environ["SEALION_INPUT_DIR"] = str(INPUT)
os.environ["SEALION_WORK_DIR"] = str(WORK)
original = importlib.import_module("aws_sealion_train_repair_v3")
CONFIG = json.loads((INPUT / "settings.json").read_text())
PLAN = json.loads((INPUT / "mixed-plan.json").read_text())
RECIPE = json.loads((INPUT / "direct-parent.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalized(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def batch_unique(rows: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    pending = list(rows)
    random.Random(seed).shuffle(pending)
    result: list[list[dict]] = []
    while pending:
        current, rest, groups, queries = [], [], set(), set()
        for row in pending:
            query = normalized(row["query"])
            if len(current) >= batch_size or row["positive_group"] in groups or query in queries:
                rest.append(row)
            else:
                current.append(row)
                groups.add(row["positive_group"])
                queries.add(query)
        if not current:
            raise RuntimeError("nonduplicate batch construction made no progress")
        result.append(current)
        pending = rest
    return result


def trie_pattern(strings: set[str]) -> str:
    """Build an efficient exact-match regex for Model2Vec token pruning."""

    end = ""
    root: dict = {}
    for value in strings:
        node = root
        for character in value:
            node = node.setdefault(character, {})
        node[end] = {}

    def emit(node: dict) -> str:
        terminal = end in node
        parts = [
            re.escape(character) + emit(child)
            for character, child in sorted(node.items())
            if character != end
        ]
        if not parts:
            return ""
        body = parts[0] if len(parts) == 1 else "(?:" + "|".join(parts) + ")"
        return "(?:" + body + ")?" if terminal else body

    return emit(root)


def main() -> None:  # noqa: C901, PLR0915
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "events").mkdir(exist_ok=False)
    preserve = original.Preservation()
    preserve.event("dependency-setup")
    original.bootstrap()

    import torch
    import torch.nn.functional as F
    from model2vec import StaticModel
    from model2vec.distill import distill
    from model2vec.quantization import DType, quantize_embeddings
    from model2vec.train.utils import get_probable_pad_token_id
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from skeletoken import TokenizerModel

    from direct_parent_protocol import (
        validate_direct_parent_recipe,
        validate_tagalog_direct_parent_recipe,
    )
    from full_rank_pca import (
        explained_variance_fraction,
        fit_full_rank_pca,
        transform_full_rank_pca_batched,
        transform_pca_projection_batched,
    )
    from multiwikiqa_compact_retrieval import (
        add_neutral_query_role,
        external_article_id_from_passage_id,
        listwise_teacher_distillation_loss,
        strip_padding_and_unknown,
        validate_measured_artifact,
        validate_teacher_distance_gate,
        validate_tagalog_teacher_distance_gate,
    )
    from multiwikiqa_evaluation import rank_parents, score
    from multiwikiqa_ranking_v5 import prepare as prepare_development
    from sealion_runtime import verify_runtime
    from sealion_training import official_training_adapter
    from tagalog_capacity_frontier import validate_candidate_recipe

    tagalog_only = RECIPE.get("language") == "tl"
    capacity_frontier = str(RECIPE.get("protocol", "")).startswith(
        "multiwikiqa-sealion-tagalog-capacity-frontier-v"
    )
    if capacity_frontier:
        validate_candidate_recipe(RECIPE)
        candidate_id = os.environ.get("SEALION_CAPACITY_CANDIDATE", "")
        candidates = {row["id"]: row for row in RECIPE["candidates"]}
        if candidate_id not in candidates:
            raise ValueError("capacity-frontier candidate was not frozen in the job environment")
        candidate = candidates[candidate_id]
        student_dimension = int(candidate["dimension"])
        vocabulary_mode = candidate["vocabulary_mode"]
    elif tagalog_only:
        validate_tagalog_direct_parent_recipe(RECIPE)
        candidate_id = "tagalog_80k_1024"
        student_dimension = 1024
        vocabulary_mode = "tagalog-custom"
    else:
        validate_direct_parent_recipe(RECIPE)
        candidate_id = "bilingual_80k_1024"
        student_dimension = 1024
        vocabulary_mode = "tagalog-custom"
    verify_runtime(CONFIG["tokenlearn"]["commit"])
    if not torch.cuda.is_available():
        raise RuntimeError("direct SEA-LION training requires a CUDA GPU")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(8)
    bucket = PLAN["bucket"]

    def get(spec: dict) -> bytes:
        response = preserve.s3.get_object(Bucket=bucket, Key=spec["key"])
        try:
            raw = response["Body"].read()
        finally:
            response["Body"].close()
        if len(raw) != spec["bytes"] or sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("frozen S3 input changed: " + spec["key"])
        return raw

    expected = json.loads((INPUT / "input-manifest.json").read_text())
    for filename, spec in expected["files"].items():
        path = INPUT / filename
        if path.stat().st_size != spec["bytes"] or original.digest(path) != spec["sha256"]:
            raise ValueError("staged input changed: " + filename)

    teacher_id = CONFIG["teacher"]["model_id"]
    teacher_revision = CONFIG["teacher"]["revision"]
    if teacher_id != "aisingapore/SEA-LION-E5-Embedding-600M":
        raise ValueError("only the approved SEA-LION parent is allowed")
    if teacher_revision != RECIPE["teacher"]["revision"]:
        raise ValueError("SEA-LION revision changed")

    if vocabulary_mode == "full-parent":
        vocabulary = None
    else:
        vocabulary_path = INPUT / "vocabulary-selected.json"
        if not vocabulary_path.exists():
            # Backwards-compatible path for the already-preserved 80k protocol.
            vocabulary_path = INPUT / "vocabulary-80000.json"
        vocabulary = json.loads(vocabulary_path.read_text())
        expected_vocabulary = int(candidate.get("vocabulary_size", 160_000)) if capacity_frontier else 80_000
        if len(vocabulary) != expected_vocabulary or len(set(vocabulary)) != expected_vocabulary:
            raise ValueError(f"direct child vocabulary must contain {expected_vocabulary} unique tokens")
    parent_snapshot = WORK / "sealion-parent"
    snapshot_download(
        repo_id=teacher_id,
        revision=teacher_revision,
        local_dir=parent_snapshot,
        local_dir_use_symlinks=False,
    )
    parent_tokenizer = AutoTokenizer.from_pretrained(parent_snapshot, use_fast=True)
    token_remove_pattern = None
    retained_tokens = set()
    if vocabulary is not None:
        tokenizer_model = TokenizerModel.from_transformers_tokenizer(parent_tokenizer).prune_added_tokens()
        # Keep the parent pieces needed to encode single-token entries. Model2Vec
        # separately adds multiword entries from ``vocabulary`` as added tokens.
        retained_tokens = {
            piece
            for word in vocabulary
            for piece in tokenizer_model.preprocessor.preprocess(word, had_word_prefix=True)
        }
        for token_id in parent_tokenizer.all_special_ids:
            token = parent_tokenizer.convert_ids_to_tokens(token_id)
            if token:
                retained_tokens.add(token)
        token_remove_pattern = r"^(?!(?:" + trie_pattern(retained_tokens) + r")$).*"
    preserve.event(
        "model2vec-tokenizer-pruning-preflight",
        candidate=candidate_id,
        selected_surface_vocabulary=len(vocabulary) if vocabulary is not None else None,
        selected_language="tl" if tagalog_only else "bilingual",
        retained_parent_token_forms=len(retained_tokens),
        mechanism="full parent tokenizer" if vocabulary is None else "official token_remove_pattern",
    )
    logging.getLogger("model2vec.tokenizer.tokenizer").setLevel(logging.ERROR)
    preserve.event(
        "model2vec-direct-distillation-started",
        candidate=candidate_id,
        vocabulary=len(vocabulary) if vocabulary is not None else "full-parent",
        dimension=student_dimension,
    )
    direct_student = distill(
        model_name=str(parent_snapshot),
        vocabulary=vocabulary,
        device="cuda",
        pca_dims=student_dimension if capacity_frontier else "auto",
        sif_coefficient=0.0001,
        token_remove_pattern=token_remove_pattern,
        trust_remote_code=False,
        quantize_to="float32",
        pooling="mean",
    )
    if (
        direct_student.dim != student_dimension
        or direct_student.base_model_name != parent_snapshot.name
        or (
            capacity_frontier and vocabulary_mode == "full-parent"
            and not 240_000 <= direct_student.embedding.shape[0] <= 260_000
        )
        or (
            capacity_frontier and vocabulary_mode == "tagalog-custom"
            and not 159_000 <= direct_student.embedding.shape[0] <= 160_010
        )
        or (
            tagalog_only and not capacity_frontier
            and not 79_000 <= direct_student.embedding.shape[0] <= 80_010
        )
        or (
            not tagalog_only
            and direct_student.embedding.shape != (79_985, 1024)
        )
    ):
        raise ValueError("direct Model2Vec child has unexpected provenance or dimension")
    preserve.event(
        "model2vec-direct-distillation-complete-transient",
        vocabulary=int(direct_student.embedding.shape[0]),
        dimension=direct_student.dim,
        retained=False,
    )

    # Development retrieval is loaded before TokenLearn so every real
    # validation checkpoint can be selected by the downstream task. Held-out
    # questions remain excluded from this cohort and are not scored here.
    evaluation_layout = json.loads(get(PLAN["evaluation_layout"]))
    all_annotations = [json.loads(line) for line in get(PLAN["annotations"]).splitlines()]
    annotations = [row for row in all_annotations if row["language"] == "tl"] if tagalog_only else all_annotations
    chunks = evaluation_layout["chunks"]
    parents = evaluation_layout["parents"]
    expected_annotations = 5_001 if tagalog_only else 10_000
    if len(chunks) != 6546 or len(parents) != 1294 or len(annotations) != expected_annotations:
        raise ValueError("evaluation cohort changed")
    three_way_partition = RECIPE["training"].get("partition") == "three-way-v8"
    if three_way_partition:
        # The reported cohort is the same 197 questions the v3/v6 controls
        # reported, but it no longer drives checkpoint selection: a disjoint
        # 129-question cohort does that instead, so the headline number stops
        # being a maximum over every evaluation the ranking loop performed.
        from tagalog_ranking_v8 import prepare_three_way

        if not tagalog_only:
            raise ValueError("three-way-v8 partition is Tagalog-only")
        split = prepare_three_way(annotations, chunks)
        development_train = split["train"]
        development_selection = split["selection"]
        development_validation = split["report"]
        skipped = split["skipped"]
        original.write_json(WORK / "development-partition-audit.json", split["audit"])
        preserve.put(
            WORK / "development-partition-audit.json",
            "reports/development-partition-audit.json",
        )
        preserve.event("development-three-way-partition", **split["audit"])
    else:
        development_train, development_validation, skipped = prepare_development(
            annotations, chunks
        )
        development_selection = []
        expected_development = (818, 197, 2) if tagalog_only else (1734, 338, 6)
        if (len(development_train), len(development_validation), len(skipped)) != expected_development:
            raise ValueError("development partition changed")
    if any(row["partition"] != "development" for row in development_validation):
        raise ValueError("held-out query entered checkpoint selection")
    if any(row["partition"] != "development" for row in development_selection):
        raise ValueError("held-out query entered checkpoint selection")
    parent_positions = {parent: index for index, parent in enumerate(parents)}
    owners = [parent_positions[chunk["parent_id"]] for chunk in chunks]

    def metric_table(document_matrix: np.ndarray, query_matrix: np.ndarray, cohort: list[dict]) -> dict:
        rows = []
        for start in range(0, len(cohort), 64):
            rankings = rank_parents(
                query_matrix[start : start + 64], document_matrix, owners, parents
            )
            for query, ranking in zip(cohort[start : start + 64], rankings, strict=True):
                rows.append({"language": query["language"], "metrics": score(query, ranking, chunks)})
        result = {}
        metric_names = (
            "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
            "AnswerSpanCoverage@1", "AnswerSpanCoverage@3", "AnswerSpanCoverage@5",
            "AnswerSpanCoverage@10",
        )
        report_languages = ("tl",) if tagalog_only else ("en", "tl", "combined")
        for language in report_languages:
            selected = rows if language == "combined" else [r for r in rows if r["language"] == language]
            result[language] = {"questions": len(selected)}
            for metric in metric_names:
                values = [r["metrics"][metric] for r in selected if r["metrics"][metric] is not None]
                result[language][metric] = float(np.mean(values)) if values else None
        return result

    def normalized_matrix(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def static_stage_metrics(static_model) -> dict:
        documents = normalized_matrix(static_model.encode(
            [row["text"] for row in chunks], batch_size=1024, show_progress_bar=True
        ))
        queries = normalized_matrix(static_model.encode(
            [row["question"] for row in development_validation],
            batch_size=1024,
            show_progress_bar=True,
        ))
        return metric_table(documents, queries, development_validation)

    preserve.event("same-parent-feature-restore-started")
    feature_rows: list[dict] = []
    feature_texts: list[str] = []
    target_parts: list[np.ndarray] = []
    arm_ranges: dict[str, tuple[int, int]] = {}
    for arm in ("external_wikipedia", "approved_articles"):
        start = len(feature_rows)
        spec = PLAN["arms"][arm]
        layout = [json.loads(line) for line in get(spec["layout"]).decode().splitlines()]
        local = 0
        for pair in spec["pairs"]:
            source_texts = json.loads(get(pair["text"]))
            source_vectors = np.load(io.BytesIO(get(pair["vectors"])), allow_pickle=False)
            rows = layout[local : local + len(source_texts)]
            if source_texts != [row["text"] for row in rows] or source_vectors.shape != (len(source_texts), 1024):
                raise ValueError("same-parent feature alignment changed")
            if source_vectors.dtype != np.float32 or not np.isfinite(source_vectors).all():
                raise ValueError("same-parent feature matrix is malformed")
            local += len(source_texts)
            if tagalog_only:
                selected = [index for index, row in enumerate(rows) if row["language"] == "tl"]
                rows = [rows[index] for index in selected]
                texts = [source_texts[index] for index in selected]
                vectors = source_vectors[selected]
            else:
                texts, vectors = source_texts, source_vectors
            feature_rows.extend({**row, "arm": arm} for row in rows)
            feature_texts.extend(texts)
            target_parts.append(vectors)
        if local != spec["windows"]:
            raise ValueError("incomplete restored feature arm")
        arm_ranges[arm] = (start, len(feature_rows))
        preserve.event(
            "same-parent-feature-arm-restored",
            arm=arm,
            source_rows=local,
            selected_rows=len(feature_rows) - start,
            language="tl" if tagalog_only else "bilingual",
        )
    target_vectors = np.concatenate(target_parts, axis=0)
    del target_parts
    if target_vectors.shape[1] != 1024 or (tagalog_only and not feature_rows):
        raise ValueError("same-parent target cohort changed")
    if tagalog_only and any(row["language"] != "tl" for row in feature_rows):
        raise ValueError("English feature entered Tagalog-only TokenLearn")
    if not tagalog_only and target_vectors.shape != (182_724, 1024):
        raise ValueError("same-parent target cohort changed")

    preserve.event(
        "target-pca-started",
        rows=len(feature_rows),
        source_dimension=1024,
        output_dimension=student_dimension,
    )
    target_pca = fit_full_rank_pca(target_vectors, device="cuda", batch_size=2048)
    retained_variance = explained_variance_fraction(target_pca, student_dimension)
    if student_dimension == 1024:
        transformed_targets = transform_full_rank_pca_batched(
            target_vectors, target_pca, device="cuda", batch_size=2048
        )
    else:
        transformed_targets = transform_pca_projection_batched(
            target_vectors,
            target_pca,
            output_dimension=student_dimension,
            device="cuda",
            batch_size=2048,
        )
    pca_path = WORK / f"target-pca-{student_dimension}.npz"
    with pca_path.open("xb") as stream:
        np.savez(
            stream,
            mean=target_pca.mean,
            components=target_pca.components[:student_dimension],
            explained_variance=target_pca.explained_variance,
            retained_variance=np.asarray([retained_variance], dtype=np.float32),
            sample_count=np.asarray([target_pca.sample_count], dtype=np.int64),
        )
    preserve.put(pca_path, f"transforms/target-pca-{student_dimension}.npz")
    preserve.event(
        "target-pca-complete",
        output_dimension=student_dimension,
        retained_variance=retained_variance,
    )
    train_indices = [i for i, row in enumerate(feature_rows) if row["partition"] == "train"]
    validation_indices = [i for i, row in enumerate(feature_rows) if row["partition"] == "validation"]
    if tagalog_only and (not train_indices or not validation_indices):
        raise ValueError("Tagalog TokenLearn article split is empty")
    if not tagalog_only and (len(train_indices), len(validation_indices)) != (164_691, 18_033):
        raise ValueError("TokenLearn article split changed")
    validation_events: list[dict] = []

    def validation_event(metrics) -> None:
        values = {key: float(value) for key, value in metrics.items()}
        validation_events.append(values)
        preserve.event("tokenlearn-validation", **values)

    def tokenlearn_retrieval_selector(model, event_index: int, metrics: dict) -> dict:
        retrieval_metrics = static_stage_metrics(model.to_static_model())
        selection_value = sum(
            retrieval_metrics[language][metric]
            for language in (("tl",) if tagalog_only else ("en", "tl"))
            for metric in ("Recall@10", "nDCG@10")
        )
        event = {
            "selection_value": float(selection_value),
            "selection_metric": "development_recall_at_10_plus_ndcg_at_10",
            "event_index": event_index,
            "val_loss": float(metrics["val_loss"]),
            "retrieval_metrics": retrieval_metrics,
            "heldout_labels_used": False,
        }
        preserve.event("tokenlearn-checkpoint-retrieval", **event)
        return event

    with official_training_adapter(
        [feature_texts[i] for i in validation_indices],
        transformed_targets[validation_indices],
        None,
        validation_event,
        checkpoint_selector=tokenlearn_retrieval_selector
        if RECIPE["training"].get("tokenlearn_checkpoint_selection") == "development-retrieval"
        else None,
    ) as tokenlearn_state:
        import model2vec.train as model2vec_train

        trainable = model2vec_train.StaticModelForSimilarity.from_static_model(
            model=direct_student,
            out_dim=student_dimension,
            n_layers=0,
            freeze_weights=True,
            pad_token=None,
        )
        trainable.fit(
            X=[feature_texts[i] for i in train_indices],
            y=torch.from_numpy(transformed_targets[train_indices]),
            batch_size=256,
            device="cuda",
            learning_rate=0.001,
        )
    tokenlearn_student = trainable.to_static_model()
    if tokenlearn_student.dim != student_dimension:
        raise ValueError("TokenLearn changed the selected student dimension")
    preserve.event(
        "tokenlearn-same-parent-complete-transient",
        validation_events=len(validation_events),
        selected_validation_index=tokenlearn_state["selected_validation_index"],
        selected_validation_loss=tokenlearn_state["best_validation_value"],
        selection_mode=tokenlearn_state.get("selection_mode"),
        selected_retrieval_value=tokenlearn_state.get("best_selection_value"),
        vocabulary=int(tokenlearn_student.embedding.shape[0]),
        retained=False,
    )
    del trainable, transformed_targets
    gc.collect()
    torch.cuda.empty_cache()

    # Model2Vec's own SIF weighting, re-applied after TokenLearn.
    #
    # ``distill`` applies ``model2vec.distill.inference.compute_weights`` --
    # ``sif / (sif + proba)`` after ``apply_pca`` -- but it *estimates* ``proba``
    # from Zipf's law, assuming vocabulary order matches frequency order.
    # TokenLearn then retrains every row, which discards that weighting, and
    # nothing re-applies it.  This re-applies the same Model2Vec formula in the
    # same order, substituting measured corpus probabilities for the Zipf
    # estimate.  Both ``compute_weights``'s formula and ``apply_pca`` are used
    # unchanged; only the source of ``proba`` differs.
    sif_reweighting = RECIPE["training"].get("post_tokenlearn_sif_reweighting")
    ranking_initial_embedding = tokenlearn_student.embedding.astype(np.float32, copy=False)
    if sif_reweighting:
        from model2vec.distill.inference import apply_pca

        sif_coefficient = float(sif_reweighting["sif_coefficient"])
        if not 0.0 < sif_coefficient < 1.0:
            raise ValueError("SIF coefficient must be between zero and one")
        vocabulary_rows = int(ranking_initial_embedding.shape[0])
        corpus_counts = np.zeros(vocabulary_rows, dtype=np.float64)
        corpus_pad_id = get_probable_pad_token_id(tokenlearn_student.tokenizer)
        for start in range(0, len(feature_texts), 1024):
            encodings = tokenlearn_student.tokenizer.encode_batch_fast(
                feature_texts[start : start + 1024], add_special_tokens=False
            )
            ids = np.fromiter(
                (
                    token
                    for encoding in encodings
                    for token in strip_padding_and_unknown(
                        encoding.ids,
                        pad_id=corpus_pad_id,
                        unknown_id=tokenlearn_student.unk_token_id,
                    )
                ),
                dtype=np.int64,
            )
            if ids.size:
                corpus_counts += np.bincount(ids, minlength=vocabulary_rows)
        total_tokens = float(corpus_counts.sum())
        if total_tokens <= 0.0:
            raise ValueError("re-regularization corpus produced no tokens")
        probabilities = corpus_counts / total_tokens
        sif_weights = sif_coefficient / (sif_coefficient + probabilities)
        ranking_initial_embedding = apply_pca(
            ranking_initial_embedding, ranking_initial_embedding.shape[1]
        ).astype(np.float32, copy=False)
        ranking_initial_embedding = (
            ranking_initial_embedding * sif_weights[:, None]
        ).astype(np.float32, copy=False)
        sif_audit = {
            "mechanism": "measured-corpus-probability SIF after full-rank PCA",
            "formula": "sif / (sif + proba)",
            "sif_coefficient": sif_coefficient,
            "corpus_passages": len(feature_texts),
            "corpus_tokens": total_tokens,
            "vocabulary_rows": vocabulary_rows,
            "tokens_never_seen": int((corpus_counts == 0).sum()),
            "weight_minimum": float(sif_weights.min()),
            "weight_maximum": float(sif_weights.max()),
            "weight_median": float(np.median(sif_weights)),
            "official_formula_source": "model2vec.distill.inference.compute_weights",
            "zipf_rank_proxy_replaced_by_measured_counts": True,
        }
        original.write_json(
            WORK / "sif-reweighting-audit.json", sif_audit
        )
        preserve.put(
            WORK / "sif-reweighting-audit.json", "reports/sif-reweighting-audit.json"
        )
        preserve.event("post-tokenlearn-sif-reweighting", **sif_audit)

    expanded, retrieval_tokenizer, query_token_id = add_neutral_query_role(
        ranking_initial_embedding,
        tokenlearn_student.tokenizer,
        token="[QRY]",
    )
    unknown_id = tokenlearn_student.unk_token_id
    # StaticModel intentionally has no ``pad_token_id`` attribute.  Use the
    # same tokenizer-aware resolver as StaticModelForSimilarity so retrieval
    # preprocessing and TokenLearn agree on which batch-padding ID to drop.
    pad_id = get_probable_pad_token_id(tokenlearn_student.tokenizer)

    all_synthetic = read_jsonl(INPUT / "synthetic-admitted.jsonl")
    synthetic = [row for row in all_synthetic if row["language"] == "tl"] if tagalog_only else all_synthetic
    expected_synthetic = int(expected["retrieval"]["synthetic_queries"])
    if len(synthetic) != expected_synthetic:
        raise ValueError("verified synthetic query count changed")
    by_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_article: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_external_article: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(feature_rows):
        by_source[(row["arm"], str(row["source_id"]))].append(index)
        by_article[(row["arm"], str(row["article_id"]))].append(index)
        if row["arm"] == "external_wikipedia":
            key = external_article_id_from_passage_id(str(row["source_id"]))
            by_external_article[(row["arm"], key)].append(index)

    training_requests: list[dict] = []
    for row in synthetic:
        positive = by_source.get(("external_wikipedia", row["positive_passage_id"]), [])
        known = by_external_article.get(("external_wikipedia", row["positive_article_id"]), [])
        if not positive or not known:
            raise ValueError("synthetic positive cannot be mapped")
        training_requests.append({
            "id": "synthetic:" + row["record_id"] + ":" + row["query_type"],
            "query": row["query"],
            "language": row["language"],
            "arm": "external_wikipedia",
            "positive_indices": positive,
            "known_positive_indices": known,
            "positive_group": "external:" + row["positive_article_id"],
        })
    for row in development_train:
        positive = by_source.get(("approved_articles", row["chunk_id"]), [])
        if not positive:
            positive = [
                index for index in range(*arm_ranges["approved_articles"])
                if feature_rows[index]["text"] == row["positive"]
            ]
        known = by_article.get(("approved_articles", row["component_id"]), [])
        if not positive or not known:
            raise ValueError("development positive cannot be mapped")
        training_requests.append({
            "id": "development:" + row["query_id"],
            "query": row["question"],
            "language": row["language"],
            "arm": "approved_articles",
            "positive_indices": positive,
            "known_positive_indices": known,
            "positive_group": "approved:" + row["component_id"],
        })
    # The three-way partition moves 129 development questions out of training
    # into the disjoint selection cohort, so ranking supervision is 689 rather
    # than the frozen 818.  ``prepare_three_way`` asserts that split size
    # independently, so both ends stay pinned to a literal.
    expected_development_pairs = (
        689 if three_way_partition else (818 if tagalog_only else 1_734)
    )
    expected_requests = expected_synthetic + expected_development_pairs
    if len(training_requests) != expected_requests:
        raise ValueError("retrieval supervision count changed")

    query_prompt = (
        "Panuto: Maghanap ng kaugnay na talata na sumasagot sa tanong.\nTanong: "
        if tagalog_only
        else "Instruct: Given a passage that is guaranteed to contain the answer, "
        "retrieve relevant passages that answer the query.\nQuery: "
    )
    teacher = SentenceTransformer(
        str(parent_snapshot), device="cuda", trust_remote_code=False
    )
    if teacher.get_sentence_embedding_dimension() != 1024:
        raise ValueError("SEA-LION teacher dimension changed")
    teacher.max_seq_length = 512
    teacher_query_targets = teacher.encode(
        [row["query"] for row in training_requests],
        batch_size=64,
        normalize_embeddings=False,
        convert_to_numpy=True,
        prompt=query_prompt,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    projected_teacher_queries = transform_pca_projection_batched(
        teacher_query_targets,
        target_pca,
        output_dimension=student_dimension,
        device="cuda",
        batch_size=2048,
    )
    projected_teacher_queries = normalized_matrix(projected_teacher_queries)
    query_vectors = normalized_matrix(teacher_query_targets)
    document_vectors = target_vectors / np.maximum(
        np.linalg.norm(target_vectors, axis=1, keepdims=True), 1e-12
    )
    del teacher_query_targets
    preserve.event(
        "query-vector-distillation-targets-complete",
        queries=len(projected_teacher_queries),
        source_dimension=1024,
        output_dimension=student_dimension,
        same_passage_pca=True,
        teacher=teacher_id,
        heldout_labels_used=False,
    )

    preserve.event("hard-negative-mining-started", queries=len(training_requests), negatives=4)
    pools: dict[tuple[str, str], np.ndarray] = {}
    heldout_components = {
        row["component_id"] for row in annotations if row["partition"] == "evaluation"
    }
    for arm in ("external_wikipedia", "approved_articles"):
        for language in (("tl",) if tagalog_only else ("en", "tl")):
            values = []
            for index in range(*arm_ranges[arm]):
                row = feature_rows[index]
                if row["language"] != language:
                    continue
                if arm == "approved_articles" and row["article_id"] in heldout_components:
                    continue
                values.append(index)
            pools[(arm, language)] = np.asarray(values, dtype=np.int64)
    mining_groups = (
        (("external_wikipedia", "tl"), ("approved_articles", "tl"))
        if tagalog_only
        else (("external_wikipedia", "en"), ("external_wikipedia", "tl"),
              ("approved_articles", "en"), ("approved_articles", "tl"))
    )
    minimum_mined = int(expected["retrieval"]["minimum_safely_mined"])

    mining_manifests: dict[str, dict] = {}

    def mine_training_examples(
        search_query_vectors: np.ndarray,
        search_document_vectors: np.ndarray,
        *,
        stage: str,
    ) -> list[dict]:
        """Mine evolving hard candidates, but always apply SEA-LION safety scores."""
        mined_rows: list[dict] = []
        skipped_mining = defaultdict(int)
        for group in mining_groups:
            request_indices = [
                i for i, row in enumerate(training_requests)
                if (row["arm"], row["language"]) == group
            ]
            pool = pools[group]
            pool_vectors = torch.from_numpy(
                np.ascontiguousarray(search_document_vectors[pool])
            ).to("cuda")
            for offset in range(0, len(request_indices), 64):
                selected = request_indices[offset : offset + 64]
                queries_gpu = torch.from_numpy(
                    np.ascontiguousarray(search_query_vectors[selected])
                ).to("cuda")
                similarities = queries_gpu @ pool_vectors.T
                _, top_positions = torch.topk(similarities, k=min(512, len(pool)), dim=1)
                top_positions = top_positions.cpu().numpy()
                for local, request_index in enumerate(selected):
                    request = training_requests[request_index]
                    positives = request["positive_indices"]
                    positive_scores = document_vectors[positives] @ query_vectors[request_index]
                    if not np.isfinite(positive_scores).all() or float(positive_scores.max()) <= 0:
                        skipped_mining["nonpositive_teacher_positive"] += 1
                        continue
                    positive_index = positives[int(np.argmax(positive_scores))]
                    ceiling = 0.95 * float(positive_scores.min())
                    known = set(request["known_positive_indices"])
                    negatives = []
                    for position in top_positions[local]:
                        candidate_index = int(pool[int(position)])
                        teacher_score = float(
                            document_vectors[candidate_index] @ query_vectors[request_index]
                        )
                        if candidate_index in known or teacher_score >= ceiling:
                            continue
                        negatives.append((candidate_index, teacher_score))
                        if len(negatives) == 4:
                            break
                    if len(negatives) != 4:
                        skipped_mining["insufficient_safe_negatives"] += 1
                        continue
                    mined_rows.append({
                        **request,
                        "request_index": request_index,
                        "positive_index": positive_index,
                        "negative_indices": [index for index, _ in negatives],
                        "teacher_scores": [
                            float(positive_scores.max()),
                            *[value for _, value in negatives],
                        ],
                    })
                del similarities, top_positions, queries_gpu
            del pool_vectors
            torch.cuda.empty_cache()
        if len(mined_rows) < minimum_mined:
            raise RuntimeError("too few positive-aware examples survived")
        mining_manifest = {
            "mining_stage": stage,
            "requests": len(training_requests),
            "mined": len(mined_rows),
            "skipped": dict(skipped_mining),
            "hard_negatives_per_query": 4,
            "known_positives_masked": True,
            "false_negative_ceiling": 0.95,
            "teacher_scores_used_for_safety": True,
            "heldout_labels_used": False,
            "minimum_safely_mined_required": minimum_mined,
        }
        filename = "mining-summary.json" if stage == "initial-teacher" else f"mining-summary-{stage}.json"
        original.write_json(WORK / filename, mining_manifest)
        preserve.put(WORK / filename, "reports/" + filename)
        preserve.event(
            "hard-negative-mining-complete"
            if stage == "initial-teacher"
            else "dynamic-hard-negative-remine-complete",
            **mining_manifest,
        )
        # The report built after the promotion gates needs this; it used to be a
        # local here, which only surfaced once a candidate actually passed both
        # gates and reached the report.
        mining_manifests[stage] = mining_manifest
        return mined_rows

    mined = mine_training_examples(
        query_vectors,
        document_vectors,
        stage="initial-teacher",
    )
    gc.collect()

    def tokenize(texts: list[str], query: bool) -> list[list[int]]:
        values = [("[QRY] " + text) if query else text for text in texts]
        encodings = retrieval_tokenizer.encode_batch_fast(values, add_special_tokens=False)
        result = [
            strip_padding_and_unknown(
                encoding.ids,
                pad_id=pad_id,
                unknown_id=unknown_id,
            )
            for encoding in encodings
        ]
        if any(not row for row in result):
            raise ValueError("student tokenizer produced an empty sequence")
        if query and any(row[0] != query_token_id for row in result):
            raise ValueError("query role token is not first and atomic")
        return result

    all_feature_tokens = tokenize(feature_texts, query=False)
    all_training_query_tokens = tokenize(
        [row["query"] for row in training_requests], query=True
    )

    def attach_training_tokens(rows: list[dict]) -> None:
        for row in rows:
            row["query_tokens"] = all_training_query_tokens[row["request_index"]]
            row["candidate_tokens"] = [all_feature_tokens[row["positive_index"]]] + [
                all_feature_tokens[index] for index in row["negative_indices"]
            ]

    attach_training_tokens(mined)

    evaluation_document_tokens = tokenize([row["text"] for row in chunks], query=False)
    validation_query_tokens = tokenize(
        [row["question"] for row in development_validation], query=True
    )
    selection_query_tokens = (
        tokenize([row["question"] for row in development_selection], query=True)
        if development_selection
        else []
    )
    parent_positions = {parent: index for index, parent in enumerate(parents)}
    owners = [parent_positions[chunk["parent_id"]] for chunk in chunks]

    class TrainableStatic(torch.nn.Module):
        def __init__(self, weights: np.ndarray, *, query_projection: bool = False):
            super().__init__()
            self.embedding = torch.nn.EmbeddingBag.from_pretrained(
                torch.from_numpy(np.ascontiguousarray(weights)),
                freeze=False,
                mode="mean",
                sparse=True,
            )
            self.query_projection: torch.nn.Linear | None = None
            if query_projection:
                dimension = int(weights.shape[1])
                layer = torch.nn.Linear(dimension, dimension, bias=False)
                # Identity start, for the same reason the [QRY] row starts at
                # zero: epoch zero must be numerically identical to having no
                # projection at all, so the arm can only diverge by learning.
                with torch.no_grad():
                    layer.weight.copy_(torch.eye(dimension))
                self.query_projection = layer

        def forward(self, sequences: list[list[int]], *, query: bool = False) -> torch.Tensor:
            flat = torch.tensor(
                [token for row in sequences for token in row], dtype=torch.long, device="cuda"
            )
            offsets = torch.tensor(
                np.cumsum([0] + [len(row) for row in sequences[:-1]]),
                dtype=torch.long,
                device="cuda",
            )
            pooled = self.embedding(flat, offsets)
            if query and self.query_projection is not None:
                pooled = self.query_projection(pooled)
            return pooled

    # A question and a passage are different kinds of text, but they share one
    # lookup table; the teacher is told which is which by an instruction prefix
    # while the student only has one [QRY] row diluted across a mean pool.  This
    # gives queries their own learned linear map instead.
    use_query_projection = bool(RECIPE["training"].get("query_projection", False))
    model = TrainableStatic(expanded, query_projection=use_query_projection).to("cuda")
    del expanded

    @torch.no_grad()
    def encoded(
        sequences: list[list[int]], *, quantized: bool, query: bool = False
    ) -> np.ndarray:
        weights = model.embedding.weight
        if quantized:
            exported = quantize_embeddings(weights.detach().cpu().numpy(), DType.Int8)
            weights = torch.from_numpy(exported).to(device="cuda", dtype=torch.float32)
        values = []
        for start in range(0, len(sequences), 512):
            batch = sequences[start : start + 512]
            flat = torch.tensor(
                [token for row in batch for token in row], dtype=torch.long, device="cuda"
            )
            offsets = torch.tensor(
                np.cumsum([0] + [len(row) for row in batch[:-1]]),
                dtype=torch.long,
                device="cuda",
            )
            pooled = F.embedding_bag(flat, weights, offsets, mode="mean")
            if query and model.query_projection is not None:
                pooled = model.query_projection(pooled)
            values.append(F.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(values)

    def unknown_audit(texts: list[str]) -> tuple[dict, list[bool]]:
        encodings = retrieval_tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        token_count = 0
        unknown_count = 0
        fully_covered = []
        for encoding in encodings:
            ids = [value for value in encoding.ids if value != pad_id]
            missing = sum(value == unknown_id for value in ids)
            token_count += len(ids)
            unknown_count += missing
            fully_covered.append(missing == 0)
        return ({
            "records": len(texts),
            "tokens": token_count,
            "unknown_tokens": unknown_count,
            "unknown_rate": unknown_count / token_count if token_count else 0.0,
            "fully_covered_records": sum(fully_covered),
            "fully_covered_fraction": sum(fully_covered) / len(fully_covered) if fully_covered else 0.0,
        }, fully_covered)

    question_coverage, question_fully_covered = unknown_audit(
        [row["question"] for row in development_validation]
    )
    document_coverage, chunk_fully_covered = unknown_audit([row["text"] for row in chunks])
    chunk_coverage_by_id = {
        row["chunk_id"]: covered for row, covered in zip(chunks, chunk_fully_covered, strict=True)
    }
    gold_fully_covered = [
        any(chunk_coverage_by_id.get(chunk_id, False) for chunk_id in row["answer_bearing_chunk_ids"])
        for row in development_validation
    ]
    coverage_audit = {
        "questions": question_coverage,
        "documents": document_coverage,
        "fully_covered_gold_questions": sum(gold_fully_covered),
        "fully_covered_gold_fraction": sum(gold_fully_covered) / len(gold_fully_covered),
        "questions_and_gold_fully_covered": sum(
            question and gold
            for question, gold in zip(question_fully_covered, gold_fully_covered, strict=True)
        ),
    }
    original.write_json(WORK / "vocabulary-coverage-audit.json", coverage_audit)
    preserve.put(WORK / "vocabulary-coverage-audit.json", "reports/vocabulary-coverage-audit.json")
    preserve.event("vocabulary-coverage-audit-complete", **coverage_audit)

    direct_stage_metrics = static_stage_metrics(direct_student)
    tokenlearn_stage_metrics = static_stage_metrics(tokenlearn_student)
    preserve.event("development-stage-direct-model2vec", metrics=direct_stage_metrics)
    preserve.event("development-stage-post-tokenlearn", metrics=tokenlearn_stage_metrics)
    del direct_student, tokenlearn_student
    gc.collect()

    teacher_documents = teacher.encode(
        [row["text"] for row in chunks], batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32, copy=False)
    teacher_development_queries = teacher.encode(
        [row["question"] for row in development_validation], batch_size=64,
        normalize_embeddings=True, convert_to_numpy=True, prompt=query_prompt,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    teacher_development_metrics = metric_table(
        teacher_documents, teacher_development_queries, development_validation
    )

    # The teacher is not needed again until held-out scoring.  At 1024d the table
    # alone is 0.95 GiB and the greedy loop holds six or seven simultaneous
    # copies on the GPU, so parking the teacher on the host is what keeps the
    # peak off the 15.4 GiB T4 ceiling.
    teacher.to("cpu")
    torch.cuda.empty_cache()

    def evaluate_development(*, quantized: bool = True) -> dict:
        return metric_table(
            encoded(evaluation_document_tokens, quantized=quantized),
            encoded(validation_query_tokens, quantized=quantized, query=True),
            development_validation,
        )

    def evaluate_selection(*, quantized: bool = True) -> dict:
        """Score the disjoint selection cohort that drives checkpoint choice."""

        if not development_selection:
            raise ValueError("no selection cohort was partitioned")
        return metric_table(
            encoded(evaluation_document_tokens, quantized=quantized),
            encoded(selection_query_tokens, quantized=quantized, query=True),
            development_selection,
        )

    def score_metrics(value: dict) -> float:
        languages = ("tl",) if tagalog_only else ("en", "tl")
        return sum(value[language][metric] for language in languages
                   for metric in ("Recall@10", "nDCG@10"))

    def nonregressing(candidate: dict, baseline: dict) -> bool:
        languages = ("tl",) if tagalog_only else ("en", "tl")
        return all(candidate[language][metric] + 1e-12 >= baseline[language][metric]
                   for language in languages for metric in ("Recall@10", "nDCG@10"))

    best_metrics = evaluate_development(quantized=True)
    best_weights = model.embedding.weight.detach().cpu().numpy().copy()

    def projection_weights() -> np.ndarray | None:
        if model.query_projection is None:
            return None
        return model.query_projection.weight.detach().cpu().numpy().copy()

    def apply_projection(values: np.ndarray | None) -> None:
        if model.query_projection is not None and values is not None:
            model.query_projection.weight.data.copy_(
                torch.from_numpy(values).to("cuda")
            )

    # The embedding table takes sparse gradients and needs SparseAdam; a dense
    # projection cannot share that optimizer, so it gets its own.
    sparse_parameters = [model.embedding.weight]
    dense_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "embedding.weight"
    ]
    best_projection = projection_weights()
    development_history = [{"epoch": 0, "status": "baseline", "metrics": best_metrics}]
    preserve.event("development-baseline-complete", metrics=best_metrics)

    controlled_repair = RECIPE["training"].get("controlled_repair", {})
    if controlled_repair.get("persist_ranking_ready_checkpoint"):
        weights_path = WORK / "ranking-ready-weights.npy"
        tokenizer_path = WORK / "ranking-ready-tokenizer.json"
        pca_path = WORK / "ranking-ready-pca.npz"
        with weights_path.open("xb") as stream:
            np.save(stream, best_weights, allow_pickle=False)
        with tokenizer_path.open("x", encoding="utf-8") as stream:
            stream.write(retrieval_tokenizer.to_str())
            stream.write("\n")
        with pca_path.open("xb") as stream:
            np.savez(
                stream,
                mean=target_pca.mean,
                components=target_pca.components,
                explained_variance=target_pca.explained_variance,
                sample_count=np.asarray([target_pca.sample_count], dtype=np.int64),
            )
        checkpoint_files = {
            "ranking-ready-weights.npy": weights_path,
            "ranking-ready-tokenizer.json": tokenizer_path,
            "ranking-ready-pca.npz": pca_path,
        }
        checkpoint_receipt = {
            "protocol": RECIPE["protocol"],
            "candidate": candidate_id,
            "dtype": str(best_weights.dtype),
            "shape": list(best_weights.shape),
            "development_baseline": best_metrics,
            "files": {
                name: {"bytes": path.stat().st_size, "sha256": original.digest(path)}
                for name, path in checkpoint_files.items()
            },
            "heldout_labels_used": False,
        }
        receipt_path = WORK / "ranking-ready-receipt.json"
        original.write_json(receipt_path, checkpoint_receipt)
        preserve.put(weights_path, "checkpoints/ranking-ready-weights.npy")
        preserve.put(tokenizer_path, "checkpoints/ranking-ready-tokenizer.json")
        preserve.put(pca_path, "checkpoints/ranking-ready-pca.npz")
        preserve.put(receipt_path, "checkpoints/ranking-ready-receipt.json")
        preserve.event(
            "ranking-ready-checkpoint-preserved",
            shape=list(best_weights.shape),
            total_bytes=sum(row["bytes"] for row in checkpoint_receipt["files"].values()),
            files=checkpoint_receipt["files"],
            heldout_labels_used=False,
        )
    del target_pca

    def objective(batch: list[dict]) -> torch.Tensor:
        queries = model([row["query_tokens"] for row in batch], query=True)
        candidates = model([
            tokens for row in batch for tokens in row["candidate_tokens"]
        ]).reshape(len(batch), 5, -1)
        teacher_scores = torch.tensor(
            [row["teacher_scores"] for row in batch], dtype=torch.float32, device="cuda"
        )
        query = F.normalize(queries, dim=1)
        candidate = F.normalize(candidates, dim=2)
        scores = torch.einsum("bd,bkd->bk", query, candidate)
        listwise = listwise_teacher_distillation_loss(scores, teacher_scores, temperature=0.05)
        candidate_ce = F.cross_entropy(
            scores * 20.0, torch.zeros(len(batch), dtype=torch.long, device="cuda")
        )
        in_batch = F.cross_entropy(
            (query @ candidate[:, 0].T) * 20.0, torch.arange(len(batch), device="cuda")
        )
        alignment = F.mse_loss(scores, teacher_scores)
        teacher_query_batch = torch.from_numpy(np.ascontiguousarray(
            projected_teacher_queries[[row["request_index"] for row in batch]]
        )).to("cuda")
        query_alignment = 1.0 - F.cosine_similarity(query, teacher_query_batch, dim=1).mean()
        query_vector_distillation_weight = float(
            RECIPE["training"].get("query_vector_distillation_weight", 0.0)
        )
        # Frozen weights are (1.0, 0.5, 0.5, 0.25).  ``teacher_score_mse`` asks
        # the student to reproduce the teacher's absolute cosine values, which is
        # a regression target rather than a ranking one; the reference static
        # retrieval model uses pure contrastive loss and no such term.
        objective_weights = RECIPE["training"].get("objective_weights", {})
        return (
            float(objective_weights.get("listwise", 1.0)) * listwise
            + float(objective_weights.get("positive_cross_entropy", 0.5)) * candidate_ce
            + float(objective_weights.get("in_batch_cross_entropy", 0.5)) * in_batch
            + float(objective_weights.get("teacher_score_mse", 0.25)) * alignment
            + query_vector_distillation_weight * query_alignment
        )

    learning_rate = (
        float(RECIPE["training"]["retrieval_learning_rate"])
        if tagalog_only else 5e-5
    )
    retrieval_epochs = (
        int(RECIPE["training"]["retrieval_epochs"])
        if tagalog_only else 3
    )
    retrieval_lr_schedule = RECIPE["training"].get("retrieval_lr_schedule", [])
    dynamic_remine_after_epochs = set(
        RECIPE["training"].get("dynamic_remine_after_epochs", [])
    )
    retrieval_early_stopping_patience = int(
        RECIPE["training"].get("retrieval_early_stopping_patience", 0)
    )
    no_gain_epochs = 0
    ranking_search = RECIPE["training"].get("ranking_search", "greedy-rollback")
    retrieval_batch_size = int(RECIPE["training"].get("retrieval_batch_size", 64))

    def scheduled_learning_rate(epoch: int) -> float:
        value = learning_rate
        for phase in retrieval_lr_schedule:
            if epoch >= int(phase["start_epoch"]):
                value = float(phase["learning_rate"])
        return value

    if ranking_search == "straight":
        # The greedy-rollback search restarts from ``best_weights`` every epoch,
        # rebuilds Adam, line-searches four interpolations and discards the
        # epoch unless it improves the *reported* cohort.  That both prevents
        # any temporarily-worse step and makes the reported number a maximum
        # over 64 evaluations of the cohort being reported.  This search keeps
        # one optimizer for the whole run, decays the learning rate on a cosine,
        # never rolls back, and chooses the checkpoint on the disjoint selection
        # cohort so the reported cohort never drives a decision.
        if not development_selection:
            raise ValueError("straight ranking search requires a selection cohort")
        if retrieval_lr_schedule or dynamic_remine_after_epochs:
            raise ValueError("straight ranking search owns its own schedule")
        optimizer = torch.optim.SparseAdam(sparse_parameters, lr=learning_rate)
        dense_optimizer = (
            torch.optim.Adam(dense_parameters, lr=learning_rate)
            if dense_parameters else None
        )
        selection_metrics = evaluate_selection(quantized=True)
        best_selection_score = score_metrics(selection_metrics)
        best_metrics = evaluate_development(quantized=True)
        best_weights = model.embedding.weight.detach().cpu().numpy().copy()
        best_epoch = 0
        development_history[0] = {
            "epoch": 0,
            "status": "baseline",
            "metrics": best_metrics,
            "selection_metrics": selection_metrics,
            "selection_score": best_selection_score,
        }
        preserve.event(
            "straight-ranking-baseline",
            report_metrics=best_metrics,
            selection_metrics=selection_metrics,
            batch_size=retrieval_batch_size,
            epochs=retrieval_epochs,
            initial_learning_rate=learning_rate,
        )
        for epoch in range(1, retrieval_epochs + 1):
            epoch_learning_rate = learning_rate * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - 1) / retrieval_epochs)
            )
            for group in optimizer.param_groups:
                group["lr"] = epoch_learning_rate
            losses = []
            model.train()
            for batch in batch_unique(mined, retrieval_batch_size, 42 + epoch):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(batch)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite retrieval objective")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            selection_metrics = evaluate_selection(quantized=True)
            report_metrics = evaluate_development(quantized=True)
            selection_score = score_metrics(selection_metrics)
            improved_selection = selection_score > best_selection_score + 1e-12
            if improved_selection:
                best_selection_score = selection_score
                best_metrics = report_metrics
                best_weights = model.embedding.weight.detach().cpu().numpy().copy()
                best_epoch = epoch
            row = {
                "epoch": epoch,
                "learning_rate": epoch_learning_rate,
                "mean_loss": float(np.mean(losses)),
                "status": "selection-improved" if improved_selection else "kept-training",
                "metrics": report_metrics,
                "selection_metrics": selection_metrics,
                "selection_score": selection_score,
                "selected_checkpoint_epoch": best_epoch,
            }
            development_history.append(row)
            preserve.event("retrieval-epoch-complete", **row)
            torch.cuda.empty_cache()
        model.embedding.weight.data.copy_(torch.from_numpy(best_weights).to("cuda"))
        preserve.event(
            "straight-ranking-complete",
            selected_checkpoint_epoch=best_epoch,
            selection_score=best_selection_score,
            report_metrics=best_metrics,
            report_cohort_used_for_selection=False,
            report_cohort_evaluations_used_for_selection=0,
        )
        del optimizer
        torch.cuda.empty_cache()

    greedy_epochs = 0 if ranking_search == "straight" else retrieval_epochs
    for epoch in range(1, greedy_epochs + 1):
        epoch_learning_rate = scheduled_learning_rate(epoch)
        phase_start = torch.from_numpy(best_weights).to("cuda")
        model.embedding.weight.data.copy_(phase_start)
        apply_projection(best_projection)
        projection_start = projection_weights()
        optimizer = torch.optim.SparseAdam(sparse_parameters, lr=epoch_learning_rate)
        dense_optimizer = (
            torch.optim.Adam(dense_parameters, lr=epoch_learning_rate)
            if dense_parameters else None
        )
        losses = []
        model.train()
        for batch in batch_unique(mined, retrieval_batch_size, 42 + epoch):
            optimizer.zero_grad(set_to_none=True)
            if dense_optimizer is not None:
                dense_optimizer.zero_grad(set_to_none=True)
            loss = objective(batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite retrieval objective")
            loss.backward()
            optimizer.step()
            if dense_optimizer is not None:
                dense_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        trained = model.embedding.weight.detach().clone()
        projection_trained = projection_weights()
        model.eval()
        candidates = []
        for alpha in (0.25, 0.5, 0.75, 1.0):
            model.embedding.weight.data.copy_(phase_start.lerp(trained, alpha))
            # The projection must move with the table it was trained against,
            # or the line search would evaluate mismatched halves of one step.
            if projection_start is not None and projection_trained is not None:
                apply_projection(
                    projection_start + alpha * (projection_trained - projection_start)
                )
            metrics = evaluate_development(quantized=True)
            candidates.append((
                alpha, metrics,
                model.embedding.weight.detach().cpu().numpy().copy(),
                projection_weights(),
            ))
        improved = [
            row for row in candidates
            if nonregressing(row[1], best_metrics)
            and score_metrics(row[1]) > score_metrics(best_metrics) + 1e-12
        ]
        if improved:
            alpha, metrics, weights, projection = max(
                improved, key=lambda row: score_metrics(row[1])
            )
            best_metrics, best_weights, status = metrics, weights, "accepted"
            best_projection = projection
            no_gain_epochs = 0
        else:
            alpha, metrics, status = 0.0, best_metrics, "rolled-back-no-nonregressing-gain"
            no_gain_epochs += 1
            if not retrieval_lr_schedule:
                learning_rate *= 0.5
        model.embedding.weight.data.copy_(torch.from_numpy(best_weights).to("cuda"))
        apply_projection(best_projection)
        row = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": scheduled_learning_rate(epoch + 1),
            "mean_loss": float(np.mean(losses)),
            "selected_alpha": alpha,
            "status": status,
            "consecutive_no_gain_epochs": no_gain_epochs,
            "metrics": metrics,
        }
        development_history.append(row)
        preserve.event("retrieval-epoch-complete", **row)
        del optimizer, dense_optimizer, phase_start, trained, candidates, improved
        torch.cuda.empty_cache()

        if epoch in dynamic_remine_after_epochs:
            model.eval()
            student_query_vectors = encoded(
            all_training_query_tokens, quantized=False, query=True
        )
            student_document_vectors = encoded(all_feature_tokens, quantized=False)
            mined = mine_training_examples(
                student_query_vectors,
                student_document_vectors,
                stage=f"after-epoch-{epoch}",
            )
            attach_training_tokens(mined)
            no_gain_epochs = 0
            del student_query_vectors, student_document_vectors
            gc.collect()
            torch.cuda.empty_cache()

        if (
            retrieval_early_stopping_patience
            and epoch >= min(dynamic_remine_after_epochs or {1})
            and epoch not in dynamic_remine_after_epochs
            and no_gain_epochs >= retrieval_early_stopping_patience
        ):
            preserve.event(
                "retrieval-early-stopped",
                epoch=epoch,
                patience=retrieval_early_stopping_patience,
                heldout_labels_used=False,
            )
            break

    model.embedding.weight.data.copy_(torch.from_numpy(best_weights).to("cuda"))
    original.write_json(WORK / "development-history.json", development_history)
    preserve.put(WORK / "development-history.json", "reports/development-history.json")

    post_retrieval_float_metrics = evaluate_development(quantized=False)
    post_export_int8_metrics = evaluate_development(quantized=True)
    # Deferred deliberately: this is a float-equality check between the in-loop
    # evaluation that chose best_metrics and this re-evaluation.  Raising here
    # would destroy a complete multi-hour run before the table reaches S3, which
    # is the failure that already cost one passing model.  The verdict is carried
    # to just after the package upload instead.
    selected_checkpoint_drifted = post_export_int8_metrics != best_metrics
    stage_metrics = {
        "direct_model2vec_float32": direct_stage_metrics,
        "post_tokenlearn_float32": tokenlearn_stage_metrics,
        "post_retrieval_float32": post_retrieval_float_metrics,
        "post_export_int8": post_export_int8_metrics,
    }
    original.write_json(WORK / "development-stage-metrics.json", stage_metrics)
    preserve.put(WORK / "development-stage-metrics.json", "reports/development-stage-metrics.json")
    preserve.event("development-stage-measurements-complete", metrics=stage_metrics)

    if controlled_repair.get("phase") == "v3-control":
        actual_control = float(post_export_int8_metrics["tl"]["Recall@10"])
        previous_best = float(controlled_repair["previous_best_recall_at_10"])
        tolerance = float(controlled_repair["control_reproduction_tolerance"])
        control_result = {
            "actual_recall_at_10": actual_control,
            "previous_best_recall_at_10": previous_best,
            "delta": actual_control - previous_best,
            "tolerance": tolerance,
            "reproduced": actual_control + tolerance >= previous_best,
            "heldout_labels_used": False,
        }
        control_path = WORK / "controlled-control-result.json"
        original.write_json(control_path, control_result)
        preserve.put(control_path, "reports/controlled-control-result.json")
        preserve.event("controlled-control-result", **control_result)
        if not control_result["reproduced"]:
            raise RuntimeError("v3 control did not reproduce before ablations")

    exported_weights = quantize_embeddings(best_weights, DType.Int8)
    export_model = StaticModel(
        vectors=exported_weights,
        tokenizer=Tokenizer.from_str(retrieval_tokenizer.to_str()),
        config={
            "model_type": "model2vec",
            "architectures": ["StaticModel"],
            "normalize": True,
            "query_prefix": "[QRY] ",
            "hidden_dim": student_dimension,
            "dimension": student_dimension,
            "dtype": "int8",
            "base_model_revision": teacher_revision,
            "direct_parent_model2vec": True,
            "tokenlearn_same_parent": True,
        },
        normalize=True,
        base_model_name=teacher_id,
        language=["tl"] if tagalog_only else ["en", "tl"],
    )
    export_dir = WORK / f"direct-sealion-{candidate_id}-int8-{student_dimension}"
    export_model.save_pretrained(export_dir)
    if best_projection is not None:
        # Written inside the export directory so the size receipt below counts
        # it: a query-side linear map is part of the shipped artifact.
        with (export_dir / "query_projection.npy").open("xb") as stream:
            np.save(stream, best_projection.astype(np.float32, copy=False),
                    allow_pickle=False)
    file_sizes = {
        path.relative_to(export_dir).as_posix(): path.stat().st_size
        for path in export_dir.rglob("*") if path.is_file()
    }
    artifact_limit = (
        int(candidate["hard_maximum_complete_artifact_bytes"])
        if capacity_frontier else 100_000_000
    )
    measured_complete_bytes = sum(file_sizes.values())
    artifact_size_receipt = {
        "candidate": candidate_id,
        "dimension": student_dimension,
        "dtype": "int8",
        "matrix_shape": list(exported_weights.shape),
        "matrix_nbytes": int(exported_weights.nbytes),
        "files": file_sizes,
        "complete_bytes": measured_complete_bytes,
        "hard_maximum_bytes": artifact_limit,
        "size_gate_passed": measured_complete_bytes < artifact_limit,
    }
    original.write_json(WORK / "artifact-size-receipt.json", artifact_size_receipt)
    preserve.put(WORK / "artifact-size-receipt.json", "reports/artifact-size-receipt.json")
    preserve.event("artifact-size-measured", **artifact_size_receipt)

    # Preserve before judging.  Any gate raising below used to destroy the only
    # copy of the package it had just measured: the export lives in container
    # scratch, SageMaker only auto-uploads /opt/ml/model on success, and the
    # container is torn down at terminal state.  A 512D run that met the quality
    # gate was lost exactly this way.  Upload unconditionally, then judge.
    for path in sorted(export_dir.rglob("*")):
        if path.is_file():
            preserve.put(path, f"model/{path.relative_to(export_dir).as_posix()}")
    preserve.event(
        "candidate-package-preserved",
        prefix="model/",
        files=file_sizes,
        complete_bytes=measured_complete_bytes,
        note="uploaded before any gate; promotion decided afterwards",
    )

    if selected_checkpoint_drifted:
        raise ValueError("selected exact-int8 development checkpoint changed")
    complete_bytes = validate_measured_artifact(
        file_sizes,
        hard_maximum_bytes=artifact_limit,
        reference_artifact_bytes=max(artifact_limit + 1, 131_200_180),
    )

    development_gate = (
        validate_tagalog_teacher_distance_gate(
            best_metrics,
            teacher_development_metrics,
            maximum_gap=0.10,
        )
        if tagalog_only
        else validate_teacher_distance_gate(
            best_metrics,
            teacher_development_metrics,
            english_floor=0.90,
            maximum_language_gap=0.10,
        )
    )
    preserve.event(
        "development-promotion-gates-passed",
        complete_artifact_bytes=complete_bytes,
        quality_gate=development_gate,
    )

    heldout = [row for row in annotations if row["partition"] == "evaluation"]
    expected_heldout = 3_984 if tagalog_only else 7_922
    if len(heldout) != expected_heldout:
        raise ValueError("held-out question count changed")
    heldout_student_tokens = tokenize([row["question"] for row in heldout], query=True)
    teacher.to("cuda")
    heldout_teacher_queries = teacher.encode(
        [row["question"] for row in heldout], batch_size=64,
        normalize_embeddings=True, convert_to_numpy=True, prompt=query_prompt,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    student_heldout_metrics = metric_table(
        encoded(evaluation_document_tokens, quantized=True),
        encoded(heldout_student_tokens, quantized=True, query=True),
        heldout,
    )
    teacher_heldout_metrics = metric_table(
        teacher_documents, heldout_teacher_queries, heldout
    )
    heldout_gate = (
        validate_tagalog_teacher_distance_gate(
            student_heldout_metrics,
            teacher_heldout_metrics,
            maximum_gap=0.10,
        )
        if tagalog_only
        else validate_teacher_distance_gate(
            student_heldout_metrics,
            teacher_heldout_metrics,
            english_floor=0.90,
            maximum_language_gap=0.10,
        )
    )

    preserve.tree(export_dir, f"models/{candidate_id}")
    report = {
        "status": "complete",
        "protocol": RECIPE["protocol"],
        "language": "tl" if tagalog_only else "bilingual",
        "teacher": {
            "model": teacher_id,
            "revision": teacher_revision,
            "dimension": 1024,
        },
        "model2vec": {
            "entrypoint": "model2vec.distill.distill",
            "candidate": candidate_id,
            "vocabulary_mode": vocabulary_mode,
            "vocabulary_requested": len(vocabulary) if vocabulary is not None else "full-parent",
            "pca_dims": student_dimension if capacity_frontier else "auto",
            "direct_parent": True,
        },
        "tokenlearn": {
            "same_parent_targets": True,
            "documents": len(feature_rows),
            "training": len(train_indices),
            "validation": len(validation_indices),
            "validation_events": validation_events,
            "selected_validation_index": tokenlearn_state["selected_validation_index"],
            "selected_validation_loss": tokenlearn_state["best_validation_value"],
            "target_pca_retained_variance": retained_variance,
        },
        "retrieval": {
            "synthetic_queries": len(synthetic),
            "development_pairs": len(development_train),
            "mining": mining_manifests.get("initial-teacher"),
            "mining_stages": mining_manifests,
            "history": development_history,
        },
        "development_stage_metrics": stage_metrics,
        "vocabulary_coverage": coverage_audit,
        "development": {
            "questions": len(development_validation),
            "student": best_metrics,
            "teacher": teacher_development_metrics,
            "gate": development_gate,
        },
        "heldout": {
            "questions": len(heldout),
            "student": student_heldout_metrics,
            "teacher": teacher_heldout_metrics,
            "gate": heldout_gate,
        },
        "artifact": {
            "dimension": student_dimension,
            "dtype": "int8",
            "files": file_sizes,
            "complete_bytes": complete_bytes,
            "hard_maximum_bytes": artifact_limit,
        },
        "heldout_used_for_training_or_checkpoint_selection": False,
        "latency_measured": False,
        "peak_rss_measured": False,
        "artifacts": preserve.index,
    }
    original.write_json(WORK / "direct-sealion-completion.json", report)
    preserve.put(WORK / "direct-sealion-completion.json", "reports/direct-sealion-completion.json")
    preserve.event(
        "direct-sealion-complete",
        complete_artifact_bytes=complete_bytes,
        heldout_student=student_heldout_metrics,
        heldout_teacher=teacher_heldout_metrics,
    )


if __name__ == "__main__":
    main()
