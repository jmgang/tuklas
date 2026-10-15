#!/usr/bin/env python3
"""Build and evaluate the compact bilingual SEA-LION static student.

This is a managed SageMaker Training entrypoint.  It preserves every stage in
new S3 keys and never reads held-out labels until development checkpoint
selection is complete.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import gc
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


INPUT = Path("/opt/ml/input/data/input")
WORK = Path("/opt/ml/compact-sota-work")
sys.path.insert(0, str(INPUT))
os.environ["SEALION_INPUT_DIR"] = str(INPUT)
os.environ["SEALION_WORK_DIR"] = str(WORK)
original = importlib.import_module("aws_sealion_train_repair_v3")
CONFIG = json.loads((INPUT / "settings.json").read_text())
PLAN = json.loads((INPUT / "mixed-plan.json").read_text())
RECIPE = json.loads((INPUT / "sota-repair.json").read_text())


def normalized(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


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
        if current:
            result.append(current)
        if len(rest) == len(pending):
            raise RuntimeError("nonduplicate batch construction made no progress")
        pending = rest
    return result


def main() -> None:  # noqa: C901, PLR0915
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "events").mkdir(exist_ok=False)
    preserve = original.Preservation()
    preserve.event("dependency-setup")
    original.bootstrap()

    import torch
    import torch.nn.functional as F
    from model2vec import StaticModel
    from model2vec.distill.distillation import TokenizerModel, create_embeddings, turn_tokens_into_ids
    from model2vec.quantization import DType, quantize_embeddings
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer
    from transformers import AutoModel, AutoTokenizer

    from compact_word_model import PAD_TOKEN, QUERY_TOKEN, UNK_TOKEN
    from full_rank_pca import (
        fit_full_rank_pca,
        model2vec_sif_weights,
        transform_full_rank_pca_batched,
    )
    from multiwikiqa_compact_retrieval import (
        external_article_id_from_passage_id,
        listwise_teacher_distillation_loss,
        validate_measured_artifact,
        validate_teacher_distance_gate,
    )
    from multiwikiqa_evaluation import rank_parents, score
    from multiwikiqa_ranking_v5 import prepare as prepare_development
    from sealion_runtime import verify_runtime
    from sealion_training import official_training_adapter

    verify_runtime(CONFIG["tokenlearn"]["commit"])
    if not torch.cuda.is_available():
        raise RuntimeError("compact SOTA training requires a CUDA GPU")
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

    # Freeze and validate small staged inputs before expensive model work.
    expected = json.loads((INPUT / "input-manifest.json").read_text())
    for filename, spec in expected["files"].items():
        path = INPUT / filename
        if path.stat().st_size != spec["bytes"] or original.digest(path) != spec["sha256"]:
            raise ValueError("staged input changed: " + filename)
    compact_tokenizer = Tokenizer.from_file(str(INPUT / "tokenizer-quality.json"))
    if compact_tokenizer.get_vocab_size(with_added_tokens=True) != 170_003:
        raise ValueError("compact tokenizer size changed")
    if compact_tokenizer.token_to_id(PAD_TOKEN) != 0 or compact_tokenizer.token_to_id(UNK_TOKEN) != 1:
        raise ValueError("compact padding/unknown IDs changed")
    if compact_tokenizer.token_to_id(QUERY_TOKEN) != 2:
        raise ValueError("atomic query role ID changed")

    # Restore all unchanged contextual passage targets and their exact layouts.
    preserve.event("feature-restore-started")
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
            texts = json.loads(get(pair["text"]))
            vectors = np.load(io.BytesIO(get(pair["vectors"])), allow_pickle=False)
            rows = layout[local : local + len(texts)]
            if texts != [row["text"] for row in rows] or vectors.shape != (len(texts), 1024):
                raise ValueError("feature text/vector alignment changed")
            if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise ValueError("feature target matrix is malformed")
            feature_rows.extend({**row, "arm": arm} for row in rows)
            feature_texts.extend(texts)
            target_parts.append(vectors)
            local += len(texts)
        if local != spec["windows"]:
            raise ValueError("incomplete restored feature arm")
        arm_ranges[arm] = (start, len(feature_rows))
        preserve.event("feature-arm-restored", arm=arm, rows=local)
    raw_targets = np.concatenate(target_parts, axis=0)
    del target_parts
    if raw_targets.shape != (182_724, 1024) or len(feature_rows) != len(feature_texts):
        raise ValueError("combined feature cohort changed")

    # Build a true 170k replacement-vocabulary Model2Vec initialization.
    preserve.event("compact-initialization-started", vocabulary=170_003, dimension=1024)
    teacher_id = CONFIG["teacher"]["model_id"]
    teacher_revision = CONFIG["teacher"]["revision"]
    hf_tokenizer = AutoTokenizer.from_pretrained(teacher_id, revision=teacher_revision, use_fast=True)
    hf_model = AutoModel.from_pretrained(teacher_id, revision=teacher_revision, trust_remote_code=False)
    source_tokenizer = TokenizerModel.from_transformers_tokenizer(hf_tokenizer).prune_added_tokens()
    ordered = [None] * compact_tokenizer.get_vocab_size(with_added_tokens=True)
    for token, token_id in compact_tokenizer.get_vocab(with_added_tokens=True).items():
        ordered[token_id] = token
    if any(token is None for token in ordered):
        raise ValueError("compact tokenizer IDs are not contiguous")
    token_ids = turn_tokens_into_ids(ordered, source_tokenizer)
    raw_token_vectors = create_embeddings(
        model=hf_model,
        tokenized=token_ids,
        device="cuda",
        pad_token_id=source_tokenizer.pad_token_id or 0,
        pooling="mean",
    ).astype(np.float32, copy=False)
    if raw_token_vectors.shape != (170_003, 1024):
        raise ValueError("teacher token distillation returned the wrong shape")
    del hf_model, hf_tokenizer, source_tokenizer, token_ids
    gc.collect()
    torch.cuda.empty_cache()
    token_pca = fit_full_rank_pca(raw_token_vectors, device="cuda", batch_size=2048)
    initial_vectors = transform_full_rank_pca_batched(
        raw_token_vectors, token_pca, device="cuda", batch_size=2048
    )
    initial_vectors *= model2vec_sif_weights(len(initial_vectors), 1e-4)[:, None]
    initial_vectors[compact_tokenizer.token_to_id(PAD_TOKEN)] = 0
    initial_vectors[compact_tokenizer.token_to_id(UNK_TOKEN)] = 0
    initial_vectors[compact_tokenizer.token_to_id(QUERY_TOKEN)] = 0
    initialization = StaticModel(
        vectors=initial_vectors,
        tokenizer=Tokenizer.from_str(compact_tokenizer.to_str()),
        config={
            "model_type": "model2vec",
            "architectures": ["StaticModel"],
            "tokenizer_name": teacher_id,
            "apply_pca": 1024,
            "sif_coefficient": 1e-4,
            "hidden_dim": 1024,
            "seq_length": 1_000_000,
            "normalize": True,
            "pooling": "mean",
            "query_prefix": "[QRY] ",
        },
        normalize=True,
        base_model_name=teacher_id,
        language=["en", "tl"],
    )
    init_dir = WORK / "compact-initialization"
    initialization.save_pretrained(init_dir)
    with (WORK / "token-pca.npz").open("xb") as stream:
        np.savez(stream, mean=token_pca.mean, components=token_pca.components,
                 explained_variance=token_pca.explained_variance,
                 sample_count=np.asarray([token_pca.sample_count], dtype=np.int64))
    preserve.put(WORK / "token-pca.npz", "transforms/token-pca.npz")
    preserve.event(
        "compact-initialization-complete-transient",
        model_bytes=int(initial_vectors.nbytes),
        retained=False,
        reason="custom model packages are retained only after the final teacher-distance gate",
    )
    del raw_token_vectors, initial_vectors, token_pca
    gc.collect()

    # Official TokenLearn document objective, with one shared full-rank target
    # PCA fitted on the complete approved union and applied to both partitions.
    preserve.event("target-full-rank-pca-started", rows=len(raw_targets), dimension=1024)
    target_pca = fit_full_rank_pca(raw_targets, device="cuda", batch_size=2048)
    transformed_targets = transform_full_rank_pca_batched(
        raw_targets, target_pca, device="cuda", batch_size=2048
    )
    with (WORK / "target-pca.npz").open("xb") as stream:
        np.savez(stream, mean=target_pca.mean, components=target_pca.components,
                 explained_variance=target_pca.explained_variance,
                 sample_count=np.asarray([target_pca.sample_count], dtype=np.int64))
    preserve.put(WORK / "target-pca.npz", "transforms/target-pca.npz")
    train_indices = [i for i, row in enumerate(feature_rows) if row["partition"] == "train"]
    validation_indices = [i for i, row in enumerate(feature_rows) if row["partition"] == "validation"]
    if (len(train_indices), len(validation_indices)) != (164_691, 18_033):
        raise ValueError("TokenLearn article split changed")
    validation_events: list[dict] = []

    def validation_event(metrics) -> None:
        value = {key: float(item) for key, item in metrics.items()}
        validation_events.append(value)
        preserve.event("tokenlearn-validation", **value)

    with official_training_adapter(
        [feature_texts[i] for i in validation_indices],
        transformed_targets[validation_indices],
        PAD_TOKEN,
        validation_event,
    ):
        # The adapter temporarily replaces model2vec.train's public training
        # class and records the compact model's pad id during construction.
        # Construct through that patched module inside the adapter lifetime;
        # a module-level class alias would bypass the adapter and leave its
        # collation state uninitialized.
        import model2vec.train as model2vec_train

        trainable = model2vec_train.StaticModelForSimilarity.from_static_model(
            model=initialization,
            out_dim=1024,
            n_layers=0,
            freeze_weights=True,
            pad_token=PAD_TOKEN,
        )
        trainable.fit(
            X=[feature_texts[i] for i in train_indices],
            y=torch.from_numpy(transformed_targets[train_indices]),
            batch_size=256,
            device="cuda",
            learning_rate=0.001,
        )
    tokenlearn_student = trainable.to_static_model()
    if tokenlearn_student.embedding.shape != (170_003, 1024):
        raise ValueError("TokenLearn changed compact vocabulary or dimension")
    tokenlearn_dir = WORK / "compact-tokenlearn"
    tokenlearn_student.save_pretrained(tokenlearn_dir)
    preserve.event(
        "tokenlearn-complete-transient",
        validation_events=len(validation_events),
        retained=False,
        reason="custom model packages are retained only after the final teacher-distance gate",
    )
    del trainable, initialization, transformed_targets
    gc.collect()
    torch.cuda.empty_cache()

    # Load frozen evaluation corpus and development labels.  Held-out labels
    # are retained in memory but are not passed to any selection function.
    evaluation_layout = json.loads(get(PLAN["evaluation_layout"]))
    annotations = [json.loads(line) for line in get(PLAN["annotations"]).splitlines()]
    chunks = evaluation_layout["chunks"]
    parents = evaluation_layout["parents"]
    if len(chunks) != 6546 or len(parents) != 1294 or len(annotations) != 10_000:
        raise ValueError("evaluation cohort changed")
    development_train, development_validation, skipped = prepare_development(annotations, chunks)
    if (len(development_train), len(development_validation), len(skipped)) != (1734, 338, 6):
        raise ValueError("development partition changed")
    if any(row["partition"] != "development" for row in development_validation):
        raise ValueError("held-out query entered checkpoint selection")

    synthetic = read_jsonl(INPUT / "synthetic-admitted.jsonl")
    if len(synthetic) != 10_827:
        raise ValueError("verified Sonnet query count changed")
    by_source: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_article: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_external_article: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(feature_rows):
        by_source[(row["arm"], str(row["source_id"]))].append(index)
        by_article[(row["arm"], str(row["article_id"]))].append(index)
        if row["arm"] == "external_wikipedia":
            corpus_article_id = external_article_id_from_passage_id(str(row["source_id"]))
            by_external_article[(row["arm"], corpus_article_id)].append(index)

    training_requests: list[dict] = []
    for row in synthetic:
        positive = by_source.get(("external_wikipedia", row["positive_passage_id"]), [])
        known = by_external_article.get(("external_wikipedia", row["positive_article_id"]), [])
        if not positive or not known:
            raise ValueError("synthetic positive cannot be mapped to frozen feature rows")
        training_requests.append({
            "id": "synthetic:" + row["record_id"] + ":" + row["query_type"],
            "query": row["query"],
            "language": row["language"],
            "arm": "external_wikipedia",
            "positive_indices": positive,
            "known_positive_indices": known,
            "positive_group": "external:" + row["positive_article_id"],
            "source": "sonnet45_synthetic",
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
            raise ValueError("development positive cannot be mapped to approved feature rows")
        training_requests.append({
            "id": "development:" + row["query_id"],
            "query": row["question"],
            "language": row["language"],
            "arm": "approved_articles",
            "positive_indices": positive,
            "known_positive_indices": known,
            "positive_group": "approved:" + row["component_id"],
            "source": "development_only",
        })
    if len(training_requests) != 12_561:
        raise ValueError("ranking supervision count changed")

    # Encode all retrieval queries with the exact SEA-LION Retrieval prompt.
    query_prompt = (
        "Instruct: Given a passage that is guaranteed to contain the answer, "
        "retrieve relevant passages that answer the query.\nQuery: "
    )
    teacher = SentenceTransformer(
        teacher_id, revision=teacher_revision, device="cuda", trust_remote_code=False
    )
    if teacher.prompts.get("Retrieval") != query_prompt or teacher.get_sentence_embedding_dimension() != 1024:
        raise ValueError("SEA-LION retrieval prompt or dimension changed")
    teacher.max_seq_length = 512
    query_vectors = teacher.encode(
        [row["query"] for row in training_requests],
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt=query_prompt,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    document_vectors = raw_targets / np.maximum(
        np.linalg.norm(raw_targets, axis=1, keepdims=True), 1e-12
    )
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Positive-aware mining: four negatives, all known positives masked, same
    # language/arm candidate pool, 95% false-negative ceiling.
    preserve.event("hard-negative-mining-started", queries=len(training_requests), negatives=4)
    pools: dict[tuple[str, str], np.ndarray] = {}
    heldout_components = {
        row["component_id"] for row in annotations if row["partition"] == "evaluation"
    }
    for arm in ("external_wikipedia", "approved_articles"):
        for language in ("en", "tl"):
            indices = []
            for index in range(*arm_ranges[arm]):
                row = feature_rows[index]
                if row["language"] != language:
                    continue
                if arm == "approved_articles" and row["article_id"] in heldout_components:
                    continue
                indices.append(index)
            pools[(arm, language)] = np.asarray(indices, dtype=np.int64)
    mined: list[dict] = []
    skipped_mining = defaultdict(int)
    for group in (("external_wikipedia", "en"), ("external_wikipedia", "tl"),
                  ("approved_articles", "en"), ("approved_articles", "tl")):
        request_indices = [i for i, row in enumerate(training_requests) if (row["arm"], row["language"]) == group]
        pool = pools[group]
        pool_vectors = torch.from_numpy(np.ascontiguousarray(document_vectors[pool])).to("cuda")
        for offset in range(0, len(request_indices), 64):
            selected = request_indices[offset : offset + 64]
            queries_gpu = torch.from_numpy(np.ascontiguousarray(query_vectors[selected])).to("cuda")
            similarities = queries_gpu @ pool_vectors.T
            top_scores, top_positions = torch.topk(similarities, k=min(512, len(pool)), dim=1)
            top_scores = top_scores.cpu().numpy()
            top_positions = top_positions.cpu().numpy()
            for local, request_index in enumerate(selected):
                request = training_requests[request_index]
                positives = request["positive_indices"]
                positive_scores = document_vectors[positives] @ query_vectors[request_index]
                if not np.isfinite(positive_scores).all() or float(positive_scores.max()) <= 0:
                    skipped_mining["nonpositive_teacher_positive"] += 1
                    continue
                positive_index = positives[int(np.argmax(positive_scores))]
                minimum_positive = float(positive_scores.min())
                ceiling = 0.95 * minimum_positive
                known = set(request["known_positive_indices"])
                negatives = []
                for position, value in zip(top_positions[local], top_scores[local], strict=True):
                    candidate = int(pool[int(position)])
                    if candidate in known or float(value) >= ceiling:
                        continue
                    negatives.append((candidate, float(value)))
                    if len(negatives) == 4:
                        break
                if len(negatives) != 4:
                    skipped_mining["insufficient_safe_negatives"] += 1
                    continue
                mined.append({
                    **request,
                    "positive_index": positive_index,
                    "negative_indices": [index for index, _ in negatives],
                    "teacher_scores": [float(positive_scores.max()), *[value for _, value in negatives]],
                })
            del similarities, top_scores, top_positions, queries_gpu
        del pool_vectors
        torch.cuda.empty_cache()
    if len(mined) < 8_000:
        raise RuntimeError("too few positive-aware training examples survived")
    mining_manifest = {
        "requests": len(training_requests),
        "mined": len(mined),
        "skipped": dict(skipped_mining),
        "hard_negatives_per_query": 4,
        "false_negative_ceiling": 0.95,
        "known_positives_masked": True,
        "heldout_labels_used": False,
    }
    original.write_json(WORK / "mining-summary.json", mining_manifest)
    preserve.put(WORK / "mining-summary.json", "reports/mining-summary.json")
    preserve.event("hard-negative-mining-complete", **mining_manifest)
    del query_vectors, document_vectors, raw_targets
    gc.collect()

    # Tokenize once; every training/evaluation forward pass uses the same
    # replacement tokenizer and explicitly excludes only the unknown token.
    unknown_id = compact_tokenizer.token_to_id(UNK_TOKEN)

    def tokenize(texts: list[str], query: bool) -> list[list[int]]:
        values = [(QUERY_TOKEN + " " + text) if query else text for text in texts]
        encodings = compact_tokenizer.encode_batch_fast(values, add_special_tokens=False)
        result = [[token_id for token_id in encoding.ids if token_id != unknown_id] for encoding in encodings]
        if any(not row for row in result):
            raise ValueError("compact tokenizer produced an empty sequence")
        return result

    all_candidate_indices = sorted({row["positive_index"] for row in mined} | {
        index for row in mined for index in row["negative_indices"]
    })
    candidate_tokens = dict(zip(
        all_candidate_indices,
        tokenize([feature_texts[index] for index in all_candidate_indices], query=False),
        strict=True,
    ))
    query_tokens = tokenize([row["query"] for row in mined], query=True)
    for row, ids in zip(mined, query_tokens, strict=True):
        row["query_tokens"] = ids
        row["candidate_tokens"] = [candidate_tokens[row["positive_index"]]] + [
            candidate_tokens[index] for index in row["negative_indices"]
        ]
    del candidate_tokens, query_tokens, all_candidate_indices, feature_texts

    evaluation_document_tokens = tokenize([row["text"] for row in chunks], query=False)
    validation_query_tokens = tokenize([row["question"] for row in development_validation], query=True)

    class TrainableStatic(torch.nn.Module):
        def __init__(self, weights: np.ndarray):
            super().__init__()
            self.embedding = torch.nn.EmbeddingBag.from_pretrained(
                torch.from_numpy(np.ascontiguousarray(weights)),
                freeze=False,
                mode="mean",
                sparse=True,
            )

        def forward(self, sequences: list[list[int]]) -> torch.Tensor:
            flat = torch.tensor([token for row in sequences for token in row], dtype=torch.long, device="cuda")
            offsets = torch.tensor(
                np.cumsum([0] + [len(row) for row in sequences[:-1]]),
                dtype=torch.long,
                device="cuda",
            )
            return self.embedding(flat, offsets)

    model = TrainableStatic(tokenlearn_student.embedding.astype(np.float32, copy=False)).to("cuda")
    del tokenlearn_student
    gc.collect()

    parent_positions = {parent: index for index, parent in enumerate(parents)}
    owners = [parent_positions[chunk["parent_id"]] for chunk in chunks]

    @torch.no_grad()
    def encoded(sequences: list[list[int]], dimension: int, quantized: bool) -> np.ndarray:
        weights = model.embedding.weight[:, :dimension]
        if quantized:
            # Use the exact Model2Vec export path.  The prior float32 torch
            # approximation differed from Model2Vec's float16-backed int8
            # conversion and allowed checkpoint metrics to differ from the
            # actual saved artifact.
            exported = quantize_embeddings(weights.detach().cpu().numpy(), DType.Int8)
            weights = torch.from_numpy(exported).to(device="cuda", dtype=torch.float32)
        output = []
        original_weight = model.embedding.weight
        # Avoid replacing the Parameter: perform pooling directly with the
        # chosen full/compact view so autograd state remains intact.
        for start in range(0, len(sequences), 512):
            batch = sequences[start : start + 512]
            flat = torch.tensor([token for row in batch for token in row], dtype=torch.long, device="cuda")
            offsets = torch.tensor(np.cumsum([0] + [len(row) for row in batch[:-1]]), dtype=torch.long, device="cuda")
            values = F.embedding_bag(flat, weights, offsets, mode="mean")
            output.append(F.normalize(values, dim=1).cpu().numpy())
        if model.embedding.weight is not original_weight:
            raise AssertionError("evaluation mutated training weights")
        return np.concatenate(output)

    def metric_table(document_vectors: np.ndarray, query_vectors: np.ndarray, cohort: list[dict]) -> dict:
        rows = []
        for start in range(0, len(cohort), 64):
            rankings = rank_parents(query_vectors[start : start + 64], document_vectors, owners, parents)
            for query, ranking in zip(cohort[start : start + 64], rankings, strict=True):
                metrics = score(query, ranking, chunks)
                rows.append({"language": query["language"], "metrics": metrics})
        result = {}
        for language in ("en", "tl", "combined"):
            selected = rows if language == "combined" else [row for row in rows if row["language"] == language]
            result[language] = {
                metric: float(np.mean([row["metrics"][metric] for row in selected]))
                for metric in ("Recall@10", "nDCG@10")
            }
            result[language]["questions"] = len(selected)
        return result

    def evaluate_development() -> dict:
        return {
            "full": metric_table(
                encoded(evaluation_document_tokens, 1024, False),
                encoded(validation_query_tokens, 1024, False),
                development_validation,
            ),
            "compact": metric_table(
                encoded(evaluation_document_tokens, 480, True),
                encoded(validation_query_tokens, 480, True),
                development_validation,
            ),
        }

    def score_metrics(value: dict) -> float:
        return sum(
            value[representation][language][metric]
            for representation in ("full", "compact")
            for language in ("en", "tl")
            for metric in ("Recall@10", "nDCG@10")
        )

    def nonregressing(candidate: dict, baseline: dict) -> bool:
        return all(
            candidate[representation][language][metric] + 1e-12
            >= baseline[representation][language][metric]
            for representation in ("full", "compact")
            for language in ("en", "tl")
            for metric in ("Recall@10", "nDCG@10")
        )

    development_history = []
    best_metrics = evaluate_development()
    best_weights = model.embedding.weight.detach().cpu().numpy().copy()
    development_history.append({"phase": "epoch-zero", "epoch": 0, "metrics": best_metrics})
    preserve.event("development-baseline-complete", metrics=best_metrics)

    def objective(batch: list[dict], dimensions: list[int]) -> torch.Tensor:
        queries = model([row["query_tokens"] for row in batch])
        candidates = model([tokens for row in batch for tokens in row["candidate_tokens"]]).reshape(len(batch), 5, -1)
        teacher_scores = torch.tensor([row["teacher_scores"] for row in batch], dtype=torch.float32, device="cuda")
        losses = []
        for dimension in dimensions:
            query = F.normalize(queries[:, :dimension], dim=1)
            candidate = F.normalize(candidates[:, :, :dimension], dim=2)
            scores = torch.einsum("bd,bkd->bk", query, candidate)
            listwise = listwise_teacher_distillation_loss(scores, teacher_scores, temperature=0.05)
            candidate_ce = F.cross_entropy(scores * 20.0, torch.zeros(len(batch), dtype=torch.long, device="cuda"))
            in_batch = F.cross_entropy(
                (query @ candidate[:, 0].T) * 20.0,
                torch.arange(len(batch), device="cuda"),
            )
            alignment = F.mse_loss(scores, teacher_scores)
            losses.append(listwise + 0.5 * candidate_ce + 0.5 * in_batch + 0.25 * alignment)
        return torch.stack(losses).mean()

    phases = [
        ("full-width", [1024], 2, 1e-4),
        ("matryoshka", [1024, 768, 512, 480, 384], 3, 5e-5),
    ]
    for phase_name, dimensions, epochs, initial_lr in phases:
        learning_rate = initial_lr
        for epoch in range(1, epochs + 1):
            phase_start = torch.from_numpy(best_weights).to("cuda")
            model.embedding.weight.data.copy_(phase_start)
            optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
            batches = batch_unique(mined, 64, 42 + epoch + len(development_history))
            losses = []
            model.train()
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = objective(batch, dimensions)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite retrieval objective")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            trained = model.embedding.weight.detach().clone()
            model.eval()
            candidates = []
            for alpha in (0.25, 0.5, 0.75, 1.0):
                model.embedding.weight.data.copy_(phase_start.lerp(trained, alpha))
                metrics = evaluate_development()
                candidates.append((alpha, metrics, model.embedding.weight.detach().cpu().numpy().copy()))
            eligible = [row for row in candidates if nonregressing(row[1], best_metrics)]
            improved = [row for row in eligible if score_metrics(row[1]) > score_metrics(best_metrics) + 1e-12]
            if improved:
                alpha, metrics, weights = max(improved, key=lambda row: score_metrics(row[1]))
                best_metrics, best_weights = metrics, weights
                model.embedding.weight.data.copy_(torch.from_numpy(best_weights).to("cuda"))
                status = "accepted"
            else:
                model.embedding.weight.data.copy_(phase_start)
                alpha, metrics, status = 0.0, best_metrics, "rolled-back-no-nonregressing-gain"
                learning_rate *= 0.5
            row = {
                "phase": phase_name,
                "epoch": epoch,
                "dimensions": dimensions,
                "learning_rate": learning_rate if status.startswith("accepted") else learning_rate * 2,
                "next_learning_rate": learning_rate,
                "mean_loss": float(np.mean(losses)),
                "selected_alpha": alpha,
                "status": status,
                "metrics": metrics,
            }
            development_history.append(row)
            preserve.event("retrieval-epoch-complete", **row)
            del phase_start, trained, candidates, eligible, improved, optimizer
            torch.cuda.empty_cache()

    model.embedding.weight.data.copy_(torch.from_numpy(best_weights).to("cuda"))
    original.write_json(WORK / "development-history.json", development_history)
    preserve.put(WORK / "development-history.json", "reports/development-history.json")

    # Materialize candidates only on the managed job's temporary volume.  No
    # custom model is retained until the exact exported artifact passes the
    # frozen held-out teacher-distance and complete-size gates.
    full_static = StaticModel(
        vectors=best_weights,
        tokenizer=Tokenizer.from_str(compact_tokenizer.to_str()),
        config={"model_type": "model2vec", "architectures": ["StaticModel"],
                "normalize": True, "query_prefix": "[QRY] ", "hidden_dim": 1024,
                "model2vec_pca_dimension": 1024, "tokenlearn_target_pca_dimension": 1024,
                "base_model_revision": teacher_revision},
        normalize=True,
        base_model_name=teacher_id,
        language=["en", "tl"],
    )
    full_dir = WORK / "selected-full-float32"
    full_static.save_pretrained(full_dir)
    compact_vectors = quantize_embeddings(best_weights[:, :480], DType.Int8)
    compact_static = StaticModel(
        vectors=compact_vectors,
        tokenizer=Tokenizer.from_str(compact_tokenizer.to_str()),
        config={"model_type": "model2vec", "architectures": ["StaticModel"],
                "normalize": True, "query_prefix": "[QRY] ", "hidden_dim": 480,
                "trained_matryoshka_dimensions": [1024, 768, 512, 480, 384],
                "export_dtype": "int8", "base_model_revision": teacher_revision},
        normalize=True,
        base_model_name=teacher_id,
        language=["en", "tl"],
    )
    compact_dir = WORK / "selected-compact-int8-480"
    compact_static.save_pretrained(compact_dir)
    file_sizes = {
        path.relative_to(compact_dir).as_posix(): path.stat().st_size
        for path in compact_dir.rglob("*") if path.is_file()
    }
    complete_bytes = validate_measured_artifact(file_sizes)
    heldout = [row for row in annotations if row["partition"] == "evaluation"]
    if len(heldout) != 7922:
        raise ValueError("held-out question count changed")
    heldout_tokens = tokenize([row["question"] for row in heldout], query=True)
    heldout_metrics = {
        "full": metric_table(
            encoded(evaluation_document_tokens, 1024, False),
            encoded(heldout_tokens, 1024, False),
            heldout,
        ),
        "compact": metric_table(
            encoded(evaluation_document_tokens, 480, True),
            encoded(heldout_tokens, 480, True),
            heldout,
        ),
    }
    teacher_reference = {
        language: {"Recall@10": CONFIG["selection"]["teacher_reference_recall_at_10"][language]}
        for language in ("en", "tl")
    }
    quality_gates = {}
    for representation in ("full", "compact"):
        quality_gates[representation] = validate_teacher_distance_gate(
            heldout_metrics[representation],
            teacher_reference,
            english_floor=CONFIG["selection"]["hard_english_recall_at_10_floor"],
            maximum_language_gap=CONFIG["selection"]["hard_maximum_teacher_gap_per_language"],
        )
    preserve.tree(compact_dir, "models/selected-compact-int8-480")
    preserve.event(
        "student-promotion-gates-passed",
        complete_bytes=complete_bytes,
        hard_maximum_bytes=100_000_000,
        quality_gates=quality_gates,
    )
    report = {
        "status": "complete",
        "teacher": {"model": teacher_id, "revision": teacher_revision, "dimension": 1024},
        "tokenizer": {"type": "replacement WordLevel", "vocabulary": 170_003,
                      "query_token": QUERY_TOKEN},
        "model2vec_pca": {"dimension": 1024, "reduction": False},
        "tokenlearn_target_pca": {"dimension": 1024, "reduction": False},
        "tokenlearn_documents": len(feature_rows),
        "synthetic_queries": len(synthetic),
        "development_pairs": len(development_train),
        "mining": mining_manifest,
        "development_history": development_history,
        "heldout": {"questions": len(heldout), "metrics": heldout_metrics},
        "teacher_distance_gate": {
            "english_recall_at_10_floor": CONFIG["selection"]["hard_english_recall_at_10_floor"],
            "maximum_teacher_gap_per_language": CONFIG["selection"]["hard_maximum_teacher_gap_per_language"],
            "teacher_reference": teacher_reference,
            "results": quality_gates,
        },
        "compact_artifact": {"dimension": 480, "dtype": "int8",
                             "files": file_sizes, "complete_bytes": complete_bytes,
                             "hard_maximum_bytes": 100_000_000},
        "latency_measured": False,
        "runtime_peak_memory_measured": False,
        "heldout_used_for_training_or_checkpoint_selection": False,
        "artifacts": preserve.index,
    }
    original.write_json(WORK / "compact-sota-completion.json", report)
    preserve.put(WORK / "compact-sota-completion.json", "reports/compact-sota-completion.json")
    preserve.event("compact-sota-complete", heldout_questions=len(heldout),
                   complete_artifact_bytes=complete_bytes, heldout_metrics=heldout_metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
