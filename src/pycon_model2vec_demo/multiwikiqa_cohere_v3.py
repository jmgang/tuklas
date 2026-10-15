"""Frozen-input validation for the MultiWikiQA Cohere multilingual v3 baseline."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from pycon_model2vec_demo.segmented_encoding import SegmentPolicy


MODEL_ID = "cohere.embed-multilingual-v3"
METRICS = (
    "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10",
    "AnswerSpanCoverage@1", "AnswerSpanCoverage@3", "AnswerSpanCoverage@5",
    "AnswerSpanCoverage@10",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol") != "multiwikiqa-cohere-multilingual-v3-eval-v1":
        raise ValueError("wrong Cohere evaluation protocol")
    if config["aws"] != {"profile": "admin-general", "region": "ap-southeast-1"}:
        raise ValueError("AWS profile or region changed")
    model = config["model"]
    required = {
        "model_id": MODEL_ID,
        "dimension": 1024,
        "document_input_type": "search_document",
        "query_input_type": "search_query",
        "truncate": "NONE",
        "maximum_texts_per_request": 96,
        "maximum_input_characters": 2048,
        "full_content": True,
    }
    for key, value in required.items():
        if model.get(key) != value:
            raise ValueError(f"Cohere setting changed: {key}")
    tokenizer = config["tokenizer"]
    required_tokenizer = {
        "repository": "CohereLabs/Cohere-embed-multilingual-v3.0",
        "revision": "4cb7185c1ad1d4931fb39122308e9bcc9e3fe043",
        "snapshot": "models/cohere-multilingual-v3-tokenizer-4cb7185c1ad1",
        "tokenizer_json_sha256": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        "sentencepiece_sha256": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        "provider_token_limit": 512,
        "safe_token_limit": 480,
        "segmentation": "lossless-original-text-offsets",
    }
    for key, value in required_tokenizer.items():
        if tokenizer.get(key) != value:
            raise ValueError(f"Cohere tokenizer setting changed: {key}")
    if tokenizer["safe_token_limit"] >= tokenizer["provider_token_limit"]:
        raise ValueError("safe token budget must remain below provider limit")
    dataset = config["dataset"]
    if (dataset.get("language"), dataset.get("partition")) != ("tl", "evaluation"):
        raise ValueError("Cohere scaffold must score Tagalog heldout only")
    if (
        dataset.get("questions"),
        dataset.get("parents"),
        dataset.get("chunks"),
        dataset.get("searchable_segments"),
        dataset.get("split_chunks"),
    ) != (3984, 1294, 6546, 6842, 296):
        raise ValueError("frozen evaluation counts changed")
    retrieval = config["retrieval"]
    if retrieval != {
        "similarity": "cosine",
        "search": "exact-exhaustive",
        "segment_reduction": "maximum-similarity-per-parent",
        "cutoff": 10,
        "unique_parent_results": True,
        "vector_pooling": False,
    }:
        raise ValueError("retrieval protocol changed")
    if tuple(config["evaluation"]["metrics"]) != METRICS:
        raise ValueError("metric table changed")
    if config["evaluation"].get("latency_deferred") is not True:
        raise ValueError("latency must remain deferred")


def _load_tokenizer(root: Path, config: Mapping[str, Any]):
    from tokenizers import Tokenizer

    frozen = config["tokenizer"]
    snapshot = root / frozen["snapshot"]
    tokenizer_json = snapshot / "tokenizer.json"
    sentencepiece = snapshot / "sentencepiece.bpe.model"
    if file_sha256(tokenizer_json) != frozen["tokenizer_json_sha256"]:
        raise ValueError("Cohere tokenizer.json checksum changed")
    if file_sha256(sentencepiece) != frozen["sentencepiece_sha256"]:
        raise ValueError("Cohere sentencepiece checksum changed")
    return Tokenizer.from_file(str(tokenizer_json))


def _segment_layout(
    root: Path, config: Mapping[str, Any], layout: dict, *, frozen_counts: bool = True
) -> dict:
    """Split chunks to fit Cohere's token budget.

    ``frozen_counts`` asserts the segmentation reproduces the frozen
    6,842-segment configuration.  Pass False only when the caller has
    deliberately re-chunked the layout to match another system's index -- the
    counts then legitimately differ, and the per-segment token ceiling (the
    check that actually protects the API call) still binds.
    """
    tokenizer = _load_tokenizer(root, config)
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=True).ids)
    content_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    safe_limit = int(config["tokenizer"]["safe_token_limit"])
    character_limit = int(config["model"]["maximum_input_characters"])
    policy = SegmentPolicy(token_count, content_count, safe_limit, character_limit)

    segments: list[dict] = []
    split_chunks = 0
    maximum_segment_tokens = 0
    for source in layout["chunks"]:
        plan = policy.plan(source["text"])
        split_chunks += int(len(plan) > 1)
        for part_index, (local_start, local_end, content_tokens) in enumerate(plan):
            text = source["text"][local_start:local_end]
            counted = token_count(text)
            maximum_segment_tokens = max(maximum_segment_tokens, counted)
            segments.append({
                **source,
                "chunk_id": f'{source["chunk_id"]}:cohere-{part_index:02d}',
                "source_chunk_id": source["chunk_id"],
                "source_part": part_index,
                "start": int(source["start"]) + local_start,
                "end": int(source["start"]) + local_end,
                "text": text,
                "cohere_tokens": counted,
                "content_tokens": content_tokens,
            })
    expected_segments = int(config["dataset"]["searchable_segments"])
    expected_splits = int(config["dataset"]["split_chunks"])
    if frozen_counts and (
        len(segments) != expected_segments or split_chunks != expected_splits
    ):
        raise ValueError("frozen Cohere segmentation counts changed")
    if maximum_segment_tokens > safe_limit:
        raise AssertionError("Cohere segment exceeds safe token budget")
    return {
        **layout,
        "chunks": segments,
        "segmentation": {
            "policy": config["tokenizer"]["segmentation"],
            "original_chunks": len(layout["chunks"]),
            "searchable_segments": len(segments),
            "split_chunks": split_chunks,
            "provider_token_limit": int(config["tokenizer"]["provider_token_limit"]),
            "safe_token_limit": safe_limit,
            "maximum_segment_tokens": maximum_segment_tokens,
            "tokenizer_repository": config["tokenizer"]["repository"],
            "tokenizer_revision": config["tokenizer"]["revision"],
        },
    }


def load_frozen_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    validate_config(config)
    dataset = config["dataset"]
    layout_path = root / dataset["layout"]
    annotations_path = root / dataset["annotations"]
    if file_sha256(layout_path) != dataset["layout_sha256"]:
        raise ValueError("shared layout checksum changed")
    if file_sha256(annotations_path) != dataset["annotations_sha256"]:
        raise ValueError("annotation checksum changed")
    original_layout = json.loads(layout_path.read_text())
    annotations = [json.loads(line) for line in annotations_path.open(encoding="utf-8")]
    queries = [
        row for row in annotations
        if row["language"] == "tl" and row["partition"] == "evaluation"
    ]
    if len(original_layout["chunks"]) != 6546 or len(original_layout["parents"]) != 1294 or len(queries) != 3984:
        raise ValueError("frozen MultiWikiQA cohort changed")
    layout = _segment_layout(root, config, original_layout)
    limit = int(config["model"]["maximum_input_characters"])
    if any(not row["text"].strip() or len(row["text"]) > limit for row in layout["chunks"]):
        raise ValueError("segmented document exceeds provider character budget")
    tokenizer = _load_tokenizer(root, config)
    query_tokens = [
        len(tokenizer.encode(row["question"], add_special_tokens=True).ids)
        for row in queries
    ]
    if any(not row["question"].strip() or len(row["question"]) > limit for row in queries):
        raise ValueError("query exceeds Cohere v3 input budget")
    if max(query_tokens) > int(config["tokenizer"]["safe_token_limit"]):
        raise ValueError("query exceeds frozen safe token budget")
    layout["segmentation"]["maximum_query_tokens"] = max(query_tokens)
    return layout, queries


def preflight_summary(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    layout, queries = load_frozen_inputs(root, config)
    return {
        "protocol": config["protocol"],
        "status": "ready-not-executed",
        "model_id": config["model"]["model_id"],
        "region": config["aws"]["region"],
        "source_chunks": layout["segmentation"]["original_chunks"],
        "searchable_segments": len(layout["chunks"]),
        "split_chunks": layout["segmentation"]["split_chunks"],
        "parents": len(layout["parents"]),
        "queries": len(queries),
        "maximum_document_characters": max(len(row["text"]) for row in layout["chunks"]),
        "maximum_query_characters": max(len(row["question"]) for row in queries),
        "maximum_document_tokens": layout["segmentation"]["maximum_segment_tokens"],
        "maximum_query_tokens": layout["segmentation"]["maximum_query_tokens"],
        "safe_token_limit": layout["segmentation"]["safe_token_limit"],
        "provider_truncation": "NONE",
        "latency_measured": False,
        "paid_inference_started": False,
    }
