import json
from pathlib import Path

from pycon_model2vec_demo.multiwikiqa_cohere_v3 import (
    load_frozen_inputs,
    preflight_summary,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multiwikiqa-cohere-multilingual-v3-eval-v1.json"


def test_scaffold_is_multilingual_v3_strict_and_quality_only():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    assert config["model"]["model_id"] == "cohere.embed-multilingual-v3"
    assert config["model"]["truncate"] == "NONE"
    assert config["model"]["document_input_type"] == "search_document"
    assert config["model"]["query_input_type"] == "search_query"
    assert config["evaluation"]["latency_deferred"] is True


def test_frozen_tagalog_heldout_is_losslessly_segmented_to_safe_token_budget():
    config = json.loads(CONFIG.read_text())
    layout, queries = load_frozen_inputs(ROOT, config)
    assert len(layout["chunks"]) == 6842
    assert len(layout["parents"]) == 1294
    assert len(queries) == 3984
    assert layout["segmentation"]["original_chunks"] == 6546
    assert layout["segmentation"]["split_chunks"] == 296
    assert layout["segmentation"]["safe_token_limit"] == 480
    assert layout["segmentation"]["maximum_segment_tokens"] <= 480
    assert layout["segmentation"]["maximum_query_tokens"] == 56
    assert {row["language"] for row in queries} == {"tl"}
    assert {row["partition"] for row in queries} == {"evaluation"}
    assert max(len(row["question"]) for row in queries) == 205

    by_source = {}
    for segment in layout["chunks"]:
        by_source.setdefault(segment["source_chunk_id"], []).append(segment)
        assert segment["start"] < segment["end"]
        assert segment["text"]
    original = json.loads((ROOT / config["dataset"]["layout"]).read_text())
    for chunk in original["chunks"]:
        pieces = sorted(by_source[chunk["chunk_id"]], key=lambda row: row["start"])
        assert "".join(piece["text"] for piece in pieces) == chunk["text"]
        assert pieces[0]["start"] == chunk["start"]
        assert pieces[-1]["end"] == chunk["end"]


def test_tokenizer_snapshot_and_provider_budget_are_pinned():
    config = json.loads(CONFIG.read_text())
    tokenizer = config["tokenizer"]
    assert tokenizer["revision"] == "4cb7185c1ad1d4931fb39122308e9bcc9e3fe043"
    assert tokenizer["safe_token_limit"] == 480
    assert tokenizer["provider_token_limit"] == 512
    assert tokenizer["tokenizer_json_sha256"] == "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"


def test_preflight_does_not_claim_or_start_paid_inference():
    config = json.loads(CONFIG.read_text())
    summary = preflight_summary(ROOT, config)
    assert summary["status"] == "ready-not-executed"
    assert summary["paid_inference_started"] is False
    assert summary["latency_measured"] is False


def test_runner_defaults_to_offline_preflight_and_defers_latency():
    source = (ROOT / "scripts/run_multiwikiqa_cohere_v3.py").read_text()
    assert 'parser.add_argument("--execute", action="store_true")' in source
    assert "if not args.execute:" in source
    assert '"search_document"' not in source
    assert '"search_query"' not in source
    assert '"latency": {"status": "deferred-not-measured"}' in source
    assert "rank_parents(" in source
    assert "load_or_encode(" in source
