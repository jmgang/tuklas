import json
from pathlib import Path

import pytest

from pycon_model2vec_demo.tagalog_capacity_frontier import candidate_gate, validate_candidate_recipe


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_capacity_frontier_recipe():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v1.json").read_text()
    )
    validate_candidate_recipe(recipe)
    assert recipe["training"]["retrieval_learning_rate"] == 0.001
    assert recipe["training"]["retrieval_epochs"] == 8
    assert recipe["training"]["ranking_data_expansion"]["minimum_safely_mined_pairs"] == 12_000


def test_frozen_full_parent_432_recipe():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v2.json").read_text()
    )
    validate_candidate_recipe(recipe)
    candidate = recipe["candidates"][0]
    assert candidate["id"] == "full_parent_250k_432"
    assert candidate["vocabulary_mode"] == "full-parent"
    assert candidate["dimension"] == 432
    assert candidate["hard_maximum_complete_artifact_bytes"] == 120_000_000
    assert recipe["training"]["ranking_data_expansion"]["minimum_safely_mined_pairs"] == 15_000


def test_frozen_full_parent_432_extended_ranking_recipe():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v3.json").read_text()
    )
    validate_candidate_recipe(recipe)
    candidate = recipe["candidates"][0]
    assert candidate["id"] == "full_parent_250k_432"
    assert candidate["vocabulary_mode"] == "full-parent"
    assert candidate["dimension"] == 432
    assert candidate["hard_maximum_complete_artifact_bytes"] == 120_000_000
    assert recipe["training"]["retrieval_epochs"] == 16
    assert recipe["training"]["ranking_data_expansion"]["minimum_safely_mined_pairs"] == 15_000


def test_frozen_full_parent_432_retrieval_selected_dynamic_recipe():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v4.json").read_text()
    )
    validate_candidate_recipe(recipe)
    training = recipe["training"]
    assert training["tokenlearn_checkpoint_selection"] == "development-retrieval"
    assert training["query_vector_distillation_weight"] == 1.0
    assert training["dynamic_remine_after_epochs"] == [8, 16, 24]
    assert training["retrieval_epochs"] == 32
    assert training["retrieval_lr_schedule"] == [
        {"start_epoch": 1, "learning_rate": 0.001},
        {"start_epoch": 9, "learning_rate": 0.0005},
        {"start_epoch": 17, "learning_rate": 0.00025},
    ]
    assert training["retrieval_early_stopping_patience"] == 4
    assert training["ranking_data_expansion"]["minimum_safely_mined_pairs"] == 15_000
    candidate = recipe["candidates"][0]
    assert candidate["vocabulary_mode"] == "full-parent"
    assert candidate["dimension"] == 432
    assert candidate["hard_maximum_complete_artifact_bytes"] == 120_000_000


def test_scaffolded_full_parent_512_recipe():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v5-512.json").read_text()
    )
    validate_candidate_recipe(recipe)
    candidate = recipe["candidates"][0]
    assert candidate == {
        "id": "full_parent_250k_512",
        "vocabulary_mode": "full-parent",
        "dimension": 512,
        "dtype": "int8",
        "hard_maximum_complete_artifact_bytes": 140_000_000,
    }
    assert recipe["training"]["tokenlearn_checkpoint_selection"] == "development-retrieval"
    assert recipe["training"]["query_vector_distillation_weight"] == 1.0
    assert recipe["training"]["dynamic_remine_after_epochs"] == [8, 16, 24]
    assert recipe["training"]["retrieval_epochs"] == 32
    assert recipe["quality"]["maximum_teacher_recall_at_10_gap"] == 0.10


def test_controlled_repair_starts_with_exact_v3_control_and_persists_checkpoint():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v6-control.json").read_text()
    )
    validate_candidate_recipe(recipe)
    training = recipe["training"]
    repair = training["controlled_repair"]
    assert training["tokenlearn_checkpoint_selection"] == "validation-loss"
    assert training["query_vector_distillation_weight"] == 0.0
    assert training["retrieval_epochs"] == 16
    assert training["retrieval_lr_schedule"] == []
    assert training["dynamic_remine_after_epochs"] == []
    assert training["retrieval_early_stopping_patience"] == 0
    assert repair["persist_ranking_ready_checkpoint"] is True
    assert repair["previous_best_recall_at_10"] == pytest.approx(0.8324873096446701)
    assert repair["future_query_alignment_weights"] == [0.05, 0.1, 0.25]
    assert recipe["quality"]["minimum_development_recall_at_10"] == pytest.approx(
        0.8746192893401015
    )


def test_concurrent_512_candidate_uses_only_v3_style_capacity_change():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-sealion-tagalog-capacity-frontier-v7-512-control.json").read_text()
    )
    validate_candidate_recipe(recipe)
    training = recipe["training"]
    candidate = recipe["candidates"][0]
    assert candidate == {
        "id": "full_parent_250k_512",
        "vocabulary_mode": "full-parent",
        "dimension": 512,
        "dtype": "int8",
        "hard_maximum_complete_artifact_bytes": 140_000_000,
    }
    assert training["tokenlearn_checkpoint_selection"] == "validation-loss"
    assert training["query_vector_distillation_weight"] == 0.0
    assert training["retrieval_epochs"] == 16
    assert training["retrieval_lr_schedule"] == []
    assert training["dynamic_remine_after_epochs"] == []
    assert training["retrieval_early_stopping_patience"] == 0
    assert training["controlled_repair"]["phase"] == "v3-512-control"


def test_latest_failed_candidate_is_red():
    report = json.loads(
        (ROOT / "runs/pycon-direct-sealion-tl-20261006-030353-2882ff/rejection-gate-input.json").read_text()
    )
    with pytest.raises(ValueError, match="teacher gap"):
        candidate_gate(report)


def test_candidate_must_pass_quality_size_and_leakage_gates():
    report = {
        "language": "tl",
        "english_records_used": False,
        "heldout_used_for_training_or_selection": False,
        "artifact": {"complete_bytes": 90_000_000},
        "development": {
            "student": {"tl": {"Recall@10": 0.90}},
            "teacher": {"tl": {"Recall@10": 0.97}},
        },
    }
    assert candidate_gate(report)["passed"] is True
    report["artifact"]["complete_bytes"] = 100_000_000
    with pytest.raises(ValueError, match="hard byte limit"):
        candidate_gate(report)
