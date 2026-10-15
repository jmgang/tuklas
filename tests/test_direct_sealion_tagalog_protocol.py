import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tagalog_only_contract_excludes_english_everywhere():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-direct-sealion-tagalog-v1.json").read_text()
    )

    assert recipe["language"] == "tl"
    assert recipe["allowed_languages"] == ["tl"]
    assert "english_recall_at_10_floor" not in recipe["quality"]
    assert recipe["quality"]["tagalog_maximum_teacher_recall_at_10_gap"] == 0.10
    assert recipe["training"]["tokenlearn_corpus_language"] == "tl"
    assert recipe["training"]["retrieval_query_language"] == "tl"
    assert recipe["training"]["retrieval_learning_rate"] == 0.001
    assert recipe["training"]["retrieval_epochs"] == 8
    assert recipe["evaluation"]["development_language"] == "tl"
    assert recipe["evaluation"]["heldout_language"] == "tl"


def test_tagalog_worker_and_launcher_cannot_reuse_bilingual_protocol():
    launcher = (ROOT / "scripts/start_multiwikiqa_direct_sealion_tagalog.py").read_text()
    worker = (ROOT / "scripts/aws_multiwikiqa_direct_sealion.py").read_text()

    assert "multiwikiqa-direct-parent-v1.json" not in launcher
    assert "multiwikiqa-direct-sealion-tagalog-v1.json" in launcher
    assert 'row["language"] == "tl"' in worker
    assert 'language=["tl"] if tagalog_only' in worker
    assert "validate_tagalog_teacher_distance_gate" in worker
