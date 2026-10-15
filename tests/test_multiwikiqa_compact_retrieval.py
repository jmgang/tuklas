import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from model2vec.train.utils import get_probable_pad_token_id

from pycon_model2vec_demo.multiwikiqa_compact_retrieval import (
    artifact_budget,
    add_neutral_query_role,
    external_article_id_from_passage_id,
    select_external_title_pairs,
    select_balanced_vocabulary,
    combined_matryoshka_retrieval_loss,
    listwise_teacher_distillation_loss,
    mine_positive_aware_hard_negatives,
    prefix_role,
    select_nonregressing_checkpoint,
    strip_padding_and_unknown,
    validate_measured_artifact,
    validate_mining_yield,
    validate_teacher_distance_gate,
    validate_tagalog_teacher_distance_gate,
    validate_quality_promotion,
    validate_compact_recipe,
)


def test_static_mean_pooling_drops_padding_and_unknown_placeholders():
    # This is the exact shape observed in the Tagalog tokenizer probe: nine
    # content positions followed by batch padding. Neither placeholder may be
    # averaged into the sentence embedding or trained as ordinary content.
    assert strip_padding_and_unknown(
        [3, 149, 9980, 112, 3, 112, 27, 3, 0, 0, 0],
        pad_id=0,
        unknown_id=3,
    ) == [149, 9980, 112, 112, 27]


def test_model2vec_padding_resolver_reads_the_tokenizer_padding_configuration():
    tokenizer = Tokenizer(WordLevel({"[PAD]": 7, "[UNK]": 9, "kumusta": 11}, unk_token="[UNK]"))
    tokenizer.enable_padding(pad_id=7, pad_token="[PAD]")

    assert get_probable_pad_token_id(tokenizer) == 7


def test_tagalog_teacher_gate_rejects_english_and_enforces_ten_point_gap():
    with pytest.raises(ValueError, match="only tl"):
        validate_tagalog_teacher_distance_gate(
            candidate={"en": {"Recall@10": 0.99}, "tl": {"Recall@10": 0.90}},
            teacher={"tl": {"Recall@10": 0.95}},
        )
    with pytest.raises(ValueError, match="Tagalog teacher gap"):
        validate_tagalog_teacher_distance_gate(
            candidate={"tl": {"Recall@10": 0.84}},
            teacher={"tl": {"Recall@10": 0.95}},
        )
    assert validate_tagalog_teacher_distance_gate(
        candidate={"tl": {"Recall@10": 0.86}},
        teacher={"tl": {"Recall@10": 0.95}},
    )["tl_gap"] == pytest.approx(0.09)


def test_mining_yield_is_relative_to_the_frozen_3468_request_curriculum():
    result = validate_mining_yield(
        requested={"development:en": 867, "development:tl": 867,
                   "synthetic:en": 867, "synthetic:tl": 867},
        mined={"development:en": 800, "development:tl": 790,
               "synthetic:en": 780, "synthetic:tl": 770},
        minimum_overall_fraction=0.75,
        minimum_slice_fraction=0.70,
    )
    assert result["requested"] == 3_468
    assert result["mined"] == 3_140

    with pytest.raises(ValueError, match="overall hard-negative yield"):
        validate_mining_yield(
            requested={"development:en": 867, "development:tl": 867,
                       "synthetic:en": 867, "synthetic:tl": 867},
            mined={"development:en": 500, "development:tl": 500,
                   "synthetic:en": 500, "synthetic:tl": 500},
            minimum_overall_fraction=0.75,
            minimum_slice_fraction=0.50,
        )


def test_teacher_distance_gate_rejects_completed_compact_candidate():
    with pytest.raises(ValueError, match="English Recall@10"):
        validate_teacher_distance_gate(
            candidate={"en": {"Recall@10": 0.7402234636871509},
                       "tl": {"Recall@10": 0.6006526104417671}},
            teacher={"en": {"Recall@10": 0.9836},
                     "tl": {"Recall@10": 0.9698}},
            english_floor=0.90,
            maximum_language_gap=0.10,
        )


def test_teacher_distance_gate_requires_both_languages():
    with pytest.raises(ValueError, match="Tagalog teacher gap"):
        validate_teacher_distance_gate(
            candidate={"en": {"Recall@10": 0.92},
                       "tl": {"Recall@10": 0.86}},
            teacher={"en": {"Recall@10": 0.9836},
                     "tl": {"Recall@10": 0.9698}},
            english_floor=0.90,
            maximum_language_gap=0.10,
        )


def test_teacher_distance_gate_accepts_only_close_student():
    result = validate_teacher_distance_gate(
        candidate={"en": {"Recall@10": 0.91},
                   "tl": {"Recall@10": 0.88}},
        teacher={"en": {"Recall@10": 0.9836},
                 "tl": {"Recall@10": 0.9698}},
        english_floor=0.90,
        maximum_language_gap=0.10,
    )
    assert result == {"en_gap": pytest.approx(0.0736), "tl_gap": pytest.approx(0.0898)}


def test_external_article_id_uses_corpus_id_not_layout_url():
    assert external_article_id_from_passage_id("en:3839438:7141:8091") == "en:3839438"
    assert external_article_id_from_passage_id("tl:123:0:950") == "tl:123"
    with pytest.raises(ValueError, match="numeric start/end offsets"):
        external_article_id_from_passage_id("https://example.test/wiki/article")


def test_measured_complete_artifact_has_a_strict_100mb_boundary():
    passing = {
        "model.safetensors": 92_000_000,
        "tokenizer.json": 7_500_000,
        "config.json": 499_999,
    }
    assert validate_measured_artifact(passing) == 99_999_999

    failing = dict(passing)
    failing["config.json"] = 500_000
    with pytest.raises(ValueError, match="measured complete artifact violates the hard size gate"):
        validate_measured_artifact(failing)


def test_measured_complete_artifact_requires_the_whole_package_manifest():
    with pytest.raises(ValueError, match="manifest is empty"):
        validate_measured_artifact({})
    with pytest.raises(ValueError, match="negative artifact size"):
        validate_measured_artifact({"model.safetensors": -1})


def test_sub_potion_artifact_is_still_rejected_when_over_100mb():
    recipe = {
        "full_dimension": 1024,
        "matryoshka_dimensions": [1024, 768, 512, 480, 448, 384, 256],
        "export_dimension": 480,
        "export_dtype": "int8",
        "query_prefix": "[QRY] ",
        "document_prefix": "",
        "minimum_size_headroom_bytes": 2_000_000,
        "hard_maximum_complete_artifact_bytes": 100_000_000,
        "preferred_complete_artifact_bytes": 32 * 1024 * 1024,
    }
    budget = artifact_budget(
        vocab_size=250_000,
        dimension=480,
        dtype="int8",
        tokenizer_bytes=9_090_501,
        metadata_bytes=4_245,
        reference_artifact_bytes=131_200_180,
    )

    assert budget.matrix_bytes == 120_000_000
    assert budget.complete_artifact_bytes == 129_094_746
    assert budget.complete_artifact_bytes < budget.reference_artifact_bytes
    with pytest.raises(ValueError, match="hard size gate"):
        validate_compact_recipe(recipe, budget)

    viable = artifact_budget(
        vocab_size=180_000,
        dimension=480,
        dtype="int8",
        tokenizer_bytes=6_600_000,
        metadata_bytes=100_000,
        reference_artifact_bytes=131_200_180,
    )
    validate_compact_recipe(recipe, viable)


def test_30mb_class_is_preferred_but_only_100mb_is_the_hard_gate():
    recipe = {
        "full_dimension": 1024,
        "matryoshka_dimensions": [1024, 768, 512, 480, 384],
        "export_dimension": 384,
        "export_dtype": "int8",
        "query_prefix": "[QRY] ",
        "document_prefix": "",
        "minimum_size_headroom_bytes": 2_000_000,
        "hard_maximum_complete_artifact_bytes": 100_000_000,
        "preferred_complete_artifact_bytes": 32 * 1024 * 1024,
    }
    compact = artifact_budget(64_000, 384, "int8", 2_500_000, 100_000, 131_200_180)
    validate_compact_recipe(recipe, compact)

    too_large = artifact_budget(128_000, 480, "int8", 5_000_000, 100_000, 131_200_180)
    validate_compact_recipe(recipe | {"export_dimension": 480}, too_large)

    over_hard_gate = artifact_budget(195_000, 480, "int8", 7_000_000, 100_000, 131_200_180)
    with pytest.raises(ValueError, match="hard size gate"):
        validate_compact_recipe(recipe | {"export_dimension": 480}, over_hard_gate)


def test_compact_recipe_rejects_untrained_export_prefix():
    recipe = {
        "full_dimension": 1024,
        "matryoshka_dimensions": [1024, 512, 384, 256],
        "export_dimension": 448,
        "export_dtype": "int8",
        "query_prefix": "[QRY] ",
        "document_prefix": "",
        "minimum_size_headroom_bytes": 2_000_000,
    }
    budget = artifact_budget(250_000, 448, "int8", 9_090_501, 4_245, 131_200_180)

    with pytest.raises(ValueError, match="export dimension must be trained"):
        validate_compact_recipe(recipe, budget)


def test_query_role_is_explicit_and_document_role_is_unchanged():
    assert prefix_role("Nasaan ang Davao?", role="query", query_prefix="[QRY] ") == "[QRY] Nasaan ang Davao?"
    assert prefix_role("Ang Davao ay...", role="document", query_prefix="[QRY] ") == "Ang Davao ay..."


def test_query_role_token_starts_neutral_and_is_atomic():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Davao": 1}, unk_token="[UNK]"))
    vectors = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    expanded, updated, token_id = add_neutral_query_role(vectors, tokenizer, token="[QRY]")

    assert token_id == 2
    assert expanded.shape == (3, 2)
    np.testing.assert_array_equal(expanded[:2], vectors)
    np.testing.assert_array_equal(expanded[2], np.zeros(2, dtype=np.float32))
    assert updated.encode("[QRY]", add_special_tokens=False).ids == [token_id]


def test_external_title_pairs_are_balanced_deterministic_and_one_per_article():
    rows = [
        {"language": "en", "source_article_id": "en:a", "source_title": "Alpha", "start": 100, "passage_id": "a:100", "text": "later"},
        {"language": "en", "source_article_id": "en:a", "source_title": "Alpha", "start": 0, "passage_id": "a:0", "text": "first"},
        {"language": "en", "source_article_id": "en:b", "source_title": "Beta", "start": 0, "passage_id": "b:0", "text": "beta"},
        {"language": "tl", "source_article_id": "tl:a", "source_title": "Alpabeto", "start": 0, "passage_id": "ta:0", "text": "una"},
        {"language": "tl", "source_article_id": "tl:b", "source_title": "Bata", "start": 0, "passage_id": "tb:0", "text": "bata"},
    ]

    selected = select_external_title_pairs(rows, per_language=2, seed=42)

    assert len(selected) == 4
    assert {row["language"] for row in selected} == {"en", "tl"}
    assert len({row["article_id"] for row in selected}) == 4
    assert next(row for row in selected if row["article_id"] == "en:a")["positive"] == "first"
    assert selected == select_external_title_pairs(list(reversed(rows)), per_language=2, seed=42)


def test_balanced_vocabulary_reserves_capacity_for_both_languages():
    english = {"shared": 100, "english-a": 90, "english-b": 80, "english-c": 70}
    tagalog = {"shared": 20, "tagalog-a": 90, "tagalog-b": 80, "tagalog-c": 70}

    selected = select_balanced_vocabulary(english, tagalog, size=5, shared_fraction=0.2)

    assert selected[0] == "shared"
    assert {"english-a", "english-b"} <= set(selected)
    assert {"tagalog-a", "tagalog-b"} <= set(selected)
    assert selected == select_balanced_vocabulary(dict(reversed(list(english.items()))), tagalog, size=5, shared_fraction=0.2)


def test_combined_loss_trains_margin_and_in_batch_ranking_at_every_dimension():
    queries = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[0.8, 0.2, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]], requires_grad=True)
    negatives = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    teacher_margins = torch.tensor([0.75, 0.70])

    value, details = combined_matryoshka_retrieval_loss(
        queries,
        positives,
        negatives,
        teacher_margins,
        dimensions=[4, 2],
        margin_weight=1.0,
        ranking_weight=1.0,
        scale=20.0,
    )

    assert torch.isfinite(value)
    assert value.item() > 0
    assert set(details) == {4, 2}
    assert all(set(row) == {"margin_mse", "mnrl", "combined"} for row in details.values())
    value.backward()
    assert queries.grad is not None
    assert torch.isfinite(queries.grad).all()


def test_combined_loss_supports_sparse_row_updates_for_large_vocabulary():
    weights = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.7, 0.3, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]],
        dtype=torch.float32,
    )
    bag = torch.nn.EmbeddingBag.from_pretrained(weights, freeze=False, sparse=True, mode="mean")

    def encode(ids):
        flat = torch.tensor([token for row in ids for token in row], dtype=torch.long)
        offsets = torch.tensor(np.cumsum([0] + [len(row) for row in ids[:-1]]), dtype=torch.long)
        return bag(flat, offsets)

    query = encode([[1], [2]])
    positive = encode([[1, 3], [2, 4]])
    negative = encode([[2, 4], [1, 3]])
    value, _ = combined_matryoshka_retrieval_loss(
        query,
        positive,
        negative,
        torch.tensor([0.6, 0.6]),
        dimensions=[4, 2],
        margin_weight=1.0,
        ranking_weight=1.0,
        scale=20.0,
    )
    value.backward()

    assert bag.weight.grad is not None and bag.weight.grad.is_sparse
    before = bag.weight.detach().clone()
    torch.optim.SparseAdam([bag.weight], lr=0.001).step()
    assert not torch.equal(before, bag.weight.detach())


def test_external_pairs_can_use_ranking_without_invented_teacher_margins():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[0.6, 0.4], [0.4, 0.6]], requires_grad=True)

    value, details = combined_matryoshka_retrieval_loss(
        query,
        positive,
        None,
        None,
        dimensions=[2],
        margin_weight=0.0,
        ranking_weight=1.0,
        scale=2.0,
    )

    assert value.item() > 0
    assert details[2]["margin_mse"] == 0.0
    value.backward()
    assert query.grad is not None


def test_hard_negative_mining_masks_positives_and_possible_false_negatives():
    selected = mine_positive_aware_hard_negatives(
        ["gold", "near-duplicate", "hard-b", "hard-a", "easy", "other-gold"],
        [0.90, 0.88, 0.81, 0.81, 0.20, 0.89],
        positive_ids=["gold", "other-gold"],
        minimum_positive_score=0.90,
        maximum_positive_fraction=0.95,
        count=2,
    )

    assert selected == [("hard-a", 0.81), ("hard-b", 0.81)]


def test_hard_negative_mining_refuses_too_few_safe_candidates():
    with pytest.raises(ValueError, match="not enough safe hard negatives"):
        mine_positive_aware_hard_negatives(
            ["gold", "near"],
            [0.90, 0.89],
            positive_ids=["gold"],
            minimum_positive_score=0.90,
            count=1,
        )


def test_listwise_distillation_uses_the_entire_teacher_ordering():
    teacher = torch.tensor([[4.0, 2.0, 0.5]])
    aligned = torch.tensor([[3.0, 1.5, 0.1]], requires_grad=True)
    reversed_order = torch.tensor([[0.1, 1.5, 3.0]], requires_grad=True)

    aligned_loss = listwise_teacher_distillation_loss(aligned, teacher, temperature=0.5)
    reversed_loss = listwise_teacher_distillation_loss(reversed_order, teacher, temperature=0.5)

    assert aligned_loss < reversed_loss
    aligned_loss.backward()
    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()


def test_checkpoint_selection_rejects_any_language_or_compact_export_regression():
    history = [
        {
            "epoch": 0,
            "metrics": {
                "full": {"en": {"Recall@10": 0.80, "nDCG@10": 0.70}, "tl": {"Recall@10": 0.68, "nDCG@10": 0.55}},
                "compact": {"en": {"Recall@10": 0.78, "nDCG@10": 0.68}, "tl": {"Recall@10": 0.66, "nDCG@10": 0.53}},
            },
        },
        {
            "epoch": 1,
            "metrics": {
                "full": {"en": {"Recall@10": 0.82, "nDCG@10": 0.72}, "tl": {"Recall@10": 0.69, "nDCG@10": 0.56}},
                "compact": {"en": {"Recall@10": 0.81, "nDCG@10": 0.71}, "tl": {"Recall@10": 0.65, "nDCG@10": 0.54}},
            },
        },
        {
            "epoch": 2,
            "metrics": {
                "full": {"en": {"Recall@10": 0.83, "nDCG@10": 0.73}, "tl": {"Recall@10": 0.70, "nDCG@10": 0.57}},
                "compact": {"en": {"Recall@10": 0.82, "nDCG@10": 0.72}, "tl": {"Recall@10": 0.68, "nDCG@10": 0.55}},
            },
        },
    ]

    selected = select_nonregressing_checkpoint(history)

    assert selected["epoch"] == 2


def test_quality_promotion_requires_every_slice_to_improve_and_close_teacher_gap():
    baseline = {
        "en": {"Recall@10": .82, "nDCG@10": .71},
        "tl": {"Recall@10": .70, "nDCG@10": .57},
        "combined": {"Recall@10": .76, "nDCG@10": .64},
    }
    teacher = {
        "en": {"Recall@10": .98, "nDCG@10": .96},
        "tl": {"Recall@10": .97, "nDCG@10": .93},
        "combined": {"Recall@10": .976, "nDCG@10": .948},
    }
    improved = {
        "en": {"Recall@10": .84, "nDCG@10": .74},
        "tl": {"Recall@10": .73, "nDCG@10": .61},
        "combined": {"Recall@10": .785, "nDCG@10": .675},
    }
    validate_quality_promotion(improved, baseline, teacher)

    tagalog_regression = {**improved, "tl": {"Recall@10": .69, "nDCG@10": .61}}
    with pytest.raises(ValueError, match="does not improve tl Recall@10"):
        validate_quality_promotion(tagalog_regression, baseline, teacher)


def test_int8_budget_rejects_false_size_claims_from_weight_file_only():
    budget = artifact_budget(250_000, 480, "int8", 9_090_501, 4_245, 131_200_180)
    assert budget.matrix_bytes < budget.reference_artifact_bytes
    assert budget.complete_artifact_bytes < budget.reference_artifact_bytes

    oversized = artifact_budget(250_000, 488, "int8", 9_090_501, 4_245, 131_200_180)
    assert oversized.matrix_bytes < oversized.reference_artifact_bytes
    assert oversized.complete_artifact_bytes < oversized.reference_artifact_bytes
    with pytest.raises(ValueError, match="hard size gate"):
        validate_compact_recipe(
            {
                "full_dimension": 1024,
                "matryoshka_dimensions": [1024, 488],
                "export_dimension": 488,
                "export_dtype": "int8",
                "query_prefix": "[QRY] ",
                "document_prefix": "",
                "minimum_size_headroom_bytes": 2_000_000,
            },
            oversized,
        )
