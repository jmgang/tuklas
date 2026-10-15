"""The v8 three-way partition must preserve the v5 reported cohort exactly."""

import json
from pathlib import Path

import pytest

from pycon_model2vec_demo.multiwikiqa_ranking_v5 import prepare as prepare_v5
from pycon_model2vec_demo.tagalog_ranking_v8 import (
    EXPECTED_TAGALOG,
    article_residue,
    prepare_three_way,
)

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "runs" / "multiwikiqa-preparation-20261004-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture(scope="module")
def cohort_inputs() -> tuple[list[dict], list[dict]]:
    queries = read_jsonl(PREPARATION / "annotations.all.jsonl")
    chunks = read_jsonl(PREPARATION / "chunks.candidate.jsonl")
    return queries, chunks


@pytest.fixture(scope="module")
def split(cohort_inputs) -> dict:
    queries, chunks = cohort_inputs
    tagalog = [row for row in queries if row["language"] == "tl"]
    return prepare_three_way(tagalog, chunks)


def test_cohort_sizes_are_frozen(split):
    assert len(split["report"]) == EXPECTED_TAGALOG["report"] == 197
    assert len(split["selection"]) == EXPECTED_TAGALOG["selection"] == 129
    assert len(split["train"]) == EXPECTED_TAGALOG["train"] == 689


def test_report_cohort_is_exactly_the_v5_validation_cohort(cohort_inputs, split):
    """The headline number must stay comparable with the preserved 83.25% run."""

    queries, chunks = cohort_inputs
    tagalog = [row for row in queries if row["language"] == "tl"]
    _, v5_validation, _ = prepare_v5(tagalog, chunks)
    assert {row["query_id"] for row in split["report"]} == {
        row["query_id"] for row in v5_validation
    }


def test_v5_training_cohort_shrinks_only_by_the_selection_cohort(cohort_inputs, split):
    queries, chunks = cohort_inputs
    tagalog = [row for row in queries if row["language"] == "tl"]
    v5_train, _, _ = prepare_v5(tagalog, chunks)
    assert len(v5_train) == 818
    moved = {row["query_id"] for row in v5_train} - {
        row["query_id"] for row in split["train"]
    }
    assert moved == {row["query_id"] for row in split["selection"]}
    assert len(moved) == 129


def test_cohorts_are_disjoint_by_question_and_article(split):
    names = ("report", "selection", "train")
    for left in names:
        for right in names:
            if left >= right:
                continue
            assert not (
                {row["query_id"] for row in split[left]}
                & {row["query_id"] for row in split[right]}
            )
            assert not (
                {row["component_id"] for row in split[left]}
                & {row["component_id"] for row in split[right]}
            )


def test_no_heldout_question_enters_any_cohort(cohort_inputs, split):
    queries, _ = cohort_inputs
    heldout = {row["query_id"] for row in queries if row["partition"] == "evaluation"}
    assert len(heldout) == 7922
    for name in ("report", "selection", "train"):
        assert not {row["query_id"] for row in split[name]} & heldout


def test_selection_cohort_is_hygiene_clean(split):
    assert split["audit"]["selection_cohort_hygiene_clean"] is True
    assert split["audit"]["report_cohort_used_for_selection"] is False


def test_known_report_cohort_heldout_text_duplicate_is_recorded(split):
    """v5 never hygiene-checked the validation cohort; v8 must surface it."""

    exceptions = split["audit"]["report_hygiene_exceptions"]
    assert len(exceptions) == 1
    assert exceptions[0]["reason"] == "evaluation question overlap"


def test_residue_predicate_matches_the_frozen_v5_hash(cohort_inputs):
    queries, chunks = cohort_inputs
    tagalog = [row for row in queries if row["language"] == "tl"]
    _, v5_validation, _ = prepare_v5(tagalog, chunks)
    assert all(article_residue(row["component_id"]) == 0 for row in v5_validation)


def test_unclean_selection_cohort_is_rejected(cohort_inputs):
    queries, chunks = cohort_inputs
    tagalog = [row for row in queries if row["language"] == "tl"]
    mutated = []
    marked = False
    for row in tagalog:
        if (
            not marked
            and row["partition"] == "development"
            and article_residue(row["component_id"]) == 1
        ):
            mutated.append(dict(row) | {"answer_spans": [{"start": 10**9, "end": 10**9 + 1}]})
            marked = True
        else:
            mutated.append(row)
    assert marked
    with pytest.raises(ValueError, match="selection cohort must be hygiene-clean"):
        prepare_three_way(mutated, chunks)
