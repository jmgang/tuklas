import json
from pathlib import Path

import pytest

from pycon_model2vec_demo.direct_parent_protocol import validate_direct_parent_recipe


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_recipe_distills_directly_from_the_same_parent():
    recipe = json.loads(
        (ROOT / "configs/multiwikiqa-direct-parent-v1.json").read_text()
    )

    validate_direct_parent_recipe(recipe)

    assert recipe["model2vec"]["vocabulary_size"] == 80000
    assert recipe["model2vec"]["full_width_pca_dims"] == "auto"
    assert recipe["tokenlearn"]["initialization_mode"] == "same-parent-direct-child"
    assert recipe["export"] == {
        "dimension": 1024,
        "dtype": "int8",
        "hard_maximum_complete_artifact_bytes": 100000000,
    }


def test_unrelated_pretrained_static_child_initialization_is_forbidden():
    recipe = {
        "teacher": {
            "model": "aisingapore/SEA-LION-E5-Embedding-600M",
            "revision": "6a97cedd22714f3d4eb7c6fd1a452bf310899683",
        },
        "model2vec": {
            "entrypoint": "model2vec.distill.distill",
            "model_name": "aisingapore/SEA-LION-E5-Embedding-600M",
            "vocabulary_size": 80000,
            "full_width_pca_dims": "auto",
        },
        "tokenlearn": {
            "featurization_model": "aisingapore/SEA-LION-E5-Embedding-600M",
            "initialization_mode": "same-parent-direct-child",
            "model_name": "aisingapore/SEA-LION-E5-Embedding-600M",
            "initialization_parent_model": "minishlab/potion-retrieval-32M",
        },
        "export": {"dimension": 1024, "dtype": "int8", "hard_maximum_complete_artifact_bytes": 100000000},
    }

    with pytest.raises(ValueError, match="same pinned parent"):
        validate_direct_parent_recipe(recipe)


def test_teacher_identity_must_match_both_model2vec_and_tokenlearn():
    recipe = {
        "teacher": {"model": "teacher/a", "revision": "pinned"},
        "model2vec": {
            "entrypoint": "model2vec.distill.distill",
            "model_name": "teacher/a",
            "vocabulary_size": 80000,
            "full_width_pca_dims": "auto",
        },
        "tokenlearn": {
            "featurization_model": "teacher/b",
            "initialization_mode": "same-parent-direct-child",
            "model_name": "teacher/a",
            "initialization_parent_model": "teacher/a",
        },
        "export": {"dimension": 1024, "dtype": "int8", "hard_maximum_complete_artifact_bytes": 100000000},
    }

    with pytest.raises(ValueError, match="same parent"):
        validate_direct_parent_recipe(recipe)


def test_tokenlearn_same_parent_direct_child_is_required():
    recipe = {
        "teacher": {"model": "teacher/a", "revision": "pinned"},
        "model2vec": {
            "entrypoint": "model2vec.distill.distill",
            "model_name": "teacher/a",
            "vocabulary_size": 80000,
            "full_width_pca_dims": "auto",
        },
        "tokenlearn": {
            "featurization_model": "teacher/a",
            "initialization_mode": "unrelated-static-child",
            "model_name": "teacher/a",
            "initialization_parent_model": "teacher/a",
        },
        "export": {"dimension": 1024, "dtype": "int8", "hard_maximum_complete_artifact_bytes": 100000000},
    }

    with pytest.raises(ValueError, match="same-parent direct child"):
        validate_direct_parent_recipe(recipe)
