"""Protocol guards for direct parent-to-Model2Vec knowledge distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_direct_parent_recipe(recipe: Mapping[str, Any]) -> None:
    """Reject any student initialization that does not start from the parent.

    The experiment requires Model2Vec and TokenLearn to use the exact same
    pinned SEA-LION parent. TokenLearn starts from the direct Model2Vec child
    produced inside this workflow; unrelated static children remain forbidden.
    """

    teacher = recipe["teacher"]
    model2vec = recipe["model2vec"]
    tokenlearn = recipe["tokenlearn"]

    parent = str(teacher["model"])
    if not str(teacher.get("revision", "")):
        raise ValueError("teacher revision must be pinned")
    if model2vec.get("entrypoint") != "model2vec.distill.distill":
        raise ValueError("Model2Vec must use model2vec.distill.distill")
    if int(model2vec.get("vocabulary_size", 0)) != 80_000:
        raise ValueError("direct child vocabulary must contain 80000 selected tokens")
    if model2vec.get("full_width_pca_dims") != "auto":
        raise ValueError("direct child must use full-rank PCA without dimensional reduction")
    if tokenlearn.get("initialization_mode") != "same-parent-direct-child":
        raise ValueError("TokenLearn must start from the same-parent direct child")
    if str(tokenlearn.get("initialization_parent_model")) != parent:
        raise ValueError("TokenLearn initialization must come from the same pinned parent")

    identities = {
        parent,
        str(model2vec.get("model_name")),
        str(tokenlearn.get("featurization_model")),
        str(tokenlearn.get("model_name")),
    }
    if identities != {parent}:
        raise ValueError("Model2Vec and TokenLearn must use the same parent")

    pruning = model2vec.get("tokenizer_pruning", {})
    if pruning.get("method") != "official model2vec token_remove_pattern":
        raise ValueError("compact direct child must use Model2Vec's supported tokenizer pruning hook")
    if int(pruning.get("expected_export_vocabulary_rows_before_query_token", 0)) != 79_985:
        raise ValueError("preflighted SEA-LION vocabulary row count changed")
    if pruning.get("manual_tokenizer_replacement") is not False:
        raise ValueError("manual tokenizer replacement is forbidden")

    export = recipe["export"]
    if int(export.get("dimension", 0)) != 1024 or export.get("dtype") != "int8":
        raise ValueError("final export must retain 1024 dimensions in int8")
    if int(export.get("hard_maximum_complete_artifact_bytes", 0)) != 100_000_000:
        raise ValueError("complete model package must use the strict 100 MB gate")


def validate_tagalog_direct_parent_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate the SEA-LION-only, Tagalog-only direct-child contract."""

    if recipe.get("protocol") != "multiwikiqa-direct-sealion-tagalog-v1":
        raise ValueError("wrong Tagalog-only protocol")
    if recipe.get("language") != "tl" or recipe.get("allowed_languages") != ["tl"]:
        raise ValueError("the direct SEA-LION workflow must be Tagalog-only")

    teacher = recipe["teacher"]
    model2vec = recipe["model2vec"]
    tokenlearn = recipe["tokenlearn"]
    parent = str(teacher["model"])
    if not str(teacher.get("revision", "")):
        raise ValueError("teacher revision must be pinned")
    if model2vec.get("entrypoint") != "model2vec.distill.distill":
        raise ValueError("Model2Vec must use model2vec.distill.distill")
    if int(model2vec.get("vocabulary_size", 0)) != 80_000:
        raise ValueError("Tagalog custom vocabulary must contain 80000 tokens")
    if model2vec.get("full_width_pca_dims") != "auto":
        raise ValueError("direct child must retain the full 1024 dimensions")
    if model2vec.get("tokenizer_pruning", {}).get("method") != "official model2vec token_remove_pattern":
        raise ValueError("Tagalog custom vocabulary must use the official pruning hook")
    if model2vec.get("tokenizer_pruning", {}).get("manual_tokenizer_replacement") is not False:
        raise ValueError("manual tokenizer replacement is forbidden")
    identities = {
        parent,
        str(model2vec.get("model_name")),
        str(tokenlearn.get("featurization_model")),
        str(tokenlearn.get("model_name")),
        str(tokenlearn.get("initialization_parent_model")),
    }
    if identities != {parent} or tokenlearn.get("initialization_mode") != "same-parent-direct-child":
        raise ValueError("Model2Vec and TokenLearn must use the same pinned SEA-LION parent")

    training = recipe["training"]
    evaluation = recipe["evaluation"]
    if training.get("tokenlearn_corpus_language") != "tl":
        raise ValueError("TokenLearn corpus must be Tagalog-only")
    if training.get("retrieval_query_language") != "tl":
        raise ValueError("retrieval training must be Tagalog-only")
    if float(training.get("retrieval_learning_rate", 0)) != 0.001:
        raise ValueError("Tagalog static retrieval training must use the tested 0.001 learning rate")
    if int(training.get("retrieval_epochs", 0)) != 8:
        raise ValueError("Tagalog static retrieval training must run eight gated epochs")
    if evaluation.get("development_language") != "tl" or evaluation.get("heldout_language") != "tl":
        raise ValueError("development and heldout evaluation must be Tagalog-only")
    if int(evaluation.get("development_questions", 0)) != 197:
        raise ValueError("Tagalog development cohort changed")
    if int(evaluation.get("heldout_questions", 0)) != 3984:
        raise ValueError("Tagalog heldout cohort changed")

    export = recipe["export"]
    if int(export.get("dimension", 0)) != 1024 or export.get("dtype") != "int8":
        raise ValueError("final Tagalog export must retain 1024 int8 dimensions")
    if int(export.get("hard_maximum_complete_artifact_bytes", 0)) != 100_000_000:
        raise ValueError("complete model package must stay strictly below 100 MB")
    quality = recipe["quality"]
    if float(quality.get("tagalog_maximum_teacher_recall_at_10_gap", -1)) != 0.10:
        raise ValueError("Tagalog Recall@10 must remain within ten points of SEA-LION")
    if "english_recall_at_10_floor" in quality:
        raise ValueError("English quality gates are forbidden in the Tagalog-only workflow")
