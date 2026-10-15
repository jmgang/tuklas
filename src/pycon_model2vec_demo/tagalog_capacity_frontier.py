"""Hard gates and diagnostics for the Tagalog SEA-LION capacity frontier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PARENT = "aisingapore/SEA-LION-E5-Embedding-600M"
REVISION = "6a97cedd22714f3d4eb7c6fd1a452bf310899683"


def validate_candidate_recipe(recipe: Mapping[str, Any]) -> None:
    protocol = recipe.get("protocol")
    if protocol not in {
        "multiwikiqa-sealion-tagalog-capacity-frontier-v1",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v2",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v3",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v4",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v5-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v6-control",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v7-512-control",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-straight-432",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-straight-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-negatives-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v9-stage1-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v10-queryhead-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v11-restore-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v12-width-544",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v12-width-640",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v13-stage1-544",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v14-stage1-1024",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v14-stage1-pairs2m-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v14-stage1-pairs2m-1024",
        # v15 holds width at the 512D that step 1 proved best and varies only
        # stage-1 *composition*.  The v14 1M-vs-2M comparison was confounded:
        # the 1M build had no cross-passage or first-sentence pairs at all, so
        # its +0.83 held-out delta cannot be attributed to volume.  These three
        # bracket the independent-crop share at 71.5%, 41.8% and 0% while
        # raising cross-passage -- the only construction whose supervision is
        # article-level, like the metric -- from 1.8% to 9.2%, 18.8% and 32.4%.
        "multiwikiqa-sealion-tagalog-capacity-frontier-v15-stage1-compose-full-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v15-stage1-compose-balanced-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v15-stage1-compose-nocrop-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v15-stage1-compose-cropmatched-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v13-stage1-constructions-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-contrastive-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-highlr-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-reference-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-greedy-432",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-greedy-512",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-sif-432",
        "multiwikiqa-sealion-tagalog-capacity-frontier-v8-sif-512",
    }:
        raise ValueError("wrong capacity-frontier protocol")
    if recipe.get("language") != "tl" or recipe.get("allowed_languages") != ["tl"]:
        raise ValueError("capacity frontier must be Tagalog-only")
    teacher = recipe["teacher"]
    if teacher.get("model") != PARENT or teacher.get("revision") != REVISION:
        raise ValueError("the pinned SEA-LION parent changed")
    candidates = recipe.get("candidates", [])
    if protocol.endswith("-v1"):
        expected = {
            "full_parent_250k_256": ("full-parent", 256, None),
            "tagalog_160k_512": ("tagalog-custom", 512, 160_000),
        }
    elif protocol.endswith("-1024"):
        # The teacher is 1024-dimensional, so at this width the target PCA is an
        # identity transform: no teacher variance is discarded at all.  Every
        # narrower candidate truncates it (512d retains 97.0%).
        expected = {"full_parent_250k_1024": ("full-parent", 1024, None)}
    elif protocol.endswith("-544"):
        expected = {"full_parent_250k_544": ("full-parent", 544, None)}
    elif protocol.endswith("-640"):
        expected = {"full_parent_250k_640": ("full-parent", 640, None)}
    elif protocol.endswith(("-v5-512", "-v7-512-control", "-512")):
        expected = {"full_parent_250k_512": ("full-parent", 512, None)}
    else:
        expected = {"full_parent_250k_432": ("full-parent", 432, None)}
    if {row.get("id") for row in candidates} != set(expected):
        raise ValueError("the fixed-byte candidate set changed")
    for candidate in candidates:
        mode, dimension, vocabulary_size = expected[candidate["id"]]
        if candidate.get("vocabulary_mode") != mode or int(candidate.get("dimension", 0)) != dimension:
            raise ValueError(f"capacity changed for {candidate['id']}")
        if vocabulary_size is not None and int(candidate.get("vocabulary_size", vocabulary_size)) != vocabulary_size:
            raise ValueError(f"vocabulary size changed for {candidate['id']}")
        if candidate.get("dtype") != "int8":
            raise ValueError("candidate export must use int8")
        required_bytes = (
            100_000_000 if protocol.endswith("-v1")
            # The width sweep is explicitly not size-constrained: the artifact
            # receipt still records measured bytes, but size must not reject a
            # candidate while the question is how far width carries quality.
            else 300_000_000 if protocol.endswith(("-544", "-640", "-1024"))
            else 140_000_000 if protocol.endswith(("-v5-512", "-v7-512-control", "-512"))
            else 120_000_000
        )
        if int(candidate.get("hard_maximum_complete_artifact_bytes", 0)) != required_bytes:
            raise ValueError(f"candidate must use the strict {required_bytes}-byte gate")
    if recipe["evaluation"].get("development_questions") != 197:
        raise ValueError("development cohort changed")
    if recipe["evaluation"].get("heldout_questions") != 3984:
        raise ValueError("heldout cohort changed")
    if recipe["training"].get("heldout_questions_forbidden") is not True:
        raise ValueError("heldout leakage guard was removed")
    v8_protocol = any(
        tag in protocol
        for tag in ("-v8", "-v9", "-v10", "-v11", "-v12", "-v13", "-v14", "-v15")
    )
    learning_rate = float(recipe["training"].get("retrieval_learning_rate", 0))
    # The v8 arms may raise the learning rate because they also raise the batch
    # size; every earlier protocol stays pinned to the frozen 0.001.
    if v8_protocol:
        # sentence-transformers/static-retrieval-mrl-en-v1 trains a static token
        # table at lr 0.2; the greedy search halves the rate on every rejected
        # epoch, so a high start self-anneals instead of diverging.
        if not 0.0 < learning_rate <= 0.2:
            raise ValueError("v8 retrieval learning rate out of range")
    elif learning_rate != 0.001:
        raise ValueError("retrieval learning rate changed")
    advanced_protocol = protocol.endswith("-v4") or protocol.endswith("-v5-512")
    control_protocol = protocol.endswith("-v6-control")
    control_512_protocol = protocol.endswith("-v7-512-control")
    if v8_protocol:
        # The v8 arms own their duration: the 512D control was still gaining a
        # question per epoch when it hit the frozen 16-epoch limit.
        if int(recipe["training"].get("retrieval_epochs", 0)) not in {16, 32, 64}:
            raise ValueError("v8 retrieval epoch count must be 16, 32 or 64")
    else:
        required_epochs = (
            32 if advanced_protocol
            else 16 if (
                protocol.endswith("-v3") or control_protocol or control_512_protocol
            )
            else 8
        )
        if int(recipe["training"].get("retrieval_epochs", 0)) != required_epochs:
            raise ValueError("retrieval epoch count changed")
    if advanced_protocol:
        training = recipe["training"]
        if training.get("tokenlearn_checkpoint_selection") != "development-retrieval":
            raise ValueError("v4 must select TokenLearn by development retrieval")
        if float(training.get("query_vector_distillation_weight", -1)) != 1.0:
            raise ValueError("v4 query-vector distillation weight changed")
        if training.get("dynamic_remine_after_epochs") != [8, 16, 24]:
            raise ValueError("v4 dynamic re-mining schedule changed")
        if training.get("retrieval_lr_schedule") != [
            {"start_epoch": 1, "learning_rate": 0.001},
            {"start_epoch": 9, "learning_rate": 0.0005},
            {"start_epoch": 17, "learning_rate": 0.00025},
        ]:
            raise ValueError("v4 learning-rate schedule changed")
        if int(training.get("retrieval_early_stopping_patience", 0)) != 4:
            raise ValueError("v4 early-stopping patience changed")
    if control_protocol:
        training = recipe["training"]
        if training.get("tokenlearn_checkpoint_selection") != "validation-loss":
            raise ValueError("controlled repair must reproduce v3 TokenLearn selection")
        if float(training.get("query_vector_distillation_weight", -1)) != 0.0:
            raise ValueError("controlled repair control cannot use query alignment")
        if training.get("dynamic_remine_after_epochs") != []:
            raise ValueError("controlled repair control cannot dynamically re-mine")
        if training.get("retrieval_lr_schedule") != []:
            raise ValueError("controlled repair control must use v3 adaptive LR")
        if int(training.get("retrieval_early_stopping_patience", -1)) != 0:
            raise ValueError("controlled repair control cannot early-stop")
        repair = training.get("controlled_repair", {})
        if repair.get("phase") != "v3-control" or repair.get("persist_ranking_ready_checkpoint") is not True:
            raise ValueError("controlled repair checkpoint contract changed")
        if float(repair.get("previous_best_recall_at_10", -1)) != 0.8324873096446701:
            raise ValueError("controlled repair baseline changed")
        if float(repair.get("control_reproduction_tolerance", -1)) != 0.01:
            raise ValueError("controlled repair tolerance changed")
        if repair.get("future_query_alignment_weights") != [0.05, 0.1, 0.25]:
            raise ValueError("controlled repair future ablations changed")
    if control_512_protocol:
        training = recipe["training"]
        if training.get("tokenlearn_checkpoint_selection") != "validation-loss":
            raise ValueError("512 control must use v3 TokenLearn selection")
        if float(training.get("query_vector_distillation_weight", -1)) != 0.0:
            raise ValueError("512 control cannot use query alignment")
        if training.get("dynamic_remine_after_epochs") != []:
            raise ValueError("512 control cannot dynamically re-mine")
        if training.get("retrieval_lr_schedule") != []:
            raise ValueError("512 control must use v3 adaptive LR")
        if int(training.get("retrieval_early_stopping_patience", -1)) != 0:
            raise ValueError("512 control cannot early-stop")
        repair = training.get("controlled_repair", {})
        if repair.get("phase") != "v3-512-control":
            raise ValueError("512 control phase changed")
        if repair.get("persist_ranking_ready_checkpoint") is not True:
            raise ValueError("512 control must preserve its ranking-ready checkpoint")
        if float(repair.get("comparison_432_recall_at_10", -1)) != 0.8324873096446701:
            raise ValueError("512 comparison baseline changed")
    if v8_protocol:
        training = recipe["training"]
        if training.get("partition") not in (None, "three-way-v8"):
            raise ValueError("v8 partition must be frozen-v5 or three-way-v8")
        if training.get("tokenlearn_checkpoint_selection") != "validation-loss":
            raise ValueError("v8 arms keep v3 TokenLearn selection")
        query_projection = bool(training.get("query_projection", False))
        alignment_weight = float(training.get("query_vector_distillation_weight", -1))
        if query_projection:
            # v4 destroyed the shared table by forcing one set of rows to satisfy
            # both document geometry and the teacher's query geometry at weight
            # 1.0.  With a separate query map the conflict disappears, so a low
            # weight is permitted here and nowhere else.
            if not 0.0 <= alignment_weight <= 0.25:
                raise ValueError("query-head alignment weight must be within [0, 0.25]")
        elif alignment_weight != 0.0:
            raise ValueError("query alignment requires a query projection")
        if training.get("dynamic_remine_after_epochs") != []:
            raise ValueError("v8 arms cannot dynamically re-mine")
        if training.get("retrieval_lr_schedule") != []:
            raise ValueError("v8 arms cannot use a phase learning-rate schedule")
        if int(training.get("retrieval_early_stopping_patience", -1)) != 0:
            raise ValueError("v8 arms cannot early-stop")
        comparison = training.get("v8_comparison", {})
        if float(comparison.get("greedy_control_recall_at_10", -1)) != 0.8324873096446701:
            raise ValueError("v8 comparison baseline changed")
        if comparison.get("report_cohort_questions") != 197:
            raise ValueError("v8 report cohort must stay the comparable 197 questions")
        if comparison.get("selection_cohort_questions") != 129:
            raise ValueError("v8 selection cohort changed")
        if training.get("ranking_search") not in ("straight", "greedy-rollback"):
            raise ValueError("v8 ranking search must be straight or greedy-rollback")
        if not isinstance(training.get("retrieval_batch_size"), int):
            raise ValueError("v8 arms must declare their batch size")
        weights = training.get("objective_weights", {})
        if weights:
            for key, value in weights.items():
                if key not in {"listwise", "positive_cross_entropy",
                               "in_batch_cross_entropy", "teacher_score_mse"}:
                    raise ValueError(f"unknown objective weight {key}")
                if not 0.0 <= float(value) <= 2.0:
                    raise ValueError(f"objective weight {key} out of range")
            ranking_mass = sum(float(weights.get(k, d)) for k, d in (
                ("listwise", 1.0), ("positive_cross_entropy", 0.5),
                ("in_batch_cross_entropy", 0.5)))
            if ranking_mass <= 0.0:
                raise ValueError("v8 objective must keep a ranking term")
        restore = training.get("ranking_restore")
        if ("-restore-" in protocol) != bool(restore):
            raise ValueError("only a restore protocol may continue from a checkpoint")
        if restore and not str(restore.get("s3_prefix", "")).endswith("/checkpoints/"):
            raise ValueError("ranking_restore must name a preserved checkpoints prefix")
        stage1 = training.get("unsupervised_pretraining")
        if ("-stage1-" in protocol) != bool(stage1):
            raise ValueError("only v9 may pretrain on unsupervised pairs")
        if stage1:
            if int(stage1.get("pairs", 0)) < 500_000:
                raise ValueError("v9 stage-1 needs at least half a million pairs")
            if int(stage1.get("batch_size", 0)) < 512:
                raise ValueError("v9 stage-1 exists to make large batches viable")
            if stage1.get("llm_generated") is not False:
                raise ValueError("v9 stage-1 pairs must not be model-generated")
            if stage1.get("heldout_questions_read") is not False:
                raise ValueError("v9 stage-1 must not read held-out questions")
        reregularized = "-sif-" in protocol
        if reregularized:
            reregularization = training.get("post_tokenlearn_sif_reweighting", {})
            sif = float(reregularization.get("sif_coefficient", -1))
            if not 0.0 < sif < 1.0:
                raise ValueError("v8c SIF coefficient out of range")
            if reregularization.get("probability_source") != "measured-corpus-counts":
                raise ValueError("v8c must measure corpus probabilities")
        elif training.get("post_tokenlearn_sif_reweighting"):
            raise ValueError("only v8c may re-regularize after TokenLearn")
    required_mined = 12_000 if protocol.endswith("-v1") else 15_000
    if int(recipe["training"]["ranking_data_expansion"].get("minimum_safely_mined_pairs", 0)) != required_mined:
        raise ValueError("expanded safely-mined supervision floor changed")
    if float(recipe["quality"].get("maximum_teacher_recall_at_10_gap", -1)) != 0.10:
        raise ValueError("teacher-distance gate changed")


def candidate_gate(report: Mapping[str, Any]) -> dict[str, float | int | bool]:
    """Reject an oversized, leaky, non-Tagalog or teacher-distant candidate."""

    if report.get("language") != "tl" or report.get("english_records_used") is not False:
        raise ValueError("candidate is not strictly Tagalog-only")
    if report.get("heldout_used_for_training_or_selection") is not False:
        raise ValueError("heldout questions entered training or selection")
    artifact_bytes = int(report["artifact"]["complete_bytes"])
    hard_maximum_bytes = int(report["artifact"].get("hard_maximum_bytes", 100_000_000))
    if artifact_bytes >= hard_maximum_bytes:
        raise ValueError("complete candidate artifact is not below its hard byte limit")
    student = float(report["development"]["student"]["tl"]["Recall@10"])
    teacher = float(report["development"]["teacher"]["tl"]["Recall@10"])
    if not 0.0 <= student <= 1.0 or not 0.0 <= teacher <= 1.0:
        raise ValueError("Recall@10 must be a fraction")
    gap = teacher - student
    if gap > 0.10:
        raise ValueError(f"Tagalog teacher gap {gap:.6f} exceeds 0.100000")
    return {
        "passed": True,
        "complete_bytes": artifact_bytes,
        "student_recall_at_10": student,
        "teacher_recall_at_10": teacher,
        "teacher_gap": gap,
    }
