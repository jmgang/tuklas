"""Guards and loss functions for the compact bilingual retrieval student.

This module is intentionally independent of AWS orchestration.  It captures the
scientific invariants that the next training job must satisfy before any paid
compute is launched:

* query and document roles are explicit;
* both teacher-margin distillation and in-batch retrieval ranking are trained at
  every exportable Matryoshka prefix;
* checkpoint selection cannot hide a regression in either language or in the
  compact export; and
* the complete saved artifact, not only ``model.safetensors``, must be smaller
  than the complete Potion reference artifact and strictly below 100 MB.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence
from collections import Counter
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


_DTYPE_BYTES = {"int8": 1, "float16": 2, "float32": 4, "float64": 8}


def validate_mining_yield(
    *,
    requested: Mapping[str, int],
    mined: Mapping[str, int],
    minimum_overall_fraction: float,
    minimum_slice_fraction: float,
) -> dict[str, object]:
    """Validate mined examples relative to the frozen request curriculum.

    The old absolute 8,000-row threshold was impossible after the curriculum
    was intentionally balanced to 3,468 requests.  Relative gates retain a
    scientifically meaningful safety check and prevent one language/source
    slice from disappearing behind an acceptable aggregate.
    """

    if set(requested) != set(mined) or not requested:
        raise ValueError("requested and mined slices must be identical and non-empty")
    if not 0.0 < minimum_overall_fraction <= 1.0 or not 0.0 < minimum_slice_fraction <= 1.0:
        raise ValueError("mining fractions must be within (0, 1]")
    requested_counts = {str(key): int(value) for key, value in requested.items()}
    mined_counts = {str(key): int(value) for key, value in mined.items()}
    for key, count in requested_counts.items():
        current = mined_counts[key]
        if count <= 0 or current < 0 or current > count:
            raise ValueError(f"invalid mining counts for {key}")
    requested_total = sum(requested_counts.values())
    mined_total = sum(mined_counts.values())
    overall_fraction = mined_total / requested_total
    if overall_fraction + 1e-12 < minimum_overall_fraction:
        raise ValueError("overall hard-negative yield is below the frozen minimum")
    slice_fractions = {
        key: mined_counts[key] / requested_counts[key]
        for key in sorted(requested_counts)
    }
    failed = [key for key, value in slice_fractions.items() if value + 1e-12 < minimum_slice_fraction]
    if failed:
        raise ValueError("hard-negative yield is below the frozen slice minimum: " + ", ".join(failed))
    return {
        "requested": requested_total,
        "mined": mined_total,
        "overall_fraction": overall_fraction,
        "slice_fractions": slice_fractions,
    }


@dataclass(frozen=True)
class ArtifactBudget:
    matrix_bytes: int
    tokenizer_bytes: int
    metadata_bytes: int
    complete_artifact_bytes: int
    reference_artifact_bytes: int

    @property
    def fits_reference(self) -> bool:
        return self.complete_artifact_bytes < self.reference_artifact_bytes


def validate_measured_artifact(
    file_sizes: Mapping[str, int],
    *,
    hard_maximum_bytes: int = 100_000_000,
    reference_artifact_bytes: int = 131_200_180,
) -> int:
    """Validate the *actual* complete model package after export.

    The caller must provide every regular file that will ship with the model,
    including the vector matrix, tokenizer, configuration and metadata.  This
    intentionally accepts sizes instead of filesystem paths so the same guard
    can validate a local directory or an immutable S3 manifest without
    modifying either one.

    The requirement is strict: an artifact of exactly 100,000,000 bytes is a
    failure, not a pass.
    """

    if not file_sizes:
        raise ValueError("complete artifact manifest is empty")
    if hard_maximum_bytes <= 0 or reference_artifact_bytes <= 0:
        raise ValueError("artifact limits must be positive")
    normalized: dict[str, int] = {}
    for raw_name, raw_size in file_sizes.items():
        name = str(raw_name).strip()
        size = int(raw_size)
        if not name:
            raise ValueError("artifact filenames must be non-empty")
        if name in normalized:
            raise ValueError(f"duplicate artifact filename: {name}")
        if size < 0:
            raise ValueError(f"negative artifact size: {name}")
        normalized[name] = size
    total = sum(normalized.values())
    if total >= hard_maximum_bytes:
        raise ValueError("measured complete artifact violates the hard size gate")
    if total >= reference_artifact_bytes:
        raise ValueError("measured complete artifact is not smaller than the Potion reference")
    return total


def validate_teacher_distance_gate(
    candidate: Mapping[str, Mapping[str, float]],
    teacher: Mapping[str, Mapping[str, float]],
    *,
    english_floor: float = 0.90,
    maximum_language_gap: float = 0.10,
) -> dict[str, float]:
    """Reject a student that is not close enough to the frozen teacher.

    This is a promotion/retention gate, not an informational comparison.  A
    rejected student must not be packaged as a deliverable model.  English has
    both an absolute floor and the same maximum teacher gap used for Tagalog.
    """

    if not 0.0 < english_floor <= 1.0 or not 0.0 <= maximum_language_gap < 1.0:
        raise ValueError("quality thresholds must be fractions in range")
    values: dict[str, tuple[float, float]] = {}
    for language in ("en", "tl"):
        try:
            student_value = float(candidate[language]["Recall@10"])
            teacher_value = float(teacher[language]["Recall@10"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing {language} Recall@10") from exc
        if not (0.0 <= student_value <= 1.0 and 0.0 <= teacher_value <= 1.0):
            raise ValueError("Recall@10 values must be fractions in range")
        values[language] = (student_value, teacher_value)

    english, teacher_english = values["en"]
    tagalog, teacher_tagalog = values["tl"]
    english_gap = teacher_english - english
    tagalog_gap = teacher_tagalog - tagalog
    if english < english_floor:
        raise ValueError(
            f"English Recall@10 {english:.6f} is below hard floor {english_floor:.6f}"
        )
    if english_gap > maximum_language_gap:
        raise ValueError(
            f"English teacher gap {english_gap:.6f} exceeds {maximum_language_gap:.6f}"
        )
    if tagalog_gap > maximum_language_gap:
        raise ValueError(
            f"Tagalog teacher gap {tagalog_gap:.6f} exceeds {maximum_language_gap:.6f}"
        )
    return {"en_gap": english_gap, "tl_gap": tagalog_gap}


def validate_tagalog_teacher_distance_gate(
    candidate: Mapping[str, Mapping[str, float]],
    teacher: Mapping[str, Mapping[str, float]],
    *,
    maximum_gap: float = 0.10,
) -> dict[str, float]:
    """Apply the promotion gate to the Tagalog-only cohort."""

    if not 0.0 <= maximum_gap < 1.0:
        raise ValueError("maximum Tagalog gap must be a fraction in range")
    if set(candidate) != {"tl"} or set(teacher) != {"tl"}:
        raise ValueError("Tagalog-only quality reports must contain only tl")
    try:
        student_value = float(candidate["tl"]["Recall@10"])
        teacher_value = float(teacher["tl"]["Recall@10"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing Tagalog Recall@10") from exc
    if not (0.0 <= student_value <= 1.0 and 0.0 <= teacher_value <= 1.0):
        raise ValueError("Recall@10 values must be fractions in range")
    gap = teacher_value - student_value
    if gap > maximum_gap:
        raise ValueError(f"Tagalog teacher gap {gap:.6f} exceeds {maximum_gap:.6f}")
    return {"tl_gap": gap}


def artifact_budget(
    vocab_size: int,
    dimension: int,
    dtype: str,
    tokenizer_bytes: int,
    metadata_bytes: int,
    reference_artifact_bytes: int,
) -> ArtifactBudget:
    """Calculate a conservative complete-artifact size before training.

    ``metadata_bytes`` includes every non-matrix, non-tokenizer file that will
    be packaged.  The final build gate must still measure the actual directory;
    this function prevents an obviously oversized recipe from being launched.
    """

    values = (vocab_size, dimension, tokenizer_bytes, metadata_bytes, reference_artifact_bytes)
    if any(value < 0 for value in values) or vocab_size == 0 or dimension == 0:
        raise ValueError("artifact sizes and dimensions must be positive")
    try:
        bytes_per_value = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported export dtype: {dtype}") from exc
    matrix_bytes = vocab_size * dimension * bytes_per_value
    complete = matrix_bytes + tokenizer_bytes + metadata_bytes
    return ArtifactBudget(
        matrix_bytes=matrix_bytes,
        tokenizer_bytes=tokenizer_bytes,
        metadata_bytes=metadata_bytes,
        complete_artifact_bytes=complete,
        reference_artifact_bytes=reference_artifact_bytes,
    )


def validate_compact_recipe(recipe: Mapping[str, object], budget: ArtifactBudget) -> None:
    full_dimension = int(recipe["full_dimension"])
    export_dimension = int(recipe["export_dimension"])
    dimensions = [int(value) for value in recipe["matryoshka_dimensions"]]  # type: ignore[index]
    if full_dimension not in dimensions:
        raise ValueError("full dimension must be trained")
    if export_dimension not in dimensions:
        raise ValueError("export dimension must be trained")
    if len(set(dimensions)) != len(dimensions) or any(value <= 0 or value > full_dimension for value in dimensions):
        raise ValueError("invalid Matryoshka dimensions")
    if recipe["export_dtype"] != "int8":
        raise ValueError("compact export must use the frozen int8 protocol")
    if not str(recipe.get("query_prefix", "")).strip():
        raise ValueError("query role must be explicit")
    if str(recipe.get("document_prefix", "")):
        raise ValueError("document text must remain unprefixed")
    if not budget.fits_reference:
        raise ValueError("complete artifact is not smaller than the Potion reference")
    hard_maximum = int(recipe.get("hard_maximum_complete_artifact_bytes", 100_000_000))
    if hard_maximum <= 0 or budget.complete_artifact_bytes >= hard_maximum:
        raise ValueError("complete artifact violates the hard size gate")
    preferred_target = int(recipe.get("preferred_complete_artifact_bytes", hard_maximum))
    if preferred_target <= 0 or preferred_target > hard_maximum:
        raise ValueError("preferred size target must be positive and no larger than the hard gate")
    minimum_headroom = int(recipe.get("minimum_size_headroom_bytes", 0))
    if minimum_headroom < 0:
        raise ValueError("minimum size headroom must be non-negative")
    if budget.reference_artifact_bytes - budget.complete_artifact_bytes < minimum_headroom:
        raise ValueError("complete artifact does not preserve the required size headroom")


def prefix_role(text: str, *, role: str, query_prefix: str) -> str:
    if role == "query":
        if not query_prefix:
            raise ValueError("query prefix must be non-empty")
        return query_prefix + text
    if role == "document":
        return text
    raise ValueError(f"unknown retrieval role: {role}")


def external_article_id_from_passage_id(passage_id: str) -> str:
    """Recover the immutable external-corpus article id from a passage id.

    External Wikipedia passage ids use ``<article-id>:<start>:<end>``.  The
    feature layout also carries a URL-valued ``article_id`` field, whereas the
    synthetic-query receipts intentionally retain the corpus-native article
    id.  Deriving the key from ``source_id`` keeps those two namespaces from
    being confused while preserving every lossless window of the passage as a
    known positive.
    """

    parts = str(passage_id).rsplit(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError("external passage id must end in numeric start/end offsets")
    return parts[0]


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def select_external_title_pairs(
    rows: Sequence[Mapping[str, object]],
    *,
    per_language: int,
    seed: int,
) -> list[dict[str, object]]:
    """Select a deterministic, language-balanced title-to-passage probe set.

    This is deliberately a *probe-data* builder, not a claim that Wikipedia
    titles are sufficient retrieval supervision.  It lets us test whether
    broad bilingual query/document training helps before paying to generate
    higher-quality synthetic questions.  One earliest passage is retained per
    source article so a long article cannot dominate the probe.
    """

    if per_language <= 0:
        raise ValueError("per-language selection must be positive")
    by_article: dict[tuple[str, str], Mapping[str, object]] = {}
    titles: dict[tuple[str, str], str] = {}
    for row in rows:
        language = str(row["language"])
        if language not in {"en", "tl"}:
            raise ValueError(f"unsupported language: {language}")
        article_id = str(row["source_article_id"])
        title = str(row["source_title"]).strip()
        text = str(row["text"]).strip()
        if not article_id or not title or not text:
            raise ValueError("article id, title and passage text are required")
        key = (language, article_id)
        normalized_title = _normalized_text(title)
        if key in titles and titles[key] != normalized_title:
            raise ValueError("one article has conflicting titles")
        titles[key] = normalized_title
        current = by_article.get(key)
        ordering = (int(row["start"]), str(row["passage_id"]))
        if current is None or ordering < (int(current["start"]), str(current["passage_id"])):
            by_article[key] = row

    selected: list[dict[str, object]] = []
    for language in ("en", "tl"):
        candidates = [
            (sha256(f"{seed}:{language}:{article_id}".encode()).hexdigest(), article_id, row)
            for (row_language, article_id), row in by_article.items()
            if row_language == language
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) < per_language:
            raise ValueError(f"not enough {language} articles for requested selection")
        for stable_hash, article_id, row in candidates[:per_language]:
            selected.append(
                {
                    "language": language,
                    "article_id": article_id,
                    "query": str(row["source_title"]).strip(),
                    "positive": str(row["text"]).strip(),
                    "passage_id": str(row["passage_id"]),
                    "selection_hash": stable_hash,
                }
            )
    return selected


def select_balanced_vocabulary(
    english_counts: Mapping[str, int],
    tagalog_counts: Mapping[str, int],
    *,
    size: int,
    shared_fraction: float = 0.25,
) -> list[str]:
    """Build a deterministic bilingual vocabulary without English crowd-out.

    Shared tokens are ranked by their weaker-language frequency, then the
    remaining capacity is divided equally between language-specific lists.
    Any unfilled slots are taken from the global frequency ranking.  This is a
    vocabulary-selection policy only; the resulting tokenizer and actual model
    directory must still pass coverage and complete-byte gates.
    """

    if size <= 0 or not 0.0 <= shared_fraction <= 1.0:
        raise ValueError("invalid vocabulary size or shared fraction")
    for counts in (english_counts, tagalog_counts):
        if any(not token or int(value) <= 0 for token, value in counts.items()):
            raise ValueError("vocabulary counts require non-empty positive entries")

    english = Counter({token: int(value) for token, value in english_counts.items()})
    tagalog = Counter({token: int(value) for token, value in tagalog_counts.items()})
    shared_limit = round(size * shared_fraction)
    shared_candidates = sorted(
        english.keys() & tagalog.keys(),
        key=lambda token: (-min(english[token], tagalog[token]), -(english[token] + tagalog[token]), token),
    )
    selected = shared_candidates[:shared_limit]
    selected_set = set(selected)

    remaining = size - len(selected)
    english_limit = remaining // 2
    tagalog_limit = remaining - english_limit
    english_only = sorted(
        (token for token in english if token not in selected_set),
        key=lambda token: (-english[token], token),
    )
    tagalog_only = sorted(
        (token for token in tagalog if token not in selected_set),
        key=lambda token: (-tagalog[token], token),
    )
    for candidates, limit in ((english_only, english_limit), (tagalog_only, tagalog_limit)):
        added = 0
        for token in candidates:
            if token not in selected_set:
                selected.append(token)
                selected_set.add(token)
                added += 1
                if added == limit:
                    break

    if len(selected) < size:
        combined = english + tagalog
        for token in sorted(combined, key=lambda value: (-combined[value], value)):
            if token not in selected_set:
                selected.append(token)
                selected_set.add(token)
                if len(selected) == size:
                    break
    if len(selected) != min(size, len(english.keys() | tagalog.keys())):
        raise ValueError("balanced vocabulary selection produced an unexpected size")
    return selected


def add_neutral_query_role(
    embeddings: np.ndarray,
    tokenizer: Tokenizer,
    *,
    token: str = "[QRY]",
) -> tuple[np.ndarray, Tokenizer, int]:
    """Add one atomic query-role token whose epoch-zero effect is neutral.

    A zero row changes only the pre-normalization magnitude of mean pooling, not
    its direction.  The retrieval optimizer can then learn a role offset without
    changing document tokenization or silently degrading the starting model.
    """

    if embeddings.ndim != 2 or embeddings.shape[0] != tokenizer.get_vocab_size(with_added_tokens=True):
        raise ValueError("embedding/tokenizer vocabulary mismatch")
    if token in tokenizer.get_vocab(with_added_tokens=True):
        raise ValueError("query-role token already exists")
    updated = Tokenizer.from_str(tokenizer.to_str())
    if updated.add_special_tokens([token]) != 1:
        raise ValueError("query-role token was not added exactly once")
    token_id = updated.token_to_id(token)
    if token_id is None or updated.encode(token, add_special_tokens=False).ids != [token_id]:
        raise ValueError("query-role token is not atomic")
    zero = np.zeros((1, embeddings.shape[1]), dtype=embeddings.dtype)
    return np.concatenate([embeddings, zero], axis=0), updated, token_id


def strip_padding_and_unknown(
    token_ids: list[int],
    *,
    pad_id: int | None,
    unknown_id: int | None,
) -> list[int]:
    """Remove batch padding and unknown placeholders before mean pooling."""

    excluded = {value for value in (pad_id, unknown_id) if value is not None}
    return [token_id for token_id in token_ids if token_id not in excluded]


def _normalized_prefix(value: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension <= 0 or dimension > value.shape[-1]:
        raise ValueError("loss dimension outside embedding width")
    return F.normalize(value[..., :dimension], p=2, dim=-1)


def combined_matryoshka_retrieval_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor | None,
    teacher_margins: torch.Tensor | None,
    *,
    dimensions: Sequence[int],
    margin_weight: float,
    ranking_weight: float,
    scale: float,
) -> tuple[torch.Tensor, dict[int, dict[str, float]]]:
    """Margin distillation plus in-batch ranking at every export prefix.

    The same positive documents serve as in-batch negatives for other queries.
    When provided, explicit mined negatives are also included in the ranking
    denominator and contribute to MarginMSE.  Batch construction must separately
    guarantee that sibling or duplicate positives are not treated as negatives.
    """

    if queries.shape != positives.shape or (negatives is not None and queries.shape != negatives.shape):
        raise ValueError("query, positive and negative tensors must have identical shapes")
    if queries.ndim != 2 or (teacher_margins is not None and teacher_margins.shape != (queries.shape[0],)):
        raise ValueError("invalid retrieval batch shapes")
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("loss dimensions must be unique and non-empty")
    if margin_weight < 0 or ranking_weight < 0 or margin_weight + ranking_weight <= 0:
        raise ValueError("at least one positive objective weight is required")
    if margin_weight and (negatives is None or teacher_margins is None):
        raise ValueError("teacher-margin training requires explicit negatives and margins")
    if scale <= 0:
        raise ValueError("ranking scale must be positive")

    total = queries.new_zeros(())
    details: dict[int, dict[str, float]] = {}
    labels = torch.arange(queries.shape[0], device=queries.device)
    for dimension in dimensions:
        query = _normalized_prefix(queries, dimension)
        positive = _normalized_prefix(positives, dimension)
        if negatives is not None:
            negative = _normalized_prefix(negatives, dimension)
            ranking_documents = torch.cat([positive, negative], dim=0)
        else:
            negative = None
            ranking_documents = positive
        if margin_weight:
            predicted_margin = (query * positive).sum(dim=1) - (query * negative).sum(dim=1)  # type: ignore[operator]
            margin = F.mse_loss(predicted_margin, teacher_margins)
        else:
            margin = query.new_zeros(())
        logits = scale * (query @ ranking_documents.T)
        ranking = F.cross_entropy(logits, labels)
        combined = margin_weight * margin + ranking_weight * ranking
        total = total + combined
        details[int(dimension)] = {
            "margin_mse": float(margin.detach().cpu()),
            "mnrl": float(ranking.detach().cpu()),
            "combined": float(combined.detach().cpu()),
        }
    return total / len(dimensions), details


def mine_positive_aware_hard_negatives(
    candidate_ids: Sequence[str],
    teacher_scores: Sequence[float],
    *,
    positive_ids: Sequence[str],
    minimum_positive_score: float,
    maximum_positive_fraction: float = 0.95,
    count: int = 4,
) -> list[tuple[str, float]]:
    """Select difficult negatives without admitting near-positive candidates.

    This follows the transferable part of NVIDIA's retrieval recipe: known
    positives are masked, candidates above a fixed fraction of the least-similar
    positive are treated as possible false negatives, and the highest-scoring
    survivors are retained.  The caller must supply *all* known positive IDs;
    this function cannot infer unlabeled relevance.
    """

    if len(candidate_ids) != len(teacher_scores):
        raise ValueError("candidate IDs and scores must have identical lengths")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    if not positive_ids or count <= 0:
        raise ValueError("known positives and a positive negative count are required")
    if not 0.0 < maximum_positive_fraction < 1.0:
        raise ValueError("false-negative fraction must be between zero and one")
    if not np.isfinite(minimum_positive_score) or minimum_positive_score <= 0:
        raise ValueError("positive-aware percentage filtering requires a positive finite score")
    if not np.isfinite(np.asarray(teacher_scores, dtype=np.float64)).all():
        raise ValueError("teacher scores must be finite")

    positives = set(positive_ids)
    ceiling = minimum_positive_score * maximum_positive_fraction
    eligible = [
        (candidate_id, float(score))
        for candidate_id, score in zip(candidate_ids, teacher_scores, strict=True)
        if candidate_id not in positives and float(score) < ceiling
    ]
    eligible.sort(key=lambda row: (-row[1], row[0]))
    if len(eligible) < count:
        raise ValueError("not enough safe hard negatives after positive-aware filtering")
    return eligible[:count]


def listwise_teacher_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Match the teacher's full candidate ranking distribution.

    A single positive/negative margin discards how the teacher orders the other
    candidates.  KL distillation over the complete candidate list preserves that
    information.  Candidate construction and positive-aware filtering remain
    separate responsibilities so held-out labels cannot leak into this loss.
    """

    if student_scores.ndim != 2 or student_scores.shape != teacher_scores.shape:
        raise ValueError("student and teacher candidate-score matrices must match")
    if student_scores.shape[1] < 2:
        raise ValueError("listwise distillation requires at least two candidates")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(student_scores).all() or not torch.isfinite(teacher_scores).all():
        raise ValueError("candidate scores must be finite")

    teacher_distribution = F.softmax(teacher_scores / temperature, dim=1)
    student_log_distribution = F.log_softmax(student_scores / temperature, dim=1)
    return F.kl_div(
        student_log_distribution,
        teacher_distribution,
        reduction="batchmean",
    ) * (temperature * temperature)


def _nonregressing(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> bool:
    for representation in ("full", "compact"):
        for language in ("en", "tl"):
            for metric in ("Recall@10", "nDCG@10"):
                current = float(candidate["metrics"][representation][language][metric])  # type: ignore[index]
                original = float(baseline["metrics"][representation][language][metric])  # type: ignore[index]
                if current + 1e-12 < original:
                    return False
    return True


def _selection_score(row: Mapping[str, object]) -> float:
    compact = row["metrics"]["compact"]  # type: ignore[index]
    return sum(float(compact[language][metric]) for language in ("en", "tl") for metric in ("Recall@10", "nDCG@10"))


def select_nonregressing_checkpoint(history: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not history or int(history[0]["epoch"]) != 0:
        raise ValueError("epoch-zero baseline is required")
    baseline = history[0]
    eligible = [row for row in history if _nonregressing(row, baseline)]
    return max(eligible, key=_selection_score)


def validate_quality_promotion(
    candidate: Mapping[str, object],
    current_student: Mapping[str, object],
    teacher: Mapping[str, object],
) -> None:
    """Require a smaller teacher gap without sacrificing either language.

    The gate is intentionally aggregate and development-only.  It does not
    promise that every individual query improves, but it prevents a combined
    average from hiding an English or Tagalog regression.
    """

    for language in ("en", "tl", "combined"):
        for metric in ("Recall@10", "nDCG@10"):
            proposed = float(candidate[language][metric])  # type: ignore[index]
            baseline = float(current_student[language][metric])  # type: ignore[index]
            ceiling = float(teacher[language][metric])  # type: ignore[index]
            if proposed <= baseline:
                raise ValueError(f"candidate does not improve {language} {metric}")
            if abs(ceiling - proposed) >= abs(ceiling - baseline):
                raise ValueError(f"candidate does not reduce the teacher gap for {language} {metric}")
